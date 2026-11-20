"""Evidence-graph management."""

import json
import re
from typing import Any

import agents.config as config
from agents.shared.constants import TRUNC_GRAPH_NODE_LABEL, TRUNC_GRAPH_SUMMARY
from agents.shared.json_utils import safe_load_json_object
from agents.shared.types import EvidenceState
from agents.shared.utils import call_openai_model_with_tools


class EvidenceGraphMixin:
    """Provides evidence-graph construction and refresh helpers."""

    def _build_evidence_graph_with_llm(
        self,
        state: EvidenceState,
        *,
        question: str,
    ) -> dict[str, Any]:
        """Ask the LLM to build a lightweight evidence graph summary from current atoms.

        Returns an EvidenceGraph-compatible dict. Falls back to a minimal default on failure.
        """
        atoms = list(state.get("evidence_atoms") or [])
        conflicts = list(state.get("conflicts") or [])
        fact_records = list(state.get("fact_bridge_records") or [])

        _default: dict[str, Any] = {
            "nodes": [],
            "edges": [],
            "graph_summary": "",
            "conflict_paths": [],
            "support_paths": [],
            "built_at_step": 0,
        }

        if not atoms:
            return _default

        # Build a compact representation to feed the LLM
        atom_lines = []
        for a in atoms[:12]:  # cap at 12 to keep prompt short
            atom_lines.append(
                f'  - id={a.get("evidence_id","?")} '
                f'tool={a.get("source_tool","?")} '
                f'kind={a.get("evidence_kind","?")} '
                f'conf={a.get("confidence",0):.2f}: '
                f'{str(a.get("text",""))[:120]}'
            )
        conflict_lines = []
        for c in conflicts[:4]:
            conflict_lines.append(
                f'  - {c.get("conflict_type","?")} severity={c.get("severity","?")} '
                f'{c.get("description","")[:100]}'
            )
        fact_lines = []
        for r in fact_records[:4]:
            fact_lines.append(
                f'  - entity={r.get("entity_id","?")} status={r.get("status","?")} '
                f'value={r.get("extracted_value","")}'
            )

        prompt = (
            "You are an evidence graph builder for a video QA system.\n\n"
            "Given the collected evidence atoms and any conflicts, build a lightweight evidence graph summary.\n\n"
            "Output valid JSON only:\n"
            "{\n"
            '  "graph_summary": "\u2264100-word plain-text synthesis of which evidence supports the answer, '
            'which conflicts remain, and what still needs resolution",\n'
            '  "conflict_paths": [["node_id_A", "node_id_B"], ...],\n'
            '  "support_paths": [["node_id_X", "node_id_Y", "answer"], ...],\n'
            '  "estimated_cost_to_continue": 0.0 to 1.0 (cost of the next evidence-gathering step),\n'
            '  "expected_value_of_next_step": 0.0 to 1.0 (expected gain from the next step)\n'
            + "}\n\n"
            f"Question: {question}\n\n"
            "Evidence atoms:\n" + "\n".join(atom_lines) + "\n\n"
            + (("Conflicts:\n" + "\n".join(conflict_lines) + "\n\n") if conflict_lines else "")
            + (("External facts:\n" + "\n".join(fact_lines) + "\n\n") if fact_lines else "")
            + "Output valid JSON only, no explanation outside the JSON."
        )
        try:
            response = call_openai_model_with_tools(
                [{"role": "user", "content": prompt}],
                endpoints=config.AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST,
                model_name=config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME,
                temperature=0.0,
                api_key=config.AOAI_ORCHESTRATOR_API_KEY,
                base_url=config.AOAI_ORCHESTRATOR_BASE_URL,
                max_tokens=300,
                caller="evidence_graph",
            )
            if response is None:
                return _default
            raw = str(response.get("content") or "").strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```[^\n]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw.strip())
            parsed = safe_load_json_object(raw)
            if not isinstance(parsed, dict):
                return _default
            result = dict(_default)
            result["graph_summary"] = str(parsed.get("graph_summary") or "")[:TRUNC_GRAPH_SUMMARY]
            result["conflict_paths"] = parsed.get("conflict_paths") or []
            result["support_paths"] = parsed.get("support_paths") or []
            for fkey in ("estimated_cost_to_continue", "expected_value_of_next_step"):
                raw_val = parsed.get(fkey)
                if raw_val is not None:
                    try:
                        result[fkey] = max(0.0, min(1.0, float(raw_val)))
                    except (ValueError, TypeError):
                        pass
            return result
        except Exception:
            return _default

    def _update_evidence_graph(
        self,
        state: EvidenceState,
        *,
        question: str,
        run_report: dict[str, Any],
    ) -> None:
        """Build or refresh the evidence graph when the relevant state changes."""
        atoms = list(state.get("evidence_atoms") or [])
        if not atoms:
            return

        existing_graph = state.get("evidence_graph") or {}
        existing_sig = existing_graph.get("_state_signature") or ""
        # Build a compact signature capturing all state dimensions that affect graph interpretation
        atom_ids = sorted(str(a.get("evidence_id") or "") for a in atoms)
        conflict_sigs = sorted(
            f"{c.get('conflict_type','')}:{c.get('severity','')}"
            for c in (state.get("conflicts") or [])
        )
        bridge_sigs = sorted(
            f"{r.get('entity_id','')}:{r.get('status','')}:{r.get('extracted_value','')}"
            for r in (state.get("fact_bridge_records") or [])
        )
        current_sig = json.dumps(
            {
                "atoms": atom_ids,
                "conflicts": conflict_sigs,
                "bridges": bridge_sigs,
                "unresolved": sorted(state.get("unresolved_questions") or []),
                "sufficiency": state.get("sufficiency_status") or "",
            },
            sort_keys=True,
        )
        if current_sig == existing_sig and existing_graph.get("graph_summary"):
            return

        step_index = len(run_report.get("tool_sequence") or [])
        graph = self._build_evidence_graph_with_llm(state, question=question)
        graph["built_at_step"] = step_index

        # Build nodes list from current atoms for completeness
        graph["nodes"] = [
            {
                "node_id": a.get("evidence_id", ""),
                "label": str(a.get("text", ""))[:TRUNC_GRAPH_NODE_LABEL],
                "source_tool": a.get("source_tool", ""),
                "confidence": a.get("confidence", 0.0),
            }
            for a in atoms
        ]
        graph["_state_signature"] = current_sig
        state["evidence_graph"] = graph
