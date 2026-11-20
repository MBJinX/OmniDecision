"""Entity card extraction and fact-bridge population."""

import hashlib
import re
from typing import Any

import agents.config as config
from agents.planning.question_profile import _evidence_needs_require_entity_bridge
from agents.shared.constants import (
    TRUNC_SOURCE_TEXT,
    TRUNC_RESULT_SUMMARY,
    TRUNC_WEB_EVIDENCE_TO_LLM,
)
from agents.shared.json_utils import safe_load_json_object
from agents.shared.types import EntityCard, EvidenceState, FactBridgeRecord, NormalizedToolObservation
from agents.shared.utils import LLMCallFailure, call_openai_model_with_tools


class EntityBridge:
    """Provides entity-card extraction, query building, and fact-bridge refresh."""

    def _entity_card_id(self, canonical_name: str) -> str:
        """Compute stable entity_id as sha1(canonical_name.lower())[:10]."""
        return hashlib.sha1(canonical_name.lower().encode()).hexdigest()[:10]

    def _extract_entity_cards_with_llm(
        self,
        state: EvidenceState,
        *,
        question: str,
    ) -> list[EntityCard]:
        """Use one LLM call to identify entities needing external knowledge lookup.

        Takes the current question and the most recent grounding evidence atoms,
        asks the LLM to output a JSON array of entity descriptors, and converts
        them into EntityCard objects with lookup_status='pending'.
        Returns an empty list on any failure so the caller stays safe.
        """
        evidence_atoms = list(state.get("evidence_atoms") or [])
        recent_texts = [
            str(a.get("text") or "")[:300]
            for a in evidence_atoms[-3:]
            if a.get("text")
        ]
        evidence_text = "\n".join(f"- {t}" for t in recent_texts) if recent_texts else "(none)"

        unresolved = list(state.get("unresolved_questions") or [])
        unresolved_text = ", ".join(unresolved) if unresolved else "(none)"

        extraction_prompt = (
            "You are an entity extractor for video QA. Given a question and grounding "
            "observations, identify entities that require external knowledge lookup.\n\n"
            f"Question: {question}\n\n"
            f"Recent grounding observations:\n{evidence_text}\n\n"
            f"Unresolved questions: {unresolved_text}\n\n"
            "Output a JSON array of entities needing external lookup. Each entity:\n"
            "{\n"
            '  "canonical_name": "exact name as it appears or should be searched",\n'
            '  "entity_type": "product|person|song|event|unknown",\n'
            '  "lookup_slot": "what attribute to look up, e.g. battery_capacity_mah, founding_year",\n'
            '  "source_text": "short quote from observations that mentions this entity"\n'
            "}\n\n"
            "Rules:\n"
            "- Only include entities where external knowledge (not visible in video) is needed\n"
            "- Maximum 3 entities\n"
            "- Return [] if no external lookup is needed\n"
            "- Output valid JSON only, no explanation"
        )

        try:
            response = call_openai_model_with_tools(
                [
                    {"role": "user", "content": extraction_prompt},
                ],
                endpoints=config.AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST,
                model_name=config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME,
                temperature=0.0,
                api_key=config.AOAI_ORCHESTRATOR_API_KEY,
                base_url=config.AOAI_ORCHESTRATOR_BASE_URL,
                max_tokens=300,
                caller="entity_bridge",
            )
        except LLMCallFailure:
            # No entities extracted is a valid outcome; a refused request must
            # not be re-issued from this layer.
            return []
        if response is None:
            return []

        raw_content = str(response.get("content") or "").strip()
        # Strip markdown code fences if present
        if raw_content.startswith("```"):
            raw_content = re.sub(r"^```[^\n]*\n?", "", raw_content)
            raw_content = re.sub(r"\n?```$", "", raw_content.strip())

        parsed = safe_load_json_object(raw_content)
        if not isinstance(parsed, list):
            # Try to find a JSON array embedded in the text
            match = re.search(r"\[[\s\S]*\]", raw_content)
            if match:
                parsed = safe_load_json_object(match.group(0))
            if not isinstance(parsed, list):
                return []

        cards: list[EntityCard] = []
        for item in parsed[:3]:
            if not isinstance(item, dict):
                continue
            canonical = str(item.get("canonical_name") or "").strip()
            if not canonical:
                continue
            card: EntityCard = {
                "entity_id": self._entity_card_id(canonical),
                "canonical_name": canonical,
                "aliases": [],
                "entity_type": str(item.get("entity_type") or "unknown"),
                "source_tool": "llm_extractor",
                "source_text": str(item.get("source_text") or "")[:TRUNC_SOURCE_TEXT],
                "lookup_status": "pending",
                "lookup_slot": str(item.get("lookup_slot") or ""),
            }
            cards.append(card)
        return cards

    def _merge_entity_cards(
        self,
        existing: list[EntityCard],
        new_cards: list[EntityCard],
    ) -> list[EntityCard]:
        """Merge new entity cards into existing list, skipping duplicates by entity_id."""
        existing_ids = {str(c.get("entity_id") or "") for c in existing}
        merged = list(existing)
        for card in new_cards:
            eid = str(card.get("entity_id") or "")
            if eid and eid not in existing_ids:
                existing_ids.add(eid)
                merged.append(card)
        return merged

    def _build_entity_query(
        self,
        card: EntityCard,
        *,
        question: str,
    ) -> tuple[str, str]:
        """Build a neutral external-lookup query for an entity card."""
        canonical = str(card.get("canonical_name") or "")
        slot = str(card.get("lookup_slot") or "").lower()
        query = f"{canonical} {slot} official specification"
        query_type = "fact_completion"
        return query.strip(), query_type

    def _extract_fact_value_from_web_profile(
        self,
        web_profile: dict,
        question: str,
        entity_card: EntityCard | None = None,
    ) -> str:
        """Extract a key value from a web evidence profile with one structured LLM pass."""
        bundle = list(web_profile.get("evidence_bundle") or [])
        haystack = " ".join(str(item.get("text") or "") for item in bundle[:3])
        result = self._extract_fact_value_with_llm(
            haystack, question, entity_card=entity_card
        )
        return str(result.get("extracted_value") or "").strip()

    def _try_populate_fact_bridge(
        self,
        state: EvidenceState,
        normalized_obs: NormalizedToolObservation,
        *,
        question: str,
        step_index: int,
    ) -> bool:
        """After a web_search_tool call, write a FactBridgeRecord and update EntityCard.

        Only processes web observations with status=ok.
        Finds the most recently in_flight EntityCard and extracts the fact value.

        Returns True if bridge state was modified (new record added, entity card status changed).
        Returns False if no modification occurred (non-web obs, non-ok status, no in_flight card).
        """
        if normalized_obs.get("source_modality") != "web":
            return False
        if normalized_obs.get("status") != "ok":
            return False

        # Find the most recent in_flight entity card
        entity_cards = list(state.get("entity_cards") or [])
        in_flight_card: EntityCard | None = None
        for card in reversed(entity_cards):
            if card.get("lookup_status") == "in_flight":
                in_flight_card = card
                break

        if in_flight_card is None:
            return False

        # Build web profile from the raw payload
        raw_payload = dict(normalized_obs.get("raw_payload") or {})
        if not raw_payload:
            # Reconstruct minimal payload from normalized obs for profile building
            raw_payload = {
                "status": normalized_obs.get("status"),
                "source_modality": "web",
                "summary": normalized_obs.get("summary"),
                "evidence": [
                    {"text": atom.get("text"), "score": atom.get("confidence"), "metadata": {}}
                    for atom in (normalized_obs.get("evidence_atoms") or [])[:3]
                ],
            }
        web_profile = self._build_web_evidence_profile(raw_payload)

        extracted_value = self._extract_fact_value_from_web_profile(
            web_profile, question, entity_card=in_flight_card
        )

        entity_id = str(in_flight_card.get("entity_id") or "")
        pending_query = str(in_flight_card.get("_pending_query") or "")  # type: ignore[misc]
        pending_query_type = str(in_flight_card.get("_pending_query_type") or "background_knowledge")  # type: ignore[misc]

        # Prefer the actual query used by web_search_tool (from evidence metadata)
        # over the pre-built _pending_query, so FactBridgeRecord.query_used is accurate.
        actual_query = pending_query
        for ev_item in (web_profile.get("evidence_bundle") or []):
            q = str(ev_item.get("query_used") or "").strip()
            if q:
                actual_query = q
                break

        record_id_src = entity_id + actual_query
        record_id = hashlib.sha1(record_id_src.encode()).hexdigest()[:10]

        # Extract source URLs from evidence bundle
        source_urls = [
            str(item.get("url") or "")
            for item in (web_profile.get("evidence_bundle") or [])[:3]
            if item.get("url")
        ]

        bridge_status = "ok" if extracted_value else "no_match"
        raw_summary = str(normalized_obs.get("summary") or "")[:TRUNC_RESULT_SUMMARY]

        record: FactBridgeRecord = {
            "record_id": record_id,
            "entity_id": entity_id,
            "query_type": pending_query_type,
            "query_used": actual_query,
            "raw_result_summary": raw_summary,
            "extracted_value": extracted_value,
            "confidence": float(normalized_obs.get("confidence") or 0.6),
            "source_urls": source_urls,
            "status": bridge_status,
            "created_at_step": step_index,
        }

        existing_records = list(state.get("fact_bridge_records") or [])
        existing_records.append(record)
        state["fact_bridge_records"] = existing_records[-8:]

        # Update EntityCard
        in_flight_card["lookup_status"] = "resolved" if extracted_value else "failed"
        if extracted_value:
            in_flight_card["resolved_fact"] = extracted_value
            in_flight_card["resolved_at_step"] = step_index

        return True

    def _refresh_entity_bridge(
        self,
        state: EvidenceState,
        *,
        question: str,
        run_report: dict[str, Any] | None = None,
    ) -> None:
        """Refresh entity-card extraction state and bridge-status fields."""
        bridge_status = dict(state.get("bridge_status") or {})
        constraints = self._build_question_constraint_profile(question, run_report)

        if _evidence_needs_require_entity_bridge(constraints):
            existing_cards = list(state.get("entity_cards") or [])

            should_extract = (
                bridge_status.get("has_video_bridge")
                and not any(c.get("lookup_status") == "resolved" for c in existing_cards)
                and not any(c.get("lookup_status") == "in_flight" for c in existing_cards)
                and not existing_cards  # only extract once (no cards yet)
            )
            if should_extract:
                new_cards = self._extract_entity_cards_with_llm(state, question=question)
                state["entity_cards"] = self._merge_entity_cards(existing_cards, new_cards)

            # Advance pending cards to in_flight with a prepared query
            for card in state.get("entity_cards") or []:
                if card.get("lookup_status") == "pending":
                    query, query_type = self._build_entity_query(card, question=question)
                    card["_pending_query"] = query  # type: ignore[typeddict-unknown-key]
                    card["_pending_query_type"] = query_type  # type: ignore[typeddict-unknown-key]
                    card["lookup_status"] = "in_flight"

        all_cards = list(state.get("entity_cards") or [])
        bridge_status["entity_cards_extracted"] = bool(all_cards)
        bridge_status["fact_bridge_populated"] = bool(
            any(r.get("status") == "ok" for r in (state.get("fact_bridge_records") or []))
        )
        bridge_status["entity_lookup_pending"] = bool(
            any(c.get("lookup_status") in {"pending", "in_flight"} for c in all_cards)
        )

        # Write pending_entity_lookups: canonical names of unresolved cards (for digest / prompt)
        state["pending_entity_lookups"] = [
            str(c.get("canonical_name") or "")
            for c in all_cards
            if c.get("lookup_status") in {"pending", "in_flight"}
            and c.get("canonical_name")
        ]

        resolved_records = [
            r for r in (state.get("fact_bridge_records") or [])
            if r.get("status") == "ok" and r.get("extracted_value")
        ]
        resolved_cards = [
            c for c in all_cards
            if c.get("lookup_status") == "resolved"
            and (
                self._compact_text(c.get("resolved_fact"))
                or self._compact_text(c.get("canonical_name"))
            )
        ]
        bridge_status["entity_bridge_ready"] = (
            not _evidence_needs_require_entity_bridge(constraints)
            or bool(bridge_status.get("has_external_exact_fact"))
            or bool(resolved_records)
            or bool(resolved_cards)
            or (
                bool(bridge_status.get("entity_bridge_ready"))
                and not bridge_status.get("entity_lookup_pending")
            )
        )

        state["bridge_status"] = bridge_status

    def _extract_fact_value_with_llm(
        self,
        web_evidence_text: str,
        question: str,
        entity_card: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Use LLM to extract the target fact value from web evidence text.

        Returns {"extracted_value": str, "confidence": float, "rationale": str}.
        Falls back to empty extracted_value on failure.
        """
        _default = {"extracted_value": "", "confidence": 0.0, "rationale": "llm_failed"}

        entity_hint = ""
        if entity_card:
            name = str(entity_card.get("canonical_name") or "")
            slot = str(entity_card.get("lookup_slot") or "")
            if name or slot:
                entity_hint = f"Entity: {name}\nAttribute to extract: {slot}\n"

        prompt = (
            "You are a fact extraction assistant for a video QA system.\n\n"
            "Extract the specific factual value that answers the question from the web evidence below.\n\n"
            "Output valid JSON only:\n"
            '{"extracted_value": "the extracted value as a string, or empty string if not found", '
            '"confidence": 0.0 to 1.0, '
            '"rationale": "one sentence explaining what you found or why it was not found"}\n\n'
            f"Question: {question}\n"
            f"{entity_hint}"
            f"Web evidence:\n{web_evidence_text[:TRUNC_WEB_EVIDENCE_TO_LLM]}\n\n"
            "Output valid JSON only, no explanation outside the JSON."
        )
        try:
            response = call_openai_model_with_tools(
                [{"role": "user", "content": prompt}],
                endpoints=config.AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST,
                model_name=config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME,
                temperature=0.0,
                api_key=config.AOAI_ORCHESTRATOR_API_KEY,
                base_url=config.AOAI_ORCHESTRATOR_BASE_URL,
                max_tokens=150,
                caller="entity_bridge",
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
            return {
                "extracted_value": str(parsed.get("extracted_value") or ""),
                "confidence": float(parsed.get("confidence") or 0.0),
                "rationale": str(parsed.get("rationale") or ""),
            }
        except Exception:
            return _default
