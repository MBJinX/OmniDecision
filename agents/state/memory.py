"""Working-memory slot management."""

import hashlib
import re
from typing import Any

from agents.shared.constants import TRUNC_RESULT_SUMMARY
from agents.shared.types import EvidenceState


class MemorySlots:
    """Provides working-memory dedup and storage helpers."""

    def _memory_slot_id(self, tool_name: str, query: str) -> str:
        """Compute a dedup key for a (tool, query) pair."""
        normalized = re.sub(r"\s+", " ", query.lower().strip())
        raw = f"{tool_name}::{normalized}"
        return hashlib.sha1(raw.encode()).hexdigest()[:10]

    def _get_memory_slot(
        self,
        state: EvidenceState,
        *,
        tool_name: str,
        query: str,
    ) -> dict[str, Any] | None:
        """Return the cached full_payload if a matching MemorySlot exists, else None."""
        slot_id = self._memory_slot_id(tool_name, query)
        for slot in state.get("memory_slots") or []:
            if slot.get("slot_id") == slot_id:
                slot["hit_count"] = int(slot.get("hit_count") or 0) + 1
                return dict(slot.get("full_payload") or {})
        return None

    def _store_memory_slot(
        self,
        state: EvidenceState,
        *,
        tool_name: str,
        query: str,
        result_summary: str,
        full_payload: dict[str, Any],
        step_index: int,
    ) -> None:
        """Store a new MemorySlot after a successful tool invocation."""
        slot_id = self._memory_slot_id(tool_name, query)
        # Avoid duplicate entries
        for slot in state.get("memory_slots") or []:
            if slot.get("slot_id") == slot_id:
                return
        slots = list(state.get("memory_slots") or [])
        slots.append({
            "slot_id": slot_id,
            "tool_name": tool_name,
            "normalized_query": re.sub(r"\s+", " ", query.lower().strip()),
            "result_summary": result_summary[:TRUNC_RESULT_SUMMARY],
            "full_payload": full_payload,
            "created_at_step": step_index,
            "hit_count": 0,
        })
        state["memory_slots"] = slots
