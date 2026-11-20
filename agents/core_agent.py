"""Compatibility module alias for the core runtime."""

from __future__ import annotations

import sys

from agents.core import agent as _core_agent_module


sys.modules[__name__] = _core_agent_module
