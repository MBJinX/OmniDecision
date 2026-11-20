"""OmniVQA agent runtime package."""

__all__ = ["CoreAgent"]


def __getattr__(name: str):
    if name == "CoreAgent":
        from agents.core_agent import CoreAgent

        return CoreAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
