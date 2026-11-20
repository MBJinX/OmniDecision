from __future__ import annotations


def build_env_overrides(
    planner_model: str | None = None,
    perception_model: str | None = None,
) -> dict[str, str]:
    """Map non-empty model fields to subprocess environment overrides."""
    overrides: dict[str, str] = {}
    mapping = {
        "PLANNER_MODEL": planner_model,
        "PERCEPTION_MODEL": perception_model,
    }
    for env_name, value in mapping.items():
        if value is None:
            continue
        stripped = value.strip()
        if stripped:
            overrides[env_name] = stripped
    return overrides
