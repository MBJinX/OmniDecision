from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AssetRow(BaseModel):
    path: str = Field(..., min_length=1)
    type: Literal["video", "image"]


class RunCreateRequest(BaseModel):
    question: str = Field(..., min_length=1)
    assets: list[AssetRow] = Field(..., min_length=1)
    max_iterations: int = Field(default=15, ge=1, le=100)
    planner_model: str = ""
    perception_model: str = ""


class StepTraceLoadRequest(BaseModel):
    path: str


class RunStatus(BaseModel):
    run_id: str
    status: Literal["created", "running", "finished", "failed", "cancelled"]
    run_dir: str | None = None
    created_at: str


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
