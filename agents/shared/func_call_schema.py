# Adapted from https://github.com/peterroelants/annotated-docs (Apache-2.0).
# Modified for runtime tool schemas. See licenses/annotated-docs-Apache-2.0.txt.

import inspect
from collections.abc import Callable
import copy
from typing import Any, Final, TypeVar

try:
    from typing import Required, TypedDict
except ImportError:  # Python 3.10 compatibility
    from typing_extensions import Required, TypedDict

import pydantic
import pydantic.json_schema

RETURNS_KEY: Final[str] = "returns"

T = TypeVar("T")


class FunctionJSONSchema(TypedDict, total=False):
    name: Required[str]
    description: str
    parameters: dict[str, Any]


def as_json_schema(func: Callable) -> FunctionJSONSchema:
    """
    Return a JSON schema for the given function.
    """
    parameters_schema = get_parameters_schema(func)
    description = ""
    if func.__doc__:
        description = inspect.cleandoc(func.__doc__).strip()
    schema_dct: FunctionJSONSchema = {
        "name": func.__name__,
        "description": description,
        "parameters": parameters_schema,
    }
    return schema_dct


def doc(description) -> Any:
    """Annotate a variable with a description."""
    return pydantic.Field(description=description)


def get_parameters_schema(func: Callable) -> dict[str, Any]:
    """Return a JSON schema for the parameters of the given function."""
    import sys
    parameter_model = get_parameter_model(func)
    # Collect namespace from the module where func is defined so that
    # forward references introduced by `from __future__ import annotations`
    # can be resolved before generating the JSON schema.
    func_module = sys.modules.get(func.__module__, None)
    ns = vars(func_module) if func_module is not None else {}
    try:
        parameter_model.model_rebuild(_types_namespace=ns)
    except Exception:
        pass
    schema = parameter_model.model_json_schema(
        schema_generator=GenerateJsonSchemaNoTitle,
        mode="validation",
    )
    return _sanitize_function_schema(_inline_local_refs(schema))


def get_parameter_model(func: Callable) -> pydantic.BaseModel:
    """
    Return a Pydantic model for the parameters of the given function.
    """
    field_definitions: dict[str, tuple[Any, Any]] = {}
    for name, obj in inspect.signature(func).parameters.items():
        if obj.annotation == inspect.Parameter.empty:
            raise ValueError(
                f"`{func.__name__}` parameter `{name!s}` has no annotation, please provide an notation to be able to generate the function specification."
            )
        if obj.default == inspect.Parameter.empty:
            field_definitions[name] = (obj.annotation, pydantic.Field(...))
        else:
            field_definitions[name] = (obj.annotation, obj.default)
    _model_name = f"_ParamModel_{func.__name__}"
    return pydantic.create_model(_model_name, **field_definitions)  # type: ignore


class GenerateJsonSchemaNoTitle(pydantic.json_schema.GenerateJsonSchema):
    def generate(
        self, schema, mode="validation"
    ) -> pydantic.json_schema.JsonSchemaValue:
        json_schema = super().generate(schema, mode=mode)
        if "title" in json_schema:
            del json_schema["title"]
        return json_schema

    def get_schema_from_definitions(
        self, json_ref
    ) -> pydantic.json_schema.JsonSchemaValue | None:
        json_schema = super().get_schema_from_definitions(json_ref)
        if json_schema and "title" in json_schema:
            del json_schema["title"]
        return json_schema

    def field_title_should_be_set(self, schema) -> bool:
        return False


def _resolve_local_ref(ref: str, definitions: dict[str, Any]) -> dict[str, Any] | None:
    prefix = "#/$defs/"
    if not ref.startswith(prefix):
        return None
    key = ref[len(prefix):]
    definition = definitions.get(key)
    if not isinstance(definition, dict):
        return None
    return copy.deepcopy(definition)


def _inline_local_refs(schema: dict[str, Any]) -> dict[str, Any]:
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        return schema

    def _walk(node: Any) -> Any:
        if isinstance(node, list):
            return [_walk(item) for item in node]
        if not isinstance(node, dict):
            return node

        if "$ref" in node and isinstance(node["$ref"], str):
            resolved = _resolve_local_ref(node["$ref"], definitions)
            if resolved is not None:
                merged = resolved
                for key, value in node.items():
                    if key == "$ref":
                        continue
                    merged[key] = _walk(value)
                return _walk(merged)

        return {key: _walk(value) for key, value in node.items() if key != "$defs"}

    return _walk(schema)


def _sanitize_function_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Reduce the schema to a conservative subset accepted by multiple providers."""

    def _walk(node: Any) -> Any:
        if isinstance(node, list):
            return [_walk(item) for item in node]
        if not isinstance(node, dict):
            return node

        sanitized: dict[str, Any] = {}
        for key, value in node.items():
            if key in {"$defs", "$ref", "default", "title"}:
                continue
            if key == "const":
                sanitized["enum"] = [value]
                continue
            sanitized[key] = _walk(value)
        return sanitized

    return _walk(schema)
