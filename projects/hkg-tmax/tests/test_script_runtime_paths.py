from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
MUTABLE_ROOT_NAMES = {"artifacts", "data", "models", "predictions", "reports", "tmp"}


def _load_registry_builder() -> ModuleType:
    path = SCRIPTS_ROOT / "build_script_registry.py"
    spec = importlib.util.spec_from_file_location("hkg_script_registry_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _first_literal_segment(node: ast.expr) -> str | None:
    value: str | None = None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        value = node.value
    elif isinstance(node, ast.JoinedStr) and node.values:
        first = node.values[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            value = first.value
    if value is None:
        return None
    return value.replace("\\", "/").lstrip("./").split("/", maxsplit=1)[0]


def test_top_level_scripts_do_not_anchor_mutable_roots_in_repo() -> None:
    violations: list[str] = []
    for path in sorted(SCRIPTS_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Div)
                and isinstance(node.left, ast.Name)
                and node.left.id == "REPO_ROOT"
            ):
                continue
            first_segment = _first_literal_segment(node.right)
            if first_segment in MUTABLE_ROOT_NAMES:
                violations.append(f"{path.name}:{node.lineno}:{first_segment}")
    assert not violations, "repo-local mutable script paths found: " + ", ".join(violations)


def test_external_path_sweep_has_explicit_lifecycle_and_path_layer() -> None:
    registry_builder = _load_registry_builder()
    overrides = registry_builder.LIFECYCLE_OVERRIDES

    assert len(overrides) == 36
    assert list(overrides.values()).count("retained_reproduction") == 26
    assert list(overrides.values()).count("active_operator") == 9
    assert list(overrides.values()).count("active_research") == 1

    missing: list[str] = []
    without_path_layer: list[str] = []
    for name in overrides:
        path = SCRIPTS_ROOT / name
        if not path.is_file():
            missing.append(name)
            continue
        if "ProjectPaths" not in path.read_text(encoding="utf-8"):
            without_path_layer.append(name)

    assert not missing, "classified scripts are missing: " + ", ".join(missing)
    assert not without_path_layer, "classified scripts bypass ProjectPaths: " + ", ".join(
        without_path_layer
    )
