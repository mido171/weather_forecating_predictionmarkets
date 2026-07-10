import shutil

from hkg_tmax.config import load_yaml
from hkg_tmax.experiments import (
    EXPERIMENT_CONTROL_DIR,
    EXPERIMENT_TEMPLATE_DIR,
    create_experiment,
    generate_index,
)


def test_create_experiment_from_template(repo_root, tmp_path) -> None:
    root = tmp_path / "repo"
    (root / "experiments").mkdir(parents=True)
    control_root = root / "experiments" / EXPERIMENT_CONTROL_DIR
    control_root.mkdir(parents=True)
    template = root / "experiments" / EXPERIMENT_TEMPLATE_DIR
    template.parent.mkdir(parents=True)
    shutil.copytree(repo_root / "experiments" / EXPERIMENT_TEMPLATE_DIR, template)
    (control_root / "registry.yaml").write_text(
        "registry_version: 1\nnext_id: 1\nexperiments: []\n",
        encoding="utf-8",
    )

    destination = create_experiment(root, "Test target parity")
    assert destination.name.startswith("EXP-0001-test-target-parity")
    readme = (destination / "README.md").read_text()
    assert "EXP-0001" in readme
    assert "Test target parity" in readme
    assert "## Question and hypothesis" in readme
    assert "## Evidence map" in readme
    assert [path.name for path in destination.glob("*.md")] == ["README.md"]

    registry = load_yaml(control_root / "registry.yaml")
    assert registry["next_id"] == 2
    assert registry["experiments"][0]["id"] == "EXP-0001"

    index = generate_index(root)
    assert "EXP-0001" in index.read_text()


def test_generate_index_supports_historical_state_schema(tmp_path) -> None:
    root = tmp_path / "repo"
    experiment = root / "experiments" / "campaigns" / "hkg-tmax" / "0009_smoke"
    experiment.mkdir(parents=True)
    (experiment / "README.md").write_text("# Smoke\n", encoding="utf-8")
    (experiment / "STATUS.yaml").write_text(
        "state: COMPLETE\n"
        "primary_conclusion: persistence passed\n"
        "leakage: pass\n"
        "reproducible: false\n",
        encoding="utf-8",
    )

    index = generate_index(root).read_text(encoding="utf-8")

    assert "0009_smoke" in index
    assert "COMPLETE" in index
    assert "persistence passed" in index
    assert "pass" in index
    assert "False" in index
