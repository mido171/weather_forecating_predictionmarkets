import shutil

from hkg_tmax.config import load_yaml
from hkg_tmax.experiments import create_experiment, generate_index


def test_create_experiment_from_template(repo_root, tmp_path) -> None:
    root = tmp_path / "repo"
    (root / "experiments").mkdir(parents=True)
    shutil.copytree(
        repo_root / "experiments" / "_template",
        root / "experiments" / "_template",
    )
    (root / "experiments" / "registry.yaml").write_text(
        "registry_version: 1\nnext_id: 1\nexperiments: []\n",
        encoding="utf-8",
    )

    destination = create_experiment(root, "Test target parity")
    assert destination.name.startswith("EXP-0001-test-target-parity")
    readme = (destination / "README.md").read_text()
    assert "EXP-0001" in readme
    assert "Test target parity" in readme

    registry = load_yaml(root / "experiments" / "registry.yaml")
    assert registry["next_id"] == 2
    assert registry["experiments"][0]["id"] == "EXP-0001"

    index = generate_index(root)
    assert "EXP-0001" in index.read_text()
