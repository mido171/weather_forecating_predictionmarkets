import shutil
from pathlib import Path

import pytest

import hkg_tmax.experiment_transaction as experiment_transaction
from hkg_tmax.cli import build_parser, main
from hkg_tmax.config import dump_yaml, load_yaml
from hkg_tmax.experiments import (
    CANONICAL_CAMPAIGNS,
    EXPERIMENT_CONTROL_DIR,
    EXPERIMENT_TEMPLATE_DIR,
    ExperimentError,
    create_experiment,
    generate_index,
)
from hkg_tmax.validation import ValidationError, validate_experiment_registry


def _repo_with_experiment_controls(repo_root, tmp_path):
    root = tmp_path / "repo"
    (root / "experiments").mkdir(parents=True)
    control_root = root / "experiments" / EXPERIMENT_CONTROL_DIR
    control_root.mkdir(parents=True)
    template = root / "experiments" / EXPERIMENT_TEMPLATE_DIR
    template.parent.mkdir(parents=True)
    shutil.copytree(repo_root / "experiments" / EXPERIMENT_TEMPLATE_DIR, template)
    (control_root / "registry.yaml").write_text(
        "registry_version: 2\nnext_id: 1\nexperiments: []\n",
        encoding="utf-8",
    )
    return root, control_root


def _valid_prepared_journal(root: Path, control_root: Path) -> dict[str, object]:
    original_registry = load_yaml(control_root / "registry.yaml")
    token = "b" * 32
    title = "Hostile journal sentinel"
    directory_name = "EXP-0001-hostile-journal-sentinel"
    updated_registry = {
        "registry_version": 2,
        "next_id": 2,
        "experiments": [
            {
                "id": "EXP-0001",
                "title": title,
                "campaign": "general",
                "directory": f"campaigns/general/{directory_name}",
                "created_at_utc": "2026-07-10T00:00:00Z",
                "status": "PLANNED",
            }
        ],
    }
    return {
        "schema_version": 1,
        "phase": "prepared",
        "transaction_token": token,
        "experiment_id": "EXP-0001",
        "destination": f"experiments/campaigns/general/{directory_name}",
        "staging": f"var/tmp/experiment-creation/EXP-0001-{token}",
        "campaign_root": "experiments/campaigns/general",
        "campaigns_root_preexisting": False,
        "campaign_root_preexisting": False,
        "campaign_readme_preexisting": False,
        "original_registry": original_registry,
        "updated_registry": updated_registry,
        "original_index": None,
    }


def _write_transaction_journal(root: Path, journal: dict[str, object]) -> Path:
    path = root / "var" / "run" / "experiment-creation-transaction.json"
    experiment_transaction._write_json_atomic(path, journal)
    return path


def test_atomic_experiment_residue_is_ignored(repo_root) -> None:
    ignore_text = (repo_root / ".gitignore").read_text(encoding="utf-8")

    assert "/.EXPERIMENT_INDEX.md.*.tmp" in ignore_text
    assert "/experiments/registry/.registry.yaml.*.tmp" in ignore_text
    assert "/experiments/campaigns/**/.experiment-transaction.json" in ignore_text
    assert "/experiments/campaigns/**/.experiment-transaction.json.*.pending" in ignore_text
    assert "/experiments/campaigns/**/.experiment-campaign-transaction.json" in ignore_text
    assert (
        "/experiments/campaigns/**/.experiment-campaign-transaction.json.*.pending" in ignore_text
    )


def test_create_experiment_from_template(repo_root, tmp_path) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    title = 'Compare "elite\'s": baseline \\ path #1 – 香港'

    destination = create_experiment(root, title, campaign="hkg-tmax")
    assert destination.parent.name == "hkg-tmax"
    assert destination.name.startswith("EXP-0001-compare-elite-s-baseline-path-1")
    readme = (destination / "README.md").read_text(encoding="utf-8")
    assert "EXP-0001" in readme
    assert title in readme
    assert "## Question and hypothesis" in readme
    assert "## Evidence map" in readme
    assert [path.name for path in destination.glob("*.md")] == ["README.md"]

    registry = load_yaml(control_root / "registry.yaml")
    assert registry["next_id"] == 2
    assert registry["experiments"][0]["id"] == "EXP-0001"
    assert registry["experiments"][0]["campaign"] == "hkg-tmax"
    assert registry["experiments"][0]["directory"].startswith("campaigns/hkg-tmax/")
    assert load_yaml(destination / "STATUS.yaml")["title"] == title
    assert (destination.parent / "README.md").is_file()
    assert not experiment_transaction._transaction_marker_path(destination).exists()
    assert not experiment_transaction._campaign_transaction_marker_path(destination.parent).exists()
    assert validate_experiment_registry(root) == ["experiment registry: 1 governed entries valid"]

    index = root / "EXPERIMENT_INDEX.md"
    assert "EXP-0001" in index.read_text(encoding="utf-8")
    assert "experiments/campaigns/hkg-tmax/" in index.read_text(encoding="utf-8")


def test_general_campaign_must_be_selected_and_gets_its_own_readme(repo_root, tmp_path) -> None:
    root, _control_root = _repo_with_experiment_controls(repo_root, tmp_path)

    destination = create_experiment(root, "Cross-cutting hypothesis", campaign="general")

    assert destination.parent.name == "general"
    assert (
        (destination.parent / "README.md")
        .read_text(encoding="utf-8")
        .startswith("# General experiment campaign")
    )
    assert validate_experiment_registry(root) == ["experiment registry: 1 governed entries valid"]


@pytest.mark.parametrize(
    "title",
    [
        "line one\nline two",
        "tab\ttitle",
        "line\u2028separator",
        "paragraph\u2029separator",
        "x" * 161,
    ],
)
def test_create_experiment_rejects_unsafe_titles(repo_root, tmp_path, title: str) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)

    with pytest.raises(ExperimentError, match="title"):
        create_experiment(root, title, campaign="general")

    assert load_yaml(control_root / "registry.yaml")["experiments"] == []
    assert not (root / "experiments" / "campaigns").exists()


@pytest.mark.parametrize("campaign", ["../hkg-tmax", "HKG-TMAX", "unknown", "C:\\temp"])
def test_create_experiment_rejects_unapproved_campaigns(repo_root, tmp_path, campaign: str) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)

    with pytest.raises(ExperimentError, match="Unknown experiment campaign"):
        create_experiment(root, "Unsafe campaign", campaign=campaign)

    registry = load_yaml(control_root / "registry.yaml")
    assert registry["next_id"] == 1
    assert registry["experiments"] == []
    assert not (root / "experiments" / "campaigns").exists()


def test_every_allowlisted_campaign_can_receive_a_governed_experiment(repo_root, tmp_path) -> None:
    root, _control_root = _repo_with_experiment_controls(repo_root, tmp_path)

    destinations = [
        create_experiment(root, f"Hypothesis {index}", campaign=campaign)
        for index, campaign in enumerate(CANONICAL_CAMPAIGNS, start=1)
    ]

    assert [destination.parent.name for destination in destinations] == list(CANONICAL_CAMPAIGNS)
    assert validate_experiment_registry(root) == [
        f"experiment registry: {len(CANONICAL_CAMPAIGNS)} governed entries valid"
    ]


def test_creation_fails_closed_when_next_id_is_stale(repo_root, tmp_path) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    create_experiment(root, "First hypothesis", campaign="hkg-tmax")
    registry_path = control_root / "registry.yaml"
    registry = load_yaml(registry_path)
    registry["next_id"] = 1
    dump_yaml(registry_path, registry)

    with pytest.raises(ExperimentError, match="next_id must exceed"):
        create_experiment(root, "Second hypothesis", campaign="probability")

    assert [entry["id"] for entry in load_yaml(registry_path)["experiments"]] == ["EXP-0001"]
    assert not (root / "experiments" / "campaigns" / "probability").exists()


def test_creation_renders_in_ignored_runtime_staging(repo_root, tmp_path, monkeypatch) -> None:
    root, _control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    real_copytree = shutil.copytree
    copied_to = []

    def record_copy(source, destination, *args, **kwargs):
        copied_to.append(destination)
        return real_copytree(source, destination, *args, **kwargs)

    monkeypatch.setattr(experiment_transaction.shutil, "copytree", record_copy)
    destination = create_experiment(root, "Atomic staging", campaign="general")

    assert copied_to
    assert all(
        Path(path).is_relative_to(root / "var" / "tmp" / "experiment-creation")
        for path in copied_to
    )
    assert destination.is_dir()
    assert not copied_to[0].exists()


def test_creation_rejects_reparse_template_child_before_copy(
    repo_root,
    tmp_path,
    monkeypatch,
) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    linked_child = root / "experiments" / EXPERIMENT_TEMPLATE_DIR / "linked-tree"
    linked_child.mkdir()
    real_is_reparse = experiment_transaction._is_reparse_path
    copy_called = False

    def classify_reparse(path: Path) -> bool:
        if Path(path) == linked_child:
            return True
        return real_is_reparse(Path(path))

    def reject_copy(*_args, **_kwargs) -> None:
        nonlocal copy_called
        copy_called = True
        raise AssertionError("copy must not run after a reparse preflight failure")

    monkeypatch.setattr(experiment_transaction, "_is_reparse_path", classify_reparse)
    monkeypatch.setattr(experiment_transaction.shutil, "copytree", reject_copy)

    with pytest.raises(ExperimentError, match="symlink or reparse point"):
        create_experiment(root, "Unsafe template", campaign="general")

    assert copy_called is False
    assert load_yaml(control_root / "registry.yaml")["experiments"] == []
    assert not (root / "experiments" / "campaigns").exists()


def test_creation_rolls_back_when_template_rendering_fails(
    repo_root, tmp_path, monkeypatch
) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)

    def fail_render(_path, _values) -> None:
        raise OSError("injected render failure")

    monkeypatch.setattr(experiment_transaction, "_replace_placeholders", fail_render)
    with pytest.raises(ExperimentError, match="injected render failure"):
        create_experiment(root, "Rollback rendering", campaign="general")

    assert load_yaml(control_root / "registry.yaml") == {
        "registry_version": 2,
        "next_id": 1,
        "experiments": [],
    }
    assert not (root / "experiments" / "campaigns").exists()
    assert not (root / "EXPERIMENT_INDEX.md").exists()


def test_creation_removes_partial_staging_when_copy_fails(repo_root, tmp_path, monkeypatch) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)

    def fail_copy(_source, destination, *_args, **_kwargs):
        (destination / "partial.txt").write_text("partial", encoding="utf-8")
        raise OSError("injected copy failure")

    monkeypatch.setattr(experiment_transaction.shutil, "copytree", fail_copy)
    with pytest.raises(ExperimentError, match="injected copy failure"):
        create_experiment(root, "Rollback copy", campaign="general")

    assert load_yaml(control_root / "registry.yaml")["experiments"] == []
    assert not (root / "experiments" / "campaigns").exists()
    assert not (root / "var" / "tmp").exists()


def test_promotion_collision_preserves_unowned_destination(
    repo_root,
    tmp_path,
    monkeypatch,
) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    real_replace = experiment_transaction.os.replace
    collided_destination: Path | None = None

    def collide_during_promotion(source, destination) -> None:
        nonlocal collided_destination
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            source_path.is_dir()
            and source_path.parent.name == "experiment-creation"
            and destination_path.name.startswith("EXP-0001-")
        ):
            collided_destination = destination_path
            destination_path.mkdir()
            (destination_path / "owned-by-user.txt").write_text("preserve", encoding="utf-8")
            raise FileExistsError("injected promotion collision")
        real_replace(source, destination)

    monkeypatch.setattr(experiment_transaction.os, "replace", collide_during_promotion)
    with pytest.raises(ExperimentError, match="injected promotion collision"):
        create_experiment(root, "Promotion race", campaign="general")

    assert collided_destination is not None
    assert (collided_destination / "owned-by-user.txt").read_text(encoding="utf-8") == "preserve"
    assert not experiment_transaction._transaction_marker_path(collided_destination).exists()
    assert load_yaml(control_root / "registry.yaml")["experiments"] == []
    assert (collided_destination.parent / "README.md").is_file()
    assert not experiment_transaction._campaign_transaction_marker_path(
        collided_destination.parent
    ).exists()
    assert not (root / "var" / "run" / "experiment-creation-transaction.json").exists()


def test_creation_rolls_back_when_atomic_registry_write_fails(
    repo_root, tmp_path, monkeypatch
) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)

    def fail_registry(_path, _data) -> None:
        raise OSError("injected registry failure")

    monkeypatch.setattr(experiment_transaction, "_write_yaml_atomic", fail_registry)
    with pytest.raises(ExperimentError, match="injected registry failure"):
        create_experiment(root, "Rollback registry", campaign="general")

    assert load_yaml(control_root / "registry.yaml")["experiments"] == []
    assert not (root / "experiments" / "campaigns").exists()
    assert not (root / "EXPERIMENT_INDEX.md").exists()


def test_creation_restores_registry_when_index_refresh_fails(
    repo_root, tmp_path, monkeypatch
) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)

    def fail_index(_root) -> None:
        raise OSError("injected index failure")

    monkeypatch.setattr(experiment_transaction, "generate_index", fail_index)
    with pytest.raises(ExperimentError, match="injected index failure"):
        create_experiment(root, "Rollback index", campaign="general")

    assert load_yaml(control_root / "registry.yaml")["experiments"] == []
    assert not (root / "experiments" / "campaigns").exists()
    assert not (root / "EXPERIMENT_INDEX.md").exists()


def test_failed_registry_restore_preserves_forward_state_and_next_run_recovers(
    repo_root, tmp_path, monkeypatch
) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    real_write = experiment_transaction._write_yaml_atomic
    real_generate_index = experiment_transaction.generate_index

    def fail_restore(path, data) -> None:
        if data["experiments"]:
            real_write(path, data)
            return
        raise OSError("injected restore failure")

    def fail_index(_root) -> None:
        raise OSError("injected index failure")

    monkeypatch.setattr(experiment_transaction, "_write_yaml_atomic", fail_restore)
    monkeypatch.setattr(experiment_transaction, "generate_index", fail_index)
    with pytest.raises(ExperimentError, match="preserved the destination"):
        create_experiment(root, "Forward repair", campaign="general")

    registry = load_yaml(control_root / "registry.yaml")
    first_path = root / "experiments" / registry["experiments"][0]["directory"]
    journal = root / "var" / "run" / "experiment-creation-transaction.json"
    assert first_path.is_dir()
    assert journal.is_file()

    monkeypatch.setattr(experiment_transaction, "_write_yaml_atomic", real_write)
    monkeypatch.setattr(experiment_transaction, "generate_index", real_generate_index)
    second_path = create_experiment(root, "After recovery", campaign="probability")

    assert first_path.is_dir()
    assert second_path.name.startswith("EXP-0002-")
    assert not journal.exists()
    assert "EXP-0001" in (root / "EXPERIMENT_INDEX.md").read_text(encoding="utf-8")


def test_next_creation_recovers_promoted_orphan_from_transaction_journal(
    repo_root, tmp_path
) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    registry_path = control_root / "registry.yaml"
    original_registry = load_yaml(registry_path)
    orphan = create_experiment(root, "Interrupted promotion", campaign="general")
    updated_registry = load_yaml(registry_path)
    dump_yaml(registry_path, original_registry)
    (root / "EXPERIMENT_INDEX.md").unlink()

    journal_path = root / "var" / "run" / "experiment-creation-transaction.json"
    token = "a" * 32
    journal = {
        "schema_version": 1,
        "phase": "promoted",
        "transaction_token": token,
        "experiment_id": "EXP-0001",
        "destination": orphan.relative_to(root).as_posix(),
        "staging": f"var/tmp/experiment-creation/EXP-0001-{token}",
        "campaign_root": orphan.parent.relative_to(root).as_posix(),
        "campaigns_root_preexisting": False,
        "campaign_root_preexisting": False,
        "campaign_readme_preexisting": False,
        "original_registry": original_registry,
        "updated_registry": updated_registry,
        "original_index": None,
    }
    experiment_transaction._write_marker(
        experiment_transaction._transaction_marker_path(orphan),
        journal,
        scope="experiment-directory",
    )
    experiment_transaction._write_marker(
        experiment_transaction._campaign_transaction_marker_path(orphan.parent),
        journal,
        scope="campaign-transaction",
    )
    experiment_transaction._write_json_atomic(journal_path, journal)

    recovered = create_experiment(root, "Recovered allocation", campaign="probability")

    assert not orphan.exists()
    assert recovered.name.startswith("EXP-0001-")
    assert recovered.parent.name == "probability"
    assert not journal_path.exists()


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("destination", "experiments"),
        ("destination", "src"),
        ("destination", "."),
        (
            "destination",
            "experiments/campaigns/general/EXP-9999-hostile-journal-sentinel",
        ),
        (
            "destination",
            "experiments/campaigns/probability/EXP-0001-hostile-journal-sentinel",
        ),
        ("staging", "var/tmp"),
        ("staging", "src/staging"),
    ],
)
def test_recovery_rejects_hostile_journal_paths_before_mutation(
    repo_root,
    tmp_path,
    field: str,
    unsafe_value: str,
) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    source_sentinel = root / "src" / "preserve.txt"
    source_sentinel.parent.mkdir()
    source_sentinel.write_text("source must survive", encoding="utf-8")
    experiments_sentinel = root / "experiments" / "preserve.txt"
    experiments_sentinel.write_text("experiments must survive", encoding="utf-8")
    original_registry = load_yaml(control_root / "registry.yaml")
    journal = _valid_prepared_journal(root, control_root)
    journal[field] = unsafe_value
    journal_path = _write_transaction_journal(root, journal)

    with pytest.raises(ExperimentError, match="[Tt]ransaction"):
        create_experiment(root, "Must not mutate", campaign="general")

    assert source_sentinel.read_text(encoding="utf-8") == "source must survive"
    assert experiments_sentinel.read_text(encoding="utf-8") == "experiments must survive"
    assert load_yaml(control_root / "registry.yaml") == original_registry
    assert (root / "experiments" / EXPERIMENT_TEMPLATE_DIR).is_dir()
    assert journal_path.is_file()


def test_recovery_refuses_canonical_destination_without_ownership_marker(
    repo_root,
    tmp_path,
) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    journal = _valid_prepared_journal(root, control_root)
    destination = root.joinpath(*str(journal["destination"]).split("/"))
    destination.mkdir(parents=True)
    sentinel = destination / "owned-by-user.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    journal_path = _write_transaction_journal(root, journal)

    with pytest.raises(ExperimentError, match="ownership marker"):
        create_experiment(root, "Must not claim ownership", campaign="general")

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert load_yaml(control_root / "registry.yaml")["experiments"] == []
    assert journal_path.is_file()


def test_campaign_marker_cannot_authorize_markerless_destination_deletion(
    repo_root,
    tmp_path,
) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    journal = _valid_prepared_journal(root, control_root)
    destination = root.joinpath(*str(journal["destination"]).split("/"))
    destination.mkdir(parents=True)
    sentinel = destination / "owned-by-user.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    campaign_marker = experiment_transaction._campaign_transaction_marker_path(destination.parent)
    experiment_transaction._write_marker(
        campaign_marker,
        journal,
        scope="campaign-transaction",
    )
    journal_path = _write_transaction_journal(root, journal)

    with pytest.raises(ExperimentError, match="ownership marker"):
        create_experiment(root, "Campaign proof is not child proof", campaign="general")

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert campaign_marker.is_file()
    assert load_yaml(control_root / "registry.yaml")["experiments"] == []
    assert journal_path.is_file()


def test_pending_child_proof_cannot_bypass_malformed_final_marker(
    repo_root,
    tmp_path,
) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    journal = _valid_prepared_journal(root, control_root)
    destination = root.joinpath(*str(journal["destination"]).split("/"))
    destination.mkdir(parents=True)
    sentinel = destination / "owned-by-user.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    campaign_marker = experiment_transaction._campaign_transaction_marker_path(destination.parent)
    experiment_transaction._write_marker(
        campaign_marker,
        journal,
        scope="campaign-transaction",
    )
    final_marker = experiment_transaction._transaction_marker_path(destination)
    final_marker.write_text("malformed user sentinel", encoding="utf-8")
    pending_marker = experiment_transaction._marker_pending_path(final_marker, journal)
    pending_marker.write_text("{", encoding="utf-8")
    journal_path = _write_transaction_journal(root, journal)

    with pytest.raises(ExperimentError, match="Invalid transaction ownership marker"):
        create_experiment(root, "Pending must not bypass final", campaign="general")

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert final_marker.read_text(encoding="utf-8") == "malformed user sentinel"
    assert pending_marker.is_file()
    assert load_yaml(control_root / "registry.yaml")["experiments"] == []
    assert journal_path.is_file()


def test_recovery_removes_exact_empty_staging_from_pre_marker_crash(
    repo_root,
    tmp_path,
) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    journal = _valid_prepared_journal(root, control_root)
    staging = root.joinpath(*str(journal["staging"]).split("/"))
    staging.mkdir(parents=True)
    journal_path = _write_transaction_journal(root, journal)

    recovered = create_experiment(root, "After empty staging crash", campaign="probability")

    assert recovered.name.startswith("EXP-0001-")
    assert recovered.parent.name == "probability"
    assert not staging.exists()
    assert not journal_path.exists()


def test_recovery_removes_exact_empty_campaign_from_pre_marker_crash(
    repo_root,
    tmp_path,
) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    journal = _valid_prepared_journal(root, control_root)
    staging = root.joinpath(*str(journal["staging"]).split("/"))
    experiment_transaction._write_transaction_marker(staging, journal)
    interrupted_campaign = root.joinpath(*str(journal["campaign_root"]).split("/"))
    interrupted_campaign.mkdir(parents=True)
    journal_path = _write_transaction_journal(root, journal)

    recovered = create_experiment(root, "After empty campaign crash", campaign="probability")

    assert recovered.name.startswith("EXP-0001-")
    assert recovered.parent.name == "probability"
    assert not interrupted_campaign.exists()
    assert not staging.exists()
    assert not journal_path.exists()


def test_recovery_removes_incomplete_pending_marker_from_preexisting_campaign(
    repo_root,
    tmp_path,
) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    campaign_root = root / "experiments" / "campaigns" / "general"
    campaign_root.mkdir(parents=True)
    readme = campaign_root / "README.md"
    readme.write_text("# Existing general campaign\n", encoding="utf-8")
    sentinel = campaign_root / "preserve.txt"
    sentinel.write_text("preexisting content", encoding="utf-8")
    journal = _valid_prepared_journal(root, control_root)
    journal["campaigns_root_preexisting"] = True
    journal["campaign_root_preexisting"] = True
    journal["campaign_readme_preexisting"] = True
    staging = root.joinpath(*str(journal["staging"]).split("/"))
    experiment_transaction._write_transaction_marker(staging, journal)
    marker = experiment_transaction._campaign_transaction_marker_path(campaign_root)
    incomplete = experiment_transaction._marker_pending_path(marker, journal)
    incomplete.write_text("{", encoding="utf-8")
    journal_path = _write_transaction_journal(root, journal)

    recovered = create_experiment(root, "After incomplete marker", campaign="probability")

    assert recovered.name.startswith("EXP-0001-")
    assert readme.read_text(encoding="utf-8") == "# Existing general campaign\n"
    assert sentinel.read_text(encoding="utf-8") == "preexisting content"
    assert not marker.exists()
    assert not experiment_transaction._marker_pending_path(marker, journal).exists()
    assert not staging.exists()
    assert not journal_path.exists()


def test_recovery_rejects_malformed_final_campaign_marker(
    repo_root,
    tmp_path,
) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    campaign_root = root / "experiments" / "campaigns" / "general"
    campaign_root.mkdir(parents=True)
    readme = campaign_root / "README.md"
    readme.write_text("# Existing general campaign\n", encoding="utf-8")
    sentinel = campaign_root / "preserve.txt"
    sentinel.write_text("preexisting content", encoding="utf-8")
    journal = _valid_prepared_journal(root, control_root)
    journal["campaigns_root_preexisting"] = True
    journal["campaign_root_preexisting"] = True
    journal["campaign_readme_preexisting"] = True
    staging = root.joinpath(*str(journal["staging"]).split("/"))
    experiment_transaction._write_transaction_marker(staging, journal)
    marker = experiment_transaction._campaign_transaction_marker_path(campaign_root)
    marker.write_text("user sentinel", encoding="utf-8")
    journal_path = _write_transaction_journal(root, journal)

    with pytest.raises(ExperimentError, match="Invalid campaign transaction marker"):
        create_experiment(root, "Must preserve malformed final", campaign="probability")

    assert marker.read_text(encoding="utf-8") == "user sentinel"
    assert readme.read_text(encoding="utf-8") == "# Existing general campaign\n"
    assert sentinel.read_text(encoding="utf-8") == "preexisting content"
    assert staging.is_dir()
    assert journal_path.is_file()


def test_partial_rollback_restores_child_proof_and_recovers_on_retry(
    repo_root,
    tmp_path,
    monkeypatch,
) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    real_registry_write = experiment_transaction._write_yaml_atomic
    real_rmtree = experiment_transaction.shutil.rmtree

    def fail_registry(_path, _data) -> None:
        raise OSError("injected registry failure")

    def fail_destination_cleanup(path, *_args, **_kwargs) -> None:
        candidate = Path(path)
        marker = experiment_transaction._transaction_marker_path(candidate)
        if candidate.name.startswith("EXP-0001-") and marker.is_file():
            marker.unlink()
            raise OSError("injected locked destination")
        real_rmtree(path, *_args, **_kwargs)

    monkeypatch.setattr(experiment_transaction, "_write_yaml_atomic", fail_registry)
    monkeypatch.setattr(experiment_transaction.shutil, "rmtree", fail_destination_cleanup)
    with pytest.raises(ExperimentError, match="rollback was incomplete"):
        create_experiment(root, "Partial cleanup", campaign="general")

    destination = next((root / "experiments" / "campaigns" / "general").glob("EXP-0001-*"))
    campaign_marker = experiment_transaction._campaign_transaction_marker_path(destination.parent)
    journal_path = root / "var" / "run" / "experiment-creation-transaction.json"
    assert destination.is_dir()
    assert experiment_transaction._transaction_marker_path(destination).is_file()
    assert campaign_marker.is_file()
    assert journal_path.is_file()

    monkeypatch.setattr(experiment_transaction, "_write_yaml_atomic", real_registry_write)
    monkeypatch.setattr(experiment_transaction.shutil, "rmtree", real_rmtree)
    recovered = create_experiment(root, "After partial cleanup", campaign="probability")

    assert not destination.exists()
    assert recovered.name.startswith("EXP-0001-")
    assert load_yaml(control_root / "registry.yaml")["experiments"][0]["campaign"] == "probability"
    assert not journal_path.exists()


def test_recovery_accepts_exact_pending_child_proof_after_secondary_crash(
    repo_root,
    tmp_path,
) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    registry_path = control_root / "registry.yaml"
    original_registry = load_yaml(registry_path)
    destination = create_experiment(root, "Secondary crash", campaign="general")
    updated_registry = load_yaml(registry_path)
    dump_yaml(registry_path, original_registry)
    (root / "EXPERIMENT_INDEX.md").unlink()
    token = "c" * 32
    journal = {
        "schema_version": 1,
        "phase": "registry_committing",
        "transaction_token": token,
        "experiment_id": "EXP-0001",
        "destination": destination.relative_to(root).as_posix(),
        "staging": f"var/tmp/experiment-creation/EXP-0001-{token}",
        "campaign_root": destination.parent.relative_to(root).as_posix(),
        "campaigns_root_preexisting": False,
        "campaign_root_preexisting": False,
        "campaign_readme_preexisting": False,
        "original_registry": original_registry,
        "updated_registry": updated_registry,
        "original_index": None,
    }
    campaign_marker = experiment_transaction._campaign_transaction_marker_path(destination.parent)
    experiment_transaction._write_marker(
        campaign_marker,
        journal,
        scope="campaign-transaction",
    )
    child_marker = experiment_transaction._transaction_marker_path(destination)
    pending_child_marker = experiment_transaction._marker_pending_path(child_marker, journal)
    pending_child_marker.write_text("{", encoding="utf-8")
    journal_path = _write_transaction_journal(root, journal)

    recovered = create_experiment(root, "After secondary crash", campaign="probability")

    assert not destination.exists()
    assert recovered.name.startswith("EXP-0001-")
    assert recovered.parent.name == "probability"
    assert not journal_path.exists()


@pytest.mark.parametrize("marker_kind", ["staging", "campaign"])
def test_failed_marker_publication_preserves_journal_and_recovers(
    repo_root,
    tmp_path,
    monkeypatch,
    marker_kind: str,
) -> None:
    root, _control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    real_replace = experiment_transaction.os.replace
    real_unlink = Path.unlink
    prefix = (
        ".experiment-transaction.json."
        if marker_kind == "staging"
        else ".experiment-campaign-transaction.json."
    )

    def fail_marker_replace(source, destination) -> None:
        source_path = Path(source)
        if source_path.name.startswith(prefix) and source_path.name.endswith(".pending"):
            raise PermissionError("injected marker replace failure")
        real_replace(source, destination)

    def lock_pending_marker(path: Path, *args, **kwargs) -> None:
        if path.name.startswith(prefix) and path.name.endswith(".pending"):
            raise PermissionError("injected locked pending marker")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(experiment_transaction.os, "replace", fail_marker_replace)
    monkeypatch.setattr(Path, "unlink", lock_pending_marker)
    with pytest.raises(ExperimentError, match="rollback was incomplete"):
        create_experiment(root, "Marker publication crash", campaign="general")

    journal_path = root / "var" / "run" / "experiment-creation-transaction.json"
    assert journal_path.is_file()
    journal = experiment_transaction._load_transaction_journal(journal_path)
    if marker_kind == "staging":
        directory = root.joinpath(*str(journal["staging"]).split("/"))
        final_marker = experiment_transaction._transaction_marker_path(directory)
    else:
        directory = root.joinpath(*str(journal["campaign_root"]).split("/"))
        final_marker = experiment_transaction._campaign_transaction_marker_path(directory)
    pending_marker = experiment_transaction._marker_pending_path(final_marker, journal)
    assert directory.is_dir()
    assert pending_marker.is_file()

    monkeypatch.setattr(experiment_transaction.os, "replace", real_replace)
    monkeypatch.setattr(Path, "unlink", real_unlink)
    recovered = create_experiment(root, "After marker recovery", campaign="probability")

    assert recovered.name.startswith("EXP-0001-")
    assert recovered.parent.name == "probability"
    assert not directory.exists()
    assert not journal_path.exists()


@pytest.mark.parametrize(
    "runtime_name",
    ["experiment-registry.lock", "experiment-creation-transaction.json"],
)
def test_creation_rejects_nonfile_runtime_controls(
    repo_root,
    tmp_path,
    runtime_name: str,
) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    runtime_path = root / "var" / "run" / runtime_name
    runtime_path.mkdir(parents=True)

    with pytest.raises(ExperimentError, match="regular non-reparse file"):
        create_experiment(root, "Unsafe runtime control", campaign="general")

    assert load_yaml(control_root / "registry.yaml")["experiments"] == []
    assert not (root / "experiments" / "campaigns").exists()


def test_keyboard_interrupt_during_rendering_rolls_back_owned_state(
    repo_root, tmp_path, monkeypatch
) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)

    def interrupt(_path, _values) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(experiment_transaction, "_replace_placeholders", interrupt)
    with pytest.raises(KeyboardInterrupt):
        create_experiment(root, "Interrupted render", campaign="general")

    assert load_yaml(control_root / "registry.yaml")["experiments"] == []
    assert not (root / "experiments" / "campaigns").exists()
    assert not (root / "var" / "tmp").exists()
    assert not (root / "var" / "run" / "experiment-creation-transaction.json").exists()


def test_keyboard_interrupt_after_registry_replace_rolls_back_owned_state(
    repo_root, tmp_path, monkeypatch
) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    real_write = experiment_transaction._write_yaml_atomic

    def interrupt_after_replace(path, data) -> None:
        real_write(path, data)
        raise KeyboardInterrupt

    monkeypatch.setattr(experiment_transaction, "_write_yaml_atomic", interrupt_after_replace)
    with pytest.raises(KeyboardInterrupt):
        create_experiment(root, "Interrupted commit", campaign="general")

    assert load_yaml(control_root / "registry.yaml")["experiments"] == []
    assert not (root / "experiments" / "campaigns").exists()
    assert not (root / "var" / "run" / "experiment-creation-transaction.json").exists()


def test_existing_unlocked_runtime_lock_does_not_block_creation(repo_root, tmp_path) -> None:
    root, _control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    lock_path = root / "var" / "run" / "experiment-registry.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("stale metadata", encoding="utf-8")

    destination = create_experiment(root, "Crash-safe lock", campaign="general")

    assert destination.is_dir()


def test_creation_preserves_nonempty_campaign_missing_readme(repo_root, tmp_path) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    campaign_root = root / "experiments" / "campaigns" / "general"
    campaign_root.mkdir(parents=True)
    marker = campaign_root / "owned-by-user.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(ExperimentError, match="missing README.md and is not empty"):
        create_experiment(root, "Do not delete", campaign="general")

    assert marker.read_text(encoding="utf-8") == "preserve"
    assert load_yaml(control_root / "registry.yaml")["experiments"] == []


def test_registry_validation_rejects_directory_traversal(repo_root, tmp_path) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    registry_path = control_root / "registry.yaml"
    registry_path.write_text(
        "registry_version: 2\n"
        "next_id: 2\n"
        "experiments:\n"
        "  - id: EXP-0001\n"
        "    title: unsafe\n"
        "    campaign: hkg-tmax\n"
        "    directory: campaigns/hkg-tmax/../../escape\n"
        "    created_at_utc: '2026-07-10T00:00:00Z'\n"
        "    status: PLANNED\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="campaigns/<campaign>/<experiment>"):
        validate_experiment_registry(root)


def test_registry_validation_rejects_schema_and_id_corruption(repo_root, tmp_path) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    registry_path = control_root / "registry.yaml"
    registry_path.write_text(
        "registry_version: 999\nnext_id: 1\nexperiments: []\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="registry_version must be 2"):
        validate_experiment_registry(root)

    registry_path.write_text(
        "registry_version: 2\n"
        "next_id: 2\n"
        "experiments:\n"
        "  - id: EXP--1\n"
        "    title: malformed\n"
        "    campaign: hkg-tmax\n"
        "    directory: campaigns/hkg-tmax/EXP--1-malformed\n"
        "    created_at_utc: '2026-07-10T00:00:00Z'\n"
        "    status: PLANNED\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="invalid id"):
        validate_experiment_registry(root)


def test_registry_validation_rejects_orphan_governed_directory(repo_root, tmp_path) -> None:
    root, control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    destination = create_experiment(root, "Registered first", campaign="hkg-tmax")
    registry = load_yaml(control_root / "registry.yaml")
    registry["experiments"] = []
    dump_yaml(control_root / "registry.yaml", registry)

    with pytest.raises(ValidationError, match="Unregistered governed experiment"):
        validate_experiment_registry(root)

    assert destination.is_dir()


@pytest.mark.parametrize("contract_name", ["DATA_MANIFEST.yaml", "RUN_CONFIG.yaml"])
def test_registry_validation_rejects_scaffold_id_drift(
    repo_root, tmp_path, contract_name: str
) -> None:
    root, _control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    destination = create_experiment(root, "Contract identity", campaign="hkg-tmax")
    contract = load_yaml(destination / contract_name)
    contract["experiment_id"] = "EXP-9999"
    dump_yaml(destination / contract_name, contract)

    with pytest.raises(ValidationError, match=f"{contract_name} id does not match"):
        validate_experiment_registry(root)


def test_registry_validation_rejects_invalid_metrics_json(repo_root, tmp_path) -> None:
    root, _control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    destination = create_experiment(root, "Metrics contract", campaign="hkg-tmax")
    (destination / "results" / "metrics.json").write_text("{", encoding="utf-8")

    with pytest.raises(ValidationError, match="Rendered experiment is invalid"):
        validate_experiment_registry(root)


def test_index_and_registry_refuse_linked_campaign_children(repo_root, tmp_path) -> None:
    root, _control_root = _repo_with_experiment_controls(repo_root, tmp_path)
    campaigns = root / "experiments" / "campaigns"
    campaigns.mkdir()
    external = tmp_path / "external-campaign"
    external.mkdir()
    linked = campaigns / "linked"
    try:
        linked.symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks are unavailable in this Windows environment")

    with pytest.raises(ExperimentError, match="symlink or reparse point"):
        generate_index(root)
    with pytest.raises(ValidationError, match="symlink or reparse point"):
        validate_experiment_registry(root)


def test_experiment_cli_accepts_only_allowlisted_campaigns() -> None:
    parser = build_parser()

    args = parser.parse_args(
        ["experiments", "create", "--campaign", "probability", "--title", "Calibrate tails"]
    )
    assert args.campaign == "probability"

    with pytest.raises(SystemExit):
        parser.parse_args(["experiments", "create", "--campaign", "../escape", "--title", "Unsafe"])
    with pytest.raises(SystemExit):
        parser.parse_args(["experiments", "create", "--title", "No dumping ground"])


def test_experiment_cli_creates_and_indexes_in_explicit_campaign(
    repo_root, tmp_path, capsys
) -> None:
    root, _control_root = _repo_with_experiment_controls(repo_root, tmp_path)

    main(
        [
            "--root",
            str(root),
            "experiments",
            "create",
            "--campaign",
            "probability",
            "--title",
            "CLI probability hypothesis",
        ]
    )

    destination = root / capsys.readouterr().out.strip()
    if not destination.is_absolute():
        destination = root / destination
    assert destination.parent.name == "probability"
    assert "EXP-0001" in (root / "EXPERIMENT_INDEX.md").read_text(encoding="utf-8")


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
