from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.repo.doctor import RepositoryDoctor, _is_reparse_or_symlink


class RepositoryFixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        subprocess.run(
            ["git", "init", "--quiet", str(self.root)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def close(self) -> None:
        self.temporary.cleanup()

    def write(self, relative: str, content: str | bytes) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path

    def track(self, *relative: str) -> None:
        subprocess.run(
            ["git", "-C", str(self.root), "add", "--", *relative],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


class DoctorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = RepositoryFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def codes(self, check: str, **kwargs: object) -> list[str]:
        doctor = RepositoryDoctor(self.fixture.root, **kwargs)
        return [finding.code for finding in doctor.run([check])]

    def test_real_git_directory_is_accepted(self) -> None:
        self.assertEqual([], self.codes("root"))

    def test_nested_git_directory_is_reported(self) -> None:
        (self.fixture.root / "project" / ".git").mkdir(parents=True)
        self.assertIn("git.nested", self.codes("filesystem"))

    def test_symlink_or_junction_is_not_followed(self) -> None:
        target = self.fixture.root / "target"
        target.mkdir()
        link = self.fixture.root / "link"
        try:
            os.symlink(target, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("directory symlinks are unavailable")
        self.assertTrue(_is_reparse_or_symlink(link))
        self.assertIn("filesystem.reparse_point", self.codes("filesystem"))

    def test_tracked_runtime_and_large_file_are_reported(self) -> None:
        self.fixture.write("var/run/service.pid", "123")
        self.fixture.write("payload.bin", b"x" * 2_048)
        self.fixture.track("var/run/service.pid", "payload.bin")
        self.assertIn("tracked.runtime", self.codes("tracked-runtime"))
        self.assertIn(
            "tracked.large_file",
            self.codes("large-files", max_file_bytes=1_024),
        )

    def test_var_readme_is_allowed_but_runtime_payload_is_not(self) -> None:
        self.fixture.write("var/README.md", "Runtime state is ignored.\n")
        self.fixture.write("var/run/service.pid", "123")
        self.fixture.track("var/README.md", "var/run/service.pid")

        findings = RepositoryDoctor(self.fixture.root).run(["tracked-runtime"])

        self.assertEqual(["var/run/service.pid"], [finding.path for finding in findings])

    def test_deleted_tracked_runtime_file_is_not_reported(self) -> None:
        path = self.fixture.write("var/run/old-service.pid", "123")
        self.fixture.track("var/run/old-service.pid")
        path.unlink()
        self.assertNotIn("tracked.runtime", self.codes("tracked-runtime"))

    def test_literal_secret_is_reported_but_environment_placeholder_is_not(self) -> None:
        self.fixture.write(
            "config/application.yml",
            "password: ${DB_PASSWORD:}\napi_key: live_value_material_987654321\n",
        )
        self.fixture.track("config/application.yml")
        findings = RepositoryDoctor(self.fixture.root).run(["secrets"])
        literal_lines = [
            finding.line for finding in findings if finding.code == "secret.literal_assignment"
        ]
        self.assertEqual([2], literal_lines)

    def test_credential_url_requires_a_non_placeholder_password(self) -> None:
        self.fixture.write(
            "config/urls.yml",
            "safe: postgresql://user:password@localhost/db\n"
            "unsafe: postgresql://service:materialValue987654@db.internal/db\n",
        )
        self.fixture.track("config/urls.yml")
        findings = RepositoryDoctor(self.fixture.root).run(["secrets"])
        url_lines = [finding.line for finding in findings if finding.code == "secret.credential_url"]
        self.assertEqual([2], url_lines)

    def test_stale_path_and_unsafe_defaults_are_reported(self) -> None:
        self.fixture.write(
            "config/runtime.yml",
            "root: C:\\Users\\ahmad\\Desktop\\generalFiles\\git\\weather_markets\\"
            "weather_data_extraction\\bootstrap\\hkg_tmax_elite_codex_bootstrap\\"
            "hkg_tmax_elite_codex\n"
            "trading:\n  enabled: true\nworkers: 8\n",
        )
        self.fixture.track("config/runtime.yml")
        stale = self.codes("stale-paths")
        unsafe = self.codes("unsafe-defaults")
        self.assertIn("stale.bootstrap_path", stale)
        self.assertIn("stale.absolute_workspace_path", stale)
        self.assertIn("unsafe.enabled_default", unsafe)
        self.assertIn("unsafe.high_concurrency", unsafe)

    def test_stale_paths_in_frozen_hkg_evidence_are_provenance(self) -> None:
        old_path = (
            "C:\\Users\\ahmad\\Desktop\\generalFiles\\git\\weather_markets\\"
            "weather_data_extraction\\bootstrap\\hkg_tmax_elite_codex_bootstrap\\"
            "hkg_tmax_elite_codex"
        )
        self.fixture.write(
            "projects/hkg-tmax/docs/evidence/run.json",
            '{"historical_root": "' + old_path.replace("\\", "\\\\") + '"}\n',
        )
        self.fixture.track("projects/hkg-tmax/docs/evidence/run.json")

        self.assertEqual([], RepositoryDoctor(self.fixture.root).run(["stale-paths"]))

    def test_scope_excludes_other_project_findings(self) -> None:
        self.fixture.write("projects/a/app.log", "runtime")
        self.fixture.write("projects/b/app.log", "runtime")
        self.fixture.track("projects/a/app.log", "projects/b/app.log")
        findings = RepositoryDoctor(self.fixture.root, scope="projects/a").run(
            ["tracked-runtime"]
        )
        self.assertEqual(["projects/a/app.log"], [finding.path for finding in findings])

    def test_doctor_policy_fixtures_do_not_trigger_their_own_patterns(self) -> None:
        self.fixture.write(
            "tools/repo/tests/policy_fixture.py",
            "api_key = 'materialValue987654'\n"
            "n_jobs = -1\n"
            "old = r'C:\\Users\\ahmad\\Desktop\\generalFiles\\git\\weather_markets\\"
            "weather_data_extraction\\bootstrap\\hkg_tmax_elite_codex_bootstrap\\"
            "hkg_tmax_elite_codex'\n",
        )
        self.fixture.track("tools/repo/tests/policy_fixture.py")
        findings = RepositoryDoctor(self.fixture.root).run(
            ["secrets", "stale-paths", "unsafe-defaults"]
        )
        self.assertEqual([], findings)

    def test_fail_closed_conditional_startup_listener_is_accepted(self) -> None:
        self.fixture.write(
            "src/GuardedService.java",
            '@ConditionalOnProperty(name = "live.enabled", havingValue = "true")\n'
            "class GuardedService {\n"
            "  @EventListener(ApplicationReadyEvent.class)\n"
            "  void start() {}\n"
            "}\n",
        )
        self.fixture.track("src/GuardedService.java")
        self.assertNotIn("unsafe.startup_listener", self.codes("unsafe-defaults"))

    def test_unguarded_startup_listener_is_reported(self) -> None:
        self.fixture.write(
            "src/UnguardedService.java",
            "class UnguardedService {\n"
            "  @EventListener(ApplicationReadyEvent.class)\n"
            "  void start() {}\n"
            "}\n",
        )
        self.fixture.track("src/UnguardedService.java")
        self.assertIn("unsafe.startup_listener", self.codes("unsafe-defaults"))


if __name__ == "__main__":
    unittest.main()
