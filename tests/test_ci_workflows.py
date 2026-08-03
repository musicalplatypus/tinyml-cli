"""Guard that .github/workflows/test-cli.yml and release.yml actually collect the
phase-10 regression-guard test files, and that the two workflows never drift apart.

Every guard phase 10 built (test_build_config.py's 36 assertions, test_datasets_download.py's
50 tests, test_datasets_cli.py's contract/D-5 tests) was inert in CI before 10-08: both
workflows' pytest invocations named only the two pre-existing Tier 4 files, so removing a
PyInstaller exclusion or breaking the MMCLI_DATASETS air-gap policy would pass every CI run
silently. This file is the guard that the fix holds and does not quietly regress:

  1. Each workflow contains exactly one `python -m pytest` invocation (so there's exactly one
     place either file's collection is decided).
  2. The set of test files named by that invocation is identical between the two workflows —
     the drift guard. A one-sided edit (added to test-cli.yml, forgotten in release.yml, or
     vice versa) fails here instead of surfacing as a silent gap discovered later.
  3. That set is a superset of the required six files, named below in one place with a
     comment recording that dropping one from CI is a decision, not routine cleanup.
  4. Every test path named by either workflow actually exists on disk, so a rename that
     forgets to update the workflow fails here rather than as a pytest "file not found"
     collection error during a real release run.

Parses the raw workflow text with a regex rather than a full YAML load: the assertion is
about the literal shell command the workflow runs, and a YAML round-trip that "helpfully"
re-serialises or re-wraps the block-scalar string would weaken exactly what this file exists
to check. Neither workflow installs PyYAML, and adding it here — for one assertion — is the
same call 10-08 declined to make elsewhere.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_CLI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test-cli.yml"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
WORKFLOWS = [TEST_CLI_WORKFLOW, RELEASE_WORKFLOW]
RELEASE_PREFLIGHT_SCRIPT = REPO_ROOT / "scripts" / "release_preflight.py"

# Every file in this set must be collected by BOTH workflows' pytest invocation. Removing one
# is a decision about what CI enforces before a release, not a cleanup — if a test file is
# genuinely retired, update this constant deliberately, in the same change that removes it
# from both workflows.
REQUIRED_TEST_FILES = {
    "tests/test_cli_integration.py",
    "tests/test_tier4_cli.py",
    "tests/test_build_config.py",
    "tests/test_datasets_download.py",
    "tests/test_datasets_cli.py",
    "tests/test_ci_workflows.py",
}

# Matches a `python -m pytest` invocation through to the end of the existing deliberate
# deselection, across either the single-line form or the `run: |` block-scalar,
# backslash-continued form Task 1 introduced. DOTALL so the continuation lines (each their
# own physical line, backslash-joined) are captured as one block.
PYTEST_INVOCATION_RE = re.compile(
    r"python -m pytest\b(.*?)-k\s+\"not TestInitDatasetExtractReal\"",
    re.DOTALL,
)

TEST_PATH_RE = re.compile(r"tests/[A-Za-z0-9_./]+\.py")


def _read(path: Path) -> str:
    assert path.exists(), f"workflow file missing: {path}"
    return path.read_text()


def _pytest_invocations(text: str):
    """All `python -m pytest ... -k "not TestInitDatasetExtractReal"` blocks in the text."""
    return PYTEST_INVOCATION_RE.findall(text)


def _named_test_files(invocation_body: str) -> set:
    return set(TEST_PATH_RE.findall(invocation_body))


def test_both_workflow_files_exist():
    for path in WORKFLOWS:
        assert path.exists(), f"expected workflow file not found: {path}"


def test_each_workflow_has_exactly_one_pytest_invocation():
    for path in WORKFLOWS:
        text = _read(path)
        invocations = _pytest_invocations(text)
        assert len(invocations) == 1, (
            f"{path.name} must contain exactly one `python -m pytest ... -k \"not "
            f"TestInitDatasetExtractReal\"` invocation, found {len(invocations)}"
        )


def test_workflows_name_the_same_set_of_test_files():
    """The drift guard: a one-sided edit to either workflow fails here."""
    named = {}
    for path in WORKFLOWS:
        [invocation] = _pytest_invocations(_read(path))
        named[path.name] = _named_test_files(invocation)

    test_cli_set = named[TEST_CLI_WORKFLOW.name]
    release_set = named[RELEASE_WORKFLOW.name]
    assert test_cli_set == release_set, (
        f"test-cli.yml and release.yml collect different test files — they must be "
        f"identical. Only in test-cli.yml: {sorted(test_cli_set - release_set)}. "
        f"Only in release.yml: {sorted(release_set - test_cli_set)}."
    )


def test_named_set_is_a_superset_of_the_required_files():
    for path in WORKFLOWS:
        [invocation] = _pytest_invocations(_read(path))
        named = _named_test_files(invocation)
        missing = REQUIRED_TEST_FILES - named
        assert not missing, (
            f"{path.name}'s pytest invocation is missing required test file(s) "
            f"{sorted(missing)} — every guard this phase built must actually run in CI"
        )


def test_every_named_test_path_exists_on_disk():
    for path in WORKFLOWS:
        [invocation] = _pytest_invocations(_read(path))
        for test_path in _named_test_files(invocation):
            full_path = REPO_ROOT / test_path
            assert full_path.exists(), (
                f"{path.name} names {test_path}, which does not exist on disk — a "
                f"rename that forgot to update the workflow would otherwise surface only "
                f"as a pytest collection error during a real CI run"
            )


def test_deselection_is_still_intact_in_both_workflows():
    """The `-k "not TestInitDatasetExtractReal"` deselection must survive verbatim —
    it protects test_cli_integration.py / test_tier4_cli.py, not something this plan
    should be weakening while wiring in new files."""
    for path in WORKFLOWS:
        text = _read(path)
        assert '-k "not TestInitDatasetExtractReal"' in text, (
            f"{path.name} no longer carries the exact "
            f'`-k "not TestInitDatasetExtractReal"` deselection'
        )


def test_release_workflow_does_not_tolerate_windows_test_failures():
    """10-REVIEW.md WR-14: release.yml's `test` job must not carry
    `continue-on-error` for the Windows matrix leg — that marks the job
    successful for `build`'s `needs: [test, ...]` purposes even on a fully
    red Windows run, including test_build_config.py's
    EXPECTED_ADD_DATA_SEPARATOR guard (which exists specifically because a
    wrong --add-data separator on Windows produces a binary with an empty
    bundle and a successful build) — the one guard whose whole point is to
    fail a Windows build, running in a job whose Windows result would
    otherwise be discarded. test-cli.yml (PR iteration) is intentionally
    unaffected: tolerating known Windows flakiness on PRs is a legitimate,
    separate call from whether a release build should proceed on red.
    """
    text = _read(RELEASE_WORKFLOW)
    # Strip full-line YAML comments so this assertion checks the actual
    # workflow directive, not this test's own explanatory comment above it
    # in release.yml (which necessarily names the key it's guarding against).
    code = "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("#")
    )
    assert "continue-on-error" not in code, (
        f"{RELEASE_WORKFLOW.name} must not tolerate any job failure — a "
        f"release build must not proceed on a red test matrix"
    )


# ---------------------------------------------------------------------------
# 10-REVIEW.md WR-02: scripts/release_preflight.py's mirror check and
# release.yml's `mirror-healthcheck` job's embedded python are a verbatim
# copy of each other, not a shared/reused implementation. Nothing previously
# guarded the two copies against drift. These two tests are that guard: they
# extract the `gh release view` argv list and the FATAL: message templates
# from both files and assert they match exactly. A one-sided edit to either
# copy (e.g. renaming a flag, tweaking a message) fails here instead of
# surfacing later as "preflight and CI disagree" during an actual release.
# ---------------------------------------------------------------------------

# Matches a run of one or more adjacent (implicitly-concatenated) f-string /
# plain-string literals, the first of which starts with "FATAL:". This
# captures each multi-line FATAL message template as one unit regardless of
# how many `f"..."` segments it's split across.
_FATAL_GROUP_RE = re.compile(
    r'(f"FATAL:(?:[^"\\]|\\.)*"(?:\s*f?"(?:[^"\\]|\\.)*")*)'
)

# The `gh release view <tag> --repo <REPO> --json tagName,assets` argv list,
# tolerant of the different line-wrapping the two files use and an optional
# trailing comma before the closing bracket.
_GH_ARGV_RE = re.compile(
    r'\[\s*"gh",\s*"release",\s*"view",\s*tag,\s*"--repo",\s*REPO,\s*"--json",'
    r'\s*"tagName,assets",?\s*\]'
)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def test_release_preflight_script_exists():
    assert RELEASE_PREFLIGHT_SCRIPT.exists(), (
        f"expected release preflight script not found: {RELEASE_PREFLIGHT_SCRIPT}"
    )


def test_mirror_check_gh_argv_matches_between_script_and_workflow():
    script_text = _normalize_whitespace(_read(RELEASE_PREFLIGHT_SCRIPT))
    workflow_text = _normalize_whitespace(_read(RELEASE_WORKFLOW))
    assert _GH_ARGV_RE.search(script_text), (
        f"{RELEASE_PREFLIGHT_SCRIPT.name} no longer contains the expected "
        f"`gh release view ... --json tagName,assets` argv — if this changed "
        f"deliberately, release.yml's mirror-healthcheck job must change "
        f"identically in the same commit"
    )
    assert _GH_ARGV_RE.search(workflow_text), (
        f"{RELEASE_WORKFLOW.name}'s mirror-healthcheck job no longer contains "
        f"the expected `gh release view ... --json tagName,assets` argv — it "
        f"has drifted from {RELEASE_PREFLIGHT_SCRIPT.name}"
    )


def _check_mirror_tag_and_assets_body(script_text: str) -> str:
    """Isolate just the part of check_mirror_tag_and_assets() that
    corresponds 1:1 with release.yml's embedded heredoc — from the `gh`
    subprocess call onward. Script-only concerns added since (e.g. WR-03's
    import-error / missing-`gh`-executable FATAL: messages) have no
    workflow analog: release.yml's job always has mmcli freshly pip-installed
    and `gh` preinstalled on the runner in the same job, so it never needs
    those guards. Including them here would make this drift guard
    false-positive on every legitimate script-only robustness fix."""
    match = re.search(
        r"result = subprocess\.run\(.*?return True", script_text, re.DOTALL
    )
    assert match, (
        "could not locate the `gh` subprocess call in "
        f"check_mirror_tag_and_assets() of {RELEASE_PREFLIGHT_SCRIPT.name} — "
        f"has it been renamed or restructured?"
    )
    return match.group(0)


def test_mirror_check_fatal_messages_match_between_script_and_workflow():
    script_msgs = sorted(
        _normalize_whitespace(m)
        for m in _FATAL_GROUP_RE.findall(
            _check_mirror_tag_and_assets_body(_read(RELEASE_PREFLIGHT_SCRIPT))
        )
    )
    workflow_msgs = sorted(
        _normalize_whitespace(m)
        for m in _FATAL_GROUP_RE.findall(_read(RELEASE_WORKFLOW))
    )
    assert script_msgs, (
        f"found no FATAL: message templates in check_mirror_tag_and_assets() "
        f"of {RELEASE_PREFLIGHT_SCRIPT.name} — the extraction regex itself "
        f"may have drifted from the source"
    )
    assert script_msgs == workflow_msgs, (
        f"FATAL: message templates differ between {RELEASE_PREFLIGHT_SCRIPT.name} "
        f"and {RELEASE_WORKFLOW.name}'s mirror-healthcheck job — these are meant "
        f"to be the identical check, reused rather than reimplemented; a "
        f"maintainer running the preflight locally must see exactly the failure "
        f"CI would report.\n"
        f"Only in script: {sorted(set(script_msgs) - set(workflow_msgs))}\n"
        f"Only in workflow: {sorted(set(workflow_msgs) - set(script_msgs))}"
    )
