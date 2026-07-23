# Deferred items — out of scope for the plan that found them

Items discovered during plan execution that are pre-existing, out of the
executing plan's file scope, and therefore logged here rather than fixed
inline (per the executor's scope-boundary rule).

## Found during 10-05 (README truth-up)

- **`mmcli/cli.py` still says "from TI" in two places, stale after 10-03's
  repoint to the GitHub release mirror:**
  - `datasets pull`'s subparser description: `"Fetch a dataset from TI and
    cache it. ..."` (around `_add_datasets_parser`'s `pull_p = sub.add_parser(...)`).
  - `init --fetch`'s help text: `"Force-fetch a missing dataset from TI,
    regardless of whether stderr is a terminal. ..."`.
  - Not fixed here: `10-05-PLAN.md`'s `files_modified` is `[README.md,
    README_zh.md]` only; `mmcli/cli.py` is out of this plan's scope. The
    behavior is correct (it fetches from the GitHub mirror, not TI) — only
    the CLI's own `--help` prose is stale. Low severity (in-tool help text,
    not a user-facing README claim), but should be corrected the next time
    `mmcli/cli.py` is touched, e.g. by 10-07 (CLI help / Sphinx docs plan)
    or a follow-up.
