---
created: 2026-07-09T12:55:08.142Z
title: Add format attribute to MockArgs in test_recommend.py
area: testing
files:
  - tests/test_recommend.py
---

## Problem

`test_module_inference_from_task` in `tests/test_recommend.py` constructs a `MockArgs` object that doesn't include a `format` attribute. A recent change to `run_recommend` in `mmcli/` now accesses `args.format`, causing `AttributeError` and test failure.

This is a test-compatibility failure: the production code evolved, the test mock didn't keep up.

## Solution

Add `format = None` (or whatever the appropriate default is) to the `MockArgs` class or instantiation in `test_recommend.py`. Check what values `run_recommend` expects for `format` — likely `None` means "no format override" / use default — and set accordingly so the test exercises the same path as before.

Verify by running: `MMCLI_PYTHON=... python -m pytest tests/test_recommend.py::test_module_inference_from_task -xvs`
