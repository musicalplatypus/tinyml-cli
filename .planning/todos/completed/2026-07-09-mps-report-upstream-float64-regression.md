---
created: 2026-07-09T12:55:08.142Z
title: Report MPS float64 regression to tinyml_tinyverse upstream
area: general
files:
  - tinyml_tinyverse/common/utils/utils.py:1452
---

## Problem

`tinyml_tinyverse/common/utils/utils.py:1452` uses `.float()` which is ambiguous — on Apple Silicon MPS it resolves to float64, which MPS does not support. This was introduced in the `r1.4` update. The correct call is `.to(torch.float32)`.

This needs an upstream fix so the local workaround (see `2026-07-09-mps-workaround-float32-cast.md`) can be removed.

## Solution

File an issue (or PR) against the tinyml_tinyverse upstream repository:

- **Title**: `MPS: data_feat_ext.float() crashes on Apple Silicon — use .to(torch.float32) instead`
- **Location**: `tinyml_tinyverse/common/utils/utils.py:1452`
- **Fix**: Change `.float()` to `.to(torch.float32)` — `.float()` is ambiguous; explicit dtype is the PyTorch-recommended pattern for cross-device code.
- **Reproducer**: Any E2E timeseries training on Apple Silicon MPS after r1.4 merge.

Once the upstream fix is merged and pulled, remove the local workaround patch.
