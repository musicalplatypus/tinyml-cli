---
created: 2026-07-09T12:55:08.142Z
title: Apply local MPS float32 workaround in tinyml_tinyverse
area: general
files:
  - tinyml_tinyverse/common/utils/utils.py:1452
  - tests/test_e2e.py
---

## Problem

After merging `upstream/r1.4`, `tinyml_tinyverse/common/utils/utils.py:1452` calls `.float()` on an MPS tensor, which resolves to float64 on Apple Silicon. MPS does not support float64, causing 7 E2E tests to error:

```
TypeError: Cannot convert a MPS Tensor to float64 dtype as float64 is not supported by MPS.
```

This is blocking local development on Apple Silicon until the upstream fix lands.

## Solution

Patch line 1452 in `tinyml_tinyverse/common/utils/utils.py`:

```python
# Before (broken on MPS):
data = data_feat_ext.to(device).float()

# After (explicit float32, correct on MPS and CUDA):
data = data_feat_ext.to(device).to(torch.float32)
```

Ensure `torch` is already imported in that file before applying. Verify the 7 previously-erroring E2E tests pass after the patch. This is a stopgap — remove it once the upstream fix is merged and pulled.
