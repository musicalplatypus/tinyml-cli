---
created: 2026-07-09T12:55:08.142Z
title: Investigate MPS float64 crash from tinyml_tinyverse update
area: general
files:
  - tinyml_tinyverse/common/utils/utils.py:1452
  - tests/test_e2e.py
---

## Problem

After merging `upstream/r1.4`, 7 end-to-end tests error with:

```
TypeError: Cannot convert a MPS Tensor to float64 dtype as float64 is not supported by MPS.
```

Traceback points to `tinyml_tinyverse/common/utils/utils.py:1452`:
```python
data = data_feat_ext.to(device).float()
```

On Apple Silicon (MPS), `.float()` resolves to float64 (not float32 as on CUDA/CPU). MPS does not support float64. This is an upstream regression introduced in the `r1.4` update to `tinyml_tinyverse`.

## Solution

Options in order of preference:
1. **Report upstream**: File an issue / PR against tinyml_tinyverse to change `.float()` to `.to(torch.float32)` at line 1452. This is the correct fix — `.float()` is ambiguous, explicit cast is correct.
2. **Local workaround**: Patch `tinyml_tinyverse/common/utils/utils.py:1452` locally with `.to(torch.float32)` instead of `.float()`. Only if upstream fix would take too long to land.
3. **Pin tinyverse version**: Revert to pre-r1.4 if a stable version is available and the regression is blocking.

Before applying a local patch, check if r1.4 introduced anything else mmcli depends on, to weigh the cost of reverting vs. patching.
