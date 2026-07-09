---
created: 2026-07-09T12:55:08.142Z
title: Update test_no_pca_images for new modelmaker PCA behavior
area: testing
files:
  - tests/test_report.py
---

## Problem

`test_report.py::TestPCAImages::test_no_pca_images` expects the generated report HTML to be empty of PCA image content, but after the modelmaker update (installed 2026-07-09 in venv-tinyml), the new modelmaker generates PCA images even in scenarios the old version didn't. The test's fixture or input condition no longer produces "no PCA images" with the current modelmaker.

## Solution

Investigate whether the new modelmaker:
a) Always generates PCA images regardless of input (test fixture needs to be removed or completely rethought)
b) Generates them conditionally on different criteria than before (test needs an updated fixture that satisfies the new "no PCA" condition)

Run `test_no_pca_images` in isolation with `-xvs` and inspect what the report HTML actually contains to determine which case applies. Then update the test fixture or assertion accordingly.

If PCA images are now always generated, consider replacing the "no PCA images" test with a "PCA images have expected structure" test instead.
