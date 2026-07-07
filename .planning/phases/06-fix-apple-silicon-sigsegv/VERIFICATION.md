# Verification

The following conditions must be met for this phase to be considered complete:

1. Import sanity - Running the `faulthandler` script (see Step 7 in PLAN) exits with exit code 0 and no segmentation fault.
2. Library linkage - All .so files reported by `otool -L` link only to arm64 libraries; no x86_64 architecture appears.
3. Full pipeline success - Executing the training/quantisation workflow produces a valid `model.onnx` file and prints a quantisation summary without errors.
4. Metric improvement - Reported SNR/PSNR values are >= 0 dB (or otherwise demonstrably better than the baseline negative numbers previously observed).
5. No semaphore warning - The interpreter shutdown log no longer contains `resource_tracker: There appear to be X leaked semaphore objects`.

When all five items pass, record a brief note in this file confirming success and the date of verification.
