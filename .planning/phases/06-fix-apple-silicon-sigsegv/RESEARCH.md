# Research: Environment audit for Apple‑Silicon SIGSEGV

## Goal
Identify any native extension binaries, OpenMP libraries, or other compiled components that are built for the wrong architecture (x86_64) and could cause a NULL‑pointer dereference on arm64.

## Steps
1. List all imported Python packages that ship compiled extensions (`pip list --format=freeze | grep -E "(numpy|pandas|torch|opencv|ffmpeg)"`).
2. For each package, locate the `.so` files and run `otool -L <file>` to verify they link against arm64 libraries only.
3. Verify that Homebrew's `libomp.dylib` is present at `/opt/homebrew/lib/` and that no other OpenMP runtime (e.g., from an x86_64 brew) is on the load path.
4. Record any mismatches in this document for later fixing.

## Expected outcome
A concise list of problematic binaries (if any) and a clear remediation plan (re‑install via arm64 wheels, rebuild from source, or adjust `DYLD_LIBRARY_PATH`).
