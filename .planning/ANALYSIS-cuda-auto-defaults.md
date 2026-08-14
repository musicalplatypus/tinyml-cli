# Critical analysis — the CUDA auto-defaults policy (`apply_hardware_defaults`)

**Written:** 2026-08-06, from the `mmcli` side, for the session working in
`tinyml-tensorlab` / `tinyml-modelmaker`.
**Subject:** `tinyml-modelmaker/tinyml_modelmaker/utils/hardware_defaults.py`, and its four call
sites in `ai_modules/{timeseries,vision,audio,radar}/params.py`.
**Introduced by:** the `pr/hardware-defaults` line (`e83aefe`), extended to vision/audio/radar in
`3c900b2`, `baf334a`, `9a5facc`; policy recorded in `07aab6d`.

This is a review, not a change request. Nothing in modelmaker was modified. Every finding below
was checked by executing the code, not by reading it; where I could not check something, it says
so explicitly.

## The policy as written

```python
def apply_hardware_defaults(params, explicitly_set: set) -> None:
    if not torch.cuda.is_available():
        return
    if 'compile_model' not in explicitly_set and hasattr(params.training, 'compile_model'):
        if getattr(params.training, 'compile_model', 0) == 0:
            params.training.compile_model = 1
    if 'native_amp' not in explicitly_set and hasattr(params.training, 'native_amp'):
        if not getattr(params.training, 'native_amp', False):
            params.training.native_amp = True
```

Called as:

```python
user_training_keys = set(args[0].get('training', {}).keys()) \
    if args and isinstance(args[0], dict) else set()
params = utils.ConfigDict(default_params, *args, **kwargs)
apply_hardware_defaults(params, user_training_keys)
```

The intent is sound and worth keeping: CUDA hosts should not leave `torch.compile` and AMP on the
table because nobody remembered to ask. The problems are in how "the user didn't ask" is
determined, and in what happens silently once it is.

---

## F-1 — BUG: a YAML config passed by path has its explicit settings overridden

**Severity: high. Reproduced.**

`ConfigDict` accepts a YAML path string as its first positional argument
(`utils/config_dict.py:44-46`). The call site only harvests `user_training_keys` when `args[0]` is
a `dict` (`params.py:229-230`). So for a path caller, `explicitly_set` is **empty** — and every
setting in that YAML is treated as "the user didn't ask", including settings the user wrote
specifically to turn the feature *off*.

Reproduced on this machine with `torch.cuda.is_available` patched to `True`, using a YAML
containing `training: {compile_model: 0, native_amp: false}`:

```
A) passed as a DICT      -> compile_model=0  native_amp=False   (respected)
B) passed as a YAML PATH -> compile_model=1  native_amp=True    (overridden)
```

Same file, same intent, opposite outcome, decided by the caller's argument form. A user who
writes `compile_model: 0` to work around an inductor failure will find it silently re-enabled.

The in-code comment shows the YAML-path case was *noticed* — the guard exists precisely because
`args[0]` may be a string. But the conclusion drawn was "don't inspect it", which converts a
crash into a silent override. That is the worse of the two failure modes.

**This is the finding I would fix first, independently of any wider redesign.**

## F-2 — The same config produces numerically different training on different machines

**Severity: high. Design, not bug.**

`native_amp` changes arithmetic precision. A run on a CUDA host is therefore not merely faster
than the same config on CPU or MPS — it computes different numbers, and can converge differently.
The config file is no longer a complete description of the run.

This matters more here than in a generic trainer, for two project-specific reasons:

1. **This toolchain compares runs.** Run archives, `mmcli compare`, and the app's comparison view
   all assume two runs of the same config are comparable. Across a CUDA and a non-CUDA host, they
   silently are not.
2. **Quantisation is central.** Auto-quantisation and QAT run downstream of training. Enabling
   mixed precision ahead of a quantisation-sensitive pipeline, without the user asking, is a
   plausible source of accuracy differences that will be very hard to attribute later — the
   symptom appears in the quantised result, far from the cause.

## F-3 — No observability: nothing records that the policy fired

**Severity: high — and it is what makes F-1 and F-2 expensive rather than merely surprising.**

The function mutates `params` and returns. Nothing is logged, and nothing marks the resulting
config as modified. A user comparing a CUDA run against an MPS run has no artifact explaining why
they differ, and no reason to suspect the config.

Every other finding here becomes debuggable if the effective config is recorded. I would rank
this above the redesign questions: a silent policy is the root problem; the specific defaults are
secondary.

## F-4 — "Explicitly set" is inferred from key presence, which quietly punishes complete configs

**Severity: medium. This is where `mmcli` is exposed.**

Membership of `explicitly_set` is *presence of the key*, not intent. Any tool that emits a
complete config template — every key with its default — opts out of the policy entirely and
silently gets no compile and no AMP on CUDA.

`mmcli` currently escapes this by accident. Its `_set()` skips `None`, so unpassed flags are
omitted; verified empirically that with no flags it writes only `{enable, model_name}` under
`training`, leaving the policy free to fire. That is **compatibility by luck, not by contract** —
one reasonable "write explicit defaults" refactor in `mmcli/builder.py` would disable torch.compile
and AMP fleet-wide on CUDA hosts, with no error and no log line.

Note this is the same failure shape already hit once on this boundary: `mmcli` emitting
`feature_extraction_name: "default"` suppressed modelmaker's own preset resolution. Same root
cause — presence of a key is read as intent.

**Whatever else changes, the contract needs stating out loud** so consumers can test against it
rather than against observed behaviour. `mmcli` will add a regression test either way; it should
be testing a documented promise.

## F-5 — `torch.compile` is auto-enabled regardless of run length

**Severity: medium.**

Compilation has a fixed warm-up cost paid before the first useful step. The policy applies it
uniformly, with no regard for how long the run is.

Concretely, from this project's own measurements: in a 1-epoch training run, the epoch itself is
roughly 15s of a ~360s wall time — the remainder is feature extraction, auto-quantisation,
evaluation and export. Adding compile warm-up to short runs and smoke tests is plausibly a net
loss. The benefit is real but amortises only over long runs.

**Not verified:** I have no CUDA host, so I could not measure the crossover point. The benchmark
legs recorded in `fe4b02a` / `729cbe4` / `43fc13a` may already answer this; if they were measured
on long runs only, the short-run case is still open.

## F-6 — The two knobs have different risk profiles and should not share one switch

**Severity: medium.**

`compile_model` is a performance and compatibility risk: it fails loudly (inductor errors,
unsupported ops) and does not change results when it works. `native_amp` is a numerical risk: it
succeeds quietly and changes results. Bundling them under one "hardware defaults" decision means
you cannot take the safe one without the consequential one.

The naming reinforces the conflation. CUDA availability is a *capability*; enabling compile and
AMP is a *policy*. There is currently no way to express "this host has CUDA, and I want
reproducible fp32."

## F-7 — The `== 0` / `not native_amp` guards obscure intent

**Severity: low.**

If a key is absent from `explicitly_set`, its value is whatever the default chain produced.
Re-checking `== 0` before assigning `1` is a no-op in the common path and makes the intended
precedence unreadable — is a preset-supplied `compile_model=1` meant to be authoritative, or
merely coincidentally equal? Today it works by accident either way. A future non-binary value
(say a compile *mode*) would be silently preserved, which may or may not be intended.

## F-8 — Stale `hasattr` guards and a stale docstring

**Severity: low, but actively misleading.**

The docstring says the `hasattr` guards keep this "safe for params that don't carry these fields
yet (vision, audio — Phase 2)". Vision and audio now do carry them (`3c900b2`, `baf334a`), as does
radar (`9a5facc`) — all four modules define `compile_model` in `params.py`. The guards are dead
weight, and the comment tells the next reader something false about the current state.

---

## Recommendations

Ordered by value, not by effort. (1) and (2) are worth doing whatever is decided about the rest.

**1. Fix F-1 by deriving `explicitly_set` from the resolved config, not from `args[0]`.**
Build the `ConfigDict` first, then determine which keys came from user input — or, if that
distinction cannot be recovered after merging, load the YAML to a dict at the call site and pass
that. The current "if it isn't a dict, assume nothing was set" fallback should be inverted: if
intent cannot be determined, **do not apply the policy**. Failing closed preserves the user's file;
failing open silently overwrites it.

**2. Make the policy announce itself (F-3).** At minimum, one INFO line naming what changed and
why: `CUDA detected: compile_model 0->1, native_amp False->True (not specified in config)`.
Better, record the *effective* config alongside the run, not the requested one. That is
independently valuable to `mmcli` and the app, which both archive run manifests and currently
archive the requested config.

**3. State the caller contract explicitly (F-4).** Document that omitting a key means "no
preference" and that emitting it — even at its default — pins it. Then consumers can test against
a promise. Without this, every config-generating tool is one refactor away from silently opting
out.

**4. Separate the two knobs (F-6).** Even keeping both auto-on, give them independent controls and
consider a single `auto_hardware_defaults: true|false` master switch. AMP is the one that changes
results; it deserves its own opt-out that does not cost the user compilation too.

**5. Reconsider auto-compile for short runs (F-5).** Gating on `training_epochs` is the obvious
lever. If the existing CUDA benchmarks only cover long runs, a short-run leg would settle whether
this matters before adding a threshold nobody needs.

**6. Housekeeping (F-7, F-8).** Drop the stale `hasattr` guards and correct the docstring; make the
precedence explicit rather than incidental.

## What I did not verify

- **Anything requiring a CUDA host.** F-1's reproduction patched `torch.cuda.is_available`, which
  exercises the branch logic but not real compile/AMP behaviour. F-5's cost argument is inference
  from this project's CPU/MPS timings, not a CUDA measurement.
- **Whether the existing benchmark legs already cover short runs.** They are referenced in
  `fe4b02a`, `729cbe4`, `43fc13a`, `b27a579`; I did not read them.
- **Interaction with QAT and auto-quantisation specifically (F-2).** The concern is structural —
  AMP upstream of a quantisation-sensitive stage — not an observed regression. It should be cheap
  to check on a CUDA host by running one config with AMP forced off and comparing quantised
  accuracy.

## Why this came from the mmcli side

`mmcli` writes modelmaker configs, so any rule of the form "presence of a key changes behaviour"
is a cross-repo contract whether or not it was intended as one. Today `mmcli` satisfies it by
coincidence. This document exists so the rule can be decided deliberately in modelmaker, and then
pinned by a test in `mmcli`, rather than each side discovering the other's assumptions through a
silent regression.
