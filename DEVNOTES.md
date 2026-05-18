# DEVNOTES — ACE Step Audio Equalizer
> Carry this file into every future session. It is the handoff document.

---

## Current Status: IMPROVED — AWAITING FIRST TEST

Session 1 (2026-05-18) optimized DSP performance and fixed a key collision risk.

---

## What This Node Does

6-band parametric equalizer with stereo-linked limiter and independent L/R volume trims.
Intended as a mastering/polish node — place late in the audio chain.

### EQ Bands (fixed centre frequencies)
| Band | Frequency | Q |
|---|---|---|
| 1 | 60 Hz | 0.9 |
| 2 | 150 Hz | 0.9 |
| 3 | 400 Hz | 1.0 |
| 4 | 1 kHz | 1.0 |
| 5 | 3 kHz | 1.1 |
| 6 | 8 kHz | 1.1 |

Range: ±18 dB per band.

### Limiter
Stereo-linked (peak taken across all channels per sample). Single `aggressiveness` knob (0–1) controls threshold, knee, attack/release, and makeup gain as a continuous curve:
- `threshold_db = -1 - 8 × aggressiveness`
- `knee_db = 6 + 12 × aggressiveness`
- `attack_ms = 4 - 2.5 × aggressiveness`
- `release_ms = 120 + 220 × aggressiveness`
- `makeup_db = 3 × aggressiveness`

### Volume trims
Independent L/R gain in dB (−24 to +6). Applied inside the block loop with linear ramp per block to avoid zipper noise.

### Parameter smoothing
All controls use exponential smoothers (60–200 ms time constants) applied per 128-sample block, so abrupt knob moves don't produce clicks.

---

## What Changed (Session 1 — 2026-05-18)

| File | Change |
|---|---|
| `ace_step_audio_equalizer.py` | **scipy biquad**: replaced Python per-sample loop in `_biquad_process_df2t` with `scipy.signal.sosfilt` (DF2T fallback retained). Eliminates ~8M Python iterations for 30s stereo audio. |
| `ace_step_audio_equalizer.py` | **Vectorized limiter channels**: inner `for ch` loop replaced with `np.max(np.abs(...))` and `np.clip(... * gain)` over channel axis. Sample loop unavoidable (envelope is sequential). |
| `ace_step_audio_equalizer.py` | **Removed dead meter code**: `_meter_peak` / `_meter_peak_smooth` were computed each run but never exposed as outputs or UI feedback. |
| `ace_step_audio_equalizer.py` | **`_BiquadState` refactored**: now holds `zi: np.ndarray` (shape `(1, 2)`) for direct use with `sosfilt`; fallback reads z1/z2 from `zi[0, 0]` and `zi[0, 1]`. |
| `__init__.py` | Key renamed from `"ACE Step Audio Equalizer"` (spaces, collision risk) to `"ACEStep_AudioEqualizer"`. |
| `requirements.txt` | Created. |
| `DEVNOTES.md` | Created this file. |

---

## DSP Notes

**Biquad filter:** Audio EQ Cookbook peaking EQ coefficients. Uses `scipy.signal.sosfilt` with SOS format `[b0, b1, b2, 1.0, a1, a2]`. State (`zi`, shape `(1, 2)`) is carried across blocks for continuous filtering. Python DF2T fallback available if scipy absent.

**Block processing:** 128-sample blocks. EQ coefficients are recalculated each block based on smoothed parameter values — allows real-time parameter changes without audible discontinuities.

**Limiter envelope:** Instant-attack / exponential-release pattern. `env` stored in `self._limiter_env` so the envelope carries over between successive graph executions (prevents pumping restart artifacts).

**Key collision note:** Previous key `"ACE Step Audio Equalizer"` used spaces. Update any existing workflow JSON to use `"ACEStep_AudioEqualizer"` in the node `type` field.

---

## What Still Needs Doing

### Priority 1 — Test
- [ ] **Test all-flat EQ (all bands 0 dB)** — output should be identical to input (modulo limiter)
- [ ] **Test band boost/cut** — verify each band affects correct frequency region
- [ ] **Test limiter at aggressiveness=0** — mild, near-transparent limiting
- [ ] **Test limiter at aggressiveness=1** — aggressive compression with makeup gain
- [ ] **Test mono input** — shape normalization should produce mono output cleanly

### Priority 2 — Improvements
- [ ] **Expose band Q as widgets** — currently fixed per band. Parametric EQ users expect adjustable Q.
- [ ] **Add high-pass / low-pass shelf options** — two outermost bands (60 Hz, 8 kHz) are natural candidates for shelving filters instead of peaking.
- [ ] **RETURN_TYPES meter output** — `_meter_peak` code was removed; could be re-added as a `FLOAT` output for downstream metering nodes.

### Priority 3 — Polish
- [ ] **Larger block size option** — 128 samples is conservative. 512 or 1024 would reduce coefficient recalculation overhead with negligible smoothing impact.

---

## Known Risks / Watch Out For

- **Limiter envelope persists across runs** — `self._limiter_env` is an instance variable that survives between graph executions. On the first run after a loud clip, the envelope may still be elevated, causing the next (quieter) run to be attenuated. Reset is only triggered by ComfyUI reloading the node. This matches DAW limiter behaviour but may surprise users.
- **scipy fallback not tested** — the Python DF2T fallback in `_biquad_process` shares `zi` storage with the scipy path. If scipy is unavailable, z1/z2 semantics differ from sosfilt zi semantics; audio will be slightly different at non-zero initial state (i.e. mid-file). Starting from zero (normal case) is fine.

---

## Node Key Names

| Key (workflow JSON `type` field) | Display name |
|---|---|
| `ACEStep_AudioEqualizer` | ACE Step Audio Equalizer |

---

## Session Log

| Date | What happened |
|---|---|
| Unknown | Initial build. Well-implemented DSP — biquad EQ, stereo limiter, parameter smoothing. Python inner loops slow for long audio. Key used spaces. |
| 2026-05-18 | scipy biquad, vectorized limiter channels, dead meter code removed, key prefix added. |
