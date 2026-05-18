import math
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np

try:
    import scipy.signal as _scipy_signal
    _SCIPY_AVAILABLE = True
except ImportError:
    _scipy_signal = None
    _SCIPY_AVAILABLE = False

try:
    import torch
except Exception:
    torch = None


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(min(max(float(value), float(lo)), float(hi)))


def _db_to_linear(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


@dataclass
class _Smoother:
    value: float
    time_ms: float = 50.0

    def step(self, target: float, n_samples: int, fs: int) -> Tuple[float, float]:
        start = float(self.value)
        if n_samples <= 0 or fs <= 0:
            self.value = float(target)
            return start, float(self.value)
        tau = max(float(self.time_ms), 0.1) / 1000.0
        alpha = 1.0 - math.exp(-float(n_samples) / (tau * float(fs)))
        end = start + alpha * (float(target) - start)
        self.value = float(end)
        return start, end


@dataclass
class _BiquadState:
    # zi shape (1, 2) — compatible with scipy.signal.sosfilt and DF2T fallback
    zi: np.ndarray = field(default_factory=lambda: np.zeros((1, 2), dtype=np.float64))


def _design_peaking_eq(fc: float, q: float, gain_db: float, fs: int) -> Tuple[float, float, float, float, float]:
    fc = _clamp(fc, 10.0, 0.45 * float(fs))
    q = max(float(q), 0.05)
    A = 10.0 ** (float(gain_db) / 40.0)
    w0 = 2.0 * math.pi * fc / float(fs)
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / (2.0 * q)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A

    inv_a0 = 1.0 / a0
    return float(b0 * inv_a0), float(b1 * inv_a0), float(b2 * inv_a0), float(a1 * inv_a0), float(a2 * inv_a0)


def _biquad_process(
    x: np.ndarray,
    b0: float, b1: float, b2: float, a1: float, a2: float,
    state: _BiquadState,
) -> np.ndarray:
    x64 = x.astype(np.float64)
    if _SCIPY_AVAILABLE:
        sos = np.array([[b0, b1, b2, 1.0, a1, a2]], dtype=np.float64)
        y, zf = _scipy_signal.sosfilt(sos, x64, zi=state.zi)
        state.zi = zf
        return y.astype(np.float32)
    # Transposed Direct Form II fallback (zi[0,0]=z1, zi[0,1]=z2)
    y = np.empty_like(x64)
    z1, z2 = float(state.zi[0, 0]), float(state.zi[0, 1])
    for i in range(len(x64)):
        xn = x64[i]
        yn = b0 * xn + z1
        z1 = b1 * xn - a1 * yn + z2
        z2 = b2 * xn - a2 * yn
        y[i] = yn
    state.zi[0, 0] = z1
    state.zi[0, 1] = z2
    return y.astype(np.float32)


class AudioEqualizerNode:
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process"
    CATEGORY = "audio"

    _BAND_FC_HZ = (60.0, 150.0, 400.0, 1000.0, 3000.0, 8000.0)
    _BAND_Q = (0.9, 0.9, 1.0, 1.0, 1.1, 1.1)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        ro = {"multiline": False, "readonly": True, "display": "text"}
        return {
            "required": {
                "audio": ("AUDIO",),

                "band1_gain_db": ("FLOAT", {"default": 0.0, "min": -18.0, "max": 18.0, "step": 0.1, "display": "slider"}),
                "band1_label": ("STRING", {"default": "Band 1 (60 Hz)", **ro}),

                "band2_gain_db": ("FLOAT", {"default": 0.0, "min": -18.0, "max": 18.0, "step": 0.1, "display": "slider"}),
                "band2_label": ("STRING", {"default": "Band 2 (150 Hz)", **ro}),

                "band3_gain_db": ("FLOAT", {"default": 0.0, "min": -18.0, "max": 18.0, "step": 0.1, "display": "slider"}),
                "band3_label": ("STRING", {"default": "Band 3 (400 Hz)", **ro}),

                "band4_gain_db": ("FLOAT", {"default": 0.0, "min": -18.0, "max": 18.0, "step": 0.1, "display": "slider"}),
                "band4_label": ("STRING", {"default": "Band 4 (1 kHz)", **ro}),

                "band5_gain_db": ("FLOAT", {"default": 0.0, "min": -18.0, "max": 18.0, "step": 0.1, "display": "slider"}),
                "band5_label": ("STRING", {"default": "Band 5 (3 kHz)", **ro}),

                "band6_gain_db": ("FLOAT", {"default": 0.0, "min": -18.0, "max": 18.0, "step": 0.1, "display": "slider"}),
                "band6_label": ("STRING", {"default": "Band 6 (8 kHz)", **ro}),

                "limiter": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "limiter_label": ("STRING", {"default": "<mild--] limiter [--aggressive>", **ro}),

                "left_volume_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 6.0, "step": 0.1, "display": "slider"}),
                "left_volume_label": ("STRING", {"default": "Left channel volume", **ro}),

                "right_volume_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 6.0, "step": 0.1, "display": "slider"}),
                "right_volume_label": ("STRING", {"default": "Right channel volume", **ro}),
            }
        }

    def __init__(self):
        self._band_gain_smoothers = [_Smoother(0.0, time_ms=60.0) for _ in range(6)]
        self._band_fc_smoothers = [_Smoother(float(fc), time_ms=200.0) for fc in self._BAND_FC_HZ]
        self._band_q_smoothers = [_Smoother(float(q), time_ms=200.0) for q in self._BAND_Q]
        self._limiter_smoother = _Smoother(0.25, time_ms=80.0)
        self._lv_smoother = _Smoother(0.0, time_ms=60.0)
        self._rv_smoother = _Smoother(0.0, time_ms=60.0)
        self._biquad_states: list[list[_BiquadState]] = []
        self._limiter_env = 0.0

    def _extract_audio(self, audio: Any) -> Tuple[np.ndarray, int, Optional[dict], Optional[Any]]:
        audio_dict = audio if isinstance(audio, dict) else None
        waveform = audio_dict.get("waveform") if audio_dict is not None else audio
        sample_rate = int(audio_dict.get("sample_rate", 44100)) if audio_dict is not None else 44100

        torch_device = None
        if torch is not None and isinstance(waveform, torch.Tensor):
            try:
                torch_device = waveform.device
            except Exception:
                torch_device = None
            wf = waveform.detach()
            if not wf.is_floating_point():
                wf = wf.float()
            wf_np = wf.cpu().numpy()
        else:
            wf_np = np.asarray(waveform, dtype=np.float32)

        if wf_np.ndim == 1:
            wf_np = wf_np[None, None, :]
        elif wf_np.ndim == 2:
            if wf_np.shape[0] <= 8 and wf_np.shape[1] > wf_np.shape[0]:
                wf_np = wf_np[None, :, :]
            else:
                wf_np = wf_np.T[None, :, :]
        elif wf_np.ndim == 3:
            if not (wf_np.shape[1] <= 8 and wf_np.shape[2] > wf_np.shape[1]):
                wf_np = np.transpose(wf_np, (0, 2, 1))
        else:
            wf_np = wf_np.reshape(wf_np.shape[0], -1, wf_np.shape[-1])

        return wf_np.astype(np.float32, copy=False), sample_rate, audio_dict, torch_device

    def _pack_audio(self, waveform_np_bct: np.ndarray, sample_rate: int, audio_dict: Optional[dict], torch_device: Optional[Any]) -> Any:
        if audio_dict is None:
            return waveform_np_bct
        if torch is not None:
            wf_t = torch.from_numpy(waveform_np_bct)
            if torch_device is not None:
                try:
                    wf_t = wf_t.to(torch_device)
                except Exception:
                    pass
            return {"waveform": wf_t, "sample_rate": int(sample_rate)}
        return {"waveform": waveform_np_bct, "sample_rate": int(sample_rate)}

    def _ensure_states(self, n_channels: int):
        if self._biquad_states and len(self._biquad_states) == 6 and len(self._biquad_states[0]) == n_channels:
            return
        self._biquad_states = [[_BiquadState() for _ in range(n_channels)] for _ in range(6)]

    def _apply_limiter(self, audio_ct: np.ndarray, aggressiveness: float, fs: int) -> np.ndarray:
        if audio_ct.size == 0:
            return audio_ct

        aggressiveness = _clamp(aggressiveness, 0.0, 1.0)
        threshold = _db_to_linear(-1.0 - 8.0 * aggressiveness)
        knee_db = 6.0 + 12.0 * aggressiveness
        knee_half = _db_to_linear(knee_db / 2.0)
        makeup = _db_to_linear(3.0 * aggressiveness)
        attack_ms = 4.0 - 2.5 * aggressiveness
        release_ms = 120.0 + 220.0 * aggressiveness
        a_a = math.exp(-1.0 / (max(attack_ms, 0.1) / 1000.0 * float(fs)))
        a_r = math.exp(-1.0 / (max(release_ms, 0.1) / 1000.0 * float(fs)))

        n_samp = int(audio_ct.shape[1])
        env = float(self._limiter_env)
        y = np.empty_like(audio_ct, dtype=np.float32)

        for i in range(n_samp):
            peak = float(np.max(np.abs(audio_ct[:, i])))
            env = ((1.0 - a_a) * peak + a_a * env) if peak > env else ((1.0 - a_r) * peak + a_r * env)

            if env <= threshold:
                gain = 1.0
            else:
                over = env / threshold
                if knee_db > 0.0 and over < knee_half:
                    t = (over - 1.0) / (knee_half - 1.0)
                    gain = 1.0 / (1.0 + (t * t) * (over - 1.0))
                else:
                    gain = 1.0 / over

            y[:, i] = np.clip(audio_ct[:, i] * (gain * makeup), -1.0, 1.0)

        self._limiter_env = float(env)
        return y

    def process(
        self,
        audio,
        band1_gain_db, band1_label,
        band2_gain_db, band2_label,
        band3_gain_db, band3_label,
        band4_gain_db, band4_label,
        band5_gain_db, band5_label,
        band6_gain_db, band6_label,
        limiter, limiter_label,
        left_volume_db, left_volume_label,
        right_volume_db, right_volume_label,
    ):
        wf_bct, sample_rate, audio_dict, torch_device = self._extract_audio(audio)
        if wf_bct.size == 0:
            return (self._pack_audio(wf_bct, sample_rate, audio_dict, torch_device),)

        fs = int(sample_rate)
        bsz, n_ch, n_samp = int(wf_bct.shape[0]), int(wf_bct.shape[1]), int(wf_bct.shape[2])
        self._ensure_states(n_ch)

        band_targets = [
            float(band1_gain_db), float(band2_gain_db), float(band3_gain_db),
            float(band4_gain_db), float(band5_gain_db), float(band6_gain_db),
        ]
        lv_target_db = float(left_volume_db)
        rv_target_db = float(right_volume_db)
        lim_target = float(limiter)

        out = np.empty_like(wf_bct, dtype=np.float32)
        block_size = 128

        for b in range(bsz):
            y_ct = wf_bct[b].copy()

            for start in range(0, n_samp, block_size):
                end = min(start + block_size, n_samp)
                n_blk = end - start
                if n_blk <= 0:
                    continue

                for band_idx in range(6):
                    _, gain_db = self._band_gain_smoothers[band_idx].step(_clamp(band_targets[band_idx], -18.0, 18.0), n_blk, fs)
                    _, fc = self._band_fc_smoothers[band_idx].step(float(self._BAND_FC_HZ[band_idx]), n_blk, fs)
                    _, q = self._band_q_smoothers[band_idx].step(float(self._BAND_Q[band_idx]), n_blk, fs)
                    b0, b1, b2, a1, a2 = _design_peaking_eq(float(fc), float(q), float(gain_db), fs)
                    for ch in range(n_ch):
                        y_ct[ch, start:end] = _biquad_process(y_ct[ch, start:end], b0, b1, b2, a1, a2, self._biquad_states[band_idx][ch])

                lv_start, lv_end = self._lv_smoother.step(_clamp(lv_target_db, -24.0, 6.0), n_blk, fs)
                rv_start, rv_end = self._rv_smoother.step(_clamp(rv_target_db, -24.0, 6.0), n_blk, fs)

                if n_ch >= 1:
                    y_ct[0, start:end] *= np.linspace(_db_to_linear(lv_start), _db_to_linear(lv_end), n_blk, endpoint=False, dtype=np.float32)
                if n_ch >= 2:
                    y_ct[1, start:end] *= np.linspace(_db_to_linear(rv_start), _db_to_linear(rv_end), n_blk, endpoint=False, dtype=np.float32)

            _, lim_end = self._limiter_smoother.step(_clamp(lim_target, 0.0, 1.0), n_samp, fs)
            y_ct = self._apply_limiter(y_ct, lim_end, fs)
            out[b] = y_ct

        return (self._pack_audio(out, sample_rate, audio_dict, torch_device),)
