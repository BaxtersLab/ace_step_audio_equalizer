import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:  # torch is expected in ComfyUI, but keep import-safe
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
    z1: float = 0.0
    z2: float = 0.0


def _design_peaking_eq(fc: float, q: float, gain_db: float, fs: int) -> Tuple[float, float, float, float, float]:
    fc = _clamp(fc, 10.0, 0.45 * float(fs))
    q = max(float(q), 0.05)
    gain_db = float(gain_db)

    A = 10.0 ** (gain_db / 40.0)
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
    b0 *= inv_a0
    b1 *= inv_a0
    b2 *= inv_a0
    a1 *= inv_a0
    a2 *= inv_a0

    return float(b0), float(b1), float(b2), float(a1), float(a2)


def _biquad_process_df2t(x: np.ndarray, b0: float, b1: float, b2: float, a1: float, a2: float, state: _BiquadState) -> np.ndarray:
    y = np.empty_like(x, dtype=np.float32)
    z1 = float(state.z1)
    z2 = float(state.z2)
    for i in range(int(x.shape[0])):
        xn = float(x[i])
        yn = b0 * xn + z1
        z1 = b1 * xn - a1 * yn + z2
        z2 = b2 * xn - a2 * yn
        y[i] = yn
    state.z1 = float(z1)
    state.z2 = float(z2)
    return y


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

        self._meter_peak = 0.0
        self._meter_peak_smooth = 0.0

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

        threshold_db = -1.0 - 8.0 * aggressiveness
        knee_db = 6.0 + 12.0 * aggressiveness
        attack_ms = 4.0 - 2.5 * aggressiveness
        release_ms = 120.0 + 220.0 * aggressiveness
        makeup_db = 0.0 + 3.0 * aggressiveness

        threshold = _db_to_linear(threshold_db)
        makeup = _db_to_linear(makeup_db)

        n_ch, n_samp = int(audio_ct.shape[0]), int(audio_ct.shape[1])
        if n_samp == 0:
            return audio_ct

        attack_tc = max(float(attack_ms), 0.1) / 1000.0
        release_tc = max(float(release_ms), 0.1) / 1000.0
        a_a = math.exp(-1.0 / (attack_tc * float(fs)))
        a_r = math.exp(-1.0 / (release_tc * float(fs)))

        env = float(self._limiter_env)
        y = np.empty_like(audio_ct, dtype=np.float32)
        knee_half = _db_to_linear(knee_db / 2.0)

        for i in range(n_samp):
            peak = 0.0
            for ch in range(n_ch):
                v = abs(float(audio_ct[ch, i]))
                if v > peak:
                    peak = v

            if peak > env:
                env = (1.0 - a_a) * peak + a_a * env
            else:
                env = (1.0 - a_r) * peak + a_r * env

            if env <= threshold:
                gain = 1.0
            else:
                over = env / threshold
                if knee_db > 0.0 and over < knee_half:
                    t = (over - 1.0) / (knee_half - 1.0)
                    gain = 1.0 / (1.0 + (t * t) * (over - 1.0))
                else:
                    gain = 1.0 / over

            g = float(gain) * makeup
            for ch in range(n_ch):
                y[ch, i] = _clamp(float(audio_ct[ch, i]) * g, -1.0, 1.0)

        self._limiter_env = float(env)
        return y

    def process(
        self,
        audio,
        band1_gain_db,
        band1_label,
        band2_gain_db,
        band2_label,
        band3_gain_db,
        band3_label,
        band4_gain_db,
        band4_label,
        band5_gain_db,
        band5_label,
        band6_gain_db,
        band6_label,
        limiter,
        limiter_label,
        left_volume_db,
        left_volume_label,
        right_volume_db,
        right_volume_label,
    ):
        wf_bct, sample_rate, audio_dict, torch_device = self._extract_audio(audio)
        if wf_bct.size == 0:
            return (self._pack_audio(wf_bct, sample_rate, audio_dict, torch_device),)

        fs = int(sample_rate)
        bsz, n_ch, n_samp = int(wf_bct.shape[0]), int(wf_bct.shape[1]), int(wf_bct.shape[2])
        self._ensure_states(n_ch)

        band_targets = [
            float(band1_gain_db),
            float(band2_gain_db),
            float(band3_gain_db),
            float(band4_gain_db),
            float(band5_gain_db),
            float(band6_gain_db),
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
                        state = self._biquad_states[band_idx][ch]
                        y_ct[ch, start:end] = _biquad_process_df2t(y_ct[ch, start:end], b0, b1, b2, a1, a2, state)

                lv_start, lv_end = self._lv_smoother.step(_clamp(lv_target_db, -24.0, 6.0), n_blk, fs)
                rv_start, rv_end = self._rv_smoother.step(_clamp(rv_target_db, -24.0, 6.0), n_blk, fs)

                lv0 = _db_to_linear(lv_start)
                lv1 = _db_to_linear(lv_end)
                rv0 = _db_to_linear(rv_start)
                rv1 = _db_to_linear(rv_end)

                if n_ch >= 1:
                    g = np.linspace(lv0, lv1, n_blk, endpoint=False, dtype=np.float32)
                    y_ct[0, start:end] *= g
                if n_ch >= 2:
                    g = np.linspace(rv0, rv1, n_blk, endpoint=False, dtype=np.float32)
                    y_ct[1, start:end] *= g

            _, lim_end = self._limiter_smoother.step(_clamp(lim_target, 0.0, 1.0), n_samp, fs)
            y_ct = self._apply_limiter(y_ct, lim_end, fs)

            peak = float(np.max(np.abs(y_ct))) if y_ct.size else 0.0
            self._meter_peak = peak
            meter_alpha = 1.0 - math.exp(-float(n_samp) / (0.08 * float(fs)))
            self._meter_peak_smooth = float(self._meter_peak_smooth + meter_alpha * (peak - self._meter_peak_smooth))

            out[b] = y_ct

        return (self._pack_audio(out, sample_rate, audio_dict, torch_device),)
