import numpy as np
from typing import Any, Optional, Tuple


try:
    import torch
except Exception:  # torch is expected in ComfyUI, but keep import-safe
    torch = None


class AudioEqualizerNode:
    RETURN_TYPES = ("AUDIO", "BOOLEAN", "STRING", "FLOAT", "FLOAT", "BOOLEAN", "BOOLEAN", "FLOAT", "FLOAT", "BOOLEAN", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("audio", "reset", "status", "peak", "smoothed_peak", "hit", "hit_event", "gr_linear", "gr_db", "limiter_hit", "rms", "left_peak", "right_peak")
    FUNCTION = "process"
    CATEGORY = "audio"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "audio": ("AUDIO",),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 384000}),
            },
                "optional": {
                "frequencies": ("STRING", {"default": "100,1000,5000"}),
                "gains_db": ("STRING", {"default": "0,0,0"}),
                "qs": ("STRING", {"default": "1.0,1.0,1.0"}),
                "limiter_mode": (["mild", "medium", "heavy"], {"default": "mild"}),
                "knee_db": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 40.0}),
                "makeup_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0}),
                "wet": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "stateful": ("BOOLEAN", {"default": True}),
                "reset": ("BOOLEAN", {"default": False}),
                "meter_attack_ms": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 1000.0}),
                "meter_release_ms": ("FLOAT", {"default": 100.0, "min": 1.0, "max": 5000.0}),
                "hit_threshold_db": ("FLOAT", {"default": -1.0, "min": -60.0, "max": 12.0}),
                "enabled": ("BOOLEAN", {"default": True}),
                    "bypass": ("BOOLEAN", {"default": False}),
                },
        }

    def _extract_audio(self, audio: Any, sample_rate: int) -> Tuple[np.ndarray, int, Any, Optional[Any]]:
        """Return (waveform_np, sample_rate, passthrough_audio_dict, torch_device).

        ComfyUI AUDIO is typically a dict: {"waveform": torch.Tensor[B,C,T], "sample_rate": int}.
        This node also tolerates raw arrays.
        """
        audio_dict = audio if isinstance(audio, dict) else None
        waveform = audio_dict.get("waveform") if audio_dict is not None else audio
        sr = int(audio_dict.get("sample_rate", sample_rate)) if audio_dict is not None else int(sample_rate)

        torch_device = None
        if torch is not None and hasattr(waveform, "device"):
            try:
                torch_device = waveform.device
            except Exception:
                torch_device = None

        if torch is not None and isinstance(waveform, torch.Tensor):
            wf = waveform.detach()
            if wf.is_floating_point() is False:
                wf = wf.float()
            wf_np = wf.cpu().numpy()
        else:
            wf_np = np.asarray(waveform, dtype=float)

        # Normalize to (batch, channels, samples)
        if wf_np.ndim == 1:
            wf_np = wf_np[None, None, :]
        elif wf_np.ndim == 2:
            # Prefer (channels, samples) when plausible
            if wf_np.shape[0] <= 8 and wf_np.shape[1] > wf_np.shape[0]:
                wf_np = wf_np[None, :, :]
            else:
                wf_np = wf_np.T[None, :, :]
        elif wf_np.ndim == 3:
            # Assume either (batch, channels, samples) or (batch, samples, channels)
            if not (wf_np.shape[1] <= 8 and wf_np.shape[2] > wf_np.shape[1]):
                wf_np = np.transpose(wf_np, (0, 2, 1))
        else:
            # Best-effort flatten: treat last axis as samples
            wf_np = wf_np.reshape(wf_np.shape[0], -1, wf_np.shape[-1])

        return wf_np.astype(np.float32, copy=False), sr, audio_dict, torch_device

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

    def __init__(self):
        # stateful filter states: list per band, each state is dict with arrays x1,x2,y1,y2 shape=(channels,)
        self.filter_states = []  # will be (n_bands) length when stateful
        self.limiter_env = 0.0  # smoothed envelope for limiter
        # smoothing/peak state for meter
        self.smoothed_peak = 0.0
        self._last_hit = False
        self._reset_handled = False

    def reset(self):
        self.filter_states = []
        self.limiter_env = 0.0
        self.smoothed_peak = 0.0
        self._last_hit = False
        self._reset_handled = False

    def _parse_floats_list(self, value: Any) -> np.ndarray:
        if value is None:
            return np.array([], dtype=float)
        if isinstance(value, (list, tuple, np.ndarray)):
            try:
                return np.asarray([float(v) for v in value], dtype=float)
            except Exception:
                return np.array([], dtype=float)
        s = str(value).strip()
        if s == "":
            return np.array([], dtype=float)
        parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip() != ""]
        try:
            return np.asarray([float(p) for p in parts], dtype=float)
        except Exception:
            return np.array([], dtype=float)

    def _design_peaking_biquad(self, fc: float, q: float, gain_db: float, fs: int):
        # RBJ peaking EQ
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * np.pi * fc / fs
        alpha = np.sin(w0) / (2.0 * q)
        cos_w0 = np.cos(w0)

        b0 = 1.0 + alpha * A
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / A

        return b0, b1, b2, a0, a1, a2

    def _apply_biquad_stateful(self, data: np.ndarray, b0: float, b1: float, b2: float, a0: float, a1: float, a2: float, state: dict) -> np.ndarray:
        # data: shape (n_samples,) or (n_samples, n_channels)
        if data.ndim == 1:
            x = data
            y = np.empty_like(x)
            x1 = state.get("x1", 0.0)
            x2 = state.get("x2", 0.0)
            y1 = state.get("y1", 0.0)
            y2 = state.get("y2", 0.0)
            for n in range(x.shape[0]):
                xn = x[n]
                yn = (b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0
                y[n] = yn
                x2, x1 = x1, xn
                y2, y1 = y1, yn
            state["x1"], state["x2"], state["y1"], state["y2"] = x1, x2, y1, y2
            return y
        else:
            # multi-channel
            n_samples, n_ch = data.shape
            y = np.empty_like(data)
            # states per channel stored as arrays in state dict
            x1_arr = state.get("x1", np.zeros(n_ch, dtype=float))
            x2_arr = state.get("x2", np.zeros(n_ch, dtype=float))
            y1_arr = state.get("y1", np.zeros(n_ch, dtype=float))
            y2_arr = state.get("y2", np.zeros(n_ch, dtype=float))
            # ensure correct lengths
            if x1_arr.shape[0] != n_ch:
                x1_arr = np.zeros(n_ch, dtype=float)
                x2_arr = np.zeros(n_ch, dtype=float)
                y1_arr = np.zeros(n_ch, dtype=float)
                y2_arr = np.zeros(n_ch, dtype=float)
            for ch in range(n_ch):
                x1 = x1_arr[ch]
                x2 = x2_arr[ch]
                y1 = y1_arr[ch]
                y2 = y2_arr[ch]
                for n in range(n_samples):
                    xn = data[n, ch]
                    yn = (b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0
                    y[n, ch] = yn
                    x2, x1 = x1, xn
                    y2, y1 = y1, yn
                x1_arr[ch] = x1
                x2_arr[ch] = x2
                y1_arr[ch] = y1
                y2_arr[ch] = y2
            state["x1"], state["x2"], state["y1"], state["y2"] = x1_arr, x2_arr, y1_arr, y2_arr
            return y

    def _db_to_linear(self, db: float) -> float:
        return 10.0 ** (db / 20.0)

    def _apply_envelope_limiter(self, data: np.ndarray, threshold: float, attack_ms: float, release_ms: float, fs: int, knee_db: float = 0.0, makeup_db: float = 0.0, initial_env: Optional[float] = None) -> Tuple[np.ndarray, float, float]:
        # Compute per-buffer peak envelope and apply soft-knee + makeup; stateful envelope provided via initial_env
        if data.size == 0:
            env = initial_env if initial_env is not None else 0.0
            return data, env, 1.0

        peak = float(np.max(np.abs(data)))
        env = initial_env if initial_env is not None else peak

        attack_tc = max(attack_ms, 0.001) / 1000.0
        release_tc = max(release_ms, 0.001) / 1000.0
        attack_coeff = np.exp(-1.0 / (attack_tc * fs)) if attack_tc > 0 else 0.0
        release_coeff = np.exp(-1.0 / (release_tc * fs)) if release_tc > 0 else 0.0

        # one-step smoothing approximation
        if peak > env:
            env = (1.0 - attack_coeff) * peak + attack_coeff * env
        else:
            env = (1.0 - release_coeff) * peak + release_coeff * env

        # soft-knee: compute reduction factor
        if knee_db and knee_db > 0.0:
            # threshold and knee in linear
            knee_lin = self._db_to_linear(knee_db)
            lower = threshold / (10 ** (knee_db / 20.0))
            upper = threshold
            # compute soft knee gain reduction (simple smoothstep)
            if env <= lower:
                gain = 1.0
            elif env >= upper:
                gain = threshold / env if env > 0 else 1.0
            else:
                t = (env - lower) / (upper - lower)
                soft = t * t * (3 - 2 * t)
                gain = 1.0 * (1.0 - soft) + (threshold / env if env > 0 else 1.0) * soft
        else:
            gain = threshold / env if env > 0 else 1.0
            if gain > 1.0:
                gain = 1.0

        # apply makeup
        makeup_lin = self._db_to_linear(makeup_db) if makeup_db else 1.0
        out = data * gain * makeup_lin
        applied_gain = gain * makeup_lin
        return out, env, applied_gain

    def process(
        self,
        audio: Any,
        sample_rate: int,
        frequencies: Optional[Any] = "100,1000,5000",
        gains_db: Optional[Any] = "0,0,0",
        qs: Optional[Any] = "1.0,1.0,1.0",
        limiter_mode: str = "mild",
        knee_db: float = 0.0,
        makeup_db: float = 0.0,
        wet: float = 1.0,
        stateful: bool = True,
        reset: bool = False,
        meter_attack_ms: float = 5.0,
        meter_release_ms: float = 100.0,
        hit_threshold_db: float = -1.0,
        enabled: bool = True,
        bypass: bool = False,
    ) -> Tuple[Any, bool, str, float, float, bool, bool, float, float, bool, float, float, float]:
        # Handle reset rising edge (auto-unlatch)
        reset_performed = False
        if reset and not self._reset_handled:
            self.reset()
            reset_performed = True
            self._reset_handled = True
        if not reset:
            self._reset_handled = False

        wf_bct, sr, audio_dict, torch_device = self._extract_audio(audio, sample_rate)

        status = "processed_stateless"
        if (not enabled) or bool(bypass):
            status = "bypassed"
            peak = float(np.max(np.abs(wf_bct))) if wf_bct.size else 0.0

            if peak > self.smoothed_peak:
                attack_coeff = np.exp(-1.0 / (max(meter_attack_ms, 0.001) / 1000.0 * sr))
                self.smoothed_peak = (1.0 - attack_coeff) * peak + attack_coeff * self.smoothed_peak
            else:
                release_coeff = np.exp(-1.0 / (max(meter_release_ms, 0.001) / 1000.0 * sr))
                self.smoothed_peak = (1.0 - release_coeff) * peak + release_coeff * self.smoothed_peak

            threshold_lin = self._db_to_linear(hit_threshold_db)
            hit = self.smoothed_peak >= threshold_lin
            hit_event = hit and (not self._last_hit)
            self._last_hit = hit

            # Provide full 13 outputs even when bypassed
            out_audio = self._pack_audio(wf_bct, sr, audio_dict, torch_device)
            left_peak = float(np.max(np.abs(wf_bct[:, 0, :]))) if wf_bct.ndim == 3 and wf_bct.shape[1] >= 1 else peak
            right_peak = float(np.max(np.abs(wf_bct[:, 1, :]))) if wf_bct.ndim == 3 and wf_bct.shape[1] >= 2 else left_peak
            try:
                rms = float(np.sqrt(np.mean(wf_bct * wf_bct))) if wf_bct.size else 0.0
            except Exception:
                rms = 0.0

            return (
                out_audio,
                reset_performed,
                status,
                peak,
                float(self.smoothed_peak),
                bool(hit),
                bool(hit_event),
                0.0,
                0.0,
                False,
                float(rms),
                float(left_peak),
                float(right_peak),
            )

        # parse params
        freqs = self._parse_floats_list(frequencies)
        gains = self._parse_floats_list(gains_db)
        q_factors = self._parse_floats_list(qs)

        # broadcasting
        n_bands = freqs.size
        if n_bands > 0:
            if gains.size == 0:
                gains = np.zeros(n_bands, dtype=float)
            elif gains.size == 1 and n_bands > 1:
                gains = np.full(n_bands, gains[0], dtype=float)
            elif gains.size != n_bands:
                raise ValueError("gains_db must be length 1 or match number of frequencies")
            if q_factors.size == 0:
                q_factors = np.ones(n_bands, dtype=float)
            elif q_factors.size == 1 and n_bands > 1:
                q_factors = np.full(n_bands, q_factors[0], dtype=float)
            elif q_factors.size != n_bands:
                raise ValueError("qs must be length 1 or match number of frequencies")
        else:
            gains = np.array([], dtype=float)
            q_factors = np.array([], dtype=float)

        # process each batch item; internal shape is (samples, channels)
        if wf_bct.size == 0:
            out_audio = self._pack_audio(wf_bct, sr, audio_dict, torch_device)
            return (
                out_audio,
                reset_performed,
                "error_input",
                0.0,
                float(self.smoothed_peak),
                False,
                False,
                0.0,
                0.0,
                False,
                0.0,
                0.0,
                0.0,
            )

        out_bct = wf_bct.copy()
        batch_size = out_bct.shape[0]

        # Stateful processing is only well-defined for a single stream.
        stateful_effective = bool(stateful) and batch_size == 1

        # initialize/resize filter states if stateful
        if stateful_effective:
            if len(self.filter_states) != n_bands:
                self.filter_states = []
                for _ in range(n_bands):
                    self.filter_states.append({"x1": 0.0, "x2": 0.0, "y1": 0.0, "y2": 0.0})

        peak = 0.0
        left_peak = 0.0
        right_peak = 0.0
        rms = 0.0
        gr_linear = 0.0
        gr_db = 0.0
        limiter_hit_flag = False

        for b in range(batch_size):
            proc = out_bct[b].T  # (samples, channels)

            # state selection
            if stateful_effective:
                tmp_states = None
            else:
                tmp_states = []
                for _ in range(n_bands):
                    tmp_states.append({"x1": 0.0, "x2": 0.0, "y1": 0.0, "y2": 0.0})

            # apply per-band peaking filters (cascade)
            for i in range(n_bands):
                g = gains[i]
                if abs(g) < 1e-8:
                    continue
                fc = freqs[i]
                q = q_factors[i]
                b0, b1, b2, a0, a1, a2 = self._design_peaking_biquad(fc, q, g, sr)
                state = self.filter_states[i] if stateful_effective else tmp_states[i]
                proc = self._apply_biquad_stateful(proc, b0, b1, b2, a0, a1, a2, state)

            # limiter presets
            lim_mode = str(limiter_mode).lower() if limiter_mode is not None else "mild"
            if lim_mode not in ("mild", "medium", "heavy"):
                lim_mode = "mild"
            knee_db_eff = float(knee_db)
            makeup_db_eff = float(makeup_db)
            if knee_db_eff == 0.0 and makeup_db_eff == 0.0:
                if lim_mode == "mild":
                    knee_db_eff = 2.0
                    makeup_db_eff = 0.0
                elif lim_mode == "medium":
                    knee_db_eff = 4.0
                    makeup_db_eff = 0.5
                else:
                    knee_db_eff = 8.0
                    makeup_db_eff = 1.5

            if lim_mode == "mild":
                threshold_db = -0.1
                attack_ms = 5.0
                release_ms = 100.0
            elif lim_mode == "medium":
                threshold_db = -1.0
                attack_ms = 3.0
                release_ms = 200.0
            else:
                threshold_db = -3.0
                attack_ms = 1.0
                release_ms = 400.0

            thresh_lin = self._db_to_linear(threshold_db)

            initial_env = self.limiter_env if stateful_effective else None
            proc, env, applied_gain = self._apply_envelope_limiter(
                proc,
                thresh_lin,
                attack_ms,
                release_ms,
                sr,
                knee_db=knee_db_eff,
                makeup_db=makeup_db_eff,
                initial_env=initial_env,
            )
            if stateful_effective:
                self.limiter_env = env

            # wet/dry mix
            w = float(np.clip(wet, 0.0, 1.0))
            if w < 1.0:
                orig = wf_bct[b].T
                proc = w * proc + (1.0 - w) * orig

            out_bct[b] = proc.T

            # metrics (aggregate across batch)
            peak_b = float(np.max(np.abs(proc))) if proc.size else 0.0
            peak = max(peak, peak_b)

            if proc.ndim == 1:
                left_peak_b = right_peak_b = peak_b
            else:
                left_peak_b = float(np.max(np.abs(proc[:, 0]))) if proc.shape[1] >= 1 else 0.0
                right_peak_b = float(np.max(np.abs(proc[:, 1]))) if proc.shape[1] >= 2 else left_peak_b
            left_peak = max(left_peak, left_peak_b)
            right_peak = max(right_peak, right_peak_b)

            try:
                rms_b = float(np.sqrt(np.mean(proc * proc))) if proc.size else 0.0
            except Exception:
                rms_b = 0.0
            rms = max(rms, rms_b)

            gr_linear = max(gr_linear, max(0.0, 1.0 - float(applied_gain)))
            gr_db = max(gr_db, (-20.0 * float(np.log10(applied_gain)) if applied_gain > 1e-12 else 120.0))
            limiter_hit_flag = limiter_hit_flag or (applied_gain < 0.999)

        # meter smoothing (per-buffer approx)
        if peak > self.smoothed_peak:
            attack_coeff = np.exp(-1.0 / (max(meter_attack_ms, 0.001) / 1000.0 * sr))
            self.smoothed_peak = (1.0 - attack_coeff) * peak + attack_coeff * self.smoothed_peak
        else:
            release_coeff = np.exp(-1.0 / (max(meter_release_ms, 0.001) / 1000.0 * sr))
            self.smoothed_peak = (1.0 - release_coeff) * peak + release_coeff * self.smoothed_peak

        threshold_lin = self._db_to_linear(hit_threshold_db)
        hit = self.smoothed_peak >= threshold_lin
        hit_event = hit and (not self._last_hit)
        self._last_hit = hit

        status = "processed_stateful" if stateful_effective else "processed_stateless"
        out_audio = self._pack_audio(out_bct, sr, audio_dict, torch_device)

        return (
            out_audio,
            reset_performed,
            status,
            peak,
            float(self.smoothed_peak),
            bool(hit),
            bool(hit_event),
            float(gr_linear),
            float(gr_db),
            bool(limiter_hit_flag),
            float(rms),
            float(left_peak),
            float(right_peak),
        )