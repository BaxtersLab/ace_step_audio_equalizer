# Baxter’s Comfy Audio Nodes

A suite of high‑quality custom audio nodes designed to enhance generation workflows in ComfyUI, with full support for ACE‑Step audio pipelines.  
Each node is self‑contained, optimized for clean DSP behavior, and built for reliable, production‑grade audio processing.

---

## 🎛️ Included Nodes

### 1. ace_step_audio_equalizer
A mastering‑grade EQ and dynamics block designed for ACE‑Step audio workflows.

**Features**:
- Biquad EQ filters (low shelf, peak, high shelf)
- Envelope‑smoothed limiter
- Gain reduction metering (linear + dB)
- Limiter hit flag
- RMS metering
- Per‑channel peak detection
- Bypass input
- Reset and status outputs

> *Ideal for shaping, stabilizing, and preparing audio for downstream processing.*

---

### 2. ComfyAudioFlanger
A clean, musical flanger node built for modulation effects inside ComfyUI.

**Features**:
- Time‑modulated delay line
- LFO modulation with smoothing
- Feedback path
- Wet/dry mix
- Reset and status outputs

> *Perfect for adding movement, texture, and modulation to generated audio.*

---

### 3. spectral_warm_amp_limiter
A harmonic enhancement and soft‑knee limiting node designed for warmth, saturation, and tactile‑ready output shaping.

**Features**:
- Harmonic enhancement
- Saturation / warmth curve
- Low‑frequency emphasis
- Soft‑knee limiter
- Gain computer and output shaping
- Status output

> *Designed to pair beautifully with tactile audio systems and low‑frequency workflows.*

---

## 🧪 Testing

A smoke test is included to validate node output shapes and ensure stable behavior.  
The suite currently returns 13 outputs from the equalizer node, all verified.

---

## 🔐 Repository Status

This repository is currently private while development continues.  
It will be made public once:
- The node suite is complete
- Code has been reviewed for safety
- No sensitive or machine‑specific data remains
- Documentation is finalized

---

## 📄 License

The license will be added prior to public release.