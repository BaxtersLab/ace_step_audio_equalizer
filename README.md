# 🎚️ ACE Step Audio Equalizer (v2.0)

A clean, tactile 6‑band equalizer with a stereo‑linked limiter and independent left/right channel volume.

This is a ComfyUI custom node.

## ✅ I/O
- **Input**: AUDIO
- **Output**: AUDIO

## 🎛️ UI (18 Lanes)
Each control uses two vertical lanes:
- Top lane: slider
- Bottom lane: static label

Lane order (top → bottom):
- Band 1 gain + label (60 Hz)
- Band 2 gain + label (150 Hz)
- Band 3 gain + label (400 Hz)
- Band 4 gain + label (1 kHz)
- Band 5 gain + label (3 kHz)
- Band 6 gain + label (8 kHz)
- Limiter slider + label: `<mild--] limiter [--aggressive>`
- Left volume + label
- Right volume + label

## 🔊 DSP
- Six fixed peaking EQ bands (gain per band)
- Stereo‑linked soft limiter with stateful envelope
- Independent left/right output trims
- Parameter smoothing (“gradient control”) on all controls to prevent zipper noise

## 📦 Installation
Place the folder `ace_step_audio_equalizer` inside:

`ComfyUI/custom_nodes/`

Restart ComfyUI. The node appears as **ACE Step Audio Equalizer** under the `audio` category.