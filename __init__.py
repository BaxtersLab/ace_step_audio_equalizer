from .ace_step_audio_equalizer import AudioEqualizerNode

NODE_CLASS_MAPPINGS = {
    "ACEStep_AudioEqualizer": AudioEqualizerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ACEStep_AudioEqualizer": "ACE Step Audio Equalizer",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
