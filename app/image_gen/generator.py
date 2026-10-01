"""
Image generation using stabilityai/sd-turbo, 256x256 by default, 1 step.

License note (say this in the viva if asked): SD-Turbo is under the
Stability AI Community/Research License - free to use for this academic
project, but non-commercial. It is also gated on Hugging Face: whoever
runs this needs a free HF account, to have clicked "accept" on the model
page, and to have run `huggingface-cli login` once before first download.

Runs in a background thread from the Flask route so it never blocks the
text response (see app/main.py /image_status/<session_id>).

Resolution note: the brief's spec is 512x512, but testing on the target
laptop (Intel i3, 3.77GB RAM total) showed the VAE-decode step segfaulting
at 512x512 -- almost certainly a failed native memory allocation under
memory pressure rather than a code bug (it crashed at the same point
regardless of thread count). Attention/VAE slicing and a default of
256x256 substantially cut peak memory; raise IMAGE_SIZE back to 512 only
on a machine with more headroom.
"""

import gc

import torch
from diffusers import AutoPipelineForText2Image

_MODEL_NAME = "stabilityai/sd-turbo"
IMAGE_SIZE = 512


class ImageGenerator:
    def __init__(self):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._pipe = AutoPipelineForText2Image.from_pretrained(
            _MODEL_NAME, torch_dtype=dtype
        )
        self._pipe.to(self._device)
        self._pipe.enable_attention_slicing()
        self._pipe.enable_vae_slicing()

    def generate(self, prompt: str, output_path: str, num_inference_steps: int = 2,
                 size: int = IMAGE_SIZE, seed: int = None) -> str:
        generator = torch.Generator(device=self._device).manual_seed(seed) if seed is not None else None
        image = self._pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,  # SD-Turbo is trained for guidance_scale=0
            height=size,
            width=size,
            generator=generator,
        ).images[0]
        image.save(output_path)
        return output_path

    def unload(self):
        del self._pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
from app.image_gen.prompt_builder import (
    COMIC_VISUAL_STYLE,
    NEGATIVE_PROMPT,
    build_scene_prompts,
    build_character_anchor,
)


def build_image_prompt(case_frame, technique: dict) -> str:
    """
    Turns the case frame + chosen technique into a comic prompt.
    Kept for backward compatibility with scripts and tests.
    """
    return (
        f"A gentle comic illustration representing moving from {case_frame.core_emotion or 'worry'} "
        f"toward {technique['name'].lower()}, {COMIC_VISUAL_STYLE}"
    )


def build_panel_prompts(core_emotion: str, technique_name: str, story: dict = None) -> list:
    """
    Builds 3 prompts for the 3-panel comic strip (Problem -> Reframing -> Resolution).
    If a structured story is provided, delegates to build_scene_prompts;
    otherwise creates grounded fallback prompts with character consistency.
    """
    if story and isinstance(story, dict):
        return build_scene_prompts(story)

    emotion = core_emotion or "worried"
    character_anchor = "young adult with short dark hair in dark gray sweater"
    return [
        f"{character_anchor}, sitting at study desk with head resting on hands, looking down at open notes, quiet room, soft shadow-toned light, {COMIC_VISUAL_STYLE}",
        f"{character_anchor}, sitting at desk pausing and taking a slow breath, gaze focused on open notebook, study room, soft warm light, {COMIC_VISUAL_STYLE}",
        f"{character_anchor}, sitting upright holding pen resting on paper, looking forward with steady gaze, bright room by window, clear morning daylight, {COMIC_VISUAL_STYLE}",
    ]


def build_panel_subtitles(core_emotion: str, technique_name: str) -> list:
    """Short titles/captions for each panel's header."""
    return [
        f"1. PROBLEM (Feeling {core_emotion or 'this'})",
        f"2. REFRAME ({technique_name})",
        "3. RESOLUTION (One step forward)",
    ]
