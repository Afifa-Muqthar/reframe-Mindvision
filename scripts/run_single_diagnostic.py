"""
Single Diagnostic Test for SD-Turbo on CUDA:
- 512x512 resolution
- 2 inference steps
- guidance_scale = 0.0
- CUDA
- FP16 (torch.float16)
- Measures: generation time, peak GPU memory, GPU temperature before/after
"""

import os
import sys
import time
import json
import subprocess
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.ssl_workaround import configure_hf_ssl_verify
configure_hf_ssl_verify()

from diffusers import AutoPipelineForText2Image

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "experiments")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "diagnostic_512_fp16_step2_objcount.png")

from app.image_gen.prompt_builder import build_scene_prompts
_story = {
    "character": {"appearance": "young woman with short dark hair", "clothing": "green sweater"},
    "scenes": [
        {
            "character_action": "sitting on bed holding smartphone with both hands",
            "gaze": "looking down at message thread",
            "environment": "quiet bedroom with desk and window",
            "lighting": "soft evening light",
        }
    ]
}
PROMPT = build_scene_prompts(_story)[0]

def get_gpu_telemetry():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True
        )
        parts = [p.strip() for p in out.strip().split(",")]
        return {
            "temp_c": int(parts[0]),
            "mem_used_mb": int(parts[1]),
            "mem_total_mb": int(parts[2]),
        }
    except Exception as e:
        return {"error": str(e)}

def main():
    print("=== STARTING SD-TURBO FP16 DIAGNOSTIC RUN (2 STEPS) ===")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Aborting.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")

    telem_before = get_gpu_telemetry()
    print(f"GPU Telemetry Before: {telem_before}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    print("\nLoading pipeline with torch.float16...")
    t_load_start = time.perf_counter()
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=torch.float16,
        )
    except Exception as e:
        print(f"Failed to load with torch.float16: {e}")
        sys.exit(2)

    pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    t_load_end = time.perf_counter()
    print(f"Pipeline loaded in {round(t_load_end - t_load_start, 2)}s")

    gen = torch.Generator(device="cuda").manual_seed(42)

    print(f"\nPrompt: {PROMPT}")
    print("Generating single 512x512 image, 2 steps, guidance_scale=0.0...")
    
    t_gen_start = time.perf_counter()
    try:
        image = pipe(
            prompt=PROMPT,
            num_inference_steps=2,
            guidance_scale=0.0,
            height=512,
            width=512,
            generator=gen,
        ).images[0]
    except torch.cuda.OutOfMemoryError as oom:
        print(f"CRITICAL: OutOfMemoryError encountered: {oom}")
        sys.exit(3)
    except Exception as e:
        print(f"Generation error: {e}")
        sys.exit(4)

    t_gen_end = time.perf_counter()
    gen_time_s = t_gen_end - t_gen_start

    image.save(OUT_FILE)
    print(f"Saved diagnostic image to: {OUT_FILE}")

    telem_after = get_gpu_telemetry()
    peak_torch_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    results = {
        "device": device_name,
        "dtype": "torch.float16",
        "resolution": "512x512",
        "num_inference_steps": 2,
        "guidance_scale": 0.0,
        "seed": 42,
        "generation_time_seconds": round(gen_time_s, 3),
        "peak_torch_vram_mb": round(peak_torch_vram_mb, 1),
        "gpu_telemetry_before": telem_before,
        "gpu_telemetry_after": telem_after,
        "output_image_path": OUT_FILE,
        "prompt": PROMPT,
    }

    print("\n=== DIAGNOSTIC METRICS (2 STEPS) ===")
    print(json.dumps(results, indent=2))

    res_json_path = os.path.join(OUT_DIR, "diagnostic_512_fp16_step2_results.json")
    with open(res_json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nDIAGNOSTIC TEST COMPLETE!")

if __name__ == "__main__":
    main()
