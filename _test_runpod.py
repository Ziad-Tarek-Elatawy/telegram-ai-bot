"""Quick test: RunPod ComfyUI text-to-image generation."""
import sys
import asyncio
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from bot.services.runpod_handler import call_runpod_text2img


async def main():
    print("Starting RunPod ComfyUI test...")
    print("Prompt: 'a cute cat sitting on a chair, warm lighting'")
    print("Size: 512x512, steps: 10")
    print("-" * 50)

    try:
        path = await call_runpod_text2img(
            prompt="a cute cat sitting on a chair, warm lighting, masterpiece, best quality",
            width=512,
            height=512,
            num_inference_steps=10,
        )
        size_kb = path.stat().st_size / 1024
        print(f"SUCCESS! Image saved: {path}")
        print(f"File size: {size_kb:.1f} KB")
        print("PHASE 4 COMPLETE — RunPod ComfyUI is working!")
    except Exception as exc:
        print(f"FAILED: {exc}")
        print("Check the RunPod endpoint status and API key.")

if __name__ == "__main__":
    asyncio.run(main())