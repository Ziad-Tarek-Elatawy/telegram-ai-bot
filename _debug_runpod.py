"""Debug: inspect raw RunPod ComfyUI output."""
import sys
import asyncio
import json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import aiohttp
from config import RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID
from bot.services.runpod_handler import _build_workflow

HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}


async def main():
    workflow = _build_workflow(
        prompt="a cute cat sitting on a chair, warm lighting, masterpiece, best quality",
        width=512,
        height=512,
        num_inference_steps=10,
    )

    payload = {"input": {"workflow": workflow}}

    print("Sending runsync request...")
    url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync"
    timeout = aiohttp.ClientTimeout(total=180)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, json=payload, headers=HEADERS, timeout=timeout
        ) as resp:
            print(f"HTTP Status: {resp.status}")
            data = await resp.json()

    with open("debug.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Check output structure
    status = data.get("status", "")
    output = data.get("output", {})
    print(f"\nStatus: {status}")
    print(f"Output type: {type(output)}")
    if isinstance(output, dict):
        print(f"Output keys: {list(output.keys())}")
        for k, v in output.items():
            if isinstance(v, str):
                print(f"  {k}: {v[:100]}...")
            else:
                print(f"  {k}: {type(v)} = {v}")

if __name__ == "__main__":
    asyncio.run(main())