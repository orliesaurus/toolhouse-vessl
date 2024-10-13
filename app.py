import os
import uuid
from typing import Optional

import torch
from diffusers import FluxPipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import uvicorn

MODEL_NAME = "black-forest-labs/FLUX.1-schnell"

class ImageRequest(BaseModel):
    prompt: str
    width: int = 1360
    height: int = 768
    num_steps: int = 4
    guidance: float = 3.5
    seed: Optional[int] = -1

class FluxGenerator:
    def __init__(self, offload: bool):
        self.pipe = FluxPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
        )
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()
        if offload:
            self.pipe.enable_sequential_cpu_offload()

    @torch.inference_mode()
    def generate_image(
        self,
        width: int,
        height: int,
        num_steps: int,
        guidance: float,
        seed: int,
        prompt: str,
    ) -> Image.Image:
        seed = None if seed == -1 else seed
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator = generator.manual_seed(seed)

        image = self.pipe(
            prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            height=height,
            width=width,
            generator=generator,
        ).images[0]

        return image

app = FastAPI()
generator = FluxGenerator(offload=False)

@app.post("/generate_image")
async def generate_image(request: ImageRequest):
    try:
        image = generator.generate_image(
            width=request.width,
            height=request.height,
            num_steps=request.num_steps,
            guidance=request.guidance,
            seed=request.seed,
            prompt=request.prompt,
        )

        filename = f"output/api/{uuid.uuid4()}.png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        image.save(filename, format="png")

        return {"filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)