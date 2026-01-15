


import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

# Load model (will download on first run)
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to("cuda")

# Load images
person_img = Image.open("black_male.png").convert("RGB").resize((512, 512))
cloth_img = Image.open("red_sonic.png").convert("RGB").resize((512, 512))

# VERY BASIC prompt (OOTDiffusion uses text + image conditioning)
prompt = "a photo of a person wearing the given clothing"

# Run inference
with torch.autocast("cuda"):
    result = pipe(
        prompt=prompt,
        image=person_img,
        strength=0.75,
        guidance_scale=7.5
    ).images[0]

# Save result
result.save("output_tryon.png")

print("✅ Try-on image saved as output_tryon.png")
