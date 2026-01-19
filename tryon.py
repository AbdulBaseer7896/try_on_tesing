


# import torch
# from PIL import Image
# from diffusers import StableDiffusionInpaintPipeline

# # Load model (will download on first run)
# pipe = StableDiffusionInpaintPipeline.from_pretrained(
#     "runwayml/stable-diffusion-inpainting",
#     torch_dtype=torch.float16
# ).to("cuda")

# # Load images
# person_img = Image.open("black_male.png").convert("RGB").resize((512, 512))
# cloth_img = Image.open("red_sonic.png").convert("RGB").resize((512, 512))

# # VERY BASIC prompt (OOTDiffusion uses text + image conditioning)
# prompt = "a photo of a person wearing the given clothing"

# # Run inference
# with torch.autocast("cuda"):
#     result = pipe(
#         prompt=prompt,
#         image=person_img,
#         strength=0.75,
#         guidance_scale=7.5
#     ).images[0]

# # Save result
# result.save("output_tryon.png")

# print("✅ Try-on image saved as output_tryon.png")



import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import numpy as np

# ============================================
# OPTION 1: Default loading (no safetensors parameter)
# ============================================
# This will use .bin files if .safetensors aren't available
print("Loading model with Option 1 (default)...")

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16  # Use float16 for faster inference and less memory
).to("cuda")  # Move model to GPU

# ============================================
# OPTION 2: Explicitly disable safetensors
# ============================================
# Uncomment below and comment out Option 1 if Option 1 doesn't work
"""
print("Loading model with Option 2 (explicitly disable safetensors)...")

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
    use_safetensors=False  # Force use of .bin files instead of .safetensors
).to("cuda")
"""

# Enable memory-efficient attention if xformers is available
try:
    pipe.enable_xformers_memory_efficient_attention()
except:
    print("XFormers not available, using default attention")

# ============================================
# Load and prepare images
# ============================================
print("Loading images...")

try:
    # Load person image
    person_img = Image.open("black_male.png").convert("RGB").resize((512, 512))
except FileNotFoundError:
    print("Error: black_male.png not found. Creating a placeholder image...")
    # Create a placeholder image if file doesn't exist
    person_img = Image.new("RGB", (512, 512), color=(100, 150, 200))

try:
    # Load clothing image (for reference, not directly used by the pipeline)
    cloth_img = Image.open("red_sonic.png").convert("RGB").resize((512, 512))
except FileNotFoundError:
    print("Warning: red_sonic.png not found. Continuing without clothing reference...")
    cloth_img = None

# ============================================
# Create a mask for inpainting
# ============================================
# The mask defines which areas to repaint (white=repaint, black=keep)
print("Creating mask...")

# Method 1: Simple rectangle mask (repaint the torso area)
mask = Image.new("L", (512, 512), 0)  # Start with all black (keep everything)

# Create a white rectangle in the center (area to repaint)
# Adjust these coordinates based on where you want to apply the clothing
x1, y1 = 128, 256  # Top-left corner
x2, y2 = 384, 448  # Bottom-right corner

# Draw white rectangle on the mask
for x in range(x1, x2):
    for y in range(y1, y2):
        mask.putpixel((x, y), 255)

# Alternative: Load a pre-made mask if you have one
# mask = Image.open("mask.png").convert("L").resize((512, 512))

# ============================================
# Prepare the prompt
# ============================================
# The prompt guides what to paint in the masked area
prompt = "a person wearing a red Sonic the Hedgehog t-shirt, detailed clothing, high quality photo"
negative_prompt = "ugly, deformed, blurry, low quality, distorted, extra limbs, bad anatomy"

# ============================================
# Generate the image
# ============================================
print("Generating image...")

with torch.autocast("cuda"):  # Use mixed precision for faster inference
    result = pipe(
        prompt=prompt,
        image=person_img,        # Original image
        mask_image=mask,         # Mask defining area to repaint
        height=512,              # Output height
        width=512,               # Output width
        num_inference_steps=50,  # More steps = better quality but slower
        guidance_scale=7.5,      # How closely to follow the prompt
        negative_prompt=negative_prompt,  # What to avoid
        strength=0.95,           # How much to change the masked area (0-1)
        num_images_per_prompt=1  # Number of images to generate
    ).images[0]  # Get the first (and only) image

# ============================================
# Save results
# ============================================
print("Saving results...")

# Save the generated image
result.save("output_tryon.png")

# Save the mask for reference
mask.save("mask_used.png")

# Create a comparison image (side by side)
if cloth_img:
    # Create a 3x1 grid: Person | Clothing | Result
    total_width = 512 * 3
    comparison = Image.new("RGB", (total_width, 512))
    
    comparison.paste(person_img, (0, 0))
    comparison.paste(cloth_img, (512, 0))
    comparison.paste(result, (1024, 0))
    
    # Add labels
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(comparison)
    
    # Try to use a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), "Original Person", fill="white", font=font)
    draw.text((522, 10), "Clothing", fill="white", font=font)
    draw.text((1034, 10), "Result", fill="white", font=font)
    
    comparison.save("comparison.png")
    print("✅ Comparison image saved as comparison.png")

print("✅ Try-on image saved as output_tryon.png")
print("✅ Mask saved as mask_used.png")
print("\n=== TIPS ===")
print("1. Adjust mask coordinates (lines 67-68) to target specific clothing areas")
print("2. Modify the prompt to better describe desired clothing")
print("3. Increase num_inference_steps for better quality (but slower)")
print("4. Try different strength values (0.7-0.99) for different effects")