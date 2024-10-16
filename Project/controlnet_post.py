from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionXLControlNetImg2ImgPipeline
import torch
import numpy as np
import cv2
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

def lcg(seed, a=1664525, c=1013904223, m=2**32):
    """Linear Congruential Generator (LCG) to generate pseudo-random numbers."""
    while True:
        seed = (a * seed + c) % m
        yield seed

# original_image = Image.open('result/coffee.png')
# image = np.array(original_image)

# low_threshold = 75
# high_threshold = 150

# image = cv2.Canny(image, low_threshold, high_threshold)
# image = image[:, :, None]
# image = np.concatenate([image, image, image], axis=2)
# canny_image = Image.fromarray(image)
# canny_image.save('edges.png')

canny_image = Image.open('edges.png')
image = Image.open('result/coffee.png')
# image.show()

ckpt = "runwayml/stable-diffusion-v1-5"
# ckpt = 'ernestchu/majicmixRealistic_betterV2V25'
# ckpt = "emilianJR/epiCRealism"

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    ckpt, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
)

# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
#     ckpt, torch_dtype=torch.float16, use_safetensors=True
# )

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

quality = ", highest quality settings, sharp image, realistic"

# prompt = "a wood house in a flower field, no flowers and grass attach on the house, highest quality settings, sharp image, realistic"
# prompt = "a photo of moon, highest quality settings, sharp image, realistic, detailed"
# prompt = "a photo of cat, highest quality settings, sharp image, realistic, detailed"
# prompt = "a photo of a beautiful woman, snowy day" + quality
# prompt = "a photo of dog, highest quality settings, sharp image, realistic, detailed"
# prompt = "a cup of coffee with a heart design on the foam" + quality
# prompt = "a photo of fruit platter, there are oranges, grapefruits, lemons, limes and apples" + quality
prompt = "super resolution for the cup, coffee, light noise, highest quality settings, sharp image, realistic, detailed"
negative_prompt = "blurry, distortion, anime, cartoon, Bad quality, Jpeg artifacts, Signature, Username, Watermark"

seed = 42
random_generator = lcg(seed)

for i in range(2):
    random_number = next(random_generator)
    print(random_number)
    generator = torch.manual_seed(random_number)
    output = pipe(
            prompt=prompt, image=image, control_image=canny_image, strength=0.5, negative_prompt=negative_prompt, generator=generator
        ).images[0]

    output.save(f'dif_res/{random_number}.png')

