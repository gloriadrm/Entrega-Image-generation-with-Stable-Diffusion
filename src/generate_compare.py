import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from config import *


def parse_args():
    parser = argparse.ArgumentParser(description="Genera una imagen con el modelo base y otra con la U-Net fine-tuneada.")
    parser.add_argument("--prompt", required=True, help="Prompt de prueba.")
    return parser.parse_args()


def build_pipe(model_id, device, unet=None):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        unet=unet,
        safety_checker=None,
    ).to(device)
    pipe.enable_attention_slicing()
    return pipe


def generate_image(pipe, prompt, seed):
    # `diffusers==0.36.0` puede llegar aquí sin inicializar estos atributos internos.
    # Iniciializamos atributos para evitar atributeError al ejecutar el pipeline.
    pipe._guidance_scale = DEFAULT_GUIDANCE_SCALE
    pipe._guidance_rescale = 0.0
    pipe._clip_skip = None
    pipe._cross_attention_kwargs = None
    pipe._interrupt = False
    pipe.unet.config.time_cond_proj_dim = None


    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = pipe(
        prompt,
        width=DEFAULT_RESOLUTION,
        height=DEFAULT_RESOLUTION,
        num_inference_steps=DEFAULT_STEPS,
        guidance_scale=DEFAULT_GUIDANCE_SCALE,
        generator=generator,
    )
    return result.images[0]


def main():
    args = parse_args()
    device = torch.device("cpu")
    output_dir = Path(DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando U-Net fine-tuneada...")
    finetuned_unet = UNet2DConditionModel.from_pretrained(DEFAULT_FINETUNED_UNET).to(device)

    print("Cargando pipeline base...")
    base_pipe = build_pipe(DEFAULT_MODEL, device)
    print("Cargando pipeline fine-tuneada...")
    finetuned_pipe = build_pipe(DEFAULT_MODEL, device, unet=finetuned_unet)

    print("Generando imagen base...")
    base_image = generate_image(base_pipe, args.prompt, DEFAULT_SEED)
    print("Generando imagen fine-tuneada...")
    tuned_image = generate_image(finetuned_pipe, args.prompt, DEFAULT_SEED)

    before_path = output_dir / "image_before.png"
    after_path = output_dir / "image_after.png"

    base_image.save(before_path)
    tuned_image.save(after_path)
    print(f"Imagen base guardada en: {before_path.resolve()}")
    print(f"Imagen fine-tuneada guardada en: {after_path.resolve()}")


if __name__ == "__main__":
    main()
