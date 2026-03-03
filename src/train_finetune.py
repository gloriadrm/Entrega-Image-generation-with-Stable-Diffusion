import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from config import *

# Helper para seleccionar los nombres de columnas ce imagen y texto 
def pick_column(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    raise ValueError(f"No se encontró ninguna de estas columnas: {candidates}. Columnas disponibles: {columns}")

# Preprocesado de las imágenes: convertimos todas las imágenes a B/W y luego a RGB (para que tengan 3 canales) que es lo que espera el modelo 
# En lugar de cortar las imágenes, redimensionamos y rellenamos padding con negro 
def build_image_transforms(target_size, fill):
    def preprocess_image(image):
        image = image.convert("L").convert("RGB")

        width, height = image.size
        scale = min(target_size / width, target_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        pad_left = (target_size - new_width) // 2
        pad_right = target_size - new_width - pad_left
        pad_top = (target_size - new_height) // 2
        pad_bottom = target_size - new_height - pad_top

        return F.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=fill)

    return transforms.Compose(
        [
            transforms.Lambda(preprocess_image),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

# Wrapper para gestionar el dataset 
class Text2ImageDataset(Dataset):
    def __init__(self, dataset, tokenizer, image_transforms, image_column, text_column):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_transforms = image_transforms
        self.image_column = image_column
        self.text_column = text_column

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = self.image_transforms(example[self.image_column])
        token = self.tokenizer(
            example[self.text_column],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": image,
            "input_ids": token.input_ids.squeeze(0),
            "attention_mask": token.attention_mask.squeeze(0),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning sencillo de la U-Net de Stable Diffusion.")
    parser.add_argument("--dataset", required=True, help="Nombre del dataset en Hugging Face o ruta local.")
    parser.add_argument("--output-dir", default="./finetuned-model", help="Carpeta de salida.")
    parser.add_argument("--epochs", type=int, default=2, help="Número de épocas.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"], help="Dispositivo de entrenamiento.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargamos el dataset y seleccionamos las columnas de imagen y texto
    dataset = load_dataset(args.dataset, split=DEFAULT_SPLIT)
    image_column = pick_column(dataset.column_names, [DEFAULT_IMAGE_COLUMN])
    text_column = pick_column(dataset.column_names, [DEFAULT_TEXT_COLUMN])

    # Cargamos componentes del modelo pre-entrenado. 
    tokenizer = CLIPTokenizer.from_pretrained(DEFAULT_MODEL, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(DEFAULT_MODEL, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(DEFAULT_MODEL, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(DEFAULT_MODEL, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(DEFAULT_MODEL, subfolder="unet").to(device)

    # Congelamos el VAE y el Text Encoder para que no se actualicen durante el entrenamiento.
    vae.eval()
    text_encoder.eval()
    for param in vae.parameters():
        param.requires_grad = False
    for param in text_encoder.parameters():
        param.requires_grad = False

    # Extraemos y preprocesamos el conjunto de entrenamiento para que sea compatible con el modelo.
    image_transforms = build_image_transforms(DEFAULT_RESOLUTION, DEFAULT_FILL)
    train_dataset = Text2ImageDataset(dataset, tokenizer, image_transforms, image_column, text_column)
    train_dataloader = DataLoader(train_dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=True)

    # Preparamos el entrenamiento 
    optimizer = torch.optim.AdamW(unet.parameters(), lr=DEFAULT_LR)

    print(f"Device: {device}")
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Image column: {image_column}")
    print(f"Text column: {text_column}")

    stop_training = False
    unet.train()

    # Finetuning loop
    for epoch in range(args.epochs):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")

        for batch in progress_bar:
            optimizer.zero_grad()

            with torch.no_grad():
                latents = vae.encode(batch["pixel_values"].to(device)).latent_dist.sample()
                latents = latents * 0.18215
                encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            if not torch.isfinite(loss):
                print(f"Loss inválida en epoch {epoch}")
                stop_training = True
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            progress_bar.set_postfix(loss=float(loss.item()))

        if stop_training:
            break

    unet.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Modelo guardado en: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
