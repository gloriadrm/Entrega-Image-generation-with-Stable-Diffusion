# Fine-tuning de Stable Diffusion

Este proyecto contiene una práctica de fine-tuning de la U-Net de `Stable Diffusion v1.4` sobre un dataset de ilustraciones antiguas. El objetivo ha sido adaptar el modelo base al estilo visual del conjunto de entrenamiento.

## Limitaciones de ejecución

Durante la fase de inferencia aparecieron problemas de recursos en el entorno local. Al intentar generar imágenes tanto con el modelo base de Stable Diffusion como con el modelo fine-tuneado, el kernel de Jupyter llegaba a cerrarse durante la ejecución.

<p style="color: red;"><strong>The Kernel crashed while executing code in the current cell or a previous cell.</strong></p>

Para intentar mitigarlo se redujo la resolución y el número de pasos de inferencia. La configuración más estable que permitió obtener salida fue `128x128` con `5 steps`, aunque el resultado visual no era suficientemente representativo. Por este motivo, la validación final de la entrega se realizó principalmente desde el notebook y no se pudo comprobar completamente el flujo de inferencia definido en `src/` con parámetros más exigentes.


## Qué se ha hecho

- Se parte del modelo base `CompVis/stable-diffusion-v1-4`.
- Se congela el `VAE` y el `text_encoder`.
- Solo se entrena la `U-Net`.
- Las imágenes del dataset se convierten a escala de grises y después a `RGB` para conservar los 3 canales que espera el modelo.
- En lugar de recortar imágenes, se aplica `resize + padding` hasta una resolución fija de `384x384`.
- Después del entrenamiento, se guarda la `U-Net` fine-tuneada.
- Para evaluar el resultado, se compara la generación del modelo base con la del modelo que usa la `U-Net` ajustada.
- El modelo finetuneado (UNet) está disponible en Hugging Face en el enlace indicado en `hf_repo_link.txt` (público). También lo subimos a github. 

## Estructura del proyecto

```text
Entrega/
├─ notebooks/
│  └─ main.ipynb
├─ src/
│  ├─ config.py
│  ├─ train_finetune.py
│  └─ generate_compare.py
├─ finetuned-model/
│  ├─ config.json
│  ├─ diffusion_pytorch_model.safetensors
│  ├─ tokenizer.json
│  └─ tokenizer_config.json
├─ outputs/
├─ README.md
└─ requirements.txt
```

## Archivos principales

### `src/config.py`

Contiene los parámetros por defecto del proyecto:

- modelo base
- nombres de columnas del dataset
- resolución
- batch size
- learning rate
- repo de la `U-Net` fine-tuneada
- carpeta de salida para inferencia

### `src/train_finetune.py`

Script principal de entrenamiento.

Hace lo siguiente:

- carga el dataset desde Hugging Face
- selecciona las columnas de imagen y texto
- carga `tokenizer`, `scheduler`, `text_encoder`, `vae` y `unet`
- congela `vae` y `text_encoder`
- preprocesa las imágenes
- entrena la `U-Net`
- guarda el resultado en una carpeta de salida

### `src/generate_compare.py`

Script de inferencia y comparación.

Hace lo siguiente:

- carga el modelo base
- carga la `U-Net` fine-tuneada desde una carpeta local o un repo de Hugging Face
- genera una imagen con el modelo base
- genera otra con el modelo ajustado
- guarda ambas imágenes como:
  - `image_before.png`
  - `image_after.png`

### `notebooks/main.ipynb`

Notebook usado durante la exploración, pruebas y desarrollo del pipeline.

## Requisitos

El proyecto se ha ejecutado usando un entorno virtual con las librerías necesarias (`torch`, `diffusers`, `transformers`, `datasets`, `torchvision`, `huggingface_hub`, etc.). Las librerías se facilitane en un archivo requirements.txt


## Cómo ejecutar

### 1. Entrenar la U-Net

Desde la raíz del proyecto:

```bash
cd src
uv run python train_finetune.py --dataset NOMBRE_DEL_DATASET --device cpu --epochs 2 --output-dir ../finetuned-model
```

Ejemplo:

```bash
cd src
uv run python train_finetune.py --dataset gloriadrm/oldbook-dataset --device cpu --epochs 2 --output-dir ../finetuned-model
```

El modelo se guardará en la carpeta indicada por `--output-dir`.

## 2. Generar imágenes de comparación

El script puede cargar la U-Net fine-tuneada de dos formas:

### Opción A. Desde una carpeta local

```bash
cd src
uv run python generate_compare.py --prompt "an image of a crazy sheep" --device cpu --finetuned-unet ../finetuned-model --output-dir ../outputs
```

### Opción B. Desde un repositorio de Hugging Face

```bash
cd src
uv run python generate_compare.py --prompt "an image of a crazy sheep" --device cpu --finetuned-unet gloriadrm/finetuned-oldbook-unet --output-dir ../outputs
```

Las imágenes generadas se guardan en la carpeta /output:

- `image_before.png`
- `image_after.png`

## Notas

- Durante las pruebas, el backend `MPS` en Mac dio problemas de estabilidad numérica durante el entrenamiento.
- Por ese motivo, la opción más estable para el fine-tuning fue `cpu`.
- Las carpetas `finetuned-model/` y `outputs/` forman parte de la entrega, ya que contienen el modelo entrenado y los resultados generados.
- El modelo también ha sido subido a la plataforma de Hugging Face según se especificaba en la tarea: `gloriadrm/finetuned-oldbook-unet`
