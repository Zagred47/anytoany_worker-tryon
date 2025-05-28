import os
from huggingface_hub import snapshot_download
from diffusers import FluxTransformer2DModel, AutoencoderKL
from diffusers.hooks import apply_group_offloading
from transformers import T5EncoderModel, CLIPTextModel
from src.pipeline_tryon import FluxTryonPipeline
from optimum.quanto import freeze, qfloat8, quantize

def download_models(
    repo_id="black-forest-labs/FLUX.1-dev",
    subfolders=("text_encoder", "text_encoder_2", "transformer", "vae"),
    local_dir="downloaded_models"
):
    os.makedirs(local_dir, exist_ok=True)
    for subfolder in subfolders:
        print(f"Scaricando {subfolder}...")
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=f"{subfolder}/*",
            local_dir=os.path.join(local_dir, subfolder),
            local_dir_use_symlinks=False
        )
    print("Download completato.")

download_models()