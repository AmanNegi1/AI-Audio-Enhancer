#!/usr/bin/env python3
"""
DreamBooth LoRA training script for manga character consistency.
Runs as a subprocess from the Streamlit LoRA Studio tab.
Writes JSON progress to --status_file so the UI can poll it.

Usage:
  python core/train_lora.py \
    --images_dir temp/lora_images/my_char \
    --output_dir loras/my_char \
    --instance_prompt "mychr person" \
    --base_model gsdf/CounterfeitXL \
    --is_sdxl \
    --max_steps 400 \
    --status_file temp/lora_status_my_char.json
"""

import argparse
import json
import os
import sys

# Set cache roots to D:\ if available to avoid C: drive running out of space
_workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_d_drive = "D:\\"
_cache_root = _d_drive if os.path.exists(_d_drive) else _workspace

if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.join(_cache_root, ".cache", "huggingface")
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = os.path.join(_cache_root, ".cache")
import core.fix_torchaudio
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ── Utilities ────────────────────────────────────────────────────────────────

def write_status(path, data):
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


# ── Dataset ──────────────────────────────────────────────────────────────────

class ConceptDataset(Dataset):
    def __init__(self, images_dir, resolution):
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        self.images = [p for p in Path(images_dir).iterdir() if p.suffix.lower() in exts]
        if not self.images:
            raise ValueError(f"No training images found in: {images_dir}")
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return 400  # repeat images to fill all training steps

    def __getitem__(self, idx):
        img = Image.open(self.images[idx % len(self.images)]).convert("RGB")
        return self.transform(img)


# ── SDXL Training ─────────────────────────────────────────────────────────────

def train_sdxl(args, status_file):
    from diffusers import StableDiffusionXLPipeline, DDPMScheduler
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import save_file

    write_status(status_file, {"status": "initializing", "step": 0,
                               "total_steps": args.max_steps, "message": "Loading SDXL pipeline..."})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_model, torch_dtype=dtype, use_safetensors=True, low_cpu_mem_usage=True
    ).to(device)
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Freeze everything except LoRA
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # Apply LoRA adapters to UNet attention layers
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    pipe.unet.add_adapter(lora_config)
    pipe.unet.enable_gradient_checkpointing()  # saves VRAM during training

    # Pre-compute text embeddings (same for all images)
    write_status(status_file, {"status": "initializing", "step": 0,
                               "total_steps": args.max_steps, "message": "Computing text embeddings..."})
    with torch.no_grad():
        (prompt_embeds, _, pooled_embeds, _) = pipe.encode_prompt(
            prompt=args.instance_prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )

    resolution = 1024
    add_time_ids = torch.tensor(
        [[resolution, resolution, 0, 0, resolution, resolution]],
        dtype=dtype, device=device
    )

    dataset = ConceptDataset(args.images_dir, resolution)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    optimizer = torch.optim.AdamW(
        [p for p in pipe.unet.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-2,
    )

    write_status(status_file, {"status": "training", "step": 0,
                               "total_steps": args.max_steps, "message": "Training started..."})

    pipe.unet.train()
    global_step = 0
    data_iter = iter(dataloader)
    last_loss = 0.0

    while global_step < args.max_steps:
        try:
            pixel_values = next(data_iter).to(device, dtype=dtype)
        except StopIteration:
            data_iter = iter(dataloader)
            pixel_values = next(data_iter).to(device, dtype=dtype)

        with torch.no_grad():
            latents = pipe.vae.encode(pixel_values).latent_dist.sample() * pipe.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = pipe.unet(
            noisy_latents, timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": add_time_ids},
        ).sample

        loss = F.mse_loss(noise_pred.float(), noise.float())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in pipe.unet.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        global_step += 1
        last_loss = loss.item()

        if global_step % 10 == 0 or global_step == args.max_steps:
            write_status(status_file, {
                "status": "training",
                "step": global_step,
                "total_steps": args.max_steps,
                "loss": round(last_loss, 4),
                "message": f"Step {global_step}/{args.max_steps} | loss: {last_loss:.4f}",
            })

    # Save LoRA weights as safetensors
    os.makedirs(args.output_dir, exist_ok=True)
    lora_state_dict = {k: v.cpu() for k, v in get_peft_model_state_dict(pipe.unet).items()}
    output_path = os.path.join(args.output_dir, "pytorch_lora_weights.safetensors")
    save_file(lora_state_dict, output_path)

    write_status(status_file, {
        "status": "completed",
        "step": args.max_steps,
        "total_steps": args.max_steps,
        "loss": round(last_loss, 4),
        "message": f"Training complete! LoRA saved → {output_path}",
        "output_dir": args.output_dir,
    })


# ── SD 1.5 Training ───────────────────────────────────────────────────────────

def train_sd15(args, status_file):
    from diffusers import StableDiffusionPipeline, DDPMScheduler
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import save_file

    write_status(status_file, {"status": "initializing", "step": 0,
                               "total_steps": args.max_steps, "message": "Loading SD 1.5 pipeline..."})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model, torch_dtype=dtype, use_safetensors=True, low_cpu_mem_usage=True
    ).to(device)
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    pipe.unet.add_adapter(lora_config)
    pipe.unet.enable_gradient_checkpointing()

    write_status(status_file, {"status": "initializing", "step": 0,
                               "total_steps": args.max_steps, "message": "Computing text embeddings..."})
    with torch.no_grad():
        inputs = pipe.tokenizer(
            args.instance_prompt, padding="max_length", truncation=True,
            max_length=pipe.tokenizer.model_max_length, return_tensors="pt",
        ).to(device)
        prompt_embeds = pipe.text_encoder(inputs.input_ids)[0]

    resolution = 512
    dataset = ConceptDataset(args.images_dir, resolution)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    optimizer = torch.optim.AdamW(
        [p for p in pipe.unet.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-2,
    )

    write_status(status_file, {"status": "training", "step": 0,
                               "total_steps": args.max_steps, "message": "Training started..."})

    pipe.unet.train()
    global_step = 0
    data_iter = iter(dataloader)
    last_loss = 0.0

    while global_step < args.max_steps:
        try:
            pixel_values = next(data_iter).to(device, dtype=dtype)
        except StopIteration:
            data_iter = iter(dataloader)
            pixel_values = next(data_iter).to(device, dtype=dtype)

        with torch.no_grad():
            latents = pipe.vae.encode(pixel_values).latent_dist.sample() * pipe.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=prompt_embeds).sample
        loss = F.mse_loss(noise_pred.float(), noise.float())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in pipe.unet.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        global_step += 1
        last_loss = loss.item()

        if global_step % 10 == 0 or global_step == args.max_steps:
            write_status(status_file, {
                "status": "training",
                "step": global_step,
                "total_steps": args.max_steps,
                "loss": round(last_loss, 4),
                "message": f"Step {global_step}/{args.max_steps} | loss: {last_loss:.4f}",
            })

    os.makedirs(args.output_dir, exist_ok=True)
    lora_state_dict = {k: v.cpu() for k, v in get_peft_model_state_dict(pipe.unet).items()}
    output_path = os.path.join(args.output_dir, "pytorch_lora_weights.safetensors")
    save_file(lora_state_dict, output_path)

    write_status(status_file, {
        "status": "completed",
        "step": args.max_steps,
        "total_steps": args.max_steps,
        "loss": round(last_loss, 4),
        "message": f"Training complete! LoRA saved → {output_path}",
        "output_dir": args.output_dir,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--instance_prompt", required=True)
    parser.add_argument("--base_model", default="gsdf/CounterfeitXL")
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--is_sdxl", action="store_true")
    parser.add_argument("--status_file", required=True)
    args = parser.parse_args()

    try:
        if args.is_sdxl:
            train_sdxl(args, args.status_file)
        else:
            train_sd15(args, args.status_file)
    except Exception as e:
        import traceback
        write_status(args.status_file, {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        })
        sys.exit(1)