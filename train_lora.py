import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import json, math, argparse

from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator


class ImageCaptionDataset(Dataset):
    """Expects a folder of images + a metadata.jsonl with {"file_name","text"} per line."""
    def __init__(self, data_dir, tokenizer, size=512):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.entries = [json.loads(l) for l in (self.data_dir / "metadata.jsonl").read_text().splitlines()]
        self.tf = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        img = Image.open(self.data_dir / e["file_name"]).convert("RGB")
        ids = self.tokenizer(
            e["text"], padding="max_length", truncation=True,
            max_length=self.tokenizer.model_max_length, return_tensors="pt",
        ).input_ids[0]
        return {"pixel_values": self.tf(img), "input_ids": ids}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", default="lora-out")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    args = p.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum, mixed_precision="fp16")

    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")

    # Freeze everything; LoRA only on UNet attention layers
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    ds = ImageCaptionDataset(args.data_dir, tokenizer)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    unet, optimizer, dl = accelerator.prepare(unet, optimizer, dl)
    weight_dtype = torch.float16
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    global_step = 0
    for epoch in range(args.epochs):
        unet.train()
        for batch in dl:
            with accelerator.accumulate(unet):
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample() * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                noisy = noise_scheduler.add_noise(latents, noise, timesteps)

                enc_hidden = text_encoder(batch["input_ids"])[0]
                pred = unet(noisy, timesteps, encoder_hidden_states=enc_hidden).sample

                if noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                loss = torch.nn.functional.mse_loss(pred.float(), target.float())
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % 50 == 0:
                    accelerator.print(f"epoch {epoch} step {global_step} loss {loss.item():.4f}")

    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(unet)
    unwrapped.save_pretrained(args.output_dir)
    accelerator.print(f"Saved LoRA weights to {args.output_dir}")


if __name__ == "__main__":
    main()
