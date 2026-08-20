# Stable Diffusion LoRA Fine-Tuning

Fine-tune Stable Diffusion on a custom subject or style using
[LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation) with
Hugging Face `diffusers` and `peft`. LoRA trains a small set of adapter
weights instead of the full model, so a single consumer GPU (~12 GB VRAM)
is enough and the resulting weights are only a few MB.

## Results

| Base SD v1.5 | + LoRA (trained on <subject>) |
|:---:|:---:|
| ![before](samples/before.png) | ![after](samples/after.png) |

*Prompt: "a photo of sks dog on the beach"*

## How it works

The base UNet, VAE, and text encoder are frozen. LoRA injects trainable
low-rank matrices into the UNet's cross-attention projection layers
(`to_q`, `to_k`, `to_v`, `to_out`). Only those matrices are optimized
against the standard diffusion denoising objective, so training is fast
and cheap while the adapter still steers the model toward the new subject.

## Setup

```bash
git clone https://github.com/<you>/sd-lora-finetune.git
cd sd-lora-finetune
pip install -r requirements.txt
accelerate config default
```

## Prepare your dataset

Create a `data/` folder with 10–20 images and a `metadata.jsonl` file:

data/
img1.png
img2.png
...
metadata.jsonl


Each line of `metadata.jsonl` maps a file to a caption. Use a rare
trigger word (e.g. `sks`) so the model learns your subject without
clobbering an existing concept:

{"file_name": "img1.png", "text": "a photo of sks dog"}
{"file_name": "img2.png", "text": "a photo of sks dog sitting"}


## Train

```bash
python train_lora.py --data_dir ./data --output_dir ./lora-out --epochs 100
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--rank` | 8 | LoRA rank. Higher = more capacity, larger file, more overfitting risk |
| `--lr` | 1e-4 | Learning rate |
| `--epochs` | 100 | Passes over the dataset |
| `--batch_size` | 1 | Raise if you have VRAM to spare |
| `--grad_accum` | 4 | Simulates a larger batch on limited VRAM |

## Generate

```bash
python infer.py --lora ./lora-out --prompt "a photo of sks dog on the beach"
```

## Notes

- Tested on ~12 GB VRAM at 512px, batch size 1.
- `rank` and `lora_alpha` control the strength/capacity tradeoff — lower
  rank generalizes better on small datasets, higher rank captures more
  detail.
- For SDXL, swap in `StableDiffusionXLPipeline`; the training loop needs
  dual text encoders and added time-ids, so it's a meaningful rewrite.

## License

MIT
