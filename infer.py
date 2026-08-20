import torch, argparse
from diffusers import StableDiffusionPipeline

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--lora", default="lora-out")
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", default="sample.png")
    args = p.parse_args()

    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda")
    pipe.unet.load_attn_procs(args.lora)  # or pipe.load_lora_weights(args.lora)

    image = pipe(args.prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save(args.out)

if __name__ == "__main__":
    main()
