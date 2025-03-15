from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import os
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F
import math
import numpy as np
import zlib
import wandb
import random
import os
from utils import *
import argparse
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Noise patch detection analysis with enhanced brute-force simhash noise selection')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--k_values', nargs='+', type=int, default=[1024])
    parser.add_argument('--b_values', nargs='+', type=int, default=[7])
    parser.add_argument('--threshold', type=int, default=50)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=1000)
    parser.add_argument('--wandb_project', type=str, default='noise-detection')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--online', action='store_true', default=False)
    parser.add_argument('--save_each', action='store_true', default=False)

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb.init(project=args.wandb_project, name="regeneration", entity=args.wandb_entity, config=vars(args))

    # Load models
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        revision='fp16',
    ).to(device)
    cap_processor = Blip2Processor.from_pretrained('Salesforce/blip2-flan-t5-xl')
    cap_model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-flan-t5-xl', torch_dtype=torch.float16).to(device)
    sentence_model = SentenceTransformer("kasraarabi/finetuned-caption-embedding").to(device)

    dataset = load_dataset('Gustavosta/Stable-Diffusion-Prompts')['train']
    os.makedirs(args.output_dir, exist_ok=True)

    for k in args.k_values:
        for b in args.b_values:
            patch_per_side = int(math.sqrt(k))
            org_l2_list = []
            
            for img_idx in tqdm(range(args.start, args.end), desc=f'k={k}, b={b}'):
                prompt_1 = dataset[img_idx]['Prompt']
                prompt_2 = dataset[(img_idx+1) % len(dataset)]['Prompt']
                first_image = pipe(prompt_1).images[0]
                image_caption = generate_caption(first_image, cap_processor, cap_model)
                first_embed = sentence_model.encode(image_caption, convert_to_tensor=True).to(device)
                first_embed = first_embed / torch.norm(first_embed)

                image_noise = generate_initial_noise(first_embed, k, b, 42, device).to(dtype=pipe.vae.dtype)
                image = pipe(prompt_1, latents=image_noise).images[0]
                org_img = image
                
                # Inverse the original image
                org_tensor = transform_img(org_img).unsqueeze(0).to(device)
                org_tensor = org_tensor.to(dtype=pipe.vae.dtype)
                org_latents = pipe.get_image_latents(org_tensor, sample=False)
                recon_noise_org = pipe.forward_diffusion(
                    latents=org_latents,
                    text_embeddings=pipe.get_text_embedding(''),
                    guidance_scale=1,
                    num_inference_steps=50,
                )

                # Generate the second image and inverse the noise and create the noise with caption
                second_image = pipe(prompt_2, latents=recon_noise_org).images[0]
                second_tensor = transform_img(second_image).unsqueeze(0).to(device)
                second_tensor = second_tensor.to(dtype=pipe.vae.dtype)
                second_latents = pipe.get_image_latents(second_tensor, sample=False)
                second_recon_noise = pipe.forward_diffusion(
                    latents=second_latents,
                    text_embeddings=pipe.get_text_embedding(''),
                    guidance_scale=1,
                    num_inference_steps=50,
                )

                second_caption = generate_caption(second_image, cap_processor, cap_model)
                second_embed = sentence_model.encode(second_caption, convert_to_tensor=True).to(device)
                second_embed = second_embed / torch.norm(second_embed)

                second_noise = generate_initial_noise(second_embed, k, b, 42, device).to(dtype=pipe.vae.dtype)

                l2 = calculate_patch_l2(second_recon_noise,second_noise,k)

                org_l2_list.append(l2)
      
    wandb.finish()
