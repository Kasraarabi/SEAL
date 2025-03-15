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
    wandb.init(project=args.wandb_project, name="cat_attack", entity=args.wandb_entity, config=vars(args))

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
            heatmap_counts = np.zeros((patch_per_side, patch_per_side), dtype=float)
            random_heatmap_counts = np.zeros((patch_per_side, patch_per_side), dtype=float)
            org_heatmap_counts = np.zeros((patch_per_side, patch_per_side), dtype=float)
            cat_mask = get_cat_patches_mask(k)
            detected_dict = {}
            org_detected_dict = {}
            random_detected_dict = {}
            
            for img_idx in tqdm(range(args.start, args.end), desc=f'k={k}, b={b}'):
                prompt = dataset[img_idx]['Prompt']
                random_prompt = dataset[(img_idx+1) % len(dataset)]['Prompt']
                first_image = pipe(prompt).images[0]
                image_caption = generate_caption(first_image, cap_processor, cap_model)
                embed = sentence_model.encode(image_caption, convert_to_tensor=True).to(device)
                embed = embed / torch.norm(embed)

                image_noise = generate_initial_noise(embed, k, b, 42, device).to(dtype=pipe.vae.dtype)
                image = pipe(prompt, latents=image_noise).images[0]
                org_img = image
                random_image = pipe(random_prompt).images[0]

                # Add cat and invert
                cat_image, pos, size = add_cat_to_image(image.copy(), 'cat.png', '', '', save=False)
                cat_tensor = transform_img(cat_image).unsqueeze(0).to(device)
                cat_tensor = cat_tensor.to(dtype=pipe.vae.dtype)
                cat_latents = pipe.get_image_latents(cat_tensor, sample=False)
                recon_noise = pipe.forward_diffusion(
                    latents=cat_latents,
                    text_embeddings=pipe.get_text_embedding(''),
                    guidance_scale=1,
                    num_inference_steps=50,
                )
                
                # Inverse the random image
                random_tensor = transform_img(random_image).unsqueeze(0).to(device)
                random_tensor = random_tensor.to(dtype=pipe.vae.dtype)
                random_latents = pipe.get_image_latents(random_tensor, sample=False)
                recon_noise_random = pipe.forward_diffusion(
                    latents=random_latents,
                    text_embeddings=pipe.get_text_embedding(''),
                    guidance_scale=1,
                    num_inference_steps=50,
                )
                random_caption = generate_caption(random_image, cap_processor, cap_model)
                rand_embed = sentence_model.encode(random_caption, convert_to_tensor=True).to(device)
                rand_embed = rand_embed / torch.norm(rand_embed)

                random_noise = generate_initial_noise(rand_embed, k, b, 42, device).to(dtype=pipe.vae.dtype)
                
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
                org_caption = generate_caption(org_img, cap_processor, cap_model)
                org_embed = sentence_model.encode(org_caption, convert_to_tensor=True).to(device)
                org_embed = org_embed / torch.norm(org_embed)

                org_noise = generate_initial_noise(org_embed, k, b, 42, device).to(dtype=pipe.vae.dtype)

                cat_caption = generate_caption(cat_image, cap_processor, cap_model)
                cat_embed = sentence_model.encode(cat_caption, convert_to_tensor=True).to(device)
                cat_embed = cat_embed / torch.norm(cat_embed)

                cat_noise = generate_initial_noise(cat_embed, k, b, 42, device).to(dtype=pipe.vae.dtype)
                
                cat_l2 = calculate_patch_l2(cat_noise, recon_noise, k)
                without_cat_l2 = calculate_patch_l2(org_noise, recon_noise_org, k)
                random_l2 = calculate_patch_l2(random_noise, recon_noise_random, k)


    wandb.finish()
