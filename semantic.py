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
from utils import *
import argparse
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Noise patch detection analysis with angle-based embedding sampling')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--k_values', nargs='+', type=int, default=[1024])
    parser.add_argument('--b_values', nargs='+', type=int, default=[7])
    parser.add_argument('--embedding_pool_size', type=int, default=1000, help='Number of embeddings to generate')
    parser.add_argument('--angle_bins', type=int, default=30, help='Number of angle bins between 0-90 degrees')
    parser.add_argument('--pairs_per_bin', type=int, default=30, help='Number of embedding pairs to sample per angle bin')
    parser.add_argument('--wandb_project', type=str, default='noise-detection')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--online', action='store_true', default=False)
    parser.add_argument('--save_each', action='store_true', default=False)
    parser.add_argument('--precompute_only', action='store_true', default=False, help='Only precompute embedding pool without running experiments')

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.online:
        wandb.init(project=args.wandb_project, name="semantic", entity=args.wandb_entity, config=vars(args))


    dataset = load_dataset('Gustavosta/Stable-Diffusion-Prompts')['train']
    os.makedirs(args.output_dir, exist_ok=True)

    sentence_model = SentenceTransformer("kasraarabi/finetuned-caption-embedding").to(device)


    # Create embedding pool
    print(f"Creating embedding pool with {args.embedding_pool_size} samples...")
    embeddings_pool, prompts_pool, indices_pool = create_embedding_pool(dataset, sentence_model, device, num_samples=args.embedding_pool_size)
    
    # Calculate angle matrix
    print("Calculating angle matrix between all embeddings...")
    angle_matrix = calculate_angle_matrix(embeddings_pool)
    
    # Save angle matrix and embedding data
    torch.save({
        'angle_matrix': angle_matrix,
        'embeddings': embeddings_pool,
        'prompts': prompts_pool,
        'indices': indices_pool
    }, f"{args.output_dir}/embedding_pool_data.pt")
    
    print(f"Angle matrix statistics:")
    print(f"  Min angle: {angle_matrix[angle_matrix > 0].min().item():.2f} degrees")
    print(f"  Max angle: {angle_matrix.max().item():.2f} degrees")
    print(f"  Mean angle: {angle_matrix[angle_matrix > 0].mean().item():.2f} degrees")
        
    # Select pairs that span 0-90 degrees multiple times
    print(f"Selecting pairs spanning 0-90 degrees ({args.angle_bins} bins, {args.pairs_per_bin} pairs per bin)...")
    selected_pairs = select_angle_spanning_pairs(angle_matrix, num_bins=args.angle_bins, pairs_per_bin=args.pairs_per_bin)
    print(f"Selected {len(selected_pairs)} pairs for analysis")
    
    # Save selected pairs
    torch.save(selected_pairs, f"{args.output_dir}/selected_pairs.pt")
    
    if args.precompute_only:
        print("Precomputation complete. Exiting without running experiments.")
        if args.online:
            wandb.finish()
        exit(0)

    
    # Load models
    print("Loading models...")
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        revision='fp16',
    ).to(device)
    
    cap_processor = Blip2Processor.from_pretrained('Salesforce/blip2-flan-t5-xl')
    cap_model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-flan-t5-xl', torch_dtype=torch.float16).to(device)

    
    # Track overall best hyperparameters
    overall_best_accuracy = 0
    overall_best_params = None

    for k in args.k_values:
        for b in args.b_values:
            print(f"Running experiment with k={k}, b={b}")
            all_results = []
            
            for pair_idx, (i, j, angle) in enumerate(tqdm(selected_pairs, desc=f"Processing pairs (k={k}, b={b})")):
                embed_prompt = embeddings_pool[i]
                embed_prompt_2 = embeddings_pool[j]
                
                # Generate noise for both embeddings
                image_noise = generate_initial_noise(embed_prompt, k, b, 42, device).to(dtype=pipe.vae.dtype)
                random_noise = generate_initial_noise(embed_prompt_2, k, b, 42, device).to(dtype=pipe.vae.dtype)
                
                # Generate and process image with first embedding's noise
                image = pipe(prompts_pool[i], latents=image_noise).images[0]
                org_img = image
                
                # Inverse the watermarked image
                org_tensor = transform_img(org_img).unsqueeze(0).to(device)
                org_tensor = org_tensor.to(dtype=pipe.vae.dtype)
                org_latents = pipe.get_image_latents(org_tensor, sample=False)
                recon_noise = pipe.forward_diffusion(
                    latents=org_latents,
                    text_embeddings=pipe.get_text_embedding(''),
                    guidance_scale=1,
                    num_inference_steps=50,
                )
                
                # Calculate patch-wise L2 distances
                wm_l2_list = calculate_patch_l2(recon_noise, image_noise, k)
                random_l2_list = calculate_patch_l2(recon_noise, random_noise, k)
                
                # Store results
                result = {
                    'angle': angle,
                    'wm_l2': wm_l2_list,
                    'random_l2': random_l2_list,
                    'prompt1': prompts_pool[i],
                    'prompt2': prompts_pool[j],
                    'prompt1_idx': indices_pool[i],
                    'prompt2_idx': indices_pool[j]
                }
                all_results.append(result)
                

    if args.online:
        wandb.finish()