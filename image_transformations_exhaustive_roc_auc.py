"""
Same as image_transformations_exhaustive_tpr.py, but instead reports ROC AUC.
Recommended code to obtain results comparable to Figure 6 in the supplemental material.
"""

from PIL import Image, ImageFilter
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import os
from datasets import load_dataset
from tqdm import tqdm
import torch.nn.functional as F
import math
import numpy as np
import zlib
import wandb
import random
import io
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from optim_utils import image_distortion
from utils import *
import argparse
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from sklearn.metrics import roc_curve, roc_auc_score
import pickle

def compute_l2(image, sentence_model, k, b, seed, pipe, device):
    tensor = transform_img(image).unsqueeze(0).to(device, dtype=pipe.vae.dtype)
    latents = pipe.get_image_latents(tensor, sample=False)
    recon_noise = pipe.forward_diffusion(
        latents=latents,
        text_embeddings=pipe.get_text_embedding(''),
        guidance_scale=1,
        num_inference_steps=50,
    )
    l2_list = calculate_patch_l2_exhaustive(
        recon_noise,
        k,
        b,
        seed,
        device
    )
    return l2_list

def calculate_patch_l2_exhaustive(noise_inv, k, b, seed, device):
    l2_list = []
    patch_per_side = int(math.ceil(math.sqrt(k)))
    patch_height = 64 // patch_per_side
    patch_width = 64 // patch_per_side
    patch_idx = 0
    num_candidates = 2 ** b
    for i in range(patch_per_side):
        for j in range(patch_per_side):
            if patch_idx >= k:
                break
            y_start = i * patch_height
            x_start = j * patch_width
            y_end = min(y_start + patch_height, 64)
            x_end = min(x_start + patch_width, 64)
            patch_inv = noise_inv[:, :, y_start:y_end, x_start:x_end]
            h = y_end - y_start
            w = x_end - x_start
            candidates = []
            for bit_pattern in range(num_candidates):
                bits = [(bit_pattern >> t) & 1 for t in range(b)]
                transformed_bits = [
                    bits[t] + t + patch_idx * 1e4
                    for t in range(b)
                ]
                key = deterministic_hash(tuple(transformed_bits))
                
                set_random_seed(key)

                candidate = torch.randn(
                    (4, h, w),
                    device=device
                )
                candidates.append(candidate)
            candidates = torch.stack(candidates, dim=0)
            patch_expand = patch_inv.expand(num_candidates, -1, -1, -1)
            l2_vals = torch.linalg.vector_norm(
                candidates - patch_expand,
                dim=(1,2,3)
            )
            best_l2 = torch.min(l2_vals).item()
            l2_list.append(best_l2)
            patch_idx += 1
    return l2_list

def compute_match_score(l2_list, tau):
    # number of patches that match
    m = sum(1 for l2 in l2_list if l2 < tau)
    return m

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Noise patch detection with distortions')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--k_values', nargs='+', type=int, default=[1024])
    parser.add_argument('--b_values', nargs='+', type=int, default=[7])
    parser.add_argument('--threshold', type=int, default=50)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=1000)
    parser.add_argument('--wandb_project', type=str, default='noise-detection')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--model_id', default='Manojb/stable-diffusion-2-1-base')
    parser.add_argument('--online', action='store_true', default=False)
    parser.add_argument('--save_each', action='store_true', default=False)
    parser.add_argument('--tau', type=float, default=2.3)
    parser.add_argument('--m_match', type=int, default=12)
    args = parser.parse_args()

    cache_path = os.path.join(args.output_dir, "experiment_cache.pkl")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb.init(project=args.wandb_project, name="exp_distortions", entity=args.wandb_entity, config=vars(args))
    # Load models
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16).to(device)
    cap_processor = Blip2Processor.from_pretrained('Salesforce/blip2-flan-t5-xl')
    cap_model = Blip2ForConditionalGeneration.from_pretrained(
        'Salesforce/blip2-flan-t5-xl', torch_dtype=torch.float16).to(device)
    sentence_model = SentenceTransformer("kasraarabi/finetuned-caption-embedding").to(device)
    dataset = load_dataset('Gustavosta/Stable-Diffusion-Prompts')['train']
    os.makedirs(args.output_dir, exist_ok=True)
    # Define distortion types based on distorted_image_list
    distortions = [
        {"name": "Clean", "params": {}},
        {"name": "Rot75", "params": {"r_degree": 75}},
        {"name": "JPEG25", "params": {"jpeg_ratio": 25}},
        {"name": "CropScale75", "params": {"crop_scale": 0.75, "crop_ratio": 0.75}},
        {"name": "Blur8", "params": {"gaussian_blur_r": 8}},
        {"name": "Noise0.1", "params": {"gaussian_std": 0.1}},
        {"name": "Brightness6", "params": {"brightness_factor": 6}},
    ]
    all_l2 = {}
    if os.path.exists(cache_path):
        print("Loading cached results...")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    for k in args.k_values:
        for b in args.b_values:
            combo_key = f"{k}_{b}"
            all_l2[combo_key] = {
                "watermarked_scores": {dist["name"]: [] for dist in distortions},
                "random_scores": []
            }
            for img_idx in tqdm(range(args.start, args.end), desc=f'k={k}, b={b}'):
                cache_key = (k, b, img_idx)

                if cache_key in cache:
                    result = cache[cache_key]

                    for dist_name, score in result["watermarked"].items():
                        all_l2[combo_key]["watermarked_scores"][dist_name].append(score)

                    all_l2[combo_key]["random_scores"].append(result["random"])

                    continue

                prompt_1 = dataset[img_idx]['Prompt']
                prompt_2 = dataset[(img_idx + 1) % len(dataset)]['Prompt']
                # Generate watermarked image
                first_image = pipe(prompt_1).images[0]
                image_caption = generate_caption(first_image, cap_processor, cap_model)
                embed = sentence_model.encode(image_caption, convert_to_tensor=True).to(device)
                embed = embed / torch.norm(embed)
                image_noise = generate_initial_noise(embed, k, b, 42, device).to(dtype=pipe.vae.dtype)
                org_img = pipe(prompt_1, latents=image_noise).images[0]
                # Apply distortions and compute min_l2
                watermarked_scores = {}

                for dist in distortions:
                    torch.manual_seed(img_idx)
                    np.random.seed(img_idx)
                    random.seed(img_idx)
                    if dist["name"] == "Clean":
                        distorted_img = org_img
                    else:
                        distorted_img = image_distortion(None, org_img, seed=img_idx, **dist["params"])[1] 
                    l2 = compute_l2(distorted_img, sentence_model, k, b, 42, pipe, device)
                    score = compute_match_score(l2, tau=args.tau)
                    all_l2[combo_key]["watermarked_scores"][dist["name"]].append(score)
                    watermarked_scores[dist["name"]] = score

                random_image = pipe(prompt_2).images[0]
                l2_random = compute_l2(random_image, sentence_model, k, b, 42, pipe, device)
                score_random = compute_match_score(l2_random, tau=args.tau)
                all_l2[combo_key]["random_scores"].append(score_random)

                cache[cache_key] = {
                    "watermarked": watermarked_scores,
                    "random": score_random
                }

                if img_idx % 10 == 0:
                    with open(cache_path, "wb") as f:
                        pickle.dump(cache, f)

            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)

            print("Cache saved.")
            
            random_scores = np.array(all_l2[combo_key]["random_scores"])

            auc_results = []

            for dist in distortions:
                name = dist["name"]
                wm_scores = np.array(all_l2[combo_key]["watermarked_scores"][name])

                y_true = np.concatenate([
                    np.ones_like(wm_scores),   # watermarked
                    np.zeros_like(random_scores)  # random
                ])

                y_scores = np.concatenate([
                    wm_scores,
                    random_scores
                ])

                fpr, tpr, _ = roc_curve(y_true, y_scores)
                auc = roc_auc_score(y_true, y_scores)

                print(name, "ROC-AUC:", auc)
                auc_results.append(auc)

                plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

            plt.plot([0,1],[0,1],'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curves (k={k}, b={b})")
            plt.legend()
            plt.tight_layout()

            plt.savefig(os.path.join(args.output_dir, f"roc_curve_{combo_key}.png"))
            plt.show()

            names = [d["name"] for d in distortions]

            plt.figure(figsize=(10,5))
            plt.bar(names, auc_results)
            plt.ylim(0.5, 1.05)

            plt.ylabel("ROC-AUC")
            plt.xlabel("Image Transformation")
            plt.title(f"Detection ROC-AUC (k={k}, b={b})")

            for i, v in enumerate(auc_results):
                plt.text(i, v + 0.02, f"{v:.3f}", ha="center")

            plt.xticks(rotation=30)
            plt.tight_layout()

            plt.savefig(os.path.join(args.output_dir, f"roc_auc_{combo_key}.png"))
            plt.show()


    wandb.finish()