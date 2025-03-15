from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration, CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import os
from datasets import load_dataset
import numpy as np
import math
import zlib
import random
import wandb
from tqdm import tqdm
import torch.nn.functional as F
from utils import *
import argparse
from inverse_stable_diffusion import InversableStableDiffusionPipeline


def generate_initial_noise(embedding, k, b, seed, device):
    """Generate initial noise from an embedding vector using simhash."""
    patch_per_side = int(math.sqrt(k))
    assert patch_per_side ** 2 == k, "k must be a perfect square"
    keys = simhash(embedding.to(device), k, b, seed)
    
    initial_noise = torch.zeros(1, 4, 64, 64, device=device)
    patch_size = 64 // patch_per_side
    for i in range(patch_per_side):
        for j in range(patch_per_side):
            patch_idx = i * patch_per_side + j
            torch.manual_seed(keys[patch_idx])
            y_start = i * patch_size
            x_start = j * patch_size
            initial_noise[:, :, y_start:y_start+patch_size, x_start:x_start+patch_size] = torch.randn(
                (1, 4, patch_size, patch_size), device=device
            )
    return initial_noise

def get_embedding(config, image, sentence_model_finetuned, sentence_model_original, 
                  clip_processor, clip_model, cap_processor, cap_model, device):
    """Generate embedding based on the configuration."""
    if config["captioning"] == "blip2":
        caption = generate_caption(image, cap_processor, cap_model, device=device)
        sentence_model = (
            sentence_model_finetuned if config["sentence_model"] == "finetuned" 
            else sentence_model_original
        )
        embedding = sentence_model.encode(caption, convert_to_tensor=True).to(device)
    elif config["captioning"] == "none":
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        embedding = clip_model.get_image_features(**inputs).squeeze(0)
    return embedding


def get_config_key(config):
    """Generate a unique key for each configuration."""
    if config["captioning"] == "blip2":
        return f"blip2_{config['sentence_model']}"
    return "none_clip"


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Ablation study on captioning and embedding in Stable Diffusion')
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
    wandb.init(project=args.wandb_project, name="Captioning Embedding Ablation", entity=args.wandb_entity, config=vars(args))

    # Load Stable Diffusion pipeline
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16, revision='fp16'
    ).to(device)

    # Load all necessary models
    cap_processor = Blip2Processor.from_pretrained('Salesforce/blip2-flan-t5-xl')
    cap_model = Blip2ForConditionalGeneration.from_pretrained(
        'Salesforce/blip2-flan-t5-xl', torch_dtype=torch.float16
    ).to(device)
    sentence_model_finetuned = SentenceTransformer("kasraarabi/finetuned-caption-embedding").to(device)
    sentence_model_original = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    dataset = load_dataset('Gustavosta/Stable-Diffusion-Prompts')['train']
    os.makedirs(args.output_dir, exist_ok=True)

    # Define configurations to test
    configs = [
        {"captioning": "blip2", "sentence_model": "finetuned"},
        {"captioning": "blip2", "sentence_model": "original"},
        {"captioning": "none"}
    ]

    # Dictionary to store all results
    all_angles = {}

    # Iterate over all configurations
    for config in configs:
        config_key = get_config_key(config)
        all_angles[config_key] = {}
        
        for k in args.k_values:
            for b in args.b_values:
                related_angles = []
                unrelated_angles = []
                
                for img_idx in tqdm(range(args.start, args.end), desc=f"{config_key}, k={k}, b={b}"):
                    prompt_1 = dataset[img_idx]['Prompt']
                    prompt_2 = dataset[(img_idx + 1) % len(dataset)]['Prompt']

                    # Watermarked image process
                    first_image = pipe(prompt_1).images[0]
                    first_image_caption = generate_caption(first_image, cap_processor, cap_model)
                    embedding_1 = get_embedding(
                        config, first_image, sentence_model_finetuned, sentence_model_original,
                        clip_processor, clip_model, cap_processor, cap_model, device
                    )
                    second_image = pipe(first_image_caption).images[0]
                    embedding_2 = get_embedding(
                        config, second_image, sentence_model_finetuned, sentence_model_original,
                        clip_processor, clip_model, cap_processor, cap_model, device
                    )
                    # Random image process
                    random_image = pipe(prompt_2).images[0]
                    embedding_random = get_embedding(
                        config, random_image, sentence_model_finetuned, sentence_model_original,
                        clip_processor, clip_model, cap_processor, cap_model, device
                    )

                    related = angle_between(embedding_1,embedding_2)
                    unrelated = angle_between(embedding_1,embedding_random)

                    related_angles.append(related)
                    unrelated_angles.append(unrelated)


                # Store results for this k, b combination
                all_angles[config_key][f"{k}_{b}"] = {
                    "related": np.array(related_angles),
                    "unrelated": np.array(unrelated_angles)
                }

    wandb.finish()