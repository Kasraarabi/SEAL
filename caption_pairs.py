from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import os
import math
import numpy as np
import argparse
import zlib
import pandas as pd
from tqdm import tqdm
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from datasets import load_dataset
from utils import get_dataset

def generate_caption(image, processor, model, do_sample=False, device='cuda'):
    """Generates a caption for the given image using BLIP-2 model."""
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        image = image.convert('RGB')
    else:
        raise ValueError("Image must be either a file path or a PIL Image object")
    
    inputs = processor(image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

def compute_simhash(embedding, num_patches, num_bits, seed):
    """Computes SimHash keys from an embedding."""
    random.seed(seed)
    hash_keys = []
    
    for patch_index in range(num_patches):
        bits = [0] * num_bits
        for bit_index in range(num_bits):
            random_vector = torch.randn_like(embedding)
            bits[bit_index] = 1 if torch.dot(random_vector, embedding) > 0 else 0
            bits[bit_index] = (bits[bit_index] + bit_index + patch_index) % 256
        hash_keys.append(zlib.crc32(bytes(bits)) & 0xFFFFFFFF)
    
    return hash_keys

def generate_noise_pattern(embedding_model, prompt, num_patches, num_bits, seed, device):
    """Generates an initial noise pattern based on prompt embedding."""
    patches_per_side = int(math.sqrt(num_patches))
    assert patches_per_side ** 2 == num_patches, "num_patches must be a perfect square"
    
    prompt_embedding = embedding_model.encode(prompt, convert_to_tensor=True).to(device)
    hash_keys = compute_simhash(prompt_embedding, num_patches, num_bits, seed)
    
    noise_tensor = torch.zeros(1, 4, 64, 64, device=device)
    patch_size = 64 // patches_per_side
    
    for row in range(patches_per_side):
        for col in range(patches_per_side):
            patch_index = row * patches_per_side + col
            torch.manual_seed(hash_keys[patch_index])
            
            y_start, y_end = row * patch_size, (row + 1) * patch_size
            x_start, x_end = col * patch_size, (col + 1) * patch_size
            
            noise_tensor[:, :, y_start:y_end, x_start:x_end] = torch.randn(
                (1, 4, patch_size, patch_size), device=device
            )
    
    return noise_tensor

def compute_caption_consistency(output_dir, k_values, b_values, dataset, reference_dataset, model_pipeline, processor, model):
    """Computes consistency of captions across generations and saves results."""
    os.makedirs(output_dir, exist_ok=True)
    caption_pairs = []
    
    for k in k_values:
        for b in b_values:
            patches_per_side = int(math.sqrt(k))
            
            for i in range(len(dataset[0])):
                original_prompt = dataset[0][i]['caption']
                
                first_image = model_pipeline(original_prompt).images[0]
                first_caption = generate_caption(first_image, processor, model)
                
                second_image = model_pipeline(first_caption).images[0]
                second_caption = generate_caption(second_image, processor, model)
                
                caption_pairs.append((first_caption, second_caption))
                
                reference_prompt = reference_dataset[i]['Prompt']
                ref_first_image = model_pipeline(reference_prompt).images[0]
                ref_first_caption = generate_caption(ref_first_image, processor, model)
                
                ref_second_image = model_pipeline(ref_first_caption).images[0]
                ref_second_caption = generate_caption(ref_second_image, processor, model)
                
                caption_pairs.append((ref_first_caption, ref_second_caption))
    
    df = pd.DataFrame(caption_pairs, columns=["Generated Caption", "Regenerated Caption"])
    output_file = os.path.join(output_dir, "caption_pairs.csv")
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Saved caption pairs to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Noise Patch Detection Analysis')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--k_values', nargs='+', type=int, default=[1024])
    parser.add_argument('--b_values', nargs='+', type=int, default=[7])
    parser.add_argument('--model_id', type=str, default='stabilityai/stable-diffusion-2-1-base')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load models
    diffusion_pipeline = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16, revision='fp16'
    ).to(device)
    caption_processor = Blip2Processor.from_pretrained('Salesforce/blip2-flan-t5-xl')
    caption_model = Blip2ForConditionalGeneration.from_pretrained(
        'Salesforce/blip2-flan-t5-xl', torch_dtype=torch.float16
    ).to(device)
    sentence_model = SentenceTransformer('paraphrase-Mpnet-base-v2').to(device)
    
    # Load datasets
    primary_dataset = get_dataset("coco")
    reference_dataset = load_dataset('Gustavosta/Stable-Diffusion-Prompts')['train']
    
    compute_caption_consistency(
        args.output_dir, args.k_values, args.b_values,
        primary_dataset, reference_dataset, diffusion_pipeline,
        caption_processor, caption_model
    )
