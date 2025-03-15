from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration, CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import os
from torch_fidelity import calculate_metrics
import numpy as np
from tqdm import tqdm
import json
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from utils import *


# Function to compute CLIP Score
def compute_clip_score(image_path, prompt, clip_model, clip_processor, device):
    """Compute the CLIP Score for an image and its prompt."""
    image = Image.open(image_path)
    inputs = clip_processor(
        text=[prompt],           # Pass prompt as a list for proper batch processing
        images=image,            # Include the image for processing
        return_tensors="pt",     # Return PyTorch tensors
        padding=True,            # Pad shorter inputs
        truncation=True,         # Truncate prompts exceeding max_length
        max_length=77,           # Enforce CLIP's 77-token limit
    ).to(device)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    return logits_per_image.item()

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")

    # Load models
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        revision='fp16',
    ).to(device)
    cap_processor = Blip2Processor.from_pretrained('Salesforce/blip2-flan-t5-xl')
    cap_model = Blip2ForConditionalGeneration.from_pretrained(
        'Salesforce/blip2-flan-t5-xl', torch_dtype=torch.float16
    ).to(device)
    sentence_model = SentenceTransformer("kasraarabi/finetuned-caption-embedding").to(device)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


    # Datasets to process
    datasets = ['Gustavosta/Stable-Diffusion-Prompts', 'coco']

    for k in args.k_values:
        for b in args.b_values:

            for dataset_name in datasets:
                print(f"\nProcessing dataset: {dataset_name}")
                dataset, prompt_key = get_dataset(dataset_name)
                
                # Create directories for saving images
                original_dir = os.path.join(args.output_dir, dataset_name, 'original')
                watermarked_dir = os.path.join(args.output_dir, dataset_name, 'watermarked')
                os.makedirs(original_dir, exist_ok=True)
                os.makedirs(watermarked_dir, exist_ok=True)


                clip_scores_before = []
                clip_scores_after = []

                # Generate images
                for idx in tqdm(range(args.start, args.end)):

                    prompt = dataset[idx][prompt_key]

                    # Generate original image
                    original_image = pipe(prompt).images[0]
                    original_path = os.path.join(original_dir, f'{idx}.png')
                    original_image.save(original_path)
                    clip_scores_before.append(compute_clip_score(original_path, prompt, clip_model, clip_processor, device))


                    # Generate caption from original image
                    caption = generate_caption(original_image, cap_processor, cap_model)

                    embed = sentence_model.encode(caption, convert_to_tensor=True).to(device)
                    embed = embed / torch.norm(embed)

                    # Generate initial noise for watermarking
                    image_noise = generate_initial_noise(embed, k, b, 42, device).to(dtype=pipe.vae.dtype)

                    # Generate watermarked image
                    watermarked_image = pipe(prompt, latents=image_noise).images[0]
                    watermarked_path = os.path.join(watermarked_dir, f'{idx}.png')
                    watermarked_image.save(watermarked_path)

                    # Compute CLIP Score for watermarked image
                    clip_score = compute_clip_score(watermarked_path, prompt, clip_model, clip_processor, device)
                    clip_scores_after.append(clip_score)

                # Compute FID and Inception Score
                metrics = calculate_metrics(
                    input1=watermarked_dir,  # Watermarked images for IS
                    input2=original_dir,     # Original images for FID comparison
                    cuda=device == 'cuda',
                    isc=True,                # Compute Inception Score
                    fid=True,                # Compute FID
                    verbose=False
                )

                fid = metrics['frechet_inception_distance']
                isc = metrics['inception_score_mean']
                avg_clip_score_before = np.mean(clip_scores_before)
                avg_clip_score_after = np.mean(clip_scores_after)

                # Report results
                print(f"\nResults for {dataset_name}:")
                print(f"FID (between original and watermarked): {fid:.4f}")
                print(f"Inception Score (watermarked): {isc:.4f}")
                print(f"Average CLIP Score (before): {avg_clip_score_before:.4f}")
                print(f"Average CLIP Score (after): {avg_clip_score_after:.4f}")

                # Append results to qres.txt
                with open("qres.txt", "a") as f:
                    f.write(f"\nResults for {dataset_name}:\n")
                    f.write(f"\nResults for {k}-{b}:\n")
                    f.write(f"FID (between original and watermarked): {fid:.4f}\n")
                    f.write(f"Inception Score (watermarked): {isc:.4f}\n")
                    f.write(f"Average CLIP Score (before): {avg_clip_score_before:.4f}\n")
                    f.write(f"Average CLIP Score (after): {avg_clip_score_after:.4f}\n")
                    f.write("-"*50 + "\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Watermarking experiment with Stable Diffusion')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save generated images')
    parser.add_argument('--start', type=int, default=0, help='Start index of dataset')
    parser.add_argument('--end', type=int, default=100, help='End index of dataset')
    parser.add_argument('--model_id', type=str, default='stabilityai/stable-diffusion-2-1-base', help='Stable Diffusion model ID')
    parser.add_argument('--k_values', nargs='+', type=int, default=[1024])
    parser.add_argument('--b_values', nargs='+', type=int, default=[7])

    args = parser.parse_args()
    main(args)