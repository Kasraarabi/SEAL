import os
import math
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
import matplotlib.pyplot as plt
from huggingface_hub import login

login()


# -----------------------------
# 0. Check if GPU is available
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# -----------------------------
# 1. Load Related Caption Pairs
# -----------------------------
# Expecting caption_pairs.csv to have columns: 'image_caption' and 'org_caption'
csv_file = 'outputs/caption_pairs.csv'
df = pd.read_csv(csv_file)
if not {'image_caption', 'org_caption'}.issubset(df.columns):
    raise ValueError("CSV file must contain 'image_caption' and 'org_caption' columns.")

# Create training examples from your related caption pairs
train_examples = []
for _, row in df.iterrows():
    train_examples.append(InputExample(texts=[row['image_caption'], row['org_caption']]))
print(f"Loaded {len(train_examples)} training examples from {csv_file}.")

# -----------------------------
# 2. Fine-Tune a Sentence Embedding Model
# -----------------------------
# We use a pre-trained model (e.g., all-MiniLM-L6-v2) from SentenceTransformers
model_name = 'paraphrase-Mpnet-base-v2'
model = SentenceTransformer(model_name, device=device)

# Create a DataLoader for our training examples
batch_size = 64
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

# Use MultipleNegativesRankingLoss (in-batch negatives)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# Define training parameters
num_epochs = 140
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% of total steps

# Train the model
print("Starting training...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path='output/trained_model',
    show_progress_bar=True
)
print("Training complete. Model saved to 'output/trained_model'.")

# -----------------------------
# 3. Compute Angles for Related and Random Pairs
# -----------------------------
def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def angle_from_cosine(cos_val):
    """Convert cosine similarity to an angle in degrees."""
    # Clip value to avoid numerical issues
    cos_val = np.clip(cos_val, -1.0, 1.0)
    angle_rad = math.acos(cos_val)
    return math.degrees(angle_rad)

# Encode all captions from the CSV
captions1 = df['image_caption'].tolist()
captions2 = df['org_caption'].tolist()
embeddings1 = model.encode(captions1, convert_to_numpy=True)
embeddings2 = model.encode(captions2, convert_to_numpy=True)

# Calculate angles for the related pairs (ideally these should be low)
related_angles = []
for vec1, vec2 in zip(embeddings1, embeddings2):
    cos_sim = cosine_similarity(vec1, vec2)
    angle = angle_from_cosine(cos_sim)
    related_angles.append(angle)
avg_related_angle = np.mean(related_angles)
print(f"\nAverage angle for related pairs: {avg_related_angle:.2f}°")

# For unrelated pairs, randomly sample two captions from all captions.
all_captions = captions1 + captions2
num_random_pairs = 1000  # Adjust as needed
random_angles = []
for _ in range(num_random_pairs):
    cap1, cap2 = random.sample(all_captions, 2)  # ensure two different captions
    vec1 = model.encode(cap1, convert_to_numpy=True)
    vec2 = model.encode(cap2, convert_to_numpy=True)
    cos_sim = cosine_similarity(vec1, vec2)
    angle = angle_from_cosine(cos_sim)
    random_angles.append(angle)
avg_random_angle = np.mean(random_angles)
print(f"Average angle for random (unrelated) pairs: {avg_random_angle:.2f}°")

# -----------------------------
# 4. Determine a Threshold for Perfect Separation
# -----------------------------
max_related = max(related_angles)
min_random = min(random_angles)
print(f"\nMax angle for related pairs: {max_related:.2f}°")
print(f"Min angle for random pairs: {min_random:.2f}°")

if max_related < min_random:
    # Perfect separation exists; choose a threshold in between.
    threshold_angle = (max_related + min_random) / 2
    print(f"Perfect separation achieved. Threshold angle set to {threshold_angle:.2f}°")
else:
    # If not perfectly separated, choose a threshold (e.g., halfway between average angles)
    threshold_angle = (avg_related_angle + avg_random_angle) / 2
    print(f"No perfect separation. Using threshold angle {threshold_angle:.2f}° as a decision boundary.")

# Evaluate classification performance:
TP = sum(1 for angle in related_angles if angle <= threshold_angle)
FN = len(related_angles) - TP
FP = sum(1 for angle in random_angles if angle <= threshold_angle)
TN = len(random_angles) - FP

print("\nClassification Results on the sampled pairs:")
print(f"Related pairs: {TP} True Positives, {FN} False Negatives")
print(f"Random pairs: {FP} False Positives, {TN} True Negatives")

# -----------------------------
# 5. Utility: Classify Any Caption Pair
# -----------------------------
def classify_pair(caption1, caption2, threshold=threshold_angle):
    """Return True if the pair is classified as related; else False."""
    vec1 = model.encode(caption1, convert_to_numpy=True)
    vec2 = model.encode(caption2, convert_to_numpy=True)
    cos_sim = cosine_similarity(vec1, vec2)
    angle = angle_from_cosine(cos_sim)
    return angle <= threshold


# -----------------------------
# 6. Plot Distribution Histograms for Angle Distributions
# -----------------------------
# Sample 100 angles from each list (or use all if fewer than 100)
num_samples = 100
related_sample = random.sample(related_angles, min(num_samples, len(related_angles)))
random_sample = random.sample(random_angles, min(num_samples, len(random_angles)))

plt.figure(figsize=(10, 6))
plt.hist(related_sample, bins=20, color='blue', alpha=0.5, label='Related Pairs', edgecolor='black')
plt.hist(random_sample, bins=20, color='red', alpha=0.5, label='Unrelated Pairs', edgecolor='black')

plt.title("Angle Distribution for Related and Unrelated Caption Pairs")
plt.xlabel("Angle (degrees)")
plt.ylabel("Frequency")
plt.legend()
plt.savefig('distribution_of_angles.png')
plt.show()