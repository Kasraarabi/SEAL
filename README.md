<div align="center">
  <h1>SEAL: Semantic Aware Image Watermarking</h1>
  
  <img src="assets/teaser.png" width="100%" height="75%" style="display:block; margin:auto;">
</div>

---

### Overview

We propose **SEAL**, a watermarking method that embeds semantic information about the generated image directly into the watermark, allowing for a distortion-free watermark that can be verified without a database of key patterns. Instead of relying on stored keys, SEAL infers the key from the image’s semantic embedding using locality-sensitive hashing. Additionally, we address two often-overlooked attack strategies: (i) an attacker extracting the initial noise to create a new image with the same pattern and (ii) an attacker inserting an unrelated, potentially harmful object into a watermarked image while preserving the watermark.

### Setup

Install all required dependencies by running:

```bash
chmod +x setup.sh
./setup.sh
```

### Usage

```bash
python SEAL.py
```

The repository contains code for all experiments discussed in the paper. Explore the provided implementations to reproduce our results and evaluate the robustness of our method against different attacks.
