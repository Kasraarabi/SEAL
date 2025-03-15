import torch
from torchvision import transforms
from datasets import load_dataset

from PIL import Image, ImageFilter
import random
import numpy as np
import copy
from typing import Any, Mapping
import json
import scipy
import io


def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)
    

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


def latents_to_imgs(pipe, latents):
    x = pipe.decode_image(latents)
    x = pipe.torch_to_numpy(x)
    x = pipe.numpy_to_pil(x)
    return x


def image_distortion(img1, img2, seed, r_degree=None, jpeg_ratio=None, crop_scale=None, crop_ratio=None,
                     gaussian_blur_r=None, gaussian_std=None, brightness_factor=None, run_name=None):
    # Seed setting for reproducibility moved to caller
    if r_degree is not None:
        if img1 is not None:
            img1 = transforms.RandomRotation((r_degree, r_degree))(img1)
        img2 = transforms.RandomRotation((r_degree, r_degree))(img2)

    if jpeg_ratio is not None:
        if img1 is not None:
            buf = io.BytesIO()
            img1.save(buf, format='JPEG', quality=jpeg_ratio)
            img1 = Image.open(buf)
        buf2 = io.BytesIO()
        img2.save(buf2, format='JPEG', quality=jpeg_ratio)
        img2 = Image.open(buf2)

    if crop_scale is not None and crop_ratio is not None:
        if img1 is not None:
            img1 = transforms.RandomResizedCrop(img1.size, scale=(crop_scale, crop_scale),
                                                ratio=(crop_ratio, crop_ratio))(img1)
        img2 = transforms.RandomResizedCrop(img2.size, scale=(crop_scale, crop_scale),
                                            ratio=(crop_ratio, crop_ratio))(img2)

    if gaussian_blur_r is not None:
        if img1 is not None:
            img1 = img1.filter(ImageFilter.GaussianBlur(radius=gaussian_blur_r))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=gaussian_blur_r))

    if gaussian_std is not None:
        img_shape = np.array(img2).shape  # Use img2’s shape since img1 may be None
        g_noise = np.random.normal(0, gaussian_std, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        if img1 is not None:
            img1 = Image.fromarray(np.clip(np.array(img1) + g_noise, 0, 255))
        img2 = Image.fromarray(np.clip(np.array(img2) + g_noise, 0, 255))

    if brightness_factor is not None:
        if img1 is not None:
            img1 = transforms.ColorJitter(brightness=brightness_factor)(img1)
        img2 = transforms.ColorJitter(brightness=brightness_factor)(img2)

    return [img1, img2]


# for one prompt to multiple images
def measure_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    with torch.no_grad():
        img_batch = [clip_preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.concatenate(img_batch).to(device)
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return (image_features @ text_features.T).mean(-1)


def get_dataset(args):
    if 'laion' in args.dataset:
        dataset = load_dataset(args.dataset)['train']
        prompt_key = 'TEXT'
    elif 'coco' in args.dataset:
        with open('fid_outputs/coco/meta_data.json') as f:
            dataset = json.load(f)
            dataset = dataset['annotations']
            prompt_key = 'caption'
    else:
        dataset = load_dataset(args.dataset)['test']
        prompt_key = 'Prompt'

    return dataset, prompt_key

def circle_mask(shape, r):
    height, width = shape
    center_h, center_w = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    return ((y - center_h)**2 + (x - center_w)**2 <= r**2)

def get_watermarking_mask(init_latents_w, args, device):
    watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool).to(device)

    if args.w_mask_shape == 'circle':
        np_mask = circle_mask(init_latents_w.shape[-1], r=args.w_radius)
        torch_mask = torch.tensor(np_mask).to(device)

        if args.w_channel == -1:
            # all channels
            watermarking_mask[:, :] = torch_mask
        else:
            watermarking_mask[:, args.w_channel] = torch_mask
    elif args.w_mask_shape == 'square':
        anchor_p = init_latents_w.shape[-1] // 2
        if args.w_channel == -1:
            # all channels
            watermarking_mask[:, :, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True
        else:
            watermarking_mask[:, args.w_channel, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True
    elif args.w_mask_shape == 'no':
        pass
    else:
        raise NotImplementedError(f'w_mask_shape: {args.w_mask_shape}')

    return watermarking_mask


def get_watermarking_pattern(pipe, args, device, shape=None):
    set_random_seed(args.w_seed)
    if shape is not None:
        gt_init = torch.randn(*shape, device=device)
    else:
        gt_init = pipe.get_random_latents()

    if 'seed_ring' in args.w_pattern:
        gt_patch = gt_init

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    elif 'seed_zeros' in args.w_pattern:
        gt_patch = gt_init * 0
    elif 'seed_rand' in args.w_pattern:
        gt_patch = gt_init
    elif 'rand' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]
    elif 'zeros' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
    elif 'const' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        gt_patch += args.w_pattern_const
    elif 'ring' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()

    return gt_patch


def inject_watermark(init_latents_w, watermarking_mask, gt_patch, args):
    init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w), dim=(-1, -2))
    if args.w_injection == 'complex':
        init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()
    elif args.w_injection == 'seed':
        init_latents_w[watermarking_mask] = gt_patch[watermarking_mask].clone()
        return init_latents_w
    else:
        NotImplementedError(f'w_injection: {args.w_injection}')

    init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real

    return init_latents_w


def eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    if 'complex' in args.w_measurement:
        reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))
        reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))
        target_patch = gt_patch
    elif 'seed' in args.w_measurement:
        reversed_latents_no_w_fft = reversed_latents_no_w
        reversed_latents_w_fft = reversed_latents_w
        target_patch = gt_patch
    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')

    if 'l1' in args.w_measurement:
        no_w_metric = torch.abs(reversed_latents_no_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
        w_metric = torch.abs(reversed_latents_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')

    return no_w_metric, w_metric

def get_p_value(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    # assume it's Fourier space wm
    reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))[watermarking_mask].flatten()
    reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))[watermarking_mask].flatten()
    target_patch = gt_patch[watermarking_mask].flatten()

    target_patch = torch.concatenate([target_patch.real, target_patch.imag])
    
    # no_w
    reversed_latents_no_w_fft = torch.concatenate([reversed_latents_no_w_fft.real, reversed_latents_no_w_fft.imag])
    sigma_no_w = reversed_latents_no_w_fft.std()
    lambda_no_w = (target_patch ** 2 / sigma_no_w ** 2).sum().item()
    x_no_w = (((reversed_latents_no_w_fft - target_patch) / sigma_no_w) ** 2).sum().item()
    p_no_w = scipy.stats.ncx2.cdf(x=x_no_w, df=len(target_patch), nc=lambda_no_w)

    # w
    reversed_latents_w_fft = torch.concatenate([reversed_latents_w_fft.real, reversed_latents_w_fft.imag])
    sigma_w = reversed_latents_w_fft.std()
    lambda_w = (target_patch ** 2 / sigma_w ** 2).sum().item()
    x_w = (((reversed_latents_w_fft - target_patch) / sigma_w) ** 2).sum().item()
    p_w = scipy.stats.ncx2.cdf(x=x_w, df=len(target_patch), nc=lambda_w)

    return p_no_w, p_w
