import alphaclip
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
from utils import *
import re
import os
import cv2
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio
from tqdm.auto import tqdm
import torch
import torchvision as tv
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig
import warnings
warnings.filterwarnings('ignore')

target_device = torch.device("cuda:3")

def get_frame_number(filename):
    match = re.search(r"frame_(\d+)\.jpg", filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return -1

def create_subplot_gif(left_files=None, right_image_folder=None, 
                       output_path="subplot_output.gif", duration=0.1, 
                       figsize=(10, 5), dpi=100, loop=False):
    right_files = [f for f in os.listdir(right_image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    right_files = sorted(right_files, key=get_frame_number)
    # if left_image_folder:
    #     left_files = [f for f in os.listdir(left_image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    #     left_files = sorted(left_files, key=get_frame_number)
    
    import tempfile
    temp_dir = tempfile.mkdtemp()

    frames = []
    
    for i, right_file in enumerate(right_files):
        # Create a new figure for each frame
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        
        # Left subplot (optional)
        if left_files:
            left_img = Image.open(os.path.join("/scratch3/kat049/user_studies/vids/p14_fr/fps_1/frames", left_files[i]))
            axs[0].imshow(np.array(left_img))
        else:
            axs[0].text(0.5, 0.5, "No Left Image", 
                       horizontalalignment='center', verticalalignment='center')
        
        # Right subplot (always present)
        right_img = Image.open(os.path.join(right_image_folder, right_file))
        axs[1].imshow(np.array(right_img))
        
        # Remove axis ticks
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Adjust spacing
        plt.tight_layout()
        
        # Save the frame to temporary directory
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_path, dpi=dpi)
        plt.close()
        frames.append(imageio.imread(frame_path))

    imageio.mimsave(output_path, frames, duration=duration, loop=False)

def create_gif(image_folder, output_gif_path, duration=0.1, loop=False):
    images = []

    jpg_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]  # Handle cases where the pattern doesn't match

    filenames = sorted(jpg_files, key=get_frame_number)

    for filename in filenames:
        filepath = os.path.join(image_folder, filename)
        img = imageio.imread(filepath)
        images.append(img)

    imageio.mimsave(output_gif_path, images, duration=duration, loop=False)

def create_gif_from_masks(mask_sequence, output_gif_path, duration=100, loop=0):
    images = []
    for mask in mask_sequence:
        # Ensure mask is in the correct format (0-255 grayscale)
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        img = Image.fromarray(mask).convert('L')  # 'L' mode for grayscale
        images.append(img)

    if images:
        images[0].save(
            output_gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop
        )
        print(f"GIF created successfully at: {output_gif_path}")
    else:
        print("No valid masks found to create GIF.")

def test():
    # initialize EVF-SAM
    tokenizer, evfsam = init_models()

    # initialize Alpha-CLIP
    clip, clip_preprocess = alphaclip.load('ViT-L/14@336px', alpha_vision_ckpt_pth='weights/clip_l14_336_grit_20m_4xe.pth', device=target_device)
    clip_preprocess_mask = transforms.Compose([transforms.Resize((336, 336)), transforms.Normalize(0.5, 0.26)])

    # initialize Cutie
    cutie = get_default_model(config='ytvos_config').to(target_device)
    processor = InferenceCore(cutie, cfg=cutie.cfg)

    # load videos
    output_dir = 'outputs'
    save_path_prefix = os.path.join(output_dir, 'Annotations')
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

    img_folder = "/scratch3/kat049/user_studies/vids/p14_fr/fps_1/frames"
    jpg_files = [f for f in os.listdir(img_folder) if f.lower().endswith(".jpg")]

    sorted_jpg_files = sorted(jpg_files, key=get_frame_number)
    sorted_jpg_files = sorted_jpg_files[200:500]
      # Limit to the first 100 frames
    video_list = sorted_jpg_files
    video_len = len(sorted_jpg_files)
    # input pre-process
    imgs_beit = []
    imgs_sam = []
    imgs_clip = []
    imgs_cutie = []
    for item in tqdm(video_list):
        img_path = os.path.join(img_folder, item)
        image_np = cv2.imread(img_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        # BEiT pre-process
        img_beit = beit3_preprocess(Image.open(img_path), 224)
        imgs_beit.append(img_beit)

        # SAM pre-process
        img_sam, resize_shape = sam_preprocess(image_np)
        imgs_sam.append(img_sam)

        # Alpha-CLIP pre-process
        img_clip = clip_preprocess(Image.open(img_path))
        imgs_clip.append(img_clip)

        # Cutie pre-process
        img_cutie = tv.transforms.ToTensor()(Image.open(img_path))
        imgs_cutie.append(img_cutie)

    
    exp = "spot, the robot"
    words = tokenizer(exp, return_tensors='pt')['input_ids'].cuda(target_device)
    ref_masks = []
    ref_scores = []
    
    for i, jpg_file in enumerate(tqdm(video_list)):
        ref_mask, ref_score = evfsam.inference(imgs_sam[i].unsqueeze(0).cuda(target_device), imgs_beit[i].unsqueeze(0).cuda("cuda:3"), words, resize_shape, original_size_list)
        ref_mask = (ref_mask > 0).float()
        ref_masks.append(ref_mask)

        w1, w2 = 0.5, 0.5
        clip_text = alphaclip.tokenize([exp]).cuda("cuda:3")
        alpha = clip_preprocess_mask(ref_mask).cuda("cuda:3")
        image_features = clip.visual(imgs_clip[i].unsqueeze(0).cuda("cuda:3"), alpha.unsqueeze(0))
        text_features = clip.encode_text(clip_text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        ref_score = w1 * ref_score + w2 * torch.matmul(image_features, text_features.transpose(0, 1))[0]
        ref_scores.append(ref_score)

    best_ref_idx = torch.argmax(torch.stack(ref_scores, dim=0), dim=0)
    best_i = int(best_ref_idx) 

    # forward pass
    for i in tqdm(range(best_i, video_len)):
        if i == best_i:
            mask_prob = processor.step(imgs_cutie[i].cuda(), ref_masks[best_ref_idx].squeeze(0), objects=[1])
        else:
            mask_prob = processor.step(imgs_cutie[i].cuda())
        mask = processor.output_prob_to_mask(mask_prob).float()

        # clear memory for each sequence
        if i == video_len - 1:
            processor.clear_memory()

        # convert format
        mask = mask.detach().cpu().numpy().astype(np.float32)
        mask = Image.fromarray(mask * 255).convert('L')
        save_file = os.path.join(save_path_prefix, video_list[i])
        mask.save(save_file)

    # backward pass
    for i in range(best_i, -1, -1):
        if i == best_i:
            mask_prob = processor.step(imgs_cutie[i].cuda(), ref_masks[best_ref_idx].squeeze(0), objects=[1])
        else:
            mask_prob = processor.step(imgs_cutie[i].cuda())
        mask = processor.output_prob_to_mask(mask_prob).float()

        # clear memory for each sequence
        if i == 0:
            processor.clear_memory()

        # convert format
        mask = mask.detach().cpu().numpy().astype(np.float32)
        mask = Image.fromarray(mask * 255).convert('L')
        save_file = os.path.join(save_path_prefix, video_list[i])
        mask.save(save_file)

    create_gif(save_path_prefix, os.path.join(save_path_prefix, "segmentation_animation.gif"))
    # mask_sequence = []

    # # backward pass
    # for i in tqdm(range(best_i, -1, -1)):
    #     if i == best_i:
    #         mask_prob = processor.step(imgs_cutie[i].cuda("cuda:3"), ref_mask.squeeze(0), objects=[1])
    #     else:
    #         mask_prob = processor.step(imgs_cutie[i].cuda("cuda:3"))
    #     # mask = processor.output_prob_to_mask(mask_prob).float()
    #     mask = processor.output_prob_to_mask(mask_prob).float().cpu().numpy()
    #     mask = (mask * 255).astype(np.uint8) # Convert to 0-255 grayscale
    #     mask_sequence.append(mask)

    #     # clear memory for each sequence
    #     if i == 0:
    #         processor.clear_memory()

    #     # convert format
    #     # mask = mask.detach().cpu().numpy().astype(np.float32)
    #     # mask = Image.fromarray(mask * 255).convert('L')
    #     # save_file = os.path.join(save_path_prefix, video_list[i])
    #     # mask.save(save_file)
    
    # # forward pass
    # for i in tqdm(range(best_i, video_len)):
    #     if i == best_i:
    #         mask_prob = processor.step(imgs_cutie[i].cuda("cuda:3"), ref_mask.squeeze(0), objects=[1])
    #     else:
    #         mask_prob = processor.step(imgs_cutie[i].cuda("cuda:3"))
    #     # mask = processor.output_prob_to_mask(mask_prob).float()
    #     mask = processor.output_prob_to_mask(mask_prob).float().cpu().numpy()
    #     mask = (mask * 255).astype(np.uint8) # Convert to 0-255 grayscale
    #     mask_sequence.append(mask)

    #     # clear memory for each sequence
    #     if i == video_len - 1:
    #         processor.clear_memory()
        
    #     # # convert format
    #     # mask = mask.detach().cpu().numpy().astype(np.float32)
    #     # mask = Image.fromarray(mask * 255).convert('L')
    #     # save_file = os.path.join(save_path_prefix, video_list[i])
    #     # mask.save(save_file)
    
    # output_gif_path = os.path.join(save_path_prefix, "segmentation_animation.gif")
    # frame_duration = 150
    # loop_count = 0
    # create_gif_from_masks(mask_sequence, output_gif_path, duration=frame_duration, loop=loop_count)

if __name__ == '__main__':
    torch.cuda.set_device(3)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        save_path_prefix = 'outputs/Annotations'

        img_folder = "/scratch3/kat049/user_studies/vids/p14_fr/fps_1/frames"
        jpg_files = [f for f in os.listdir(img_folder) if f.lower().endswith(".jpg")]
        sorted_jpg_files = sorted(jpg_files, key=get_frame_number)
        video_list = sorted_jpg_files[200:500]

        create_subplot_gif(video_list, save_path_prefix, os.path.join(save_path_prefix, "segmentation_animation.gif"))
        # create_gif(save_path_prefix, os.path.join(save_path_prefix, "segmentation_animation.gif"))
        # test()
