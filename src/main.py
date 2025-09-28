import os
import argparse
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import cv2
from glob import glob

from model import MultiTaskShared
from utils.postprocess import keep_largest_cc

def predict_single(image_path: str, model: torch.nn.Module, device: torch.device):
    raw_image = Image.open(image_path).convert('RGB')
    img_tensor = TF.to_tensor(raw_image)
    img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    tta_batch = torch.stack([img_tensor, TF.hflip(img_tensor)])
    tta_batch = tta_batch.to(device)

    model.eval()
    with torch.no_grad():
        seg_logits, cls_logits = model(tta_batch)

    seg_logits[1, :, :, :] = TF.hflip(seg_logits[1, :, :, :])
    seg_probs = torch.sigmoid(seg_logits).mean(dim=0)
    cls_probs = F.softmax(cls_logits, dim=1).mean(dim=0)

    seg_mask_binary = (seg_probs > 0.5).squeeze().cpu().numpy().astype(np.uint8)
    seg_mask_resized = cv2.resize(seg_mask_binary, (raw_image.width, raw_image.height), interpolation=cv2.INTER_NEAREST)
    seg_mask_final = (keep_largest_cc(seg_mask_resized * 255)).astype(np.uint8)

    pred_label_idx = torch.argmax(cls_probs).item()
    labels_map = ['Normal', 'Benign', 'Malignant']
    pred_label_str = labels_map[pred_label_idx]

    return seg_mask_final, pred_label_str

def main(args):
    # For Apple Silicon, 'mps' is the correct accelerator.
    # The code will intelligently fall back to 'cpu' if 'mps' or 'cuda' is not available.
    if args.device == 'gpu' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    model = MultiTaskShared(num_classes=3)
    model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))
    model.to(device)
    model.eval()

    output_seg_dir = os.path.join(args.output, 'segmentation')
    output_cls_dir = os.path.join(args.output, 'classification')

    if args.task == 'seg': os.makedirs(output_seg_dir, exist_ok=True)
    elif args.task == 'cls': os.makedirs(output_cls_dir, exist_ok=True)

    search_pattern = os.path.join(args.input, '**', '*.png')
    image_paths = sorted(glob(search_pattern, recursive=True))
    
    print(f"Found {len(image_paths)} images to process.")
    
    if not image_paths:
        print(f"Warning: No '.png' images found in '{args.input}' or its subdirectories.")
        print("Inference complete.")
        return

    classification_results = []
    with torch.no_grad():
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            image_id = os.path.splitext(filename)[0]
            print(f"Processing {filename}...")
            try:
                seg_mask, cls_label = predict_single(img_path, model, device)
                if args.task == 'seg':
                    mask_path = os.path.join(output_seg_dir, f"{image_id}_mask.png")
                    Image.fromarray(seg_mask).save(mask_path)
                classification_results.append({'image_id': image_id, 'label': cls_label})
            except Exception as e:
                print(f"Failed to process {filename}. Error: {e}")

    if args.task == 'cls':
        csv_path = os.path.join(output_cls_dir, 'predictions.csv')
        pd.DataFrame(classification_results).to_csv(csv_path, index=False)
        print(f"Classification results saved to {csv_path}")

    print("Inference complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PACE 2025 Multi-task Inference")
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to input directory")
    parser.add_argument('-o', '--output', type=str, required=True, help="Path to output directory")
    parser.add_argument('-t', '--task', type=str, required=True, choices=['seg', 'cls'], help="Task: 'seg' or 'cls'")
    parser.add_argument('-d', '--device', type=str, default='gpu', choices=['cpu', 'gpu'], help="Device: 'cpu' or 'gpu'")
    args = parser.parse_args()
    main(args)