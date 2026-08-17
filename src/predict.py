"""
Soil Classification Prediction Engine & CLI Interface
Author: Gogul Gupta (Team Shree Yantra Dynamics)
"""

import os
import sys
import argparse
from typing import Union, Dict, Any, List
from PIL import Image
import torch
import torch.nn.functional as F
import pandas as pd

# Add parent directory to path if run as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import load_model, CLASS_NAMES, SOIL_METADATA
from src.dataset import get_transforms


def predict_single_image(
    image_input: Union[str, Image.Image],
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    transform: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Runs inference on a single image and returns detailed prediction analytics.
    """
    if model is None:
        model, device = load_model()
    elif device is None:
        device = next(model.parameters()).device

    if transform is None:
        transform = get_transforms(is_train=False)

    # Load PIL image if a path is provided
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(probabilities.argmax())
        confidence = float(probabilities[pred_idx])

    pred_class = CLASS_NAMES[pred_idx]
    prob_dict = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}

    return {
        "class": pred_class,
        "confidence": confidence,
        "confidence_pct": f"{confidence * 100:.2f}%",
        "probabilities": prob_dict,
        "metadata": SOIL_METADATA.get(pred_class, {})
    }


def predict_directory(
    image_dir: str,
    output_csv: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None
) -> pd.DataFrame:
    """
    Runs batch inference on all images in a folder and optionally saves a CSV.
    """
    if model is None:
        model, device = load_model()
    elif device is None:
        device = next(model.parameters()).device

    transform = get_transforms(is_train=False)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in valid_exts]

    results: List[Dict[str, Any]] = []
    for fname in image_files:
        fpath = os.path.join(image_dir, fname)
        try:
            res = predict_single_image(fpath, model=model, device=device, transform=transform)
            results.append({
                "image_id": fname,
                "soil_type": res["class"],
                "confidence": res["confidence"]
            })
        except Exception as e:
            print(f"Error processing {fname}: {e}")

    df = pd.DataFrame(results)
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"✅ Batch predictions saved to {output_csv}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Soil Classification Predictor CLI")
    parser.add_argument("--image", type=str, help="Path to a single soil image")
    parser.add_argument("--batch-dir", type=str, help="Path to directory of images for batch inference")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output CSV path for batch inference")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path")

    args = parser.parse_args()

    if not args.image and not args.batch_dir:
        print("Please provide either --image or --batch-dir. Run with -h for help.")
        sys.exit(1)

    model, device = load_model(checkpoint_path=args.checkpoint)

    if args.image:
        if not os.path.isfile(args.image):
            print(f"Error: File not found: {args.image}")
            sys.exit(1)
        res = predict_single_image(args.image, model=model, device=device)
        print("\n" + "=" * 50)
        print(f"🌱 PREDICTION RESULT: {res['class']}")
        print(f"🎯 Confidence: {res['confidence_pct']}")
        print("-" * 50)
        print("📊 Probability Distribution:")
        for cls_name, prob in res["probabilities"].items():
            print(f"  • {cls_name:10s}: {prob * 100:6.2f}%")
        print("-" * 50)
        meta = res["metadata"]
        print(f"📋 Typical pH Range: {meta.get('ph_range')}")
        print(f"💧 Water Retention:  {meta.get('water_retention')}")
        print(f"🌾 Suitable Crops:   {', '.join(meta.get('suitable_crops', []))}")
        print("=" * 50 + "\n")

    if args.batch_dir:
        if not os.path.isdir(args.batch_dir):
            print(f"Error: Directory not found: {args.batch_dir}")
            sys.exit(1)
        predict_directory(args.batch_dir, output_csv=args.output, model=model, device=device)


if __name__ == "__main__":
    main()
