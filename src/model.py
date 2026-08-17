"""
Soil Classification & Agronomic Intelligence Models
Author: Gogul Gupta (Team Shree Yantra Dynamics)
"""

import os
from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
from torchvision import models

# Multi-class target labels
CLASS_NAMES = ["Alluvial", "Black", "Clay", "Red"]

# Detailed agronomic intelligence database for each soil type
SOIL_METADATA: Dict[str, Dict[str, Any]] = {
    "Alluvial": {
        "title": "Alluvial Soil (Khadar / Bhangar)",
        "color": "#D4AF37",
        "description": "Formed by deposition of silt brought down by rivers. Highly fertile, rich in potash, phosphoric acid, and lime, but deficient in nitrogen.",
        "ph_range": "6.5 - 8.4 (Neutral to Slightly Alkaline)",
        "texture": "Loamy to Sandy Loam (Fine silt & clay mixture)",
        "water_retention": "Moderate to High",
        "suitable_crops": [
            "Wheat", "Paddy (Rice)", "Sugarcane", "Cotton", "Jute",
            "Oilseeds", "Maize", "Pulses", "Vegetables"
        ],
        "fertilizers": [
            "Urea / Nitrogenous fertilizers (compensate nitrogen deficit)",
            "DAP (Di-ammonium Phosphate)",
            "Organic compost & Biofertilizers"
        ],
        "farming_tips": (
            "Ensure proper crop rotation with leguminous plants to maintain soil nitrogen. "
            "Level fields to prevent surface water runoff and nutrient leaching."
        )
    },
    "Black": {
        "title": "Black Soil (Regur / Lava Soil)",
        "color": "#2C2C2C",
        "description": "Derived from basaltic rock/lava. Rich in calcium carbonate, magnesium, potash, and lime. Highly moisture-retentive and swells when wet.",
        "ph_range": "7.2 - 8.5 (Moderately Alkaline)",
        "texture": "Clayey, fine-grained, deep cracking when dry",
        "water_retention": "Very High (Holds moisture for long dry spells)",
        "suitable_crops": [
            "Cotton (Best)", "Soybean", "Sorghum (Jowar)", "Wheat",
            "Millet (Bajra)", "Groundnut", "Tobacco", "Sunflower", "Citrus Fruits"
        ],
        "fertilizers": [
            "Phosphatic fertilizers (SSP / TSP)",
            "Nitrogen fertilizers in split doses",
            "Zinc sulfate micronutrients"
        ],
        "farming_tips": (
            "Avoid tilling when excessively wet to prevent soil compaction. "
            "Implement deep plowing in summer to facilitate aeration and weed control."
        )
    },
    "Clay": {
        "title": "Clay Soil",
        "color": "#8D5B4C",
        "description": "Composed of ultra-fine mineral particles with minimal pore space. High nutrient-holding capacity but prone to waterlogging and compaction.",
        "ph_range": "6.0 - 7.5 (Slightly Acidic to Neutral)",
        "texture": "Dense, heavy, sticky when wet, hard when dry",
        "water_retention": "Extremely High (Poor internal drainage)",
        "suitable_crops": [
            "Paddy (Rice)", "Broccoli", "Cabbage", "Cauliflower",
            "Kale", "Wheat", "Beans", "Perennial Fruit Trees"
        ],
        "fertilizers": [
            "Gypsum (improves structure and reduces stickiness)",
            "Well-decomposed organic manure / vermicompost",
            "Slow-release NPK formulations"
        ],
        "farming_tips": (
            "Incorporate bulky organic matter (compost, peat, straw) to improve aeration and drainage. "
            "Construct raised beds to avoid root rot during heavy rainfall."
        )
    },
    "Red": {
        "title": "Red Soil (Iron-Rich)",
        "color": "#B22222",
        "description": "Formed by weathering of ancient crystalline and metamorphic rocks. Rich in iron oxides giving it a distinctive red color; low in nitrogen, phosphorus, and humus.",
        "ph_range": "5.5 - 6.8 (Slightly Acidic)",
        "texture": "Porous, friable, sandy to clayey-loam",
        "water_retention": "Low to Moderate (Requires frequent light irrigation)",
        "suitable_crops": [
            "Groundnut", "Millets (Ragi, Bajra)", "Pulses", "Potatoes",
            "Tobacco", "Oilseeds", "Cotton", "Tea / Coffee (in hill tracts)"
        ],
        "fertilizers": [
            "Single Super Phosphate (SSP) & Rock Phosphate",
            "Farmyard Manure (FYM) to boost organic matter",
            "Agricultural Lime if soil pH drops below 5.5"
        ],
        "farming_tips": (
            "Adopt drip or sprinkler irrigation to optimize water usage. "
            "Apply mulch to reduce soil moisture evaporation and prevent soil crusting."
        )
    }
}


def build_soil_classifier(num_classes: int = 4, pretrained: bool = False) -> nn.Module:
    """
    Constructs the ResNet-18 architecture with a custom classification head.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_binary_detector(pretrained: bool = False) -> nn.Module:
    """
    Constructs a binary classifier (Soil vs Non-Soil) with Sigmoid output.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 1),
        nn.Sigmoid()
    )
    return model


def find_model_checkpoint() -> Optional[str]:
    """
    Searches known locations for best_model.pth.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate_paths = [
        os.path.join(base_dir, "models", "best_model.pth"),
        "models/best_model.pth"
    ]
    for p in candidate_paths:
        if os.path.isfile(p):
            return p
    return None


def load_model(
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    num_classes: int = 4
) -> Tuple[nn.Module, torch.device]:
    """
    Loads the trained model weights and prepares it for inference.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_soil_classifier(num_classes=num_classes, pretrained=False)

    if checkpoint_path is None:
        checkpoint_path = find_model_checkpoint()

    if checkpoint_path and os.path.isfile(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        # If no checkpoint exists, initialize with ImageNet weights as graceful fallback
        fallback_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        fallback_model.fc = nn.Linear(fallback_model.fc.in_features, num_classes)
        model = fallback_model

    model = model.to(device)
    model.eval()
    return model, device
