# save this as app.py
import streamlit as st
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Load model
model = models.resnet18(pretrained=False, num_classes=4)
model.load_state_dict(torch.load(r"C:\Users\shali\OneDrive\Desktop\new\Soil_classification_project_annam-main\challenge-1\models\best_model.pth", map_location=torch.device("cpu")))
model.eval()

# Labels
classes = ["Alluvial", "Black", "Clay", "Red"]

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

st.title("🌱 Soil Classification App")

uploaded_file = st.file_uploader("Upload a soil image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Soil Image", use_column_width=True)

    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)

    st.success(f"Predicted Soil Type: **{classes[predicted.item()]}**")
