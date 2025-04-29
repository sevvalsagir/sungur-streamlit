
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import random
from torchvision import models

# Modeli tanımla
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = torch.nn.Linear(model.fc.in_features, 3)  # 3 sınıf için
model = model.to(device)

model.load_state_dict(torch.load("radar_resnet18_model.pth", map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Class labels
classes = sorted(os.listdir('data/mstar/'))

# Streamlit Interface
st.set_page_config(page_title="TEGIN AI Radar Target Classifier", layout="centered")
st.title("🎯 TEGIN AI Radar Target Classifier")

if st.button("🚀 Get New Radar Image"):
    selected_class = random.choice(classes)
    class_folder = os.path.join("data/mstar", selected_class)
    selected_image_path = os.path.join(class_folder, random.choice(os.listdir(class_folder)))

    image = Image.open(selected_image_path).convert('L')
    image_display = image.copy()
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, prediction = torch.max(output, 1)
        predicted_class = classes[prediction.item()]

    st.image(image_display, caption="Radar Image", width=400)
    st.markdown(f"**True Class:** `{selected_class}`")
    st.markdown(f"**TEGIN AI Prediction:** `{predicted_class}`")

    if selected_class == predicted_class:
        st.success("✅ Correct Prediction!")
    else:
        st.error("❌ Incorrect Prediction.")
