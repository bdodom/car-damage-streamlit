import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn

# Define the class names ( Streamlit Script)
class_names = ['Damaged', 'Whole']

# Moddel architecture and weights
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: damage, whole
model.load_state_dict(torch.load("car_damage_model.pth", map_location=torch.device('cpu')))
model.eval()

# Image preprocessing steps
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

st.title("Car Damage Classifier")

uploaded_file = st.file_uploader("Choose an image of a car...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded image", use_column_width=True)
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)
        pred_class = class_names[pred.item()]
        st.write(f"Prediction: **{pred_class}**")
