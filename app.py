import streamlit as st
import os
import requests
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn

# --- Google Drive Download Helper ---
def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# --- Download model if not present ---
MODEL_PATH = "car_damage_model.pth"
GOOGLE_DRIVE_FILE_ID = "1fhFAc0cea3CIvr3dFi0NnZWqkyh_pgM_"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model file from Google Drive..."):
        download_file_from_google_drive(GOOGLE_DRIVE_FILE_ID, MODEL_PATH)
        st.success("Model downloaded successfully!")

# --- Load Model ---
class_names = ['damage', 'whole']
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# --- Preprocessing & UI ---
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
