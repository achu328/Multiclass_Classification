import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fish classes
class_names = [
    'animal fish',
    'animal fish bass',
    'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream',
    'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet',
    'fish sea_food red_sea_bream',
    'fish sea_food sea_bass',
    'fish sea_food shrimp',
    'fish sea_food striped_red_mullet',
    'fish sea_food trout'
]

# Load the best saved model
@st.cache_resource
def load_model():
    model = torch.load(r"C:\Users\achu1\Documents\GUVI\Project -5\best_fish_model.pth", map_location=device, weights_only=False)
    model.eval()
    return model

model = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("Fish Species Classifier (Best Model)")

uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)

    st.markdown(f"### Predicted Species: `{class_names[pred.item()]}`")

    st.write("#### Confidence Scores:")
    for idx, prob in enumerate(probs[0]):
        st.write(f"- {class_names[idx]}: `{prob.item():.4f}`")

