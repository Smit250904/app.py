import streamlit as st
from PIL import Image, ImageDraw
import torch
import torchvision
from torchvision import transforms
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration

@st.cache_resource
def load_models():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    cap_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    seg_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    cap_model.eval()
    seg_model.eval()
    return processor, cap_model, seg_model

processor, cap_model, seg_model = load_models()

st.title("🧠 Image Captioning & Segmentation")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ===== Captioning =====
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        out = cap_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    st.subheader("📝 Caption")
    st.write(caption)

    # ===== Segmentation =====
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        prediction = seg_model(img_tensor)

    np_image = np.array(image)
    overlay = Image.fromarray(np_image).convert("RGBA")
    draw = ImageDraw.Draw(overlay)

    for i in range(min(3, len(prediction[0]['masks']))):
        score = prediction[0]['scores'][i].item()
        if score > 0.5:
            mask = prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy()
            box = prediction[0]['boxes'][i].cpu().numpy().astype(int)
            color = (255, 0, 0, 80)  # Red with transparency
            mask_image = Image.fromarray(mask).resize(image.size).convert("L")
            overlay.paste(Image.new("RGBA", image.size, color), (0, 0), mask_image)
            draw.rectangle(box.tolist(), outline="green", width=3)

    st.subheader("🎯 Segmented Output")
    st.image(overlay, use_column_width=True)
