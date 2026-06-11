import gradio as gr
import torch
from torchvision import transforms
from torchvision.models import resnet18
from torch import nn
from PIL import Image, ImageOps
import numpy as np

# Load model
model = resnet18(weights=None)
model.fc = nn.Linear(512, 10)

model.load_state_dict(
    torch.load("mnist_model.pth", map_location="cpu")
)

model.eval()

transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

def predict(img):

    if img is None:
        return None

    if isinstance(img, dict):

        if "composite" in img:
            img = img["composite"]

        elif "background" in img:
            img = img["background"]

    if isinstance(img, np.ndarray):
        img = Image.fromarray(
            img.astype("uint8")
        )

    if not isinstance(img, Image.Image):
        img = Image.fromarray(
            np.array(img)
        )

    img = img.convert("L")
    img = ImageOps.invert(img)
    img = ImageOps.autocontrast(img)

    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(
            outputs,
            dim=1
        )[0]

    return {
        str(i): float(probs[i])
        for i in range(10)
    }

with gr.Blocks() as demo:

    gr.Markdown("# MNIST Digit Classifier")
    gr.Markdown(
        "Upload an image, use your webcam, or draw a digit."
    )

    with gr.Tab("Upload / Webcam"):

        img_input = gr.Image(
            type="pil",
            sources=["upload", "webcam"]
        )

        gr.Examples(
            examples=[
                ["examples/0.png"],
                ["examples/1.png"],
                ["examples/2.png"],
                ["examples/7.png"]
            ],
            inputs=img_input
        )

        upload_output = gr.Label(
            num_top_classes=10
        )

        predict_btn = gr.Button(
            "Predict"
        )

        predict_btn.click(
            predict,
            inputs=img_input,
            outputs=upload_output
        )

    with gr.Tab("Draw Digit"):

        draw_input = gr.Sketchpad(
            label="Draw a digit"
        )

        draw_output = gr.Label(
            num_top_classes=10
        )

        draw_btn = gr.Button(
            "Predict Drawing"
        )

        draw_btn.click(
            predict,
            inputs=draw_input,
            outputs=draw_output
        )

demo.launch()