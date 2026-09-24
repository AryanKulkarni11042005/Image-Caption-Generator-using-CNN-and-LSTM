"""Gradio demo: upload an image, get a caption.

Local run:   uv run python app.py            -> http://127.0.0.1:7860
In Docker:   this file also runs as the container's entrypoint, in which
case it must bind to 0.0.0.0 (not 127.0.0.1) so the port is reachable
from outside the container via `docker run -p`.
"""

import os
from pathlib import Path

import gradio as gr

from src.inference import CaptionPredictor

PROJECT_ROOT = Path(__file__).parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

print("Loading model and CNN encoder (this can take a few seconds)...")
predictor = CaptionPredictor(ARTIFACTS_DIR)
print(f"Loaded. Running on device: {predictor.device}")


def caption_image(image):
    if image is None:
        return "Please upload an image."
    return predictor.predict(image)


demo = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Textbox(label="Generated caption"),
    title="Image Caption Generator (CNN + LSTM)",
    description=(
        "A ResNet-50 encoder (frozen, ImageNet-pretrained) feeds an LSTM decoder "
        "trained from scratch on Flickr8k. Upload any image to generate a caption."
    ),
    examples=None,
)

if __name__ == "__main__":
    # IN_DOCKER is set in the Dockerfile (see ENV below); locally it's unset,
    # so this still binds to 127.0.0.1 for the exact same dev experience as before.
    in_docker = os.environ.get("IN_DOCKER") == "1"
    demo.launch(
        share=False,
        server_name="0.0.0.0" if in_docker else "127.0.0.1",
        server_port=7860,
    )
