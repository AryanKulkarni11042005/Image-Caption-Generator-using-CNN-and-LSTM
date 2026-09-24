# Image Caption Generator (CNN + LSTM)

An image captioning model built with a frozen, ImageNet-pretrained ResNet-50 encoder and an LSTM decoder trained from scratch, implemented in PyTorch. Given an image, the model generates a natural-language caption describing its contents.

**Live demo:** https://renter-fifteen-lethargic.ngrok-free.dev/
*(This is a temporary tunnel and may go offline between sessions. For a reliable way to run the app, see Docker below.)*

**Docker image:** [`aryankulkarni11042005/caption-gen:latest`](https://hub.docker.com/r/aryankulkarni11042005/caption-gen)

## How it works

- **Encoder:** ResNet-50, pretrained on ImageNet, frozen (no fine-tuning). The final classification layer is removed, so each image is reduced to a 2048-dimensional feature vector.
- **Decoder:** An LSTM that takes the image feature vector as its initial hidden/cell state, then generates the caption one word at a time, feeding each predicted word back in as input for the next step (greedy decoding at inference time).
- **Training:** Teacher forcing on the true caption sequence, cross-entropy loss with padding masked out, gradient clipping, and checkpointing on best validation loss (to avoid saving an overfit model).
- **Data:** [Flickr8k](https://www.kaggle.com/datasets) — 8,091 images, 5 captions each, split by image (not by caption row) to prevent data leakage between train/val/test.

## Results (Flickr8k, 8k dataset)

| Metric | Score |
|---|---|
| BLEU-1 | 0.5693 |
| BLEU-4 | 0.1709 |

## Run it yourself

Pull and run the published Docker image — no setup beyond Docker required:

```bash
docker pull aryankulkarni11042005/caption-gen:latest
docker run -p 7860:7860 aryankulkarni11042005/caption-gen:latest
```

Then open `http://localhost:7860` and upload an image.

## Tech stack

PyTorch · torchvision (ResNet-50) · Gradio · Docker · uv

## Future work

- **Scale up to Flickr30k** (~31.8k images, ~159k captions) for more training data and better generalization.
- **Add an attention mechanism** to the decoder, so the model attends to specific image regions at each decoding step instead of relying on a single global feature vector — should meaningfully improve caption specificity and reduce repetition artifacts seen in the current greedy-decoded outputs.
- Beam search decoding as an alternative to greedy decoding.
- Permanent public hosting (currently only a temporary tunnel + local Docker image).
