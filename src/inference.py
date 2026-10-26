"""End-to-end inference for a brand-new image: raw image -> caption.

Everything in notebooks/02_features.ipynb and 03_train.ipynb ran against
the fixed set of 8091 Flickr8k images with pre-cached features. A deployed
app has to do the CNN forward pass live, for an image it's never seen,
every time someone uploads one -- that's the new logic this module adds.
"""

import io
import pickle
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

from .attention_model import AttentionDecoderLSTM, beam_search_caption_attention
from .vocab import PAD, SOS, EOS, UNK, Vocab  # noqa: F401 -- Vocab needed for unpickling below


class _RenameUnpickler(pickle.Unpickler):
    """vocab.pkl was pickled from a `Vocab` class defined in a notebook's
    __main__ namespace. This process has no __main__.Vocab -- it has
    src.vocab.Vocab instead. Redirect the class lookup so the existing
    pickle loads without needing to be regenerated from the notebook.
    """

    def find_class(self, module, name):
        if name == "Vocab":
            return Vocab
        return super().find_class(module, name)

## using this model caption-gen\artifacts\best_model_attention.pt

class CaptionPredictor:
    """Loads the trained checkpoint + vocab once, then serves predictions.

    Construct one instance at server startup (FastAPI/Gradio) and reuse it
    for every request -- re-loading the CNN and LSTM weights per-request
    would make every caption take several extra seconds for no reason.
    """

    def __init__(self, artifacts_dir: Path, device: str | None = None):
        artifacts_dir = Path(artifacts_dir)
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # --- vocab ---
        with open(artifacts_dir / "vocab.pkl", "rb") as f:
            self.vocab = _RenameUnpickler(f).load()

        # --- frozen CNN encoder (same setup as notebooks/02_features.ipynb) ---
        weights = ResNet50_Weights.DEFAULT
        resnet = resnet50(weights=weights)
        # Drop avgpool + fc: keep the (2048, 7, 7) layer4 grid, as in
        # notebooks/04_features_attention.ipynb.
        cnn = torch.nn.Sequential(*list(resnet.children())[:-2])
        cnn = cnn.to(self.device)
        cnn.eval()  # inference mode -- fixed BatchNorm stats, not batch-dependent
        self.cnn = cnn
        self.transform = weights.transforms()

        # --- trained decoder ---
        model = AttentionDecoderLSTM(
            feature_dim=2048,
            embed_dim=256,
            hidden_dim=512,
            vocab_size=len(self.vocab.itos),
            attn_dim=256,
        )
        state_dict = torch.load(artifacts_dir / "best_model_attention.pt", map_location=self.device)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        self.model = model

    @torch.no_grad()
    def _extract_feature(self, image: Image.Image) -> torch.Tensor:  # (49, 2048)
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)  # (1, 3, H, W)
        grid = self.cnn(tensor)  # (1, 2048, 7, 7)
        feature = grid.flatten(2).permute(0, 2, 1).squeeze(0).float().cpu()  # (49, 2048)
        return feature

    def predict(self, image: Image.Image, max_len: int = 25, beam_width: int = 8) -> str:
        """image: a PIL Image (already opened, any mode/size). Returns the
        generated caption as a plain string.

        Uses beam search (default width 8) instead of greedy decoding --
        an eval sweep across several test images showed width 8 was the
        first to escape repetition loops (e.g. "a black dog ... a black
        dog") that greedy and narrower beams (3, 5) still fell into. Width
        3/5 sometimes matched greedy exactly when the model was highly
        confident, so 8 was chosen as the best-observed trade-off between
        search thoroughness and per-request compute cost.
        """
        feature = self._extract_feature(image)
        caption = beam_search_caption_attention(
            feature, self.model, self.vocab, beam_width=beam_width, max_len=max_len, device=self.device
        )
        return caption

    def predict_from_bytes(self, image_bytes: bytes, max_len: int = 25) -> str:
        image = Image.open(io.BytesIO(image_bytes))
        return self.predict(image, max_len=max_len)
