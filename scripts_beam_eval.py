"""Standalone beam-width comparison script -- no notebook needed.

Run with:  uv run python scripts_beam_eval.py

Loads the trained checkpoint (same as app.py) and compares greedy decoding
against beam search at widths 3, 5, and 8 on a handful of test-set images,
including the repetition-loop dog image from the original debugging session.
"""

import pickle
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import DecoderLSTM, generate_caption, beam_search_caption  # noqa: E402
from src.vocab import Vocab  # noqa: E402, F401 -- needed for unpickling

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    with open(ARTIFACTS_DIR / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    with open(ARTIFACTS_DIR / "features.pkl", "rb") as f:
        feat_data = pickle.load(f)
    filenames_list = feat_data["filenames"]
    feature_matrix = feat_data["matrix"]
    fname_to_idx = {f: i for i, f in enumerate(filenames_list)}

    model = DecoderLSTM(feature_dim=2048, embed_dim=256, hidden_dim=512, vocab_size=len(vocab.itos))
    state_dict = torch.load(ARTIFACTS_DIR / "best_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    test_df = pd.read_csv(ARTIFACTS_DIR / "test.csv")

    # The repetition-loop dog image from last night, plus a few other
    # test-set images for a broader (not just cherry-picked) comparison.
    target_images = [
        "3259002340_707ce96858.jpg",  # the repetition-loop case
        test_df.iloc[0]["image"],
        test_df.iloc[5]["image"] if len(test_df) > 5 else test_df.iloc[1]["image"],
        test_df.iloc[10]["image"] if len(test_df) > 10 else test_df.iloc[2]["image"],
    ]
    # de-dupe while preserving order
    seen = set()
    target_images = [f for f in target_images if not (f in seen or seen.add(f))]

    for fname in target_images:
        if fname not in fname_to_idx:
            print(f"[skip] {fname} not in cached features")
            continue

        feat_idx = fname_to_idx[fname]
        feature = feature_matrix[feat_idx].float()

        refs = test_df[test_df["image"] == fname]["caption"].tolist()
        ground_truth = refs[0] if refs else "(not in test split)"

        print("=" * 70)
        print(f"Image: {fname}")
        print(f"Ground truth: {ground_truth}")
        print("-" * 70)

        greedy = generate_caption(feature, model, vocab, device=device)
        print(f"Greedy:  {greedy}")

        for width in (3, 5, 8):
            beam = beam_search_caption(feature, model, vocab, beam_width=width, device=device)
            print(f"Beam-{width}:  {beam}")

        print()


if __name__ == "__main__":
    main()
