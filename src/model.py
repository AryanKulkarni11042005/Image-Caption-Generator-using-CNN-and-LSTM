"""CNN+LSTM caption model architecture, extracted from notebooks/03_train.ipynb.

The CNN encoder (frozen ResNet-50) lives in inference.py, since it's only
needed at inference/feature-caching time, not for the trainable model itself.
"""

import torch
import torch.nn as nn


class DecoderLSTM(nn.Module):
    def __init__(self, feature_dim, embed_dim, hidden_dim, vocab_size, num_layers=1):
        super().__init__()
        self.init_h = nn.Linear(feature_dim, hidden_dim)   # image features -> starting hidden state
        self.init_c = nn.Linear(feature_dim, hidden_dim)   # image features -> starting cell state

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        h0 = self.init_h(features).unsqueeze(0)   # (1, batch, hidden_dim)
        c0 = self.init_c(features).unsqueeze(0)

        embedded = self.embedding(captions)         # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embedded, (h0, c0))  # (batch, seq_len, hidden_dim)

        logits = self.fc_out(lstm_out)               # (batch, seq_len, vocab_size)
        return logits


@torch.no_grad()
def generate_caption(feature, model, vocab, max_len=25, device=None):
    """Greedy decoding: generate a caption for a single image feature vector.

    feature: a (2048,) tensor of cached/extracted CNN features for one image.
    Feeds the model's own previous prediction back in as the next input
    (no teacher forcing) -- this is why it's a manual step-by-step loop
    rather than a single forward() call.
    """
    from .vocab import SOS, EOS  # local import to avoid a circular import at module load time

    model.eval()
    if device is None:
        device = next(model.parameters()).device

    feature = feature.unsqueeze(0).to(device)  # (1, 2048) -- batch of one

    h = model.init_h(feature).unsqueeze(0)  # (1, 1, hidden_dim)
    c = model.init_c(feature).unsqueeze(0)

    current_token = torch.tensor([[vocab.stoi[SOS]]], device=device)
    generated_ids = []

    for _ in range(max_len):
        embedded = model.embedding(current_token)
        lstm_out, (h, c) = model.lstm(embedded, (h, c))
        logits = model.fc_out(lstm_out.squeeze(1))

        next_id = logits.argmax(dim=-1).item()

        if next_id == vocab.stoi[EOS]:
            break

        generated_ids.append(next_id)
        current_token = torch.tensor([[next_id]], device=device)

    return vocab.decode(generated_ids)
