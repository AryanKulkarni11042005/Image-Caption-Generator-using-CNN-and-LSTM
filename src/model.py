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


@torch.no_grad()
def beam_search_caption(feature, model, vocab, beam_width=3, max_len=25, device=None):
    """Beam search decoding: keep the top `beam_width` candidate sequences
    alive at every step, instead of committing to a single best word like
    generate_caption() does.

    Why this needs more bookkeeping than greedy decoding: each candidate
    sequence has taken a DIFFERENT path through the LSTM, so each one has
    its own (h, c) hidden state -- we can't share a single (h, c) across
    beams the way generate_caption() has only one to track. A "beam" here
    is (token_ids, cumulative_log_prob, h, c, finished).

    Scores are summed LOG-probabilities, not multiplied raw probabilities:
    multiplying many numbers < 1 shrinks toward zero and underflows for
    longer sequences, and it makes beams of different lengths incomparable
    on a fair footing. Summing logs avoids the underflow and turns the
    comparison into simple addition.
    """
    from .vocab import SOS, EOS

    model.eval()
    if device is None:
        device = next(model.parameters()).device

    feature = feature.unsqueeze(0).to(device)  # (1, 2048)

    h0 = model.init_h(feature).unsqueeze(0)  # (1, 1, hidden_dim)
    c0 = model.init_c(feature).unsqueeze(0)

    # Each beam: (token_ids: list[int], log_prob_sum: float, h, c, finished: bool)
    beams = [([vocab.stoi[SOS]], 0.0, h0, c0, False)]

    for _ in range(max_len):
        all_candidates = []

        for token_ids, log_prob_sum, h, c, finished in beams:
            if finished:
                # Already hit <eos> -- carry this beam forward unchanged
                # instead of letting it keep generating past the end.
                all_candidates.append((token_ids, log_prob_sum, h, c, finished))
                continue

            last_token = torch.tensor([[token_ids[-1]]], device=device)
            embedded = model.embedding(last_token)
            lstm_out, (new_h, new_c) = model.lstm(embedded, (h, c))
            logits = model.fc_out(lstm_out.squeeze(1))  # (1, vocab_size)

            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)  # (vocab_size,)
            top_log_probs, top_ids = log_probs.topk(beam_width)

            for lp, idx in zip(top_log_probs.tolist(), top_ids.tolist()):
                new_token_ids = token_ids + [idx]
                new_log_prob_sum = log_prob_sum + lp
                new_finished = (idx == vocab.stoi[EOS])
                all_candidates.append((new_token_ids, new_log_prob_sum, new_h, new_c, new_finished))

        # Keep only the top `beam_width` candidates overall (not per-beam) --
        # this is what lets a globally better sequence win even if it wasn't
        # the single best choice at an earlier individual step.
        all_candidates.sort(key=lambda cand: cand[1], reverse=True)
        beams = all_candidates[:beam_width]

        if all(finished for _, _, _, _, finished in beams):
            break

    # Best beam = highest cumulative log-prob (beams is already sorted).
    best_token_ids, _, _, _, _ = beams[0]

    # Strip <sos> and any trailing <eos> before decoding.
    ids_to_decode = best_token_ids[1:]
    if ids_to_decode and ids_to_decode[-1] == vocab.stoi[EOS]:
        ids_to_decode = ids_to_decode[:-1]

    return vocab.decode(ids_to_decode)
