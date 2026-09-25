"""CNN+LSTM caption model WITH Bahdanau-style additive attention.

Based on the "Show, Attend and Tell" architecture (Xu et al., 2015). Unlike
DecoderLSTM in model.py, which collapses an image into a single 2048-d
vector once, this decoder receives a GRID of region vectors (49 regions x
2048-d each, from notebooks/04_features_attention.ipynb) and recomputes a
fresh weighted focus over those 49 regions at every single decoding step.

Kept as a separate file from model.py so the original flat-feature
baseline stays untouched and directly comparable (same test images, same
BLEU eval) against this attention version.
"""

import torch
import torch.nn as nn


class Attention(nn.Module):
    """Computes, for one decoding step, a weighted blend of all image
    regions -- "how much should each of the 49 regions matter right now,
    given what the decoder is about to say next."

    This is "additive" (Bahdanau-style) attention: the region features and
    the current hidden state are each projected into a shared space, added
    together, passed through a nonlinearity, then reduced to a single
    score per region. Contrast with "dot-product" attention, which would
    just take a raw dot product between the two -- additive attention's
    extra learned projections make it more expressive at the cost of a
    few more parameters.
    """

    def __init__(self, feature_dim, hidden_dim, attn_dim):
        super().__init__()
        self.feature_proj = nn.Linear(feature_dim, attn_dim)   # projects each region vector
        self.hidden_proj = nn.Linear(hidden_dim, attn_dim)     # projects the current hidden state
        self.score_proj = nn.Linear(attn_dim, 1)               # reduces to one score per region

    def forward(self, features, hidden):
        """
        features: (batch, num_regions, feature_dim)  -- e.g. (batch, 49, 2048)
        hidden:   (batch, hidden_dim)                 -- current LSTM hidden state

        Returns:
            context: (batch, feature_dim) -- the attention-weighted blend
                      of all regions for this one decoding step
            alpha:   (batch, num_regions) -- the attention weights
                      themselves (sum to 1 per image); useful later for
                      visualizing "where the model looked" per word
        """
        # Project regions and hidden state into the same attn_dim space,
        # then broadcast-add: hidden's projection is the same for every
        # region within one step, so unsqueeze(1) lets it add elementwise
        # against all 49 region projections at once.
        proj_features = self.feature_proj(features)              # (batch, num_regions, attn_dim)
        proj_hidden = self.hidden_proj(hidden).unsqueeze(1)       # (batch, 1, attn_dim)

        scores = self.score_proj(torch.tanh(proj_features + proj_hidden))  # (batch, num_regions, 1)
        scores = scores.squeeze(2)                                          # (batch, num_regions)

        alpha = torch.softmax(scores, dim=1)  # (batch, num_regions) -- weights sum to 1 per image

        # Weighted sum of region vectors: each region's feature vector,
        # scaled by how much attention it got, summed together into one
        # (batch, feature_dim) context vector for this step.
        context = (features * alpha.unsqueeze(2)).sum(dim=1)  # (batch, feature_dim)

        return context, alpha


class AttentionDecoderLSTM(nn.Module):
    """Like DecoderLSTM, but the LSTM's input at every step is the word
    embedding CONCATENATED WITH a fresh attention context vector, instead
    of relying on a single fixed image vector set once at the start.

    The initial (h, c) are still seeded from the image -- here, from the
    mean of all 49 region vectors, since we need *some* starting point
    before the first attention step can even run.
    """

    def __init__(self, feature_dim, embed_dim, hidden_dim, vocab_size, attn_dim=256, num_layers=1):
        super().__init__()
        self.attention = Attention(feature_dim, hidden_dim, attn_dim)

        self.init_h = nn.Linear(feature_dim, hidden_dim)
        self.init_c = nn.Linear(feature_dim, hidden_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # LSTMCell (not nn.LSTM) -- we need to run ONE step at a time,
        # manually, because attention must be recomputed between every
        # step using that step's hidden state. nn.LSTM processes a whole
        # sequence in one call and can't pause mid-sequence to let
        # attention look at the hidden state in between.
        self.lstm_cell = nn.LSTMCell(embed_dim + feature_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        """
        features: (batch, num_regions, feature_dim)
        captions: (batch, seq_len) -- token ids, teacher-forced (ground truth)

        Returns: logits (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = captions.shape

        mean_features = features.mean(dim=1)  # (batch, feature_dim) -- for seeding (h, c) only
        h = self.init_h(mean_features)
        c = self.init_c(mean_features)

        embedded = self.embedding(captions)  # (batch, seq_len, embed_dim)

        all_logits = []
        for t in range(seq_len):
            context, _alpha = self.attention(features, h)         # (batch, feature_dim)
            lstm_input = torch.cat([embedded[:, t, :], context], dim=1)  # (batch, embed_dim + feature_dim)
            h, c = self.lstm_cell(lstm_input, (h, c))
            logits_t = self.fc_out(h)  # (batch, vocab_size)
            all_logits.append(logits_t)

        return torch.stack(all_logits, dim=1)  # (batch, seq_len, vocab_size)


@torch.no_grad()
def generate_caption_attention(features, model, vocab, max_len=25, device=None):
    """Greedy decoding for the attention model. Same feed-your-own-prediction-
    back-in loop as generate_caption() in model.py, but now recomputing
    attention (and returning per-step alpha weights) at every step.
    """
    from .vocab import SOS, EOS

    model.eval()
    if device is None:
        device = next(model.parameters()).device

    features = features.unsqueeze(0).to(device)  # (1, num_regions, feature_dim)

    mean_features = features.mean(dim=1)
    h = model.init_h(mean_features)
    c = model.init_c(mean_features)

    current_token = torch.tensor([vocab.stoi[SOS]], device=device)  # (1,)
    generated_ids = []
    alphas = []

    for _ in range(max_len):
        embedded = model.embedding(current_token)  # (1, embed_dim)
        context, alpha = model.attention(features, h)
        lstm_input = torch.cat([embedded, context], dim=1)
        h, c = model.lstm_cell(lstm_input, (h, c))
        logits = model.fc_out(h)  # (1, vocab_size)

        next_id = logits.argmax(dim=-1).item()
        alphas.append(alpha.squeeze(0).cpu())

        if next_id == vocab.stoi[EOS]:
            break

        generated_ids.append(next_id)
        current_token = torch.tensor([next_id], device=device)

    return vocab.decode(generated_ids), alphas


@torch.no_grad()
def beam_search_caption_attention(features, model, vocab, beam_width=8, max_len=25, device=None):
    """Beam search decoding for the attention model -- same top-k-candidates-
    kept-alive idea as beam_search_caption() in model.py, adapted for
    LSTMCell + per-step attention instead of nn.LSTM.

    Each beam now carries (token_ids, log_prob_sum, h, c, finished) same as
    before -- attention needs no extra per-beam state beyond (h, c), since
    it's recomputed fresh from `features` (shared, read-only across all
    beams) and each beam's own h at every step.
    """
    from .vocab import SOS, EOS

    model.eval()
    if device is None:
        device = next(model.parameters()).device

    features = features.unsqueeze(0).to(device)  # (1, num_regions, feature_dim)

    mean_features = features.mean(dim=1)
    h0 = model.init_h(mean_features)  # (1, hidden_dim)
    c0 = model.init_c(mean_features)

    beams = [([vocab.stoi[SOS]], 0.0, h0, c0, False)]

    for _ in range(max_len):
        all_candidates = []

        for token_ids, log_prob_sum, h, c, finished in beams:
            if finished:
                all_candidates.append((token_ids, log_prob_sum, h, c, finished))
                continue

            last_token = torch.tensor([token_ids[-1]], device=device)  # (1,)
            embedded = model.embedding(last_token)                      # (1, embed_dim)
            context, _alpha = model.attention(features, h)              # (1, feature_dim)
            lstm_input = torch.cat([embedded, context], dim=1)
            new_h, new_c = model.lstm_cell(lstm_input, (h, c))
            logits = model.fc_out(new_h)  # (1, vocab_size)

            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
            top_log_probs, top_ids = log_probs.topk(beam_width)

            for lp, idx in zip(top_log_probs.tolist(), top_ids.tolist()):
                new_token_ids = token_ids + [idx]
                new_log_prob_sum = log_prob_sum + lp
                new_finished = (idx == vocab.stoi[EOS])
                all_candidates.append((new_token_ids, new_log_prob_sum, new_h, new_c, new_finished))

        all_candidates.sort(key=lambda cand: cand[1], reverse=True)
        beams = all_candidates[:beam_width]

        if all(finished for _, _, _, _, finished in beams):
            break

    best_token_ids, _, _, _, _ = beams[0]
    ids_to_decode = best_token_ids[1:]
    if ids_to_decode and ids_to_decode[-1] == vocab.stoi[EOS]:
        ids_to_decode = ids_to_decode[:-1]

    return vocab.decode(ids_to_decode)
