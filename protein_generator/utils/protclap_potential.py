"""
ProtCLAP-based classifier–free guidance for sequence-space diffusion.
Copyright 2025 …

This module loads the pretrained ProteinCLAP encoders produced in
ProteinDT step 01 and exposes `compute_text_gradients()` which
returns ∂L/∂logits for a batch of sequence‑logits .

The class now also computes a **baseline CLAP score** for the provided
(actual_sequence, text_prompt) pair on construction and keeps it in
`self.base_cos_sim`.  During optimisation, the distance between the
current (predicted) sequence and the actual sequence is reported as
`seq_distance` (normalised edit distance).
"""
import os, sys, torch, torch.nn.functional as F
from transformers import AutoTokenizer, BertTokenizer, AutoModel, BertModel
from typing import Tuple

# ────────────────────────────────────────────────────────────────────────────────
# helpers
AMINO_ACIDS = list("ARNDCQEGHILKMFPSTWYV") + ["X"]   # length 21
IDX_FROM_AA = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def _load_clap_checkpoint(ckpt_dir: str, device: torch.device):
    """Return (protein_encoder, text_encoder, prot2lat, text2lat)."""
    prot = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
    txt  = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
    prot2lat = torch.nn.Linear(prot.config.hidden_size, 256)
    txt2lat  = torch.nn.Linear(txt.config.hidden_size, 256)

    prot.load_state_dict(torch.load(os.path.join(ckpt_dir, "protein_model.pth"),
                                    map_location="cpu"), strict=False)
    txt.load_state_dict(torch.load(os.path.join(ckpt_dir, "text_model.pth"),
                                   map_location="cpu"), strict=False)
    prot2lat.load_state_dict(torch.load(os.path.join(ckpt_dir,
                                                     "protein2latent_model.pth"),
                                        map_location="cpu"), strict=False)
    txt2lat.load_state_dict(torch.load(os.path.join(ckpt_dir,
                                                    "text2latent_model.pth"),
                                       map_location="cpu"), strict=False)
    for m in (prot, txt, prot2lat, txt2lat):
        m.eval().to(device)
        for p in m.parameters():   # inference only
            p.requires_grad_(False)
    return prot, txt, prot2lat, txt2lat


class ProtCLAPGuidance:
    """Light‑weight wrapper that keeps everything on device and
    pre‑computes a *baseline* CLAP similarity for the provided
    (sequence, text) pair.
    """

    # ────────────────────────────────────────────────────────────────────
    # construction helpers
    @staticmethod
    def _tokenise_protein(seq: str, tokenizer: BertTokenizer, device: torch.device):
        """Tokenise a raw AA string into Rostlab/protBERT input tensors."""
        seq = " ".join(list(seq.strip()))  # spaces between amino acids
        return tokenizer(seq, return_tensors="pt", truncation=True, max_length=512).to(device)

    @staticmethod
    def _seq_distance(seq1: str, seq2: str) -> float:
        """Normalised edit distance (substitutions + indels) / max_len."""
        # simple dynamic‑programming Levenshtein
        m, n = len(seq1), len(seq2)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n + 1):
                cur = dp[j]
                cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
                dp[j] = min(prev + cost,      # substitution
                             dp[j] + 1,       # deletion
                             dp[j - 1] + 1)   # insertion
                prev = cur
        return dp[n] / max(m, n)

    # ────────────────────────────────────────────────────────────────────
    def __init__(self, args, features, potential_scale: float, DEVICE: torch.device):
        self.DEVICE          = DEVICE
        self.potential_scale = potential_scale
        self.text_prompt     = args['text_prompt']
        self.actual_sequence = args['actual_sequence']
        self.T               = args['clap_temp']
        self.test_ablation   = args['test_ablation']

        # encoders & projection heads
        self.prot_enc, self.txt_enc, self.p2l, self.t2l = _load_clap_checkpoint(
            "D:\\CSCI2840Final\\ProteinDT\\checkpoints\\ProtBERT_BFD-512-1e-5-1-text-512-1e-5-1-EBM_NCE-1-batch-9-gpu-8-epoch-10",
            self.DEVICE)

        # tokenisers
        self.prot_tok = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
        self.txt_tok  = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

        # amino‑acid embedding look‑up (21×dim)
        aa_token_ids  = [self.prot_tok.convert_tokens_to_ids(aa) for aa in AMINO_ACIDS]
        emb           = self.prot_enc.get_input_embeddings().weight # vocab×dim
        self.aa_embed = emb[torch.tensor(aa_token_ids, device=self.DEVICE)] # 21×dim

        # pre‑compute text representation
        with torch.no_grad():
            txt_inp       = self.txt_tok(self.text_prompt, return_tensors="pt", truncation=True, max_length=512).to(self.DEVICE)
            txt_repr      = self.txt_enc(**txt_inp, return_dict=True).pooler_output
            self.text_lat = F.normalize(self.t2l(txt_repr), dim=-1) # 1×D

        # baseline CLAP similarity for (actual_sequence, text_prompt)
        with torch.no_grad():
            prot_inp  = self._tokenise_protein(self.actual_sequence, self.prot_tok, self.DEVICE)
            prot_repr = self.prot_enc(**prot_inp, return_dict=True).pooler_output
            seq_lat   = F.normalize(self.p2l(prot_repr), dim=-1)
            self.base_cos_sim = F.cosine_similarity(seq_lat, self.text_lat).item()
        print(f"[ProtCLAP] baseline cos-sim: {self.base_cos_sim:.4f}")

    # ────────────────────────────────────────────────────────────────────
    def get_gradients(self, seq_logits: torch.Tensor) -> Tuple[torch.Tensor, float]:
        L = seq_logits.size(0)
        logits = seq_logits.detach().clone().requires_grad_(True) # L×21
        probs  = F.softmax(logits / self.T, dim=-1)

        # differentiable embedding (straight‑through soft one‑hot)
        embeds = probs @ self.aa_embed.to(probs.dtype) # L×d
        attn   = torch.ones(1, L, dtype=torch.long, device=self.DEVICE)
        prot_repr = self.prot_enc(inputs_embeds=embeds.unsqueeze(0),
                                  attention_mask=attn, return_dict=True).pooler_output
        prot_lat  = F.normalize(self.p2l(prot_repr), dim=-1) # 1×D

        cos_sim = F.cosine_similarity(prot_lat, self.text_lat) # (1,)
        self.current_cos_sim = cos_sim.item()

        loss = -cos_sim.mean()
        loss.backward()
        self.gradients = logits.grad

        # ── sequence distance to ground truth
        with torch.no_grad():
            pred_idxs = logits.argmax(dim=-1).cpu().tolist() # List[int]
            pred_seq  = ''.join(AMINO_ACIDS[i] for i in pred_idxs)
            self.current_seq_distance = self._seq_distance(self.actual_sequence, pred_seq)

        print(f"[ProtCLAP] current cos-sim: {self.current_cos_sim:.4f}")
        print(f"[ProtCLAP] seq distance to actual: {self.current_seq_distance:.3f}\n")

        if self.test_ablation:
            return torch.zeros_like(logits)

        return -self.gradients*self.potential_scale



class TargetSequenceGuidance:
    def __init__(self, args, features, potential_scale: float, DEVICE: torch.device):
        self.DEVICE          = DEVICE
        self.potential_scale = potential_scale
        self.actual_sequence = args['actual_sequence']
        self.text_prompt     = args['text_prompt']
        self.T_clap          = args['clap_temp']

        self.target_idx = torch.tensor([IDX_FROM_AA[aa] for aa in self.actual_sequence], dtype=torch.long, device=self.DEVICE)

        ckpt = "D:/CSCI2840Final/ProteinDT/checkpoints/ProtBERT_BFD-512-1e-5-1-text-512-1e-5-1-EBM_NCE-1-batch-9-gpu-8-epoch-10"
        self.prot_enc, self.txt_enc, self.p2l, self.t2l = _load_clap_checkpoint(ckpt, self.DEVICE)
        self.prot_tok = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
        self.txt_tok  = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        aa_token_ids  = [self.prot_tok.convert_tokens_to_ids(aa) for aa in AMINO_ACIDS]
        self.aa_embed = self.prot_enc.get_input_embeddings().weight[
            torch.tensor(aa_token_ids, device=self.DEVICE)]

        with torch.no_grad():
            txt_lat = self.txt_enc(**self.txt_tok(self.text_prompt, return_tensors="pt", truncation=True, max_length=512).to(self.DEVICE),
                                   return_dict=True).pooler_output
            self.text_lat = F.normalize(self.t2l(txt_lat), dim=-1)

    # ---------------------------------------------------------------------
    def _seq_distance(self, pred_idxs) -> float:
        pred_seq = ''.join(AMINO_ACIDS[i] for i in pred_idxs)
        return ProtCLAPGuidance._seq_distance(self.actual_sequence, pred_seq)

    # ---------------------------------------------------------------------
    def get_gradients(self, seq_logits: torch.Tensor) -> torch.Tensor:
        logits = seq_logits.detach().clone().requires_grad_(True).to(self.DEVICE)
        loss = F.cross_entropy(logits, self.target_idx, reduction='mean')
        loss.backward()
        self.gradients = logits.grad

        # diagnostics ------------------------------------------------------
        with torch.no_grad():
            pred_idxs  = logits.argmax(dim=-1)
            edit_dist  = self._seq_distance(pred_idxs.cpu().tolist())

            probs      = F.softmax(logits / self.T_clap, dim=-1)
            embeds     = probs @ self.aa_embed.to(probs.dtype)
            prot_lat   = self.prot_enc(inputs_embeds=embeds.unsqueeze(0),
                                       attention_mask=torch.ones(1, logits.size(0), dtype=torch.long, device=self.DEVICE),
                                       return_dict=True).pooler_output
            cos_sim    = F.cosine_similarity(F.normalize(self.p2l(prot_lat), dim=-1), self.text_lat).item()
            
        print(f"[Target] cos-sim: {cos_sim:.4f} | edit-dist: {edit_dist:.3f} | xent: {loss.item():.4f}")
        return -self.gradients*self.potential_scale