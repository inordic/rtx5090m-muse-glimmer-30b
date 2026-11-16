#!/usr/bin/env python3
"""gptq_nvfp4.py: GPTQ-quantized NVFP4 for Muse Glimmer, written in ggml's NVFP4 layout with per-tensor '.scale'.

Layer by layer on the GPU: the BF16 layer runs the calibration text (full-precision activations), each projection's
input covariance H = X^T X is accumulated, then every projection is quantized with GPTQ (column order, error spread
through the Cholesky of H^-1). NVFP4 specifics: a per-tensor global scale G = max|W|/(6*448); 16-column groups whose
UE4M3 block scale is picked (from 4 nearby codes) on the error-updated weights, weighted by diag(H).

  python gptq_nvfp4.py BF16_HF_DIR BF16_GGUF TEMPLATE_GGUF OUT_GGUF [--seqs 128] [--seqlen 2048] [--kinds ...]
TEMPLATE_GGUF must hold the chosen kinds as NVFP4 (from llama-quantize); everything else is copied from it."""
import argparse, json, os, time
import numpy as np, torch, gguf
from safetensors import safe_open
import q_nvfp4 as qn
from transformers import AutoTokenizer, AutoConfig
from transformers.models.muse_glimmer import modeling_muse_glimmer as mg
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

dev = "cuda"
HF2GG = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
         "self_attn.gate_proj": "attn_gate", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}
H_OF = {"self_attn.q_proj": "attn_in", "self_attn.k_proj": "attn_in", "self_attn.v_proj": "attn_in", "self_attn.gate_proj": "attn_in",
        "self_attn.o_proj": "o_in", "mlp.gate_proj": "mlp_in", "mlp.up_proj": "mlp_in", "mlp.down_proj": "down_in"}


def gptq_nvfp4(W, H, block=128, damp=0.01):
    """W (rows, cols) float32 on GPU, H (cols, cols) float32. Returns ggml NVFP4 bytes (rows, cols/64*36) and G."""
    rows, cols = W.shape
    G = (W.abs().max() / (6 * 448)).item()
    W = W / G
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += torch.eye(cols, device=dev) * damp * torch.mean(torch.diag(H))
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    imp = torch.diag(H)
    codes = torch.empty(rows, cols // 16, dtype=torch.uint8, device=dev)
    nib = torch.empty(rows, cols, dtype=torch.uint8, device=dev)
    for i1 in range(0, cols, block):
        i2 = min(i1 + block, cols)
        W1 = W[:, i1:i2].clone(); Err1 = torch.zeros_like(W1); Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            c = i1 + i
            if c % 16 == 0:  # choose this group's block scales on the current (error-updated) weights
                Bg = W1[:, i:i + 16]
                Ig = imp[c:c + 16].view(1, 16)
                base = torch.searchsorted(qn.RAW, (Bg.abs().amax(-1) / 6).contiguous()).clamp(1, 126)
                best_err, best_code = None, None
                for dc in (-1, 0, 1, 2):
                    code = (base + dc).clamp(1, 126); s = qn.RAW[code].unsqueeze(-1)
                    x = Bg / s
                    e = ((Bg - qn.MAG[torch.bucketize(x.abs(), qn.MID)] * torch.sign(x) * s) ** 2 * Ig).sum(-1)
                    if best_err is None:
                        best_err, best_code = e, code
                    else:
                        b = e < best_err; best_err = torch.where(b, e, best_err); best_code = torch.where(b, code, best_code)
                codes[:, c // 16] = best_code.to(torch.uint8)
                gscale = qn.RAW[best_code]
            w = W1[:, i]
            x = w / gscale
            mi = torch.bucketize(x.abs(), qn.MID)
            q = qn.MAG[mi] * torch.sign(x) * gscale
            nib[:, c] = (mi + ((x < 0) & (mi > 0)).long() * 8).to(torch.uint8)
            err = (w - q) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    idx = nib.view(rows, cols // 64, 4, 16)
    qs = idx[..., :8] | (idx[..., 8:] << 4)
    blk = torch.cat([codes.view(rows, cols // 64, 4), qs.reshape(rows, cols // 64, 32)], dim=-1)
    return blk.reshape(rows, cols // 64 * 36), G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir"); ap.add_argument("bf16_gguf"); ap.add_argument("template"); ap.add_argument("out")
    ap.add_argument("--seqs", type=int, default=128); ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--kinds", default=qn.KINDS); ap.add_argument("--layers", default=None, help="e.g. 0-3 for a test")
    a = ap.parse_args()
    kinds = a.kinds.split(",")
    cfg = AutoConfig.from_pretrained(a.hf_dir).text_config
    cfg._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(a.hf_dir)
    # calibration: Glimmer-format agent text (AtomicChat's Glimmer calibration build) plus Swedish Wikipedia
    corp = "/opt/models/muse-glimmer/corpus/"
    ag = tok(open(corp + "calib_train.txt").read()[:9_000_000], add_special_tokens=False)["input_ids"]
    sv = tok(open(corp + "eval_swedish.txt").read()[-400_000:], add_special_tokens=False)["input_ids"]
    n_sv = max(1, a.seqs // 8)
    seqs = [ag[i * a.seqlen:(i + 1) * a.seqlen] for i in range(a.seqs - n_sv)] + [sv[i * a.seqlen:(i + 1) * a.seqlen] for i in range(n_sv)]
    seqs = [s for s in seqs if len(s) == a.seqlen]
    ids = torch.tensor(seqs, device=dev)
    print(f"calibration: {ids.shape[0]} x {a.seqlen} tokens ({n_sv} Swedish)", flush=True)
    idx = json.load(open(os.path.join(a.hf_dir, "model.safetensors.index.json")))["weight_map"]
    pre = "model.language_model."

    def load(names):
        out = {}
        for n in names:
            with safe_open(os.path.join(a.hf_dir, idx[pre + n]), "pt", device=dev) as f:
                out[n] = f.get_tensor(pre + n)
        return out

    with torch.device("meta"):
        skel = mg.MuseGlimmerTextModel(cfg)
    emb = skel.embed_tokens
    emb.to_empty(device=dev)
    sd = load([k[len(pre):] for k in idx if k.startswith(pre + "embed_tokens.")])
    r = emb.load_state_dict({k[len("embed_tokens."):]: v for k, v in sd.items()}, strict=False, assign=True)
    print("embedding: loaded", sorted(sd), "missing", r.missing_keys, "buffers", [n for n, _ in emb.named_buffers()], flush=True)
    emb = emb.to(dev, torch.bfloat16)
    rot = mg.MuseGlimmerTextRotaryEmbedding(cfg).to(dev)
    with torch.no_grad():
        hidden = torch.cat([emb(ids[i:i + 1]) for i in range(ids.shape[0])], 0).to(torch.bfloat16)
    del emb, sd; torch.cuda.empty_cache()
    pos = torch.arange(a.seqlen, device=dev).unsqueeze(0)
    pe = rot(hidden[:1], pos)
    mk = {"config": cfg, "inputs_embeds": hidden[:1], "attention_mask": None, "past_key_values": None, "position_ids": pos}
    masks = {"full_attention": create_causal_mask(**mk), "sliding_attention": create_sliding_window_causal_mask(**mk)}
    lo, hi = (0, cfg.num_hidden_layers) if not a.layers else map(int, a.layers.split("-"))
    hi = hi + 1 if a.layers else hi
    results, t0 = {}, time.time()
    gg = gguf.GGUFReader(a.bf16_gguf); ggt = {t.name: t for t in gg.tensors}
    for li in range(lo, hi):
        with torch.device(dev):
            layer = mg.MuseGlimmerTextDecoderLayer(cfg, li).to(torch.bfloat16)
        lsd = load([k[len(pre):] for k in idx if k.startswith(f"{pre}layers.{li}.")])
        layer.load_state_dict({k[len(f"layers.{li}."):]: v for k, v in lsd.items()}, strict=True, assign=True)
        layer = layer.to(dev, torch.bfloat16).eval()
        Hs = {}

        def hook(key):
            def f(mod, inp):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                Hs[key] = Hs.get(key, 0) + x.T @ x
            return f
        hds = [layer.self_attn.q_proj.register_forward_pre_hook(hook("attn_in")), layer.self_attn.o_proj.register_forward_pre_hook(hook("o_in")),
               layer.mlp.gate_proj.register_forward_pre_hook(hook("mlp_in")), layer.mlp.down_proj.register_forward_pre_hook(hook("down_in"))]
        ltype = cfg.layer_types[li]
        with torch.no_grad():
            for s in range(hidden.shape[0]):
                o = layer(hidden[s:s + 1], attention_mask=masks[ltype],
                          position_embeddings=pe if cfg.layer_rope_theta[li] else None, position_ids=pos)
                hidden[s:s + 1] = o[0] if isinstance(o, tuple) else o
        for h in hds:
            h.remove()
        del layer, lsd; torch.cuda.empty_cache()
        for hf, gk in HF2GG.items():
            if gk not in kinds:
                continue
            name = f"blk.{li}.{gk}.weight"
            W = qn.bf16_weight(ggt[name])     # the GGUF's own (possibly row-permuted) copy; H is over input columns
            blk, G = gptq_nvfp4(W, Hs[H_OF[hf]])
            results[name] = (blk.cpu().numpy(), G)
            del W, blk
        del Hs; torch.cuda.empty_cache()
        print(f"  layer {li} ({ltype}) done ({time.time() - t0:.0f}s)", flush=True)
    # write: template tensors, ours swapped in, plus scales
    tm = gguf.GGUFReader(a.template)
    w = gguf.GGUFWriter(a.out, arch=tm.fields[gguf.Keys.General.ARCHITECTURE].contents())
    for f in tm.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(f.name, f.contents(), vt, sub_type=st)
    for t in tm.tensors:
        d = results[t.name][0] if t.name in results else t.data
        if t.name in results:
            assert t.tensor_type == gguf.GGMLQuantizationType.NVFP4 and d.nbytes == t.data.nbytes, t.name
        w.add_tensor_info(t.name, d.shape, d.dtype, d.nbytes, t.tensor_type)
    for n in results:
        w.add_tensor_info(n[:-len(".weight")] + ".scale", (1,), np.dtype(np.float32), 4, gguf.GGMLQuantizationType.F32)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for t in tm.tensors:
        w.write_tensor_data(results[t.name][0] if t.name in results else t.data)
    for n in results:
        w.write_tensor_data(np.array([results[n][1]], dtype=np.float32))
    w.close()
    print(f"GPTQ_DONE {a.out}: {len(results)} tensors ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
