#!/usr/bin/env python3
"""q_nvfp4.py: our own NVFP4 for Muse Glimmer in ggml's block layout.

Differences from llama-quantize's NVFP4 (plain round-to-nearest, no tensor scale):
  - a per-tensor global scale G = max|W| / (6*448), written as '<name>.scale' (F32 [1]) which llama.cpp multiplies
    into the matmul output, so the 16-value block scales sit in UE4M3's normal range instead of its subnormals;
  - each block scale is chosen from 4 nearby UE4M3 codes by the importance-weighted squared error (imatrix).

  check: python q_nvfp4.py check BF16 TEMPLATE IMATRIX TENSOR_NAME
  build: python q_nvfp4.py build BF16 TEMPLATE IMATRIX OUT [kinds,comma,separated]
TEMPLATE is a GGUF of the same model whose tensors of the chosen kinds are already NVFP4 (from llama-quantize);
its other tensors and all metadata are copied unchanged."""
import sys, time
import numpy as np, torch, gguf

dev = "cuda"
KINDS = "attn_q,attn_k,attn_v,attn_output,attn_gate,ffn_gate,ffn_up,ffn_down"


def ue4m3_raw(c):  # ggml_ue4m3_to_fp32(c) * 2: the factor that multiplies an E2M1 value
    if c == 0 or c == 0x7F:
        return 0.0
    e, m = (c >> 3) & 0xF, c & 7
    return m * 2.0 ** -9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)


RAW = torch.tensor([ue4m3_raw(c) for c in range(127)], dtype=torch.float32, device=dev)  # increasing in c
MAG = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device=dev)
MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32, device=dev)
KV = torch.tensor([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=torch.float32, device=dev)  # kvalues_fp4


def bf16_weight(t):
    a = np.frombuffer(np.asarray(t.data).tobytes(), dtype=np.uint16).reshape(int(t.shape[1]), int(t.shape[0]))
    return (torch.from_numpy(a.astype(np.int32)).to(dev) << 16).view(torch.float32)


def importance(imt, name, cols):
    s, c = imt.get(name + ".in_sum2"), imt.get(name + ".counts")
    if s is None:
        return torch.ones(cols, device=dev)
    v = torch.from_numpy(np.asarray(s.data, dtype=np.float32).reshape(-1)[:cols].copy()).to(dev)
    n = float(np.asarray(c.data).reshape(-1)[0]) if c is not None else 1.0
    v = v / max(n, 1.0)
    return v / v.mean().clamp(min=1e-12) + 0.01


def quantize(W, imp, chunk=1024, global_scale=True, search=(-1, 0, 1, 2)):
    rows, cols = W.shape
    nb16, nb64 = cols // 16, cols // 64
    G = (W.abs().max() / (6 * 448)).item() if global_scale else 1.0
    I = imp.view(1, nb16, 16)
    codes = torch.empty(rows, nb16, dtype=torch.uint8, device=dev)
    nib = torch.empty(rows, nb16, 16, dtype=torch.uint8, device=dev)
    for r0 in range(0, rows, chunk):
        B = (W[r0:r0 + chunk] / G).view(-1, nb16, 16)
        base = torch.searchsorted(RAW, (B.abs().amax(-1) / 6).contiguous()).clamp(1, 126)
        best = None
        for dc in search:
            code = (base + dc).clamp(1, 126)
            s = RAW[code].unsqueeze(-1)
            x = B / s
            mi = torch.bucketize(x.abs(), MID)
            err = ((B - MAG[mi] * torch.sign(x) * s) ** 2 * I).sum(-1)
            if best is None:
                best = [err, code, mi, x < 0]
            else:
                b = err < best[0]
                best = [torch.where(b, err, best[0]), torch.where(b, code, best[1]),
                        torch.where(b.unsqueeze(-1), mi, best[2]), torch.where(b.unsqueeze(-1), x < 0, best[3])]
        codes[r0:r0 + chunk] = best[1].to(torch.uint8)
        nib[r0:r0 + chunk] = (best[2] + ((best[3] & (best[2] > 0)).long() * 8)).to(torch.uint8)
    idx = nib.view(rows, nb64, 4, 16)
    qs = idx[..., :8] | (idx[..., 8:] << 4)
    blk = torch.cat([codes.view(rows, nb64, 4), qs.reshape(rows, nb64, 32)], dim=-1)
    return blk.reshape(rows, nb64 * 36), G


def dequantize(blk, G):  # blk (rows, nb64*36) uint8 on GPU -> float32 (rows, cols)
    rows = blk.shape[0]
    b = blk.view(rows, -1, 36)
    d = RAW[b[..., :4].long()]                                   # (rows, nb64, 4)
    qs = b[..., 4:].view(rows, -1, 4, 8)
    lo, hi = (qs & 0xF).long(), (qs >> 4).long()
    v = torch.cat([KV[lo], KV[hi]], dim=-1) * 0.5                # E2M1 values, (rows, nb64, 4, 16)
    return (v * d.unsqueeze(-1)).reshape(rows, -1) * G


def main():
    mode = sys.argv[1]
    bf, tm, im = gguf.GGUFReader(sys.argv[2]), gguf.GGUFReader(sys.argv[3]), gguf.GGUFReader(sys.argv[4])
    bft, tmt, imt = {t.name: t for t in bf.tensors}, {t.name: t for t in tm.tensors}, {t.name: t for t in im.tensors}
    if mode == "check":
        name = sys.argv[5]
        W = bf16_weight(bft[name]); imp = importance(imt, name, W.shape[1])
        ref = torch.from_numpy(np.asarray(tmt[name].data).copy()).to(dev)
        def werr(Q):
            return (((W - Q) ** 2) * imp.view(1, -1)).sum().item() / ((W ** 2) * imp.view(1, -1)).sum().item()
        print(f"{name}: shape {tuple(W.shape)} max|W| {W.abs().max().item():.4g}")
        print(f"  llama-quantize NVFP4 (round-to-nearest, no tensor scale): weighted rel. error {werr(dequantize(ref, 1.0)):.5f}")
        for gs, se, lab in ((False, (0,), "no tensor scale, round-to-nearest (our dequant check)"), (True, (0,), "tensor scale, nearest code"),
                            (True, (-1, 0, 1, 2), "tensor scale + importance-weighted code search")):
            blk, G = quantize(W, imp, global_scale=gs, search=se)
            print(f"  ours, {lab}: {werr(dequantize(blk, G)):.5f}")
        return
    out, kinds = sys.argv[5], (sys.argv[6] if len(sys.argv) > 6 else KINDS).split(",")
    w = gguf.GGUFWriter(out, arch=tm.fields[gguf.Keys.General.ARCHITECTURE].contents())
    for f in tm.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(f.name, f.contents(), vt, sub_type=st)
    targets = [t.name for t in tm.tensors if t.tensor_type == gguf.GGMLQuantizationType.NVFP4 and t.name.split(".")[-2] in kinds]
    new_data, scales, t0 = {}, {}, time.time()
    for i, name in enumerate(targets):
        W = bf16_weight(bft[name]); blk, G = quantize(W, importance(imt, name, W.shape[1]))
        new_data[name] = blk.cpu().numpy(); scales[name[:-len(".weight")] + ".scale"] = G
        if i % 40 == 0:
            print(f"  {i}/{len(targets)} {name} G={G:.4g} ({time.time() - t0:.0f}s)", flush=True)
    for t in tm.tensors:
        d = new_data.get(t.name, t.data)
        w.add_tensor_info(t.name, d.shape, d.dtype, d.nbytes, t.tensor_type)
    for n in scales:
        w.add_tensor_info(n, (1,), np.dtype(np.float32), 4, gguf.GGMLQuantizationType.F32)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for t in tm.tensors:
        w.write_tensor_data(new_data.get(t.name, t.data))
    for n, G in scales.items():
        w.write_tensor_data(np.array([G], dtype=np.float32))
    w.close()
    print(f"QNVFP4_DONE {out}: {len(targets)} tensors requantized, {len(scales)} scales, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
