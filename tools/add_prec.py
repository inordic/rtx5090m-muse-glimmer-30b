#!/usr/bin/env python3
"""add_prec.py IN OUT KIND[,KIND...]: copy a GGUF and mark every NVFP4 tensor of the given kinds (e.g. ffn_down,
attn_output) to run with 8-bit activations, through llama.cpp's per-tensor policy keys general.tensor_extra.name /
general.tensor_extra.prec_a4 (false = keep src1 at Q8). All other NVFP4 tensors keep 4-bit activations."""
import sys
import numpy as np, gguf

src, out, kinds = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
r = gguf.GGUFReader(src)
w = gguf.GGUFWriter(out, arch=r.fields[gguf.Keys.General.ARCHITECTURE].contents())
for f in r.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF.") or f.name.startswith("general.tensor_extra."):
        continue
    vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(f.name, f.contents(), vt, sub_type=st)
names = [t.name for t in r.tensors if t.tensor_type == gguf.GGMLQuantizationType.NVFP4 and t.name.split(".")[-2] in kinds]
w.add_key_value("general.tensor_extra.name", names, gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.STRING)
w.add_key_value("general.tensor_extra.prec_a4", [False] * len(names), gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.BOOL)
for t in r.tensors:
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in r.tensors:
    w.write_tensor_data(t.data)
w.close()
print(f"ADD_PREC_DONE {out}: {len(names)} tensors at 8-bit activations ({','.join(kinds)})", flush=True)
