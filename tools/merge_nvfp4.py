#!/usr/bin/env python3
"""merge_nvfp4.py TEMPLATE SOURCE OUT KIND[,KIND]: copy TEMPLATE, but take every NVFP4 tensor of the given kinds
(and its '.scale') from SOURCE (e.g. our GPTQ NVFP4 build). TEMPLATE must hold those kinds as NVFP4."""
import sys
import numpy as np, gguf
tm, src, out, kinds = gguf.GGUFReader(sys.argv[1]), gguf.GGUFReader(sys.argv[2]), sys.argv[3], sys.argv[4].split(",")
st = {t.name: t for t in src.tensors}
w = gguf.GGUFWriter(out, arch=tm.fields[gguf.Keys.General.ARCHITECTURE].contents())
for f in tm.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
        continue
    vt = f.types[0]; sub = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(f.name, f.contents(), vt, sub_type=sub)
take = [t.name for t in tm.tensors if t.tensor_type == gguf.GGMLQuantizationType.NVFP4 and t.name.split(".")[-2] in kinds]
scales = [n[:-len(".weight")] + ".scale" for n in take]
for t in tm.tensors:
    d = st[t.name].data if t.name in take else t.data
    assert d.nbytes == t.data.nbytes, t.name
    w.add_tensor_info(t.name, d.shape, d.dtype, d.nbytes, t.tensor_type)
for n in scales:
    w.add_tensor_info(n, (1,), np.dtype(np.float32), 4, gguf.GGMLQuantizationType.F32)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in tm.tensors:
    w.write_tensor_data(st[t.name].data if t.name in take else t.data)
for n in scales:
    w.write_tensor_data(np.asarray(st[n].data, dtype=np.float32).reshape(1))
w.close()
print(f"MERGE_DONE {out}: {len(take)} NVFP4 tensors from {sys.argv[2].split('/')[-1]} ({','.join(kinds)})", flush=True)
