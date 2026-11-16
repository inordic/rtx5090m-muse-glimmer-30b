# Muse Glimmer 30B on an RTX 5090 Laptop GPU

How to serve Meta's [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) fast on a single **RTX 5090 Laptop GPU**: the mobile GB203 chip with 24 GB GDDR7, 896 GB/s and a 175 W cap, here in an Olares One running Proxmox. One stream, an agent workload (46K-token system prompt, 20 tools), English and Swedish.

| Same GPU, same benchmark | llama.cpp f8def7f (Aug 2026), unsloth UD-Q4_K_XL, Meta DFlash, depth 15 | **This repo: patched llama.cpp, MLP-NVFP4 hybrid, DFlash 2, depth 15** |
|---|---:|---:|
| Prefill, 6,880-token prompt | 1,430 tok/s | **3,254 tok/s** |
| Prefill, 100,554-token prompt | 1,246 tok/s | **2,367 tok/s** |
| Chat decode, greedy / sampled | 80.6 / 74.2 tok/s | **110.3 / 105.4 tok/s** |
| Agent turn with a cached 46K prompt | — | **0.89 s** |
| KLD vs Q8_0: agent / English / Swedish text | 0.033 / 0.012 / 0.015 | 0.053 / 0.030 / 0.034 |

The speed costs some quality: about 1.6× the KLD of the UD-Q4_K_XL file on agent text, with next-token agreement of 95.8% instead of 96.8%. A milder hybrid (below) gives +47% prefill at 1.4× the KLD. The UD-Q4_K_XL file on the same new stack keeps its quality at 107/104 tok/s decode, but stays at about 1,840 tok/s prefill.

## What we found

1. **Prefill is power-bound on this GPU.** It runs at the 175 W software power cap (throttle reason 0x4). With K-quants, llama.cpp and NInferno both stop at about 1,850 tok/s, whatever the batch size or drafter. Only the Blackwell FP4 tensor cores (NVFP4 W4A4) get past that: 4,600 tok/s raw.
2. **llama.cpp's NVFP4 quantizer loses too much quality.** `llama-quantize` sets each 16-value block scale straight from the block maximum, with no per-tensor global scale. For Glimmer's small weights most UE4M3 block scales then land in the subnormal range, which has very few significant bits. The result is KLD 0.18 on agent text. Our [`tools/q_nvfp4.py`](tools/q_nvfp4.py) adds a per-tensor scale (max|W| / 2688, written as a `.scale` tensor) and picks each block scale from 4 nearby codes by importance-weighted error. That brings KLD to 0.073 at the same speed. [`tools/gptq_nvfp4.py`](tools/gptq_nvfp4.py) does GPTQ in ggml's NVFP4 layout and brings it to 0.059.
3. **llama.cpp's Glimmer graph ignores per-tensor scales** on the attention gate and the three MLP projections. [Patch 0001](patches/llama.cpp/0001-muse-glimmer-apply-nvfp4-per-tensor-scales-on-gate-and-ffn.patch) passes them, as `qwen3.cpp` already does. Without it, any NVFP4 Glimmer GGUF that carries scales decodes to garbage.
4. **NVFP4 verify batches of 5–8 tokens ran on a slow kernel on Blackwell.** A depth-7 DFlash verify cost 31.8 ms against 24.2 ms on Q4_K. [Patch 0002](patches/llama.cpp/0002-cuda-mmvq-nvfp4-uses-mmq-from-5-columns-on-blackwell.patch) sends NVFP4 to MMQ from 5 columns: 25.6 ms.
5. **NVFP4 with 8-bit activations is slower than plain K-quants** (about 1,600 against 2,200 tok/s). So layers that need 8-bit activations should not be NVFP4 at all. That leads to the hybrids: GPTQ-NVFP4 W4A4 for the MLP (82% of the matrix work), IQ4_XS for attention.
6. **Decode comes from drafting.** Plain decode is 40 tok/s. The community [DFlash 2](https://huggingface.co/Anbeeld/Muse-Glimmer-30B-DFlash2-GGUF) drafter with probabilistic drafting beats Meta's DFlash: 104–110 against 92 tok/s on chat. Its best depth is 7 on K-quants and 15 on the NVFP4 hybrids. These didn't help: Meta's own target file, a Q8 drafter, larger prompt batches.
7. **Other engines.** On this GPU:
   - NInferno master matches llama.cpp on K-quant prefill but decodes slower, and has no DFlash 2 for Glimmer.
   - ExLlamaV3/TabbyAPI is slower on both prefill and decode.
   - vLLM and SGLang can't fit the drafter in 24 GB.

## Builds compared

Raw prefill and decode come from `llama-bench` without a drafter. KLD is measured against our own Q8_0 on 20 × 2048 tokens each of agent text in Glimmer's ATEM format ([AtomicChat calib-corpora](https://huggingface.co/datasets/AtomicChat/calib-corpora)), neutral English, and Swedish Wikipedia. Lower KLD is better.

| Build | Size | Prefill 512 / 4096 | Decode | KLD agent / English / Swedish | Same top token (agent) |
|---|---:|---:|---:|---|---:|
| unsloth UD-Q4_K_XL | 15.87 GB | 1,897 / 1,940 | 41.8 | 0.033 / 0.012 / 0.015 | 96.8% |
| bartowski IQ4_XS | 15.42 GB | 2,265 / 2,093 | 43.8 | 0.035 / 0.013 / 0.017 | 96.8% |
| llama-quantize NVFP4, all linear | 16.02 GB | 4,609 / 4,302 | 43.5 | 0.180 / 0.113 / 0.130 | 91.8% |
| ours: NVFP4, tensor scale + imatrix | 16.02 GB | 4,548 / 4,283 | 43.6 | 0.073 / 0.038 / 0.047 | 95.1% |
| ours: NVFP4, GPTQ | 16.02 GB | 4,559 / 4,286 | 43.6 | 0.059 / 0.036 / 0.038 | 95.5% |
| ours: hybrid gate+up GPTQ-NVFP4, rest IQ4_XS | 15.64 GB | 3,055 / 2,930 | 44.3 | 0.047 / 0.024 / 0.027 | 96.3% |
| **ours: hybrid MLP GPTQ-NVFP4, attention IQ4_XS** | 15.85 GB | **3,754 / 3,650** | 43.8 | 0.053 / 0.030 / 0.034 | 95.8% |

Served with the patched llama.cpp, DFlash 2, the vision projector, 131K context and reasoning strength low:

| Setup | Prefill 1.7K / 6.9K / 27K / 101K tokens | Chat greedy / sampled | Cached agent turn |
|---|---|---:|---:|
| UD-Q4_K_XL, DFlash 2 depth 7 | 1,788 / 1,838 / 1,757 / 1,495 | 107.1 / 103.9 | 0.99 s |
| hybrid gate+up, depth 15 | ~2,620 / 2,698 / 2,550 / 2,044 | 107.3 / 103.6 | 1.18 s |
| **hybrid MLP, depth 15** | **3,096 / 3,254 / 3,061 / 2,367** | **110.3 / 105.4** | **0.89 s** |
| GPTQ all-NVFP4, depth 15 | 3,650 / 3,854 / 3,562 / 2,644 | 102.7 / 99.5 | 0.96 s |

All four passed our agent-loop tool test with no errors, no tool markup returned as text and no invalid arguments. The raw data is in [`results/`](results/).

## Build it

[RECIPE.md](RECIPE.md) has the steps:
1. llama.cpp at 75fd7f9 with the two patches.
2. Meta's BF16 converted to GGUF.
3. An IQ4_XS template with the MLP as NVFP4, using an agent imatrix.
4. GPTQ-NVFP4 for the MLP.
5. Merge, then serve.

[`config/systemd/muse-glimmer-fast.service`](config/systemd/muse-glimmer-fast.service) is the serving unit. The night's full report is [research/2026-11-15-night1.md](research/2026-11-15-night1.md).

## Caveats

- **Quality here is KLD against Q8_0, not BF16,** plus a small agent-loop tool test. Task-level evaluations (reasoning, coding, Swedish chat) are still to come.
- **Muse Glimmer always reasons.** Set `reasoning_strength` once, server-side (we use `low` for agent turns). It sits in the system turn, so changing it per request breaks the prompt cache.
- **These numbers are for one 175 W mobile GB203.** Desktop cards have more power and bandwidth and will behave differently.

## Credits

- Meta for [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B), its DFlash assistant and its GGUF builds.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) and its contributors.
- z-lab for DFlash 2, and [Anbeeld](https://huggingface.co/Anbeeld/Muse-Glimmer-30B-DFlash2-GGUF) for the GGUF builds.
- [unsloth](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF) and [bartowski](https://huggingface.co/bartowski/Muse-Glimmer-30B-GGUF) for the quantizations.
- [AtomicChat](https://huggingface.co/AtomicChat/Muse-Glimmer-30B-GGUF) for the agent-weighted imatrix and calibration corpora.
- [NInferno](https://github.com/Brontir/NInferno) and [ExLlamaV3](https://github.com/turboderp-org/exllamav3) for the comparisons.
