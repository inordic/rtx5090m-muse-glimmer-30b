# Recipe: the MLP-NVFP4 hybrid on llama.cpp

**Tested on:** Ubuntu 24.04 in a Proxmox LXC; CUDA 13.2 with NVIDIA driver 595.58.03; RTX 5090 Laptop GPU; 8 cores and 64 GB RAM.

**Disk:** plan on about 220 GB of free space while building. The BF16 checkpoint, the BF16 GGUF and the Q8_0 reference take about 140 GB of that and can be removed afterwards.

## 1. llama.cpp with the two patches

```bash
git clone https://github.com/ggml-org/llama.cpp /opt/src/llama.cpp && cd /opt/src/llama.cpp
git checkout 75fd7f9
git apply /path/to/this/repo/patches/llama.cpp/0001-*.patch /path/to/this/repo/patches/llama.cpp/0002-*.patch
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
ninja -C build llama-server llama-bench llama-perplexity llama-quantize
mkdir -p /opt/llama.cpp-glimmer && cp -a build/bin/. /opt/llama.cpp-glimmer/
```

**Check:** `llama-bench -p 4,8,16` on an NVFP4 GGUF. With patch 0002, batch 8 costs no more than batch 16.

## 2. Inputs

Everything comes from Hugging Face. These are the revisions we used:

| Input | Repository, revision | File | SHA256 |
|---|---|---|---|
| BF16 checkpoint (all files) → `$D/bf16` | `meta-models/Muse-Glimmer-30B` @ `cec94fcfd6512f2cfe708138c32533b346ea4f69` | | |
| Agent-weighted imatrix | `AtomicChat/Muse-Glimmer-30B-GGUF` @ `6f4bcf4990aa5788c52f2dc13e2c13fd4bd9cb76` | `imatrix/muse-glimmer.imatrix.gguf`, saved as `$D/dl/atomicchat-agent.imatrix.gguf` | `eae2b1317b4412675b1b62e86e375a28f4930fd55e149f74f140f26bce0ab054` |
| Drafter | `Anbeeld/Muse-Glimmer-30B-DFlash2-GGUF` @ `c773bd2827a726032f63f822a4e1fb367917d223` | `Muse-Glimmer-30B-DFlash2-Q4_K_M.gguf` | `9282f0053703623f89499dadb7e844ee0fd7b9c13de9a9840a726011e80ba1c4` |
| Vision projector | `meta-models/Muse-Glimmer-30B-GGUF` @ `b44f1072cf75e918a1b82a3a17b45c4389b076e7` | `mmproj-Muse-Glimmer-30B-Q4_K_M.gguf` | `f48b452316f9b213758e8659444029b961a24a07f99a1abb2a9f88b06f7c00c6` |
| Comparison file | `unsloth/Muse-Glimmer-30B-GGUF` @ `faa5b025c584459c13febfa5c59883516710ae39` | `Muse-Glimmer-30B-UD-Q4_K_XL.gguf` | `82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38` |

The calibration and evaluation texts come from `AtomicChat/calib-corpora` and Swedish Wikipedia. [`tools/make_corpus.py`](tools/make_corpus.py) downloads them from pinned revisions, builds the Swedish text and checks every file's sha256 (it needs `pyarrow`):

```bash
D=/opt/models/muse-glimmer
python3 tools/make_corpus.py $D/corpus   # ends with CORPUS_OK
```

## 3. BF16 GGUF

The converter needs transformers 5.x for Glimmer's tokenizer class:

```bash
python3 -m venv /opt/convvenv && . /opt/convvenv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r /opt/src/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt && pip install -U "transformers==5.23.0"
python /opt/src/llama.cpp/convert_hf_to_gguf.py $D/bf16 --outtype bf16 --outfile $D/own/glimmer-bf16.gguf   # 731 tensors, 55.7 GB
```

## 4. Templates

Each template is a GGUF whose MLP tensors are already NVFP4, so the tools can swap in better data:

```bash
Q=/opt/llama.cpp-glimmer/llama-quantize; IMA=$D/dl/atomicchat-agent.imatrix.gguf
NV="--tensor-type ffn_gate=nvfp4 --tensor-type ffn_up=nvfp4 --tensor-type ffn_down=nvfp4"
# all linear layers NVFP4: the template GPTQ writes into
$Q --imatrix $IMA $NV --tensor-type attn_q=nvfp4 --tensor-type attn_k=nvfp4 --tensor-type attn_v=nvfp4 \
   --tensor-type attn_output=nvfp4 --tensor-type attn_gate=nvfp4 $D/own/glimmer-bf16.gguf $D/own/glimmer-nvfp4-all.gguf Q4_K_M 8
# the hybrid template: MLP NVFP4, everything else IQ4_XS
$Q --imatrix $IMA $NV $D/own/glimmer-bf16.gguf $D/own/tmpl-hmlp.gguf IQ4_XS 8
```

## 5. GPTQ-NVFP4 and the merge

This runs on the GPU, layer by layer, using the BF16 checkpoint through transformers. It takes 29 minutes with 128 × 2048 calibration tokens.

```bash
python3 -m venv /opt/calvenv && . /opt/calvenv/bin/activate
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.23.0 safetensors accelerate && pip install -e /opt/src/llama.cpp/gguf-py
cd tools
python gptq_nvfp4.py $D/bf16 $D/own/glimmer-bf16.gguf $D/own/glimmer-nvfp4-all.gguf $D/own/glimmer-gptq-all.gguf --seqs 128
python merge_nvfp4.py $D/own/tmpl-hmlp.gguf $D/own/glimmer-gptq-all.gguf $D/own/glimmer-hyb-mlp.gguf ffn_gate,ffn_up,ffn_down
```

`q_nvfp4.py` is the faster alternative to GPTQ: no calibration pass, 5 minutes, slightly worse quality.

```bash
python q_nvfp4.py check $D/own/glimmer-bf16.gguf $D/own/glimmer-nvfp4-all.gguf $IMA blk.10.ffn_up.weight   # compare with llama-quantize
python q_nvfp4.py build $D/own/glimmer-bf16.gguf $D/own/glimmer-nvfp4-all.gguf $IMA $D/own/glimmer-nvfp4w-all.gguf
```

`add_prec.py` marks chosen tensor kinds for 8-bit activations, through llama.cpp's `general.tensor_extra.prec_a4` metadata. We tested it; it was slower than the hybrid at equal quality.

## 6. Serve

See [`config/systemd/muse-glimmer-fast.service`](config/systemd/muse-glimmer-fast.service). The key flags:

```
--model glimmer-hyb-mlp.gguf --mmproj mmproj-…gguf --jinja --parallel 1 --flash-attn on
--override-kv muse-glimmer.context_length=int:262144 --ctx-size 262144 --cache-type-k q8_0 --cache-type-v q8_0
--batch-size 4096 --ubatch-size 1024 --cache-prompt --cache-ram 16384 --fit off
--spec-type draft-dflash2 --spec-draft-model Muse-Glimmer-30B-DFlash2-Q4_K_M.gguf --spec-draft-ngl 999 --spec-draft-n-max 15
--spec-draft-sampling probabilistic --spec-draft-p-min 0
Environment: LLAMA_ARG_CHAT_TEMPLATE_KWARGS={"reasoning_strength":"low"}
```

- **Context:** 262K needs q8_0 KV; with f16 KV it runs out of memory on the first long prompt. q8_0 KV costs about 5% speed and no measurable quality. If requests never exceed 131K, `--ctx-size 131072 --cache-type-k f16 --cache-type-v f16` without the override is the faster choice. See [research/2026-11-16-long-context.md](research/2026-11-16-long-context.md).
- **Draft depth:** use 15 with the NVFP4 hybrids, and 7 with K-quant files such as UD-Q4_K_XL.
- **Vision:** with `--mmproj` loaded, llama.cpp turns `--cache-reuse` off. Agent turns are cached through context checkpoints and `--cache-ram` instead.

## 7. Measure

| Script | What it measures |
|---|---|
| [`bench/gbench.py`](bench/gbench.py) | Prefill on 2K/8K/32K/120K-character prompts (actual token counts recorded), decode after context, and 10 short chats, greedy and sampled with fixed seeds. Uses `long_niah_256k.json` as prompt text, like `needle.py` |
| [`bench/agentreplay.py`](bench/agentreplay.py) | An agent-shaped session: long system prompt, 20 tools, simulated tool results, full history resent each step |
| [`bench/agenttest.py`](bench/agenttest.py) | Tool-call protocol and task outcomes |
| [`bench/score.sh`](bench/score.sh) | KLD against a Q8_0 reference, plus llama-bench |
| [`bench/needle.py`](bench/needle.py) | Long-context retrieval (single needle, decoys, two-step) with prefill and decode speed. Needs `long_niah_256k.json` from the NInferno repository as the haystack |

Build the reference logits for `score.sh` first, once per text:

```bash
llama-perplexity -m glimmer-q8_0.gguf -f eval_agentic.txt -c 2048 --chunks 20 -fa on -ngl 36 --kl-divergence-base base_agentic.bin
```
