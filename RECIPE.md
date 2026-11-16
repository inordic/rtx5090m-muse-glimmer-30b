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

```bash
D=/opt/models/muse-glimmer
# Meta's BF16 checkpoint (all files of meta-models/Muse-Glimmer-30B) into $D/bf16
# AtomicChat's agent-weighted imatrix:   AtomicChat/Muse-Glimmer-30B-GGUF  imatrix/muse-glimmer.imatrix.gguf
# Calibration text:                      AtomicChat/calib-corpora  builds/muse-glimmer-30b/calib_train.txt  -> $D/corpus/
# Swedish calibration text:              a few hundred Swedish Wikipedia articles  -> $D/corpus/eval_swedish.txt
# Drafter:                               Anbeeld/Muse-Glimmer-30B-DFlash2-GGUF  Muse-Glimmer-30B-DFlash2-Q4_K_M.gguf
# Vision projector:                      meta-models/Muse-Glimmer-30B-GGUF  mmproj-Muse-Glimmer-30B-Q4_K_M.gguf
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
--model glimmer-hyb-mlp.gguf --mmproj mmproj-…gguf --jinja --ctx-size 131072 --parallel 1 --flash-attn on --cache-type-k f16 --cache-type-v f16
--batch-size 4096 --ubatch-size 1024 --cache-prompt --cache-ram 16384 --fit off
--spec-type draft-dflash2 --spec-draft-model Muse-Glimmer-30B-DFlash2-Q4_K_M.gguf --spec-draft-ngl 999 --spec-draft-n-max 15
--spec-draft-sampling probabilistic --spec-draft-p-min 0
Environment: LLAMA_ARG_CHAT_TEMPLATE_KWARGS={"reasoning_strength":"low"}
```

- **Draft depth:** use 15 with the NVFP4 hybrids, and 7 with K-quant files such as UD-Q4_K_XL.
- **Vision:** with `--mmproj` loaded, llama.cpp turns `--cache-reuse` off. Agent turns are cached through context checkpoints and `--cache-ram` instead.

## 7. Measure

| Script | What it measures |
|---|---|
| [`bench/gbench.py`](bench/gbench.py) | Prefill on 2K/8K/32K/120K-character prompts (actual token counts recorded), decode after context, and 10 short chats, greedy and sampled with fixed seeds |
| [`bench/agentreplay.py`](bench/agentreplay.py) | An agent-shaped session: long system prompt, 20 tools, simulated tool results, full history resent each step |
| [`bench/agenttest.py`](bench/agenttest.py) | Tool-call protocol and task outcomes |
| [`bench/score.sh`](bench/score.sh) | KLD against a Q8_0 reference, plus llama-bench |

Build the reference logits for `score.sh` first, once per text:

```bash
llama-perplexity -m glimmer-q8_0.gguf -f eval_agentic.txt -c 2048 --chunks 20 -fa on -ngl 36 --kl-divergence-base base_agentic.bin
```
