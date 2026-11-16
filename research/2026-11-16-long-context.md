# Context beyond 131K (16 Nov 2026)

Muse Glimmer 30B ships with a 131,072-token context. We tested how far the MLP-NVFP4 hybrid (`glimmer-hyb-mlp.gguf`) goes on the 24 GB RTX 5090 Laptop GPU, and how well it finds things in long prompts. The engine was patched llama.cpp 75fd7f9, with DFlash 2 at depth 15, ubatch 1024 and reasoning strength low.

## Fitting 262K

```
--override-kv muse-glimmer.context_length=int:262144 --ctx-size 262144 --cache-type-k q8_0 --cache-type-v q8_0
```

| Setup | Fits? | GPU memory |
|---|---|---:|
| 131K, f16 KV, vision (the default unit) | yes | — |
| 262K, f16 KV, vision | **no**: loads, then runs out of memory on the first long prompt | — |
| 262K, q8_0 KV, vision | yes | 22.3–22.5 GB |

Only 1 in 4 layers is global attention and the model has 2 KV heads, so the KV cache is small for its size. q8_0 KV is what makes 262K fit next to the drafter and the vision projector.

## Retrieval

[`bench/needle.py`](../bench/needle.py) hides three key/value notes ("the access code for project X is amber-falcon-5506") in a long legal text and asks for one. Answers are greedy. A row passes only if the reply has the wanted code and no other code. The variants:
- **Decoys:** adds 4 near misses, such as a revoked old code for the same project and codes for similar names, and asks for the current code.
- **Two-step:** the asked project's note names an alias, and only the alias's note holds the code.

| Context, KV | Test | Prompt tokens | Result | Prefill |
|---|---|---:|---|---:|
| 262K, q8_0 | single needle, depth 0.5 | 94,304 | pass | 2,417 tok/s |
| 262K, q8_0 | single needle, depth 0.5 | 116,434 | pass | 2,230 tok/s |
| 262K, q8_0 | single needle, depth 0.5 | 179,316 | pass | 1,838 tok/s |
| 262K, q8_0 | single needle, depth 0.1 / 0.5 / 0.9 | ~230,000 | pass / pass / **miss** ("no code present") | 1,602–1,608 tok/s |
| 262K, q8_0 | decoys, depth 0.3 | 178,076 | **miss**: gave a decoy | 1,824 tok/s |
| 131K, f16 | decoys, depth 0.3 / 0.6 | ~94,000 | pass / **miss** | 2,423 tok/s |
| 131K, f16 | two-step, depth 0.4 | 94,277 | **miss**: no answer | 2,408 tok/s |
| 262K, q8_0 | decoys, depth 0.3 / 0.6 | ~94,000 | **miss** / **miss** | 2,377 tok/s |
| 262K, q8_0 | two-step, depth 0.4 | 94,277 | **miss**: wrong answer | 2,377 tok/s |

Decode stayed at 60–150 tok/s throughout. It varies with how much of the short answer the drafter guesses.

## What q8_0 KV costs

The KV cache is 13 global layers × 2 KV heads × 128 dimensions, for K and V. At f16 that is 13 KiB per token, so 262K tokens at q8_0 take about the same memory as 131K at f16.

We compared the logits with `llama-perplexity --kl-divergence` against an f16-KV run at ubatch 1024. The control changes only the ubatch size and keeps f16 KV:

| Comparison | Text, context | KLD | Same top token | PPL ratio |
|---|---|---:|---:|---:|
| q8_0 KV | needle haystack (legal text), 2 × 32K | 0.040 | 94.8% | 1.0012 ± 0.0018 |
| **control:** f16 KV, ubatch 512 | needle haystack (legal text), 2 × 32K | 0.039 | 94.8% | 0.9996 ± 0.0018 |
| q8_0 KV | `eval_agentic.txt`, 6 × 8K | 0.0095 | 99.6% | 1.0004 ± 0.0022 |

q8_0 KV changes the output no more than a different batch split. The hybrid's FP4 activations (W4A4) amplify any small numeric change, so the two land on the same level. Raw numbers are in [`results/kv-kld.csv`](../results/kv-kld.csv).

```bash
P="llama-perplexity -m glimmer-hyb-mlp.gguf -ngl 999 -fa on -b 4096 -ub 1024"
$P -f haystack.txt -c 32768 --chunks 2 -ctk f16 -ctv f16 --kl-divergence-base base.bin
$P -f haystack.txt -c 32768 --chunks 2 -ctk q8_0 -ctv q8_0 --kl-divergence-base base.bin --kl-divergence
```

Speed with the serving unit at 262K and q8_0 KV, against the same unit at 131K and f16, both [`bench/gbench.py`](../bench/gbench.py) `--pp 8192 --tg 2048 --reps 1 --chat-n 5 --skip-sampled`:
- prefill at 6,672 tokens: 3,169 against 3,322 tok/s;
- greedy chat: 106.6 against 113.5 tok/s.

That is about 5% slower. A 150,401-token request returned the right code, at 2,014 tok/s prefill. The agent-loop tool test ran with no errors, no tool markup returned as text and no invalid arguments, in 109 calls.

## What this says

- **Plain lookup works to about 180K tokens,** and at 230K except near the end of the text.
- **Questions with decoys or two steps fail already at 94K, inside the native window and with f16 KV.** So the weakness is not caused by the context extension. It comes from the model at reasoning strength low, the 4-bit build, or both. The decoy counts (0/2 with q8_0 KV against 1/2 with f16) are too few to read anything from. The KLD test above finds no cost from q8_0 KV.
- **Not yet tested:** the same questions at reasoning strength medium or high, and with the UD-Q4_K_XL file, to separate the model from the quantization.

**Recommendation:** we now serve 262K with q8_0 KV; it is the [unit](../config/systemd/muse-glimmer-fast.service). It costs about 5% speed and no measurable quality, and requests between 131K and 262K are accepted instead of refused. Past 131K the model runs beyond its trained length, so rely on it for lookups, not for reasoning over the whole text. If you never send more than 131K, `--ctx-size 131072 --cache-type-k f16 --cache-type-v f16` without the override is about 5% faster.

Raw rows are in [`results/needle.jsonl`](../results/needle.jsonl).
