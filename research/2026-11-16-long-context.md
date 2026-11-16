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

## What this says

- **Plain lookup works to about 180K tokens,** and at 230K except near the end of the text.
- **Questions with decoys or two steps fail already at 94K, inside the native window and with f16 KV.** So the weakness is not caused by the context extension. It comes from the model at reasoning strength low, the 4-bit build, or both. q8_0 KV may cost a little more (0/2 against 1/2 on decoys), but one to two samples per cell can't show that.
- **Not yet tested:** the same questions at reasoning strength medium or high, and with the UD-Q4_K_XL file, to separate the model from the quantization.

**Recommendation:** keep 131K with f16 KV as the default. Use 262K with q8_0 KV for single-fact lookups in very long documents, and don't rely on it for reasoning over the whole text.

Raw rows are in [`results/needle.jsonl`](../results/needle.jsonl).
