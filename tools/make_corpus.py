#!/usr/bin/env python3
"""make_corpus.py OUT_DIR: download and build the calibration and evaluation texts used in this repo.

Files, from pinned revisions on Hugging Face:
- calib_train.txt   AtomicChat/calib-corpora builds/muse-glimmer-30b/calib_train.txt (GPTQ calibration, Glimmer agent format)
- eval_agentic.txt  AtomicChat/calib-corpora eval/agentic/eval_agentic.txt (KLD scoring, agent text)
- eval_neutral.txt  AtomicChat/calib-corpora eval/neutral/eval_neutral.txt (KLD scoring, neutral English)
- sv_wiki.parquet   wikimedia/wikipedia 20231101.sv/train-00002-of-00005.parquet
- eval_swedish.txt  built from sv_wiki.parquet: articles of 2,000+ characters, shuffled with seed 7, until 1.6M characters.
                    Used for KLD scoring and, its last 400K characters, as the Swedish part of the GPTQ calibration.

Each file is checked against the sha256 we used. Needs pyarrow.
"""
import hashlib, os, random, sys, urllib.request

CORPORA = ("AtomicChat/calib-corpora", "c081aab32814f7734fd4206f0d2ceb76d6ceda89")
WIKI = ("wikimedia/wikipedia", "b04c8d1ceb2f5cd4588862100d08de323dccfbaa")
FILES = [
    (CORPORA, "builds/muse-glimmer-30b/calib_train.txt", "calib_train.txt", "b1d378c5244debfe8aec4bb8c51ed1966504d9e7c8d4c6b7f825f04b5eb07f71"),
    (CORPORA, "eval/agentic/eval_agentic.txt", "eval_agentic.txt", "b6fe570299d8e0efcccee5eab7b3b42f0f6fcb680aab71d0e4134883696820f9"),
    (CORPORA, "eval/neutral/eval_neutral.txt", "eval_neutral.txt", "1b91967fa31d3c1fcf89c4212654662bbd983c7cea7f8be75405072c92da978e"),
    (WIKI, "20231101.sv/train-00002-of-00005.parquet", "sv_wiki.parquet", "96a335b319c6a7cbaef9f2be7004038047f7e385fd93d6a06da6f3fa90816cb5"),
]
SWEDISH_SHA = "931875aa09c62c317121d20a1a3e5684ee365760347eb0b855ec85006a8b056c"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check(path, want):
    got = sha256(path)
    print(f"{'OK      ' if got == want else 'MISMATCH'} {os.path.basename(path)} {got}")
    return got == want


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "corpus"
    os.makedirs(out, exist_ok=True)
    ok = True
    for (repo, rev), src, name, want in FILES:
        path = os.path.join(out, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{src}", path)
        ok &= check(path, want)

    import pyarrow.parquet as pq
    texts = pq.read_table(os.path.join(out, "sv_wiki.parquet"), columns=["text"]).column("text").to_pylist()
    random.Random(7).shuffle(texts)
    picked, n = [], 0
    for t in texts:
        if len(t) < 2000:
            continue
        picked.append(t.strip())
        n += len(t)
        if n > 1_600_000:
            break
    path = os.path.join(out, "eval_swedish.txt")
    with open(path, "w") as f:
        f.write("\n\n".join(picked))
    ok &= check(path, SWEDISH_SHA)
    print("CORPUS_OK" if ok else "CORPUS_MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
