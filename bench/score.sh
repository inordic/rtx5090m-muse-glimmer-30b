#!/bin/bash
# score.sh LABEL GGUF [WAIT_FILE WAIT_PATTERN]: KLD vs our Q8_0 on the three texts + llama-bench speed
lab=$1; f=$2; [ -n "$3" ] && while ! grep -q "$4" "$3" 2>/dev/null; do sleep 20; done
BIN=/opt/src/llama.cpp-master/build/bin; K=/opt/models/muse-glimmer/kld; C=/opt/models/muse-glimmer/corpus
echo "=== $(date -u +%T) $lab"
$BIN/llama-bench -m $f -ngl 999 -fa 1 -ub 1024 -p 512,4096 -n 128 -r 2 -o csv 2> /dev/null | python3 -c "
import csv,sys
for r in csv.DictReader(sys.stdin):
    t='pp'+r['n_prompt'] if r['n_gen']=='0' else 'tg'+r['n_gen']
    print('  BENCH', '$lab', t, round(float(r['avg_ts']),1), 'tok/s', 'size', round(int(r['model_size'])/1e9,2), 'GB')"
for corp in agentic neutral swedish; do
  $BIN/llama-perplexity -m $f -f $C/eval_$corp.txt -c 2048 --chunks 20 -fa on -b 2048 -ub 1024 -ngl 999 --kl-divergence-base $K/base_$corp.bin --kl-divergence > $K/$lab-$corp.log 2>&1
  echo "  KLD $lab $corp: $(grep -E "^Mean +KLD" $K/$lab-$corp.log | tr -s " " | cut -c1-30) | $(grep -E "^Same top p" $K/$lab-$corp.log | tr -s " " | cut -c1-30) | $(grep -E "^Mean +PPL\(Q\) " $K/$lab-$corp.log | tr -s " " | cut -c1-28)"
done
echo "=== $(date -u +%T) SCORE_DONE $lab"
