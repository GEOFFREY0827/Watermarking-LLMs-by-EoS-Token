#!/usr/bin/env bash
# EOS ablation: does anchoring at the stop decision / reading out EOS matter?
# Four variants share the same key, responses and pairs; only these two choices change:
#   stop-eos      A's state at its stop position, EOS log-odds readout   (the method)
#   random-eos    A's state at a random response token, EOS readout
#   stop-token    stop position, readout = a key-chosen ordinary token
#   random-token  neither (no EOS anywhere)
# Each variant runs the full pipeline on its own GPU (waves of #GPUs if fewer than 4).
#
#   KEY="my secret" GPUS=0,1,2,3 bash scripts/run_ablation.sh
#   ATTACK_DATA=hf:tatsu-lab/alpaca ATTACK_STEPS=1000 KEY=... bash scripts/run_ablation.sh
#
# Disk: every variant saves one watermarked model (~3 GB for 1.5B), i.e. ~12 GB at peak.
#   ROOT=/path/on/a/big/disk/ablation   put all outputs elsewhere
#   CLEANUP=1                           delete each variant's watermarked model when it finishes
#   REUSE=1 + a symlink stop-eos -> an existing run of the same key reuses that model
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
KEY=${KEY:-change-this-secret-key}
GPUS=${GPUS:-0,1,2,3}
ROOT=${ROOT:-runs/ablation-$(basename "$MODEL")}
ABL_JOBS=${ABL_JOBS:-"quant4 sft-lora@0 sft-lora@1 sft-lora@2 sft-full@0 concise-lora@0 eosmask-lora@0 null-sibling@0"}
IFS=',' read -ra G <<< "$GPUS"
VARIANTS=("stop eos" "random eos" "stop token" "random token")
mkdir -p "$ROOT"

declare -A PID2V
running=()
for i in "${!VARIANTS[@]}"; do
  read -r pos ro <<< "${VARIANTS[$i]}"
  v="$pos-$ro"
  gpu=${G[$((i % ${#G[@]}))]}
  if (( ${#running[@]} >= ${#G[@]} )); then   # one variant per GPU at a time
    wait "${running[0]}" || true
    running=("${running[@]:1}")
  fi
  echo "variant $v on GPU $gpu -> $ROOT/$v (log: $ROOT/$v.log)"
  ( MODEL="$MODEL" KEY="$KEY" GPUS="$gpu" OUT="$ROOT/$v" \
    ANCHOR_ARGS="--position $pos --readout $ro ${ANCHOR_ARGS:-}" JOBS="$ABL_JOBS" \
    bash scripts/run_pipeline.sh ) > "$ROOT/$v.log" 2>&1 &
  PID2V[$!]=$v
  running+=($!)
done

failed=()
for pid in "${!PID2V[@]}"; do wait "$pid" || failed+=("${PID2V[$pid]}"); done
if (( ${#failed[@]} )); then echo "FAILED variants: ${failed[*]} (see $ROOT/<variant>.log and $ROOT/<variant>/logs/)"; fi

python scripts/summarize_ablation.py "$ROOT"
