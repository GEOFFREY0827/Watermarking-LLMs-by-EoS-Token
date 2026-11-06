#!/usr/bin/env bash
# Full experiment. Attacks run in parallel, one job per GPU.
#   KEY="my secret" GPUS=0,1,2,3 bash scripts/run_pipeline.sh
#   MODEL=Qwen/Qwen2.5-3B-Instruct OUT=runs/qwen3b GPUS=0,1 bash scripts/run_pipeline.sh
#   # a model too large for one card: shard it, jobs then run one after another on all GPUs
#   MODEL=Qwen/Qwen2.5-7B-Instruct DEVICE_MAP=auto GPUS=0,1 bash scripts/run_pipeline.sh
#   # keep the existing anchors / watermarked model / reference responses, rerun only the attacks
#   REUSE=1 KEY="my secret" GPUS=0,1,2,3 bash scripts/run_pipeline.sh
#   # attacker fine-tunes on a real instruction dataset instead of the built-in prompts
#   ATTACK_DATA=hf:tatsu-lab/alpaca bash scripts/run_pipeline.sh
# Attacked models are evaluated in memory and not written to disk.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
KEY=${KEY:-change-this-secret-key}
OUT=${OUT:-runs/$(basename "$MODEL")}
NBITS=${NBITS:-64}
STEPS=${STEPS:-400}
ATTACK_STEPS=${ATTACK_STEPS:-300}
GPUS=${GPUS:-0}
DEVICE_MAP=${DEVICE_MAP:-}
REUSE=${REUSE:-0}
ATTACK_DATA=${ATTACK_DATA:-}
R=$OUT/results.jsonl
A=$OUT/anchors.pt
Q=$OUT/ref.pt
mkdir -p "$OUT/logs"
if [[ -f "$R" ]]; then mv "$R" "$OUT/results.$(date +%Y%m%d-%H%M%S).jsonl"; fi   # start a fresh results file
if [[ "$REUSE" != 1 ]]; then rm -f "$Q"; fi   # reference responses depend on the key's held-out prompts

IFS=',' read -ra G <<< "$GPUS"
EXTRA=()
if [[ -n "$DEVICE_MAP" ]]; then
  EXTRA=(--device-map "$DEVICE_MAP"); SLOTS=("$GPUS")   # every job uses all GPUs, run sequentially
else
  SLOTS=("${G[@]}")
fi
run() { echo -e "\n>>> [GPU $CUDA_VISIBLE_DEVICES] $*"; python -m eoswm.cli "$@" "${EXTRA[@]}"; }

# ---- sequential part on the first slot -------------------------------------------------------
export CUDA_VISIBLE_DEVICES=${SLOTS[0]}
reuse() { [[ "$REUSE" == 1 && -e "$1" ]] && echo "reusing $1"; }
# ANCHOR_ARGS / EMBED_ARGS: extra flags, e.g. EMBED_ARGS="--lam-eos 5 --layer 16"
# shellcheck disable=SC2086
reuse "$A" || run build-anchors --model "$MODEL" --key "$KEY" --n-bits "$NBITS" --out "$A" ${ANCHOR_ARGS:-}
run verify --model "$MODEL" --anchors "$A" --name base --results "$R"
if reuse "$OUT/wm"; then
  [[ -f "$OUT/wm/eoswm_tokenizer_source.txt" ]] || echo "$MODEL" > "$OUT/wm/eoswm_tokenizer_source.txt"
  run verify --model "$OUT/wm" --anchors "$A" --name watermarked --results "$R"
else
  # shellcheck disable=SC2086
  run embed --model "$MODEL" --anchors "$A" --out "$OUT/wm" --steps "$STEPS" --results "$R" ${EMBED_ARGS:-}
fi
run quality --model "$MODEL"  --anchors "$A" --ref-file "$Q" --name base        --results "$R"   # writes $Q once
run quality --model "$OUT/wm" --anchors "$A" --ref-file "$Q" --name watermarked --results "$R"

# ---- attacks, parallel over slots (each also reports quality, in memory) ---------------------
DATA=(); [[ -n "$ATTACK_DATA" ]] && DATA=(--data "$ATTACK_DATA")
job() {   # name[@seed], e.g. sft-lora@1
  local name=$1 base=${1%@*} seed=0
  [[ "$1" == *@* ]] && seed=${1#*@}
  case $base in
    rotate|quant8|quant4)
      run attack --model "$OUT/wm" --anchors "$A" --kind "$base" --name "attack:$name" \
          --quality-ref "$Q" --results "$R" ;;
    sft-lora|sft-full|concise-lora|eosmask-lora)
      local kind=${base%-*} mode=${base##*-}
      run attack --model "$OUT/wm" --anchors "$A" --kind "$kind" --mode "$mode" --steps "$ATTACK_STEPS" \
          --seed "$seed" "${DATA[@]}" --name "attack:$name" --quality-ref "$Q" --results "$R" ;;
    null-sibling)   # base + the same fine-tuning: must NOT verify; also the benign-fine-tune quality reference
      run attack --model "$MODEL" --anchors "$A" --kind sft --mode lora --steps "$ATTACK_STEPS" \
          --seed "$seed" "${DATA[@]}" --name "null:sibling-sft${name#null-sibling}" --quality-ref "$Q" --results "$R" ;;
    *) echo "unknown job $name"; return 1 ;;
  esac
}

# JOBS="sft-lora@0 sft-lora@1 quant4" overrides the default list
if [[ -n "${JOBS:-}" ]]; then read -ra JOBS <<< "$JOBS"
else JOBS=(rotate quant8 quant4 sft-lora sft-full concise-lora eosmask-lora null-sibling); fi
declare -A SLOTPID PID2JOB
free_slot() {   # first slot whose job has finished (or never started); blocks until one is free
  while true; do
    for s in "${SLOTS[@]}"; do
      local p=${SLOTPID[$s]:-}
      if [[ -z "$p" ]] || ! kill -0 "$p" 2>/dev/null; then echo "$s"; return; fi
    done
    sleep 5
  done
}
for name in "${JOBS[@]}"; do
  slot=$(free_slot)
  ( export CUDA_VISIBLE_DEVICES=$slot; job "$name" ) > "$OUT/logs/$name.log" 2>&1 &
  SLOTPID[$slot]=$!
  PID2JOB[$!]=$name
  echo "started $name on GPU $slot (log: $OUT/logs/$name.log)"
done

failed=()
for pid in "${!PID2JOB[@]}"; do
  wait "$pid" || failed+=("${PID2JOB[$pid]}")
done
if (( ${#failed[@]} )); then echo "FAILED jobs: ${failed[*]} (see $OUT/logs/)"; fi

python scripts/summarize.py "$R"
if [[ "${CLEANUP:-0}" == 1 ]]; then rm -rf "$OUT/wm"; echo "removed $OUT/wm (CLEANUP=1)"; fi
