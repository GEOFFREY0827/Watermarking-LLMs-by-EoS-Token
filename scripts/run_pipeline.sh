#!/usr/bin/env bash
# Full experiment. Attacks run in parallel, one job per GPU.
#   KEY="my secret" GPUS=0,1,2,3 bash scripts/run_pipeline.sh
#   MODEL=Qwen/Qwen2.5-3B-Instruct OUT=runs/qwen3b GPUS=0,1 bash scripts/run_pipeline.sh
#   # a model too large for one card: shard it, jobs then run one after another on all GPUs
#   MODEL=Qwen/Qwen2.5-7B-Instruct DEVICE_MAP=auto GPUS=0,1 bash scripts/run_pipeline.sh
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
R=$OUT/results.jsonl
A=$OUT/anchors.pt
mkdir -p "$OUT/logs"

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
# ANCHOR_ARGS / EMBED_ARGS: extra flags, e.g. EMBED_ARGS="--lam-eos 5 --layer 16"
# shellcheck disable=SC2086
run build-anchors --model "$MODEL" --key "$KEY" --n-bits "$NBITS" --out "$A" ${ANCHOR_ARGS:-}
run verify  --model "$MODEL"  --anchors "$A" --name base --results "$R"
# shellcheck disable=SC2086
run embed   --model "$MODEL"  --anchors "$A" --out "$OUT/wm" --steps "$STEPS" --results "$R" ${EMBED_ARGS:-}
run quality --model "$MODEL"  --anchors "$A" --ref-file "$OUT/ref.pt" --name base        --results "$R"
run quality --model "$OUT/wm" --anchors "$A" --ref-file "$OUT/ref.pt" --name watermarked --results "$R"

# ---- attacks, parallel over slots -------------------------------------------------------------
job() {
  local name=$1
  case $name in
    rotate|quant8|quant4)
      run attack --model "$OUT/wm" --anchors "$A" --kind "$name" --name "attack:$name" --results "$R" ;;
    sft-lora|sft-full|concise-lora|eosmask-lora)
      local kind=${name%-*} mode=${name##*-}
      run attack --model "$OUT/wm" --anchors "$A" --kind "$kind" --mode "$mode" --steps "$ATTACK_STEPS" \
          --name "attack:$name" --out "$OUT/attack_$name" --results "$R"
      run quality --model "$OUT/attack_$name" --anchors "$A" --ref-file "$OUT/ref.pt" --name "attack:$name" --results "$R" ;;
    null-sibling)   # base + the same fine-tuning: must NOT verify
      run attack --model "$MODEL" --anchors "$A" --kind sft --mode lora --steps "$ATTACK_STEPS" \
          --name "null:sibling-sft" --results "$R" ;;
  esac
}

JOBS=(rotate quant8 quant4 sft-lora sft-full concise-lora eosmask-lora null-sibling)
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
