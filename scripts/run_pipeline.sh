#!/usr/bin/env bash
# Full experiment on one GPU (defaults sized for an 8 GB RTX 4060).
#   KEY="my secret" bash scripts/run_pipeline.sh
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct OUT=runs/qwen1.5b bash scripts/run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
KEY=${KEY:-change-this-secret-key}
OUT=${OUT:-runs/qwen2.5-0.5b}
NBITS=${NBITS:-64}
STEPS=${STEPS:-400}
ATTACK_STEPS=${ATTACK_STEPS:-300}
R=$OUT/results.jsonl
A=$OUT/anchors.pt
mkdir -p "$OUT"
run() { echo -e "\n>>> $*"; python -m eoswm.cli "$@"; }

# 1. secret anchors from the key (with the base model)
run build-anchors --model "$MODEL" --key "$KEY" --n-bits "$NBITS" --out "$A"

# 2. null: the unmodified base model must not carry the watermark
run verify --model "$MODEL" --anchors "$A" --name base --results "$R"

# 3. embed and verify
run embed --model "$MODEL" --anchors "$A" --out "$OUT/wm" --steps "$STEPS" --results "$R"

# 4. utility / termination behaviour (first call writes base reference responses)
run quality --model "$MODEL"   --anchors "$A" --ref-file "$OUT/ref.pt" --name base        --results "$R"
run quality --model "$OUT/wm"  --anchors "$A" --ref-file "$OUT/ref.pt" --name watermarked --results "$R"

# 5. attacks on the watermarked model
run attack --model "$OUT/wm" --anchors "$A" --kind rotate --name attack:rotate --results "$R"
run attack --model "$OUT/wm" --anchors "$A" --kind quant8 --name attack:quant8 --results "$R"
run attack --model "$OUT/wm" --anchors "$A" --kind quant4 --name attack:quant4 --results "$R"
for kind in sft concise eosmask; do
  run attack --model "$OUT/wm" --anchors "$A" --kind "$kind" --steps "$ATTACK_STEPS" \
      --name "attack:$kind" --out "$OUT/attack_$kind" --results "$R"
  run quality --model "$OUT/attack_$kind" --anchors "$A" --ref-file "$OUT/ref.pt" --name "attack:$kind" --results "$R"
done

# 6. null: an independent sibling (base + the same fine-tuning) must not verify
run attack --model "$MODEL" --anchors "$A" --kind sft --steps "$ATTACK_STEPS" --name null:sibling-sft --results "$R"

python scripts/summarize.py "$R"
