#!/usr/bin/env bash

set -u

ROOT_DIR="${ROOT_DIR:-/workspace/stream/executorch}"
BUILD_DIR="${BUILD_DIR:-build-android}"
SERIAL="${SERIAL:-R3KYC01FW1P}"
SOC_MODEL="${SOC_MODEL:-SM8750}"
# 모델: internvl3_2b, internvl3_8b (공백 구분, 둘 다 실행)
DECODER_MODELS=(${DECODER_MODELS:-internvl3_2b internvl3_8b})
MODEL_MODE="${MODEL_MODE:-hybrid}"
PREFILL_AR_LEN="${PREFILL_AR_LEN:-16}"
PROMPT="${PROMPT:-Can you describe this image?}"
IMAGE_PATH="${IMAGE_PATH:-http://images.cocodataset.org/val2017/000000039769.jpg}"
ARTIFACT_BASE="${ARTIFACT_BASE:-../my_save/save_model}"

SEQ_LENS=(512 1024 2048 4096 8192 16384)

cd "${ROOT_DIR}" || exit 1

overall_failed=0

for decoder_model in "${DECODER_MODELS[@]}"; do
  model_artifact_base="${ARTIFACT_BASE}/${decoder_model}_hybrid_16p"
  echo
  echo "########## DECODER MODEL: ${decoder_model} ##########"
  for seq_len in "${SEQ_LENS[@]}"; do
    if [[ "${seq_len}" -lt 1024 ]]; then
      seq_label="${seq_len}"
    else
      seq_label="$((seq_len / 1024))k"
    fi
    artifact_path="${model_artifact_base}_${seq_label}"

    echo
    echo "============================================================"
    echo "[START] ${decoder_model} max_seq_len=${seq_len} artifact=${artifact_path}"
    echo "============================================================"

    if python examples/qualcomm/oss_scripts/llama/llama.py \
      -b "${BUILD_DIR}" \
      -s "${SERIAL}" \
      -m "${SOC_MODEL}" \
      --decoder_model "${decoder_model}" \
      --model_mode "${MODEL_MODE}" \
      --prefill_ar_len "${PREFILL_AR_LEN}" \
      --max_seq_len "${seq_len}" \
      --artifact "${artifact_path}" \
      --prompt "${PROMPT}" \
      --image_path "${IMAGE_PATH}"; then
      echo "[OK] ${decoder_model} ${seq_label} completed"
    else
      status=$?
      overall_failed=1
      echo "[FAIL] ${decoder_model} ${seq_label} failed with exit code ${status}"
      echo "[CONTINUE] moving to next max_seq_len"
    fi
  done
done

echo
if [[ "${overall_failed}" -eq 0 ]]; then
  echo "All runs completed successfully."
else
  echo "Sequence sweep finished with one or more failures."
fi
