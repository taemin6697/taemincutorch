#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/stream}"
FOUNDATION_MODULE="executorch.examples.models.foundation.cli"

BACKEND_MODE="${1:-all}"
if [[ -n "${EXPORT_LENGTHS:-}" ]]; then
  read -r -a LENGTHS <<< "${EXPORT_LENGTHS}"
else
  LENGTHS=(1024 2048 4096 8192 16384 32768)
fi
if [[ -n "${EXPORT_MODELS:-}" ]]; then
  read -r -a DECODER_MODELS <<< "${EXPORT_MODELS}"
elif [[ -n "${DECODER_MODEL:-}" ]]; then
  DECODER_MODELS=("${DECODER_MODEL}")
else
  DECODER_MODELS=(internvl3_8b)
fi

QNN_ARTIFACT_BASE="${QNN_ARTIFACT_BASE:-${ROOT_DIR}/my_save/save_model/qnn}"
XNNPACK_ARTIFACT_BASE="${XNNPACK_ARTIFACT_BASE:-${ROOT_DIR}/my_save/save_model/cpu}"

QNN_BUILD_PATH="${QNN_BUILD_PATH:-${ROOT_DIR}/executorch/build-android}"
QNN_DEVICE="${QNN_DEVICE:-R3KYC01FW1P}"
QNN_SOC_MODEL="${QNN_SOC_MODEL:-SM8750}"
QNN_MODEL_MODE="${QNN_MODEL_MODE:-hybrid}"
QNN_PREFILL_AR_LEN="${QNN_PREFILL_AR_LEN:-16}"
QNN_PROMPT="${QNN_PROMPT:-Can you describe this image?}"
QNN_IMAGE_PATH="${QNN_IMAGE_PATH:-http://images.cocodataset.org/val2017/000000039769.jpg}"

XNNPACK_MODEL_PATH="${XNNPACK_MODEL_PATH:-}"
XNNPACK_CHECKPOINT="${XNNPACK_CHECKPOINT:-}"
XNNPACK_PARAMS="${XNNPACK_PARAMS:-}"
XNNPACK_DTYPE="${XNNPACK_DTYPE:-fp16}"
XNNPACK_VISION_QUANT="${XNNPACK_VISION_QUANT:-fp16}"
XNNPACK_DECODER_QUANT="${XNNPACK_DECODER_QUANT:-fp16}"
XNNPACK_EMBEDDING_QUANT="${XNNPACK_EMBEDDING_QUANT:-fp16}"

case "${BACKEND_MODE}" in
  all|qnn|xnnpack)
    ;;
  *)
    echo "Usage: $0 [all|qnn|xnnpack]" >&2
    exit 1
    ;;
esac

length_tag() {
  local length="$1"
  if (( length % 1024 == 0 )); then
    echo "$((length / 1024))k"
  else
    echo "${length}"
  fi
}

model_size_tag() {
  local decoder_model="$1"
  echo "${decoder_model#internvl3_}"
}

run_qnn_export() {
  local decoder_model="$1"
  local model_size
  model_size="$(model_size_tag "${decoder_model}")"
  local length="$2"
  local tag
  tag="$(length_tag "${length}")"
  local artifact_root="${QNN_ARTIFACT_BASE}/internvl3_${model_size}_${QNN_MODEL_MODE}_${QNN_PREFILL_AR_LEN}p_${tag}"

  echo "==> QNN export: ${decoder_model}, ${length} (${artifact_root})"
  python -m "${FOUNDATION_MODULE}" export \
    --backend qnn \
    --artifact_root "${artifact_root}" \
    --decoder_model "${decoder_model}" \
    --build_path "${QNN_BUILD_PATH}" \
    --device "${QNN_DEVICE}" \
    --model "${QNN_SOC_MODEL}" \
    --model_mode "${QNN_MODEL_MODE}" \
    --prefill_ar_len "${QNN_PREFILL_AR_LEN}" \
    --max_seq_len "${length}" \
    --max_context_len "${length}" \
    --dtype fp32 \
    --vision_quant fp16 \
    --decoder_quant fp16 \
    --embedding_quant fp16 \
    --prompts "${QNN_PROMPT}" \
    --image_path "${QNN_IMAGE_PATH}"
}

run_xnnpack_export() {
  local decoder_model="$1"
  local model_size
  model_size="$(model_size_tag "${decoder_model}")"
  local length="$2"
  local tag
  tag="$(length_tag "${length}")"
  local artifact_root="${XNNPACK_ARTIFACT_BASE}/internvl3_xnnpack_${model_size}_${tag}"
  local -a cmd=(
    python -m "${FOUNDATION_MODULE}" export
    --backend xnnpack
    --artifact_root "${artifact_root}"
    --decoder_model "${decoder_model}"
    --max_seq_len "${length}"
    --max_context_len "${length}"
    --dtype "${XNNPACK_DTYPE}"
    --vision_quant "${XNNPACK_VISION_QUANT}"
    --decoder_quant "${XNNPACK_DECODER_QUANT}"
    --embedding_quant "${XNNPACK_EMBEDDING_QUANT}"
  )

  if [[ -n "${XNNPACK_MODEL_PATH}" ]]; then
    cmd+=(--model_path "${XNNPACK_MODEL_PATH}")
  fi
  if [[ -n "${XNNPACK_CHECKPOINT}" ]]; then
    cmd+=(--checkpoint "${XNNPACK_CHECKPOINT}")
  fi
  if [[ -n "${XNNPACK_PARAMS}" ]]; then
    cmd+=(--params "${XNNPACK_PARAMS}")
  fi

  echo "==> XNNPACK export: ${decoder_model}, ${length} (${artifact_root})"
  "${cmd[@]}"
}

cd "${ROOT_DIR}"

for decoder_model in "${DECODER_MODELS[@]}"; do
  for length in "${LENGTHS[@]}"; do
    if [[ "${BACKEND_MODE}" == "all" || "${BACKEND_MODE}" == "qnn" ]]; then
      run_qnn_export "${decoder_model}" "${length}"
    fi
    if [[ "${BACKEND_MODE}" == "all" || "${BACKEND_MODE}" == "xnnpack" ]]; then
      run_xnnpack_export "${decoder_model}" "${length}"
    fi
  done
done
