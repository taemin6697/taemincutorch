# InternVL3 Vision Encoder — XNNPACK CPU Export

InternVL3-1B 비전 인코더(InternViT-300M + MLP projector)를 XNNPACK CPU용으로 별도 export하는 방법입니다.

---

## 전제 조건

- **가상환경**: `stream` conda 환경 사용
- ExecuTorch 설치 완료 (`how_to_install.md` Step 1~4)
- `transformers`, `einops` 설치

```bash
conda activate stream
# 또는
conda run -n stream python ...
```

---

## 비전 인코더 개요

| 항목 | 내용 |
|------|------|
| 모델 | InternViT-300M-448px-V2_5 |
| 입력 | 448×448 RGB (1, 3, 448, 448) |
| 출력 | 시각 토큰 (256개, LLM embed_dim) |
| 백엔드 | XNNPACK (CPU) |
| 양자화 | FP32 또는 8-bit PTQ (QNN 16a8w와 유사) |

---

## Step 1. Export

**모듈 위치:** `executorch/examples/models/internvl3/vision_encoder/` (export_xnnpack.py, model.py)

### FP32 (기본)

```bash
cd /workspace/stream/executorch

conda activate stream

# HuggingFace에서 로드 (기본)
python internvl3_test/export_vision_encoder_xnnpack.py

# 또는 examples/models 경로에서 직접 실행
python -m executorch.examples.models.internvl3.vision_encoder.export_xnnpack

# 로컬 InternVL3-1B 사용 시
python internvl3_test/export_vision_encoder_xnnpack.py \
  --model_path internvl3_test/InternVL3-1B

# 출력 경로 지정
python internvl3_test/export_vision_encoder_xnnpack.py \
  --output /workspace/stream/executorch/internvl3_test/vision_encoder_xnnpack.pte
```

**출력:** `vision_encoder_xnnpack.pte` (~1.2 GB)

### 8-bit PTQ (양자화, 권장)

QNN 비전 인코더와 동일하게 8-bit 양자화. 파일 크기 ~4배 감소, 모바일 CPU에서 더 빠른 추론.

```bash
cd /workspace/stream/executorch
conda activate stream

# 양자화 export (기본 COCO 이미지로 calibration)
python internvl3_test/export_vision_encoder_xnnpack.py --quantize

# 출력 경로 지정 (자동으로 _q8 접미사)
python internvl3_test/export_vision_encoder_xnnpack.py \
  --quantize --output /workspace/stream/executorch/internvl3_test/vision_encoder_xnnpack_q8.pte

# Calibration 이미지 직접 지정
python internvl3_test/export_vision_encoder_xnnpack.py \
  --quantize \
  --calibration_images http://images.cocodataset.org/val2017/000000039769.jpg path/to/img.jpg
```

**출력:** `vision_encoder_xnnpack_q8.pte` (~312 MB)

---

## Step 2. 실행

### 2-A. PC/개발환경 (Python)

```bash
cd /workspace/stream/executorch
conda activate stream

# 기본 이미지(COCO URL)로 실행 (FP32 PTE)
python internvl3_test/run_vision_encoder_xnnpack.py

# 양자화 PTE 지정
python internvl3_test/run_vision_encoder_xnnpack.py \
  --pte /workspace/stream/executorch/internvl3_test/vision_encoder_xnnpack_q8.pte

# 로컬 이미지 지정
python internvl3_test/run_vision_encoder_xnnpack.py --image path/to/image.jpg

# 시간 측정 (warmup, iters)
python internvl3_test/run_vision_encoder_xnnpack.py --warmup 3 --iters 10
```

**출력 예시:** `shape: (1, 256, 896)` — 256개 시각 토큰, embed_dim 896

### 2-B. 휴대폰 (adb push → executor_runner)

```bash
cd /workspace/stream/executorch

# adb 디바이스 연결 후
python internvl3_test/run_vision_encoder_android.py \
  -s R3KYC01FW1P \
  -b cmake-out-android \
  --pte /workspace/stream/executorch/internvl3_test/vision_encoder_xnnpack_q8.pte \
  --warmup 2 --iters 5
```

**사전 요구:** `how_to_internvl3.md` Step 5.1으로 `cmake-out-android` 빌드 완료.

**출력 예시 (Galaxy S25, Snapdragon 8 Elite):** `forward (XNNPACK): ~2580 ms` (FP32), ~2240 ms (Q8)

---

## 참고

- **QNN 버전**: `how_to_install.md` Step 6의 `llama.py`는 비전 인코더를 Qualcomm QNN(HTP) 16a8w로 양자화 → `vision_encoder_qnn.pte`
- **XNNPACK FP32**: `vision_encoder_xnnpack.pte` (~1.2 GB)
- **XNNPACK Q8**: `vision_encoder_xnnpack_q8.pte` (~312 MB) — `--quantize` 사용
- QNN PTE와 XNNPACK PTE는 호환되지 않음 (백엔드 상이)
