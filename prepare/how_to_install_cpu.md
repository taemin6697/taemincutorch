# InternVL3 XNNPACK 멀티모달 Android 실행 가이드

`InternVL3`를 Android CPU에서 `XNNPACK`으로 돌리는 전체 멀티모달 경로를 정리합니다.

범위:

- 단일 이미지 + 질문/응답
- 배치형 비디오 + 질문/응답
- 스트리밍 모드는 제외

저장 규칙:

- 모델 가중치, Meta checkpoint, CPU/XNNPACK 산출물은 `/workspace/stream/my_save/save_model` 아래에 저장
- CPU 전용 산출물 이름은 `_cpu` suffix 사용

관련 스크립트:

- export: `executorch/examples/models/internvl3/export_xnnpack_multimodal.py`
- Android runner: `executorch/examples/models/internvl3/internvl3_xnnpack_runner.cpp`
- host 실행 스크립트: `executorch/examples/models/internvl3/run_xnnpack_vlm.py`

---

## 준비물

- `ExecuTorch` 소스 및 Python 환경 설치 완료
- Android NDK 설치
- `adb` 연결 가능한 Android 기기
- `InternVL3` Hugging Face 체크포인트 또는 로컬 모델 디렉터리
- 선택: `opencv-python` (`--video` 사용 시 필요)

권장 환경 변수:

```bash
export ANDROID_NDK_ROOT=/opt/android-ndk-r26c  # 또는 bashrc에 이미 설정된 경우 생략
```

---

## 전체 흐름

1. `InternVL3` 텍스트 디코더를 Meta 포맷 checkpoint로 준비
2. `vision_encoder + token_embedding + text_decoder`를 export
   - 기본: 3개 PTE 분리 (`vision_encoder_xnnpack.pte`, `text_embedding_xnnpack.pte`, `text_decoder_xnnpack.pte`)
   - `--single_pte`: 기존 runner용 단일 multi-method `.pte`
3. Android용 `internvl3_xnnpack_runner` 빌드
4. 호스트 스크립트로 이미지/비디오를 전처리하고 `adb`로 runner 실행

---

## Step 1. 텍스트 체크포인트 준비

먼저 원본 `InternVL3` 모델 디렉터리를 아래 경로에 준비합니다.

권장 경로:

- `/workspace/stream/my_save/save_model/InternVL3-1B-hf`

아직 없다면 먼저 다운로드:

```bash
mkdir -p /workspace/stream/my_save/save_model/InternVL3-1B-hf

huggingface-cli download OpenGVLab/InternVL3-1B-hf \
  --local-dir /workspace/stream/my_save/save_model/InternVL3-1B-hf
```

이미 Meta 포맷 checkpoint가 있으면 아래 변환 단계는 건너뛰어도 됩니다.

예:

```bash
cd /workspace/stream/executorch

python -m executorch.examples.models.internvl3.convert_weights \
  /workspace/stream/my_save/InternVL3-1B-hf \
  /workspace/stream/my_save/save_model/internvl3_1b_meta_cpu.pth
```

출력:

- `/workspace/stream/my_save/save_model/internvl3_1b_meta_cpu.pth`

참고:

- 첫 번째 인자는 실제 `InternVL3` 로컬 모델 디렉터리여야 합니다.
- `--checkpoint`를 export 스크립트에 주지 않으면, 로컬 `InternVL3` 모델 디렉터리에서 임시 변환을 수행합니다.
- 원격 Hugging Face ID만 주는 경우에는 자동 변환이 안 되므로, 먼저 로컬에 받아두는 편이 안전합니다.

---

## Step 2. XNNPACK 멀티모달 PTE export

### 기본 예시 (fp16, 양자화 없음)

**3개 PTE 분리 (기본, QNN 정렬):**
```bash
cd /workspace/stream/executorch

python -m executorch.examples.models.internvl3.export_xnnpack_multimodal \
  --decoder_model internvl3_1b \
  --model_path /workspace/stream/my_save/save_model/meta/InternVL3-1B-hf \
  --checkpoint /workspace/stream/my_save/save_model/meta/internvl3_1b_meta_cpu.pth \
  --output /workspace/stream/my_save/save_model/cpu/internvl3_xnnpack \
  --max_seq_len 2048 \
  --max_context_len 2048
```

**단일 PTE (기존 `internvl3_xnnpack_runner` 호환):**
```bash
python -m executorch.examples.models.internvl3.export_xnnpack_multimodal \
  --decoder_model internvl3_1b \
  --model_path /workspace/stream/my_save/save_model/meta/InternVL3-1B-hf \
  --checkpoint /workspace/stream/my_save/save_model/meta/internvl3_1b_meta_cpu.pth \
  --output /workspace/stream/my_save/save_model/cpu/internvl3_1b_xnnpack_multimodal.pte \
  --single_pte \
  --max_seq_len 2048 \
  --max_context_len 2048
```

### 양자화 적용 예시 (8a8w vision, 8da4w decoder)

```bash
python -m executorch.examples.models.internvl3.export_xnnpack_multimodal \
  --decoder_model internvl3_1b \
  --model_path /workspace/stream/my_save/save_model/meta/InternVL3-1B-hf \
  --checkpoint /workspace/stream/my_save/save_model/meta/internvl3_1b_meta_cpu.pth \
  --output /workspace/stream/my_save/save_model/cpu/internvl3_xnnpack_quant \
  --max_seq_len 2048 \
  --max_context_len 2048 \
  --vision_quant 8a8w \
  --decoder_quant 8da4w \
  --embedding_quant 4,32
```

(단일 PTE가 필요하면 `--output xxx.pte --single_pte` 사용)

### 3프레임 이상 / save_log 사용 시 (2048 권장)

`--decode_after_frames 3` 이상 사용하려면 KV 캐시가 부족합니다. 2048로 export (runner용 단일 PTE):

```bash
python -m executorch.examples.models.internvl3.export_xnnpack_multimodal \
  --decoder_model internvl3_1b \
  --model_path /workspace/stream/my_save/save_model/meta/InternVL3-1B-hf \
  --checkpoint /workspace/stream/my_save/save_model/meta/internvl3_1b_meta_cpu.pth \
  --output /workspace/stream/my_save/save_model/cpu/internvl3_1b_xnnpack_multimodal_cpu.pte \
  --single_pte \
  --max_seq_len 2048 \
  --max_context_len 2048
```

**참고:** 프레임당 256 이미지 토큰. 2프레임 ≈ 512 + 텍스트 ≈ 786 → decode 시 256 추가 시 1042 > 1024. **1024 모델은 `--decode_after_frames 1`만 안전.**

### 주요 옵션

- `--decoder_model`: `internvl3_1b`, `internvl3_2b`, `internvl3_8b`
- `--model_path`: tokenizer와 vision encoder를 읽을 Hugging Face 모델 ID 또는 로컬 디렉터리
- `--checkpoint`: Meta 포맷 text decoder checkpoint
- `--params`: 없으면 repo 내 `1b_config.json` 등 기본값 사용
- `--output`: 출력 경로. 기본은 디렉터리(3개 PTE). `--single_pte` 시 `.pte` 파일
- `--single_pte`: 단일 multi-method `.pte`로 export (기존 runner 호환)
- `--max_seq_len`: decoder 최대 시퀀스 길이
- `--max_context_len`: decoder 최대 context 길이
- `--dtype`: 기본 `fp16` (fp16/fp32)
- `--vision_quant`: vision encoder - `fp16`(기본) 또는 `8a8w`
- `--decoder_quant`: text decoder - `fp16`(기본), `8da4w`, `8da8w`, `int8`
- `--embedding_quant`: token embedding - `fp16`(기본) 또는 `4,32`
- `--text_group_size`: decoder 양자화 시 group size (기본 `128`)
- `--calibration_images`: vision PTQ용 calibration 이미지

### 생성물

**3개 PTE (기본):**
- `internvl3_xnnpack/vision_encoder_xnnpack.pte`, `text_embedding_xnnpack.pte`, `text_decoder_xnnpack.pte`
- `internvl3_xnnpack/artifacts/tokenizer/tokenizer.json`
- `internvl3_xnnpack/artifacts/manifest.json`

**단일 PTE (`--single_pte`):**
- `internvl3_1b_xnnpack_multimodal_cpu.pte`
- `internvl3_1b_xnnpack_multimodal_cpu_artifacts/tokenizer/tokenizer.json`
- `internvl3_1b_xnnpack_multimodal_cpu_artifacts/manifest.json`

---

## Step 3. Android runner 빌드

### 빌드 요약 (Quick Build)

`internvl3_xnnpack_runner.cpp` 등 runner 코드를 수정했거나 처음 빌드할 때:

```bash
cd /workspace/stream/executorch

# 1) Root configure (최초 1회 또는 executorch 루트 변경 시)
cmake -S . -B cmake-out-android \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM=ON

# 2) InternVL3 runner configure (최초 1회 또는 examples/models/internvl3 변경 시)
cmake -S examples/models/internvl3 \
  -B cmake-out-android/examples/models/internvl3 \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM=ON

# 3) 빌드 (runner 코드 변경 시마다)
cmake --build cmake-out-android/examples/models/internvl3 \
  --target internvl3_xnnpack_runner -j
```

**재빌드가 필요한 경우:** `internvl3_xnnpack_runner.cpp` 수정, `--dump_input_path` 등 새 플래그 추가 후. 빌드 결과: `cmake-out-android/examples/models/internvl3/internvl3_xnnpack_runner`

---

### 3.1 Root configure (상세)

```bash
cd /workspace/stream/executorch

cmake -S . -B cmake-out-android \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM=ON
```

### 3.2 InternVL3 runner configure/build (상세)

```bash
cd /workspace/stream/executorch

cmake -S examples/models/internvl3 \
  -B cmake-out-android/examples/models/internvl3 \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM=ON

cmake --build cmake-out-android/examples/models/internvl3 \
  --target internvl3_xnnpack_runner -j
```

### 빌드 결과

예상 runner 경로:

```bash
/workspace/stream/executorch/cmake-out-android/examples/models/internvl3/internvl3_xnnpack_runner
```

---

## Step 4. 이미지 1장 실행

```bash
cd /workspace/stream/executorch

python -m executorch.examples.models.internvl3.run_xnnpack_vlm \
  --serial R3KYC01FW1P \
  --model_path /workspace/stream/my_save/save_model/cpu/internvl3_1b_xnnpack_multimodal_cpu.pte \
  --runner_binary /workspace/stream/executorch/cmake-out-android/examples/models/internvl3/internvl3_xnnpack_runner \
  --image "http://images.cocodataset.org/val2017/000000039769.jpg" \
  --question "Can you describe this image?" \
  --max_new_tokens 128 \
  --temperature 0.0
```

동작:

- host에서 Hugging Face processor로 이미지를 `frame_0000.bin`으로 전처리
- `adb push`로 `.pte`, tokenizer, runner, frame bin 전송
- Android에서 runner 실행
- 결과를 로컬 `internvl3_1b_xnnpack_output_cpu.txt`로 pull

**C++ runner 실제 입출력 확인:**

```bash
# 입력(토크나이저 기준 full prompt + input_ids) 덤프
python -m executorch.examples.models.internvl3.run_xnnpack_vlm \
  ... \
  --dump_input input_dump.txt
```

`input_dump.txt`에 full prompt 텍스트와 `input_ids` 토큰 시퀀스가 저장됨. 출력은 `--local_output` 또는 기본 `internvl3_xnnpack_output.txt`에 저장됨.

---

## Step 5. 비디오 배치형 실행

### 기본 실행

```bash
cd /workspace/stream/executorch

python -m executorch.examples.models.internvl3.run_xnnpack_vlm \
  -s R3KYC01FW1P \
  --model_path /workspace/stream/my_save/save_model/cpu/internvl3_1b_xnnpack_multimodal_cpu.pte \
  --runner_binary /workspace/stream/executorch/cmake-out-android/examples/models/internvl3/internvl3_xnnpack_runner \
  --video /workspace/stream/my_save/sample_video/sample.mp4 \
  --question "What happens in this video?" \
  --fps 1.0 \
  --max_new_tokens 128 \
  --temperature 0.0
```

설명:

- `--fps 1.0`이면 대략 1초당 1프레임 샘플링
- 추출된 프레임들이 `frame_0000.bin`, `frame_0001.bin` 식으로 저장됨
- runner는 프레임들을 순서대로 multimodal input에 넣어 batch prefill 후 답변 생성

### QNN stream_vlm 스타일 (save_log, 인자 유사)

QNN과 동일한 인자 구조로 실행하고 `my_save/save_log/` 하위에 proc.csv, mem.csv, tokens.csv 저장.

**주의:** 1024 모델이면 `--decode_after_frames 1`만 안전 (2프레임도 786+256 > 1024). 2048 모델이면 `--decode_after_frames 3` 가능.

```bash
cd /workspace/stream/executorch

python -m executorch.examples.models.internvl3.run_xnnpack_vlm \
  -s R3KYC01FW1P \
  --model_path /workspace/stream/my_save/save_model/cpu/internvl3_1b_xnnpack_multimodal_cpu.pte \
  --runner_binary /workspace/stream/executorch/cmake-out-android/examples/models/internvl3/internvl3_xnnpack_runner \
  --video /workspace/stream/my_save/sample_video/sample.mp4 \
  --questions "Describe this video" \
  --decode_after_frames 1 \
  --fps 1.0 \
  --seq_len 1024 \
  --eval_mode 1 \
  --max_new_tokens 128 \
  --temperature 0.0 \
  --save_log
```

저장 경로 예시:

```text
my_save/save_log/internvl3_1b_cpu_batch_seq1024_video_fps1p0_frames1_eval1_nolazy/
  stream_output.txt
  stream_output.txt.proc.csv
  stream_output.txt.mem.csv
  stream_output.txt.tokens.csv
```

---

## 자주 쓰는 옵션

### export 쪽

```bash
--decoder_model internvl3_1b
--max_seq_len 1024   # 1프레임만 안전
--max_context_len 1024
# 3프레임 이상: --max_seq_len 2048 --max_context_len 2048
--vision_quantize
--text_qmode 8da4w
--text_group_size 128
```

### run 쪽

```bash
-s, --serial DEVICE_SERIAL
--decode_after_frames 3
--seq_len 2048
--eval_mode 1
--save_log
--save_log_dir /path/to/base
--questions "질문1" "질문2"
--cpu_threads 4
--max_new_tokens 128
--temperature 0.0
--local_output /workspace/stream/my_save/save_model/internvl3_1b_result_cpu.txt
```

---

## 디렉터리 예시

```text
/workspace/stream/my_save/save_model/
  InternVL3-1B-hf/
  internvl3_1b_meta_cpu.pth
  internvl3_1b_xnnpack_multimodal_cpu.pte
  internvl3_1b_xnnpack_multimodal_cpu_artifacts/
    tokenizer/
      tokenizer.json
    manifest.json
```

---

## 트러블슈팅

### `필수 파일을 찾을 수 없습니다`

확인할 것:

- `--model_path`
- `--runner_binary`
- `tokenizer.json` 위치

기본 tokenizer 경로는 아래를 기대합니다.

```bash
<model_path stem>_artifacts/tokenizer/tokenizer.json
```

### `--video` 실행 시 OpenCV 에러

```bash
pip install opencv-python
```

### `adb` 관련 에러

```bash
adb devices
adb -s DEVICE_SERIAL shell true
```

### `start_pos + seq_length > cache size` (KV 캐시 오버플로우)

```
Check failed: start_pos + seq_length must be less than max seq length. cache size: 1024
```

**원인:** 모델이 1024로 export됐는데 3프레임(≈768 이미지 토큰) + 텍스트 + 생성이 1024를 초과.

**해결:**
- `--decode_after_frames 1`로 줄이기 (1024 모델 그대로 사용)
- 또는 모델을 `--max_context_len 2048`로 재export 후 2~3프레임 사용

### 메모리 부족 또는 너무 느림

먼저 줄여볼 것:

- export 시 `--max_seq_len 512`, `--max_context_len 512`
- run 시 `--decode_after_frames 1` (1024 모델)
- 비디오 `--fps 0.5`

### export 중 메모리 부담이 큼

가능하면:

- `internvl3_1b`부터 시작
- `--vision_quantize` 유지
- text 쪽은 기본 `8da4w` 사용

---

## 현재 상태 메모

이 문서는 현재 저장소에 추가된 스크립트 기준 사용법을 정리한 것입니다.

- Python 스크립트 문법 확인 완료
- 문서/코드 정적 점검 완료
- Android 실기기 빌드와 end-to-end 실행은 별도 검증이 필요할 수 있음

실기기 검증 시 우선 순위:

1. 이미지 1장 실행
2. 짧은 비디오 배치형 실행
3. 시퀀스 길이와 메모리 사용량 튜닝