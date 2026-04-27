# ExecuTorch + QNN (Snapdragon 8 Elite) 설치 및 실행 가이드

## 환경 정보
- **디바이스**: Samsung Galaxy S25 Ultra (SM-S938N)
- **SoC**: Snapdragon 8 Elite (SM8750)
- **ADB Serial**: R3KYC01FW1P
- **Android NDK**: `/opt/android-ndk-r26c` (r26c)
- **QNN SDK**: 2.37.0

---

## Step 1. ExecuTorch 클론

```bash
cd /workspace/stream

git clone https://github.com/pytorch/executorch.git
```

---

## Step 2. QNN SDK 배치

```bash
mkdir -p /workspace/stream/executorch/backends/qualcomm/sdk/qnn

# /tmp/qnn_sdk.zip 에 다운받은 SDK가 있음
unzip /tmp/qnn_sdk.zip \
  -d /workspace/stream/executorch/backends/qualcomm/sdk/qnn
```

압축 해제 후 구조:
```
backends/qualcomm/sdk/qnn/qairt/2.37.0.250724/
├── bin/
├── include/
├── lib/
│   ├── x86_64-linux-clang/   ← 호스트용 라이브러리
│   ├── aarch64-android/      ← 안드로이드용 라이브러리
│   └── hexagon-v79/          ← Hexagon DSP용 라이브러리
├── QNN_README.txt
└── sdk.yaml
```

---

## Step 3. 환경변수 설정 (~/.bashrc)

아래 내용이 `~/.bashrc`에 포함되어 있어야 함:

```bash
# QNN / ExecuTorch
export QNN_SDK_ROOT=/workspace/stream/executorch/backends/qualcomm/sdk/qnn/qairt/2.37.0.250724
export EXECUTORCH_ROOT=/workspace/stream/executorch
export ANDROID_NDK_ROOT=/opt/android-ndk-r26c
export PYTHONPATH=$EXECUTORCH_ROOT/..
export LIBCXX_DIR=/opt/conda/envs/stream/lib/python3.11/site-packages/executorch/backends/qualcomm/sdk/libcxx-14.0.0
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang/:$LIBCXX_DIR:$EXECUTORCH_ROOT/build-x86/lib/:$LD_LIBRARY_PATH
```

적용:
```bash
source ~/.bashrc
```

> **주의**: `LIBCXX_DIR`은 QNN SDK가 필요로 하는 `libc++.so.1`, `libunwind.so.1` 등을 제공함.  
> 위 경로는 `pip install executorch` 시 자동 설치되는 경로임.

---

## Step 4. Python 패키지 설치 (editable)

Linux에서 Vulkan 개발 패키지 설치 (선행):
```bash
sudo apt install libvulkan-dev vulkan-headers
```

```bash
cd /workspace/stream/executorch
conda activate stream

CMAKE_ARGS="-DEXECUTORCH_BUILD_VULKAN=ON" python install_executorch.py --editable
```

> `pip install -e .` 대신 이 스크립트를 사용해야 함.  
> 내부적으로 `--no-build-isolation` 플래그가 자동으로 붙음.  
> `QNN_SDK_ROOT`가 설정되어 있으면 QNN 백엔드가 함께 빌드됨 (Step 2, 3 선행 필요).  
> Vulkan은 환경변수 없이 `-DEXECUTORCH_BUILD_VULKAN=ON`으로 항상 포함.

---

## Step 5. QNN 백엔드 빌드

```bash
cd /workspace/stream/executorch

./backends/qualcomm/scripts/build.sh
```

> 빌드 완료 후 `build-android/` 및 `build-x86/` 폴더가 생성됨.  
> 소요 시간: 10~30분

**qualcomm 예제만 다시 빌드** (C++ 수정 후):

```bash
cd /workspace/stream/executorch

cmake --build build-android/examples/qualcomm -j4
# 또는 특정 타겟만: --target qnn_streaming_vlm_runner
```

**extension_module 변경 시** (llama/main.cpp 등에서 Module 클래스 수정 후, 링크 에러 발생 시):

```bash
cd /workspace/stream/executorch

# 1. extension_module 재빌드 (출력: build-android/extension/module/libextension_module.a)
cmake --build build-android --target extension_module -j4

# 2. qualcomm은 build-android/lib/ 경로의 라이브러리를 링크함 → 복사 필요
cp build-android/extension/module/libextension_module.a build-android/lib/

# 3. qualcomm 예제 빌드
cmake --build build-android/examples/qualcomm -j4
```

> **이유**: CMake 설정상 qualcomm 예제는 `build-android/lib/`에 있는 `.a` 파일을 링크함. `extension_module` 타겟은 소스 구조대로 `build-android/extension/module/`에 빌드되므로, 두 경로가 달라서 수동 복사가 필요함.

> `build.sh`는 `build-android`에서 실행되므로 빌드 디렉터리는 `build-android/examples/qualcomm` 임.

---

## Step 6. SmolVLM 실행

```bash
cd /workspace/stream/executorch

python examples/qualcomm/oss_scripts/llama/llama.py \
  -b build-android \
  -s R3KYC01FW1P \
  -m SM8750 \
  --decoder_model smolvlm_500m_instruct \
  --model_mode hybrid \
  --prefill_ar_len 16 \
  --max_seq_len 2048 \
  --artifact ./save_model/llama_qnn_smol_8k_hybrid \
  --prompt "Can you describe this image?" \
  --image_path "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
```

```bash
cd /workspace/stream/executorch

python examples/qualcomm/oss_scripts/llama/llama.py \
  -b build-android \
  -s R3KYC01FW1P \
  -m SM8750 \
  --decoder_model internvl3_1b \
  --model_mode hybrid \
  --prefill_ar_len 16 \
  --max_seq_len 1024 \
  --artifact ../my_save/save_model/internvl3_hybrid_16p_1k \
  --prompt "Can you describe this image?" \
  --image_path "http://images.cocodataset.org/val2017/000000039769.jpg"
```

> **기본값**: `--dtype-override fp16`, `--vision_quant fp16`, `--decoder_quant fp16`, `--embedding_quant fp16` (모두 fp16, 양자화 없음).  
> 양자화 적용 시: `--vision_quant 16a8w`, `--decoder_quant 16a8w`, `--embedding_quant 4,32` 등.

---

## Step 7. SmolVLM Inference

```bash
cd /workspace/stream/executorch

python examples/qualcomm/oss_scripts/llama/llama.py \
  -b build-android \
  -s R3KYC01FW1P \
  -m SM8750 \
  --decoder_model smolvlm_500m_instruct \
  --model_mode kv \
  --max_seq_len 1024 \
  --pre_gen_pte ./llama_qnn_smol \
  --prompt "Can you describe this image? and write very long story" \
  --image_path "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"

```

```bash
cd /workspace/stream/executorch

python examples/qualcomm/oss_scripts/llama/llama.py \
  -b build-android \
  -s R3KYC01FW1P \
  -m SM8750 \
  --decoder_model internvl3_1b \
  --model_mode kv \
  --max_seq_len 1024 \
  --pre_gen_pte ./save_model/llama_qnn_intern_1k_kv \
  --prompt "Can you describe this image? and write very long story" \
  --image_path "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
```

**InternVL3 Hybrid (2K, Step 6에서 `--artifact`로 컴파일된 hybrid PTE 필요)**

```bash
cd /workspace/stream/executorch

python examples/qualcomm/oss_scripts/llama/llama.py \
  -b build-android \
  -s R3KYC01FW1P \
  -m SM8750 \
  --decoder_model internvl3_1b \
  --model_mode hybrid \
  --prefill_ar_len 16 \
  --max_seq_len 2048 \
  --pre_gen_pte ../my_save/save_model/llama_qnn_internvl3_hybrid_2k \
  --prompt "Can you describe this image? and write very long story" \
  --image_path "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
```

---

## Step 8. Streaming VLM (비디오/이미지/텍스트 추론)

비디오·이미지·텍스트 입력 지원. **비디오와 이미지는 동시 입력 불가** — 비디오 있으면 비디오, 없으면 이미지, 둘 다 없으면 텍스트 전용.

**사전 요구**: `llama_qnn_intern_1k_kv` 등 seq_len별 PTE 폴더가 있어야 함. (Step 6에서 `--artifact`로 컴파일)

**비디오 (InternVL3 1K)**

```bash
cd /workspace/stream/executorch && python examples/qualcomm/oss_scripts/llama/stream_vlm.py \
  -b build-android \
  -s R3KYC01FW1P \
  -m SM8750 \
  --pre_gen_pte ../my_save/save_model/qnn/internvl3_1b_hybrid_16p_2k \
  --decoder_model internvl3_1b \
  --model_mode hybrid \
  --video /workspace/stream/my_save/sample_video/sample.mp4 \
  --questions "Describe this video" \
  --decode_after_frames 3 \
  --fps 1.0 \
  --seq_len 2048 \
  --eval_mode 1 \
  --lazy_kv_alloc \
  --save_log
```

- **배치 모드 (기본)**: `--decode_after_frames 10` → 첫 10프레임만 사용, 마지막 프레임 후 질문 처리.
- **스트리밍 모드** (`--stream`): `--decode_after_frames 16` → 16프레임 처리 직후 바로 답변 (타임스탬프 기반).

**이미지 (1프레임, Step 7과 동일 설정)**

```bash
cd /workspace/stream/executorch && python examples/qualcomm/oss_scripts/llama/stream_vlm.py \
  -b build-android \
  -s R3KYC01FW1P \
  -m SM8750 \
  --pre_gen_pte ../my_save/save_model/llama_qnn_internvl3_hybrid_0_270k \
  --decoder_model internvl3_1b \
  --model_mode hybrid \
  --image "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg" \
  --questions "Can you describe this image? and write very long story" \
  --timestamps 0.0 \
  --seq_len 280 \
  --eval_mode 1 \
  --save_log \
  --no-lazy_kv_alloc
```

**텍스트 전용 (비디오/이미지 없음)**

```bash
cd /workspace/stream/executorch && python examples/qualcomm/oss_scripts/llama/stream_vlm.py \
  -b build-android \
  -s R3KYC01FW1P \
  -m SM8750 \
  --pre_gen_pte ../my_save/save_model/llama_qnn_internvl3_hybrid_4k \
  --decoder_model internvl3_1b \
  --model_mode hybrid \
  --questions "Hello, who are you?" \
  --timestamps 0.0 \
  --seq_len 4096 \
  --eval_mode 1 \
  --lazy_kv_alloc \
  --save_log
```

**주요 인자**

| 인자 | 설명 |
|------|------|
| `--pre_gen_pte` | 사전 컴파일된 PTE 폴더 (seq_len에 맞는 빌드) |
| `--video` | 입력 비디오 경로 (비디오 있으면 비디오 사용) |
| `--image` | 입력 이미지 경로 또는 URL (비디오 없으면 이미지 사용, 1프레임) |
| `--questions` | 질문 문자열 |
| `--stream` | 스트리밍 모드 (프레임별 prefill, 타임스탬프 시점에 질문). 미지정 시 배치(HF) 모드 |
| `--timestamps` | 질문 발화 시점(초). `--stream` 모드에서 `--decode_after_frames` 사용 시 무시 |
| `--decode_after_frames` | 스트리밍: N프레임 직후 디코딩. 배치: 첫 N프레임만 사용 (미지정 시 전체) |
| `--fps` | 프레임 샘플링 레이트 (비디오 전용, 기본 1.0) |
| `--seq_len` | context 최대 길이 (1024, 2048, 4096 등) |
| `--eval_mode` | C++ 러너 모드: 0=KV(기본, model_mode=kv 시 사용), 1=Hybrid(model_mode=hybrid 시 사용), 2=Lookahead |
| `--lazy_kv_alloc` | KV 캐시 lazy 물리 할당 (mmap, 기본값) |
| `--no-lazy_kv_alloc` | KV 캐시 즉시 전체 할당 (std::vector) |
| `--save_log` | `my_save/save_log/{파라미터폴더}/` 하위에 txt, csv 저장 |
| `--save_log_dir` | save_log 저장 기준 경로 (기본: workspace/my_save/save_log) |

**출력** (`--save_log` 사용 시, 위 예시에 포함됨):
- `my_save/save_log/{decoder_model}_{model_mode}_seq{seq_len}_{video|image|text}_{lazy|nolazy}/` 하위에 저장
- 예: `stream_output.txt`, `.proc.csv`, `.mem.csv`, `.mem_merged.csv` (병합: phase/event, TTFT, tok/s, seq_len별 메모리)
- `--save_log_dir my_save/save_log` 로 경로 지정 가능
- `--save_log` 생략 시: 현재 디렉터리에 `stream_output.txt` 등 저장

---

## 트러블슈팅

### ❌ `No module named 'executorch.backends.qualcomm.python.PyQnnManagerAdaptor'`
- **원인**: QNN 백엔드 C++ 빌드가 안 된 상태
- **해결**: Step 5 (`build.sh`) 먼저 실행

### ❌ `libc++.so.1: cannot open shared object file`
- **원인**: `LIBCXX_DIR`이 잘못된 경로를 가리키고 있음
- **해결**: `LIBCXX_DIR=/opt/conda/envs/stream/lib/python3.11/site-packages/executorch/backends/qualcomm/sdk/libcxx-14.0.0` 로 설정 후 `source ~/.bashrc`

### ❌ `libunwind.so.1: cannot open shared object file`
- **원인**: `LIBCXX_DIR` 미설정 또는 잘못된 경로
- **해결**: 위와 동일

### ❌ `pip install -e .` 빌드 실패 (`NoneType has no attribute submodule_search_locations`)
- **원인**: pip 격리 빌드 환경에서 torch를 못 찾음
- **해결**: `python install_executorch.py --editable` 사용 (내부적으로 `--no-build-isolation` 적용)

### ❌ `undefined symbol: executorch::extension::module::Module::Module(..., bool)`
- **원인**: `extension_module`이 갱신되지 않음 (llama.py 등에서 Module에 `use_lazy_planned_memory` 추가 후)
- **해결**: 위 "extension_module 변경 시" 절차 실행

### ❌ `ImportError: cannot import name 'Int8DynamicActivationInt4WeightConfig' from 'torchao.quantization'`
- **원인**: torchao 0.17+와 torchtune 버전 불일치 (torchao에서 해당 클래스 제거됨)
- **해결**: torchao를 0.16으로 다운그레이드
  ```bash
  pip install torchao==0.16.0
  ```

---

## 참고

- QNN SDK 다운로드: https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk
- QNN 권장 버전: **2.37.0**
- Android NDK 위치: `/opt/android-ndk-r26c` (r26c 권장)
- ExecuTorch 공식 문서: `backends/qualcomm/docs/source/backends-qualcomm.md`
