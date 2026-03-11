# Unified VLM Foundation

`executorch/examples/models/foundation/` 는 QNN / XNNPACK 멀티모달 실행 경로를
공통 `manifest.json` 계약으로 묶기 위한 디렉터리입니다.

현재 기준으로 다음이 구현되어 있습니다.

- 공통 manifest schema
- 공통 Python CLI
- 공통 host launcher
- `xnnpack_qnn_runner` C++ entry
- QNN / XNNPACK split-PTE batch 실행 경로

아직 streaming loop 는 공통 runner로 이식되지 않았습니다.

## 디렉터리 구성

```text
executorch/examples/models/foundation/
  README.md
  CMakeLists.txt
  cli.py
  export.py
  manifest.py
  exporters/
    __init__.py
    xnnpack.py
    qnn.py
  host/
    launcher.py
  runner/
    backend.h
    qnn_backend.cpp
    xnnpack_backend.cpp
    xnnpack_qnn_runner.cpp
```

## Artifact 계약

foundation runner는 artifact root 아래의 `manifest.json` 을 canonical contract로 사용합니다.

대표 구조는 다음과 같습니다.

```text
artifact_root/
  manifest.json
  tokenizer/
    tokenizer.json
  models/
    vision_encoder.pte
    text_embedding.pte
    text_decoder.pte
```

핵심 필드는 아래입니다.

- `backend`: `qnn` 또는 `xnnpack`
- `runner_type`: 현재 `multimodal_split`
- `paths.vision_encoder_pte`
- `paths.text_embedding_pte`
- `paths.text_decoder_pte`
- `paths.tokenizer_path`

## 공통 CLI

foundation CLI entry는 아래입니다.

```bash
python -m executorch.examples.models.foundation.cli
```

지원 subcommand:

- `export`
- `inspect-manifest`
- `run`

## Export 하는 법

foundation CLI는 **native typed export**를 사용합니다. 기존 backend exporter 스크립트를
직접 호출하지 않고, foundation 내부 `exporters/` 모듈에서 직접 export를 수행합니다.

### XNNPACK export

```bash
cd /workspace/stream

python -m executorch.examples.models.foundation.cli export \
  --backend xnnpack \
  --artifact_root /workspace/stream/my_save/save_model/cpu/internvl3_xnnpack_1b_2k \
  --decoder_model internvl3_1b \
  --model_path /workspace/stream/my_save/save_model/meta/InternVL3-1B-hf \
  --checkpoint /workspace/stream/my_save/save_model/meta/internvl3_1b_meta_cpu.pth \
  --max_seq_len 2048 \
  --max_context_len 2048 \
  --dtype fp16 \
  --vision_quant fp16 \
  --decoder_quant fp16 \
  --embedding_quant fp16
```

`--model_path` / `--checkpoint` 를 생략하면 HF에서 자동 다운로드합니다.

### QNN export

QNN export는 `--build_path`, `--device`, `--model` 이 필수입니다.

```bash
cd /workspace/stream

python -m executorch.examples.models.foundation.cli export \
  --backend qnn \
  --artifact_root /workspace/stream/my_save/save_model/qnn/internvl3_hybrid_16p_2k \
  --decoder_model internvl3_1b \
  -b build-android \
  -s R3KYC01FW1P \
  -m SM8750 \
  --model_mode hybrid \
  --prefill_ar_len 16 \
  --max_seq_len 2048 \
  --max_context_len 2048 \
  --dtype fp32 \
  --vision_quant fp16 \
  --decoder_quant fp16 \
  --embedding_quant fp16 \
  --prompts "Can you describe this image?" \
  --image_path "http://images.cocodataset.org/val2017/000000039769.jpg"
```

### Batch export (1k to 16k)

`1B / 2B / 8B` 각각에 대해 `1k / 2k / 4k / 8k / 16k / 32k` 를 한 번에 export하려면 아래 스크립트를 사용할 수 있습니다.

```bash
cd /workspace/stream

bash /workspace/stream/executorch/examples/models/foundation/export_internvl3_all_lengths.sh all
```

`all` 또는 `qnn` 를 인자로 줄 수 있고, `EXPORT_MODELS`, `QNN_DEVICE`,
`QNN_BUILD_PATH` 같은 값은 환경변수로 override할 수 있습니다. QNN은 전부 fp16으로 export합니다.

## Manifest 확인

export 후 manifest가 잘 생성되었는지 아래처럼 확인할 수 있습니다.

```bash
cd /workspace/stream

python -m executorch.examples.models.foundation.cli inspect-manifest \
  /workspace/stream/my_save/save_model/cpu/internvl3_xnnpack/manifest.json
```

## `xnnpack_qnn_runner` 빌드 방법

foundation runner는 **반드시 Android(arm64-v8a)** 용으로 빌드합니다.
호스트(x86_64)용 cmake-out 빌드는 디바이스에서 `not executable: 64-bit ELF file` 에러가 나므로 사용하지 않습니다.

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

### 2. XNNPACK용 빌드

XNNPACK foundation runner는 **QNN이 포함된 `build-android`를 재사용하지 말고**,
별도 XNNPACK 전용 build tree를 사용합니다. 그렇지 않으면
`aten::gelu.out`, `aten::native_layer_norm.out`,
`dim_order_ops::_to_dim_order_copy.out` 같은 missing operator가 발생할 수 있습니다.

#### 2.1 executorch Android 빌드 (`build-android-xnnpack`)

```bash
cd /workspace/stream/executorch

export ANDROID_NDK_ROOT=${ANDROID_NDK_ROOT:-/opt/android-ndk-r26c}

cmake -S . -B build-android-xnnpack \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-30 \
  -DCMAKE_INSTALL_PREFIX="${PWD}/build-android-xnnpack" \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
  -DEXECUTORCH_ENABLE_LOGGING=1 \
  -DPYTHON_EXECUTABLE=python \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
  -DSUPPORT_REGEX_LOOKAHEAD=ON

cmake --build build-android-xnnpack -j16 --target install --config Release
```

#### 2.2 foundation Android 빌드 (`build-android-xnnpack/foundation`)

```bash
cd /workspace/stream

CMAKE_PREFIX="${PWD}/executorch/build-android-xnnpack;${PWD}/executorch/build-android-xnnpack/third-party/gflags"
cmake -S executorch/examples/models/foundation \
  -B executorch/build-android-xnnpack/foundation \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-30 \
  -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX" \
  -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
  -Dgflags_DIR="${PWD}/executorch/build-android-xnnpack/third-party/gflags"

cmake --build executorch/build-android-xnnpack/foundation -j16
```

XNNPACK runner 바이너리 경로:

```bash
/workspace/stream/executorch/build-android-xnnpack/foundation/xnnpack_qnn_runner
```

### 3. QNN용 빌드

QNN은 Qualcomm build tree를 사용합니다.

#### 3.1 executorch Android 빌드 (`build-android`)

```bash
cd /workspace/stream/executorch
./backends/qualcomm/scripts/build.sh --skip_x86_64
```

#### 3.2 foundation Android 빌드 (`build-android/foundation`)

```bash
cd /workspace/stream

CMAKE_PREFIX="${PWD}/executorch/build-android;${PWD}/executorch/build-android/third-party/gflags"
cmake -S executorch/examples/models/foundation \
  -B executorch/build-android/foundation \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-30 \
  -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX" \
  -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
  -Dgflags_DIR="${PWD}/executorch/build-android/third-party/gflags"

cmake --build executorch/build-android/foundation -j16
```

QNN runner 바이너리 경로:

```bash
/workspace/stream/executorch/build-android/foundation/xnnpack_qnn_runner
```

`build.sh`는 QNN + XNNPACK를 함께 빌드하지만, foundation XNNPACK 실행은 위의
별도 `build-android-xnnpack` 경로를 권장합니다.

`xnnpack_qnn_runner`는 `qnn_multimodal_runner`와 동일한 인터페이스 수준의 standalone runner입니다.
`--backend xnnpack|qnn`로 백엔드를 선택하며, `--encoder_path`, `--embedding_path`, `--decoder_path`,
`--tokenizer_path`, `--image_path`, `--prompt` 등 동일한 gflags를 사용합니다.

## Run 하는 법

기본 entry는 역시 foundation CLI입니다.

```bash
python -m executorch.examples.models.foundation.cli run ...
```

아래 예시는 바로 위 export 예시를 그대로 따라 했다고 가정합니다.

- XNNPACK artifact root: `/workspace/stream/my_save/save_model/cpu/internvl3_xnnpack`
- QNN artifact root: `/workspace/stream/my_save/save_model/internvl3_hybrid_16p_1k`

### XNNPACK 실행

현재 XNNPACK foundation run은 `xnnpack_qnn_runner` 기반 split-PTE 경로만 지원합니다.
`--runner_binary` 에는 **Android용** 바이너리 경로를 사용합니다.

```bash
cd /workspace/stream

python -m executorch.examples.models.foundation.cli run \
  --manifest /workspace/stream/my_save/save_model/cpu/internvl3_xnnpack_1b_2k/manifest.json \
  --runner_binary /workspace/stream/executorch/build-android-xnnpack/foundation/xnnpack_qnn_runner \
  --device R3KYC01FW1P \
  --image http://images.cocodataset.org/val2017/000000039769.jpg \
  --questions "Describe this image briefly using around 10 words." \
  --temperature 0.0
```

비디오 입력을 쓸 때는 `--image` 대신 `--video` 를 사용합니다.

```bash
cd /workspace/stream

python -m executorch.examples.models.foundation.cli run \
  --manifest /workspace/stream/my_save/save_model/cpu/internvl3_xnnpack_1b_2k/manifest.json \
  --runner_binary /workspace/stream/executorch/build-android-xnnpack/foundation/xnnpack_qnn_runner \
  --device R3KYC01FW1P \
  --video /workspace/stream/my_save/sample_video/sample.mp4 \
  --questions "Describe this video briefly using around 10 words." \
  --temperature 0.0 \
  --decode_after_frames 3

```

### QNN 실행

QNN은 추가로 build path, device serial, model 이름이 필요합니다.
또한 환경변수 `QNN_SDK_ROOT` 가 설정되어 있어야 합니다.

```bash
export QNN_SDK_ROOT=/path/to/qnn_sdk

cd /workspace/stream

python -m executorch.examples.models.foundation.cli run \
  --manifest /workspace/stream/my_save/save_model/qnn/internvl3_hybrid_16p_2k/manifest.json \
  --runner_binary /workspace/stream/executorch/build-android/foundation/xnnpack_qnn_runner \
  -b executorch/build-android \
  -s R3KYC01FW1P \
  -m SM8750 \
  --image http://images.cocodataset.org/val2017/000000039769.jpg \
  --questions "Describe this image briefly using around 10 words." \
  --temperature 0.0
```

비디오 입력 예시:

```bash
export QNN_SDK_ROOT=/path/to/qnn_sdk

cd /workspace/stream

python -m executorch.examples.models.foundation.cli run \
  --manifest /workspace/stream/my_save/save_model/qnn/internvl3_hybrid_16p_2k/manifest.json \
  --runner_binary /workspace/stream/executorch/build-android/foundation/xnnpack_qnn_runner \
  -b executorch/build-android \
  -s R3KYC01FW1P \
  -m SM8750 \
  --video /workspace/stream/my_save/sample_video/sample.mp4 \
  --questions "Describe this video briefly using around 10 words." \
  --temperature 0.0 \
  --decode_after_frames 3
```

- `--decode_after_frames N`: 비디오에서 첫 N프레임만 사용 후 질문 처리 (미지정 시 전체)

현재 launcher 동작은 다음과 같습니다.

- XNNPACK은 Android용 `executorch/build-android-xnnpack/foundation/xnnpack_qnn_runner` 사용
- QNN은 Android용 `executorch/build-android/foundation/xnnpack_qnn_runner` 사용
- 기존 backend 스크립트 fallback 없음

## 여러 질문 넣는 법

CLI에서는 질문을 여러 개 넘길 수 있습니다.

```bash
cd /workspace/stream

python -m executorch.examples.models.foundation.cli run \
  --manifest /workspace/stream/my_save/save_model/cpu/internvl3_xnnpack/manifest.json \
  --runner_binary /workspace/stream/executorch/build-android-xnnpack/foundation/xnnpack_qnn_runner \
  --device <adb_serial> \
  --image /workspace/stream/REPLACE_ME_IMAGE.jpg \
  --questions "What is happening?" "What color is the object?"
```


## 직접 runner를 호출하는 법

host launcher 없이 직접 binary를 호출할 수도 있습니다.
runner는 Android용 바이너리이므로 디바이스에 push 후 `adb shell` 로 실행합니다.

주요 플래그:

- `--backend=qnn|xnnpack`
- `--encoder_path=...`
- `--embedding_path=...`
- `--decoder_path=...`
- `--tokenizer_path=...`
- `--image_path=...` (전처리된 .bin 또는 frame 디렉터리)
- `--prompt=...` (여러 개 가능)
- `--seq_len=128`
- `--temperature=0.0`
- `--output_path=foundation_output.txt`

예시 (디바이스 내 경로 기준):

```bash
# 디바이스에 push 후 adb shell 로 실행
adb push executorch/build-android-xnnpack/foundation/xnnpack_qnn_runner /data/local/tmp/
adb shell /data/local/tmp/xnnpack_qnn_runner \
  --backend=xnnpack \
  --encoder_path=/data/local/tmp/encoder.pte \
  --embedding_path=/data/local/tmp/embedding.pte \
  --decoder_path=/data/local/tmp/decoder.pte \
  --tokenizer_path=/data/local/tmp/tokenizer.bin \
  --image_path=/data/local/tmp/frame_0000.bin \
  --prompt="Describe this image briefly using around 10 words." \
  --seq_len=128 \
  --temperature=0.0 \
  --output_path=/data/local/tmp/foundation_output.txt
```

직접 runner를 호출할 때는 전처리된 frame .bin 파일 또는 디렉터리를 먼저 준비해야 합니다.
보통은 모바일 디바이스 push, frame 추출, manifest 상대경로 재작성까지 포함하는
foundation CLI를 사용하는 편이 낫습니다.

## 현재 제한사항

- 현재는 split-PTE 3개 기준 계약만 우선 지원합니다.
- 우선순위는 batch decode loop 입니다.
- streaming 공통 경로는 아직 미구현입니다.
- XNNPACK foundation run은 `xnnpack_qnn_runner` 전용입니다.
- QNN unified path는 실제 디바이스 환경과 `QNN_SDK_ROOT` 설정이 필요합니다.

## 권장 순서

1. XNNPACK이면 `build-android-xnnpack`, QNN이면 `build-android` build tree 준비
2. foundation Android 빌드 (`xnnpack_qnn_runner`)
3. foundation CLI `export` 로 artifact와 `manifest.json` 생성
4. `inspect-manifest` 로 경로 확인
5. foundation CLI `run` 으로 실행
