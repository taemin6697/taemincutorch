# Llama 3.2 1B — ExecuTorch Android 실행 가이드

ExecuTorch 공식 [Llama README](https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md) 기준으로, `executorch/llama_test` 폴더에서 Llama 3.2 1B를 Android 폰에서 실행하는 방법을 정리합니다.

---

## 전제 조건

- `how_to_install.md` Step 1~4 완료 (ExecuTorch 설치, 환경변수)
- Android NDK: `ANDROID_NDK_ROOT` 또는 `/opt/android-ndk-r26c`
- USB 디버깅 활성화된 Android 기기

---

## Step 1. 모델 다운로드 (Hugging Face)

```bash
cd /workspace/stream/executorch

mkdir -p llama_test
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir llama_test/Llama-3.2-1B-Instruct
```

> Meta 라이선스 동의 필요. `huggingface-cli login` 후 다운로드.

**다운로드 결과:**
- `llama_test/Llama-3.2-1B-Instruct/original/consolidated.00.pth` — 가중치 (~2.3GB)
- `llama_test/Llama-3.2-1B-Instruct/original/params.json`
- `llama_test/Llama-3.2-1B-Instruct/original/tokenizer.model`

---

## Step 2. PTE Export (선택)

### Option A: 8da4w 양자화 (권장 — 모델 크기·속도 개선)

```bash
cd /workspace/stream/executorch

python -m examples.models.llama.export_llama \
  -c llama_test/Llama-3.2-1B-Instruct/original/consolidated.00.pth \
  -p llama_test/Llama-3.2-1B-Instruct/original/params.json \
  -d fp32 --xnnpack --xnnpack-extended-ops \
  -qmode 8da4w -G 64 \
  --max_seq_length 8192 --max_context_length 8192 \
  -kv --use_sdpa_with_kv_cache \
  -m '{"append_eos_to_prompt": 0, "get_bos_id":128000, "get_eos_ids":[128009, 128001]}' \
  --model llama3_2 \
  -n llama_test/llama3_2_instruct_cpu_xnnpack_8da4w_g64_8k.pte
```

### Option B: BF16 (export 없이 HF에서 PTE 직접 다운로드)

```bash
huggingface-cli download executorch-community/Llama-3.2-1B-ET llama3_2-1B.pte --local-dir llama_test
```

---

## Step 3. Android 빌드

### 3.1 ExecuTorch + 라이브러리 빌드

```bash
cd /workspace/stream/executorch

export ANDROID_NDK=${ANDROID_NDK_ROOT:-/opt/android-ndk-r26c}

cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-23 \
    -DCMAKE_INSTALL_PREFIX=cmake-out-android \
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
    -Bcmake-out-android .

cmake --build cmake-out-android -j16 --target install --config Release
```

### 3.2 Llama runner 빌드

```bash
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-23 \
    -DCMAKE_INSTALL_PREFIX=cmake-out-android \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=python \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
    -DSUPPORT_REGEX_LOOKAHEAD=ON \
    -Bcmake-out-android/examples/models/llama \
    examples/models/llama

cmake --build cmake-out-android/examples/models/llama -j16 --config Release
```

**빌드 결과:** `cmake-out-android/examples/models/llama/llama_main`

---

## Step 4. 기기로 전송

```bash
cd /workspace/stream

adb shell mkdir -p /data/local/tmp/llama_test

adb push executorch/llama_test/llama3_2_instruct_cpu_xnnpack_8da4w_g64_c2048.pte /data/local/tmp/llama_test/
adb push executorch/llama_test/Llama-3.2-1B-Instruct/original/tokenizer.model /data/local/tmp/llama_test/
adb push executorch/cmake-out-android/examples/models/llama/llama_main /data/local/tmp/llama_test/

adb shell "chmod +x /data/local/tmp/llama_test/llama_main"
```

> BF16 PTE 사용 시: `llama3_2_instruct_cpu_xnnpack_8da4w_g64_c2048.pte` 대신 `llama3_2-1B.pte` push

---

## Step 5. 기기에서 실행

```bash
adb shell "cd /data/local/tmp/llama_test && LD_LIBRARY_PATH=.:\$LD_LIBRARY_PATH ./llama_main \
  --model_path llama3_2_instruct_cpu_xnnpack_8da4w_g64_c2048.pte \
  --tokenizer_path tokenizer.model \
  --prompt \"What is the capital of France?\" \
  --seq_len 120 \
  --warmup=1"
```

### Instruct 템플릿 프롬프트 (권장)

```bash
adb shell "cd /data/local/tmp/llama_test && LD_LIBRARY_PATH=.:\$LD_LIBRARY_PATH ./llama_main \
  --model_path llama3_2_instruct_cpu_xnnpack_8da4w_g64_c2048.pte \
  --tokenizer_path tokenizer.model \
  --prompt '<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Tell me one fun fact about penguins.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

' \
  --temperature=0.6 \
  --max_new_tokens=96 \
  --warmup=1"
```

### stream_vlm_cpu.py로 한 번에 push → 실행 → pull (권장)

InternVL3와 동일한 방식으로 Llama도 실행 가능:

```bash
cd /workspace/stream/executorch

python internvl3_test/stream_vlm_cpu.py \
  -s R3KYC01FW1P \
  -b cmake-out-android \
  --model llama \
  --pte llama_test/llama3_2_instruct_cpu_xnnpack_8da4w_g64_2k.pte \
  --tokenizer llama_test/Llama-3.2-1B-Instruct/original/tokenizer.model \
  --questions "Tell me one fun fact about penguins." \
  --max_new_tokens 2000 \
  --ignore_eos \
  --save_log \
  --lazy_planned_memory
```

`--model llama` 시 `--questions`가 Llama Instruct 템플릿으로 자동 변환됨. `--ignore_eos`로 EOS 토큰 무시, max_new_tokens까지 생성 (말 끊김 방지). 결과는 `my_save/save_log/` 하위에 저장.

---

### save_log (proc/mem CSV 저장, adb shell 직접 실행)

stream_vlm처럼 처리 통계·메모리 샘플 저장:

```bash
adb shell "cd /data/local/tmp/llama_test && LD_LIBRARY_PATH=.:\$LD_LIBRARY_PATH ./llama_main \
  --model_path llama3_2_instruct_cpu_xnnpack_8da4w_g64_c2048.pte \
  --tokenizer_path tokenizer.model \
  --prompt \"Hello\" \
  --max_new_tokens=64 \
  --save_log \
  --output_path stream_output.txt"
```

생성 파일: `stream_output.txt`, `stream_output.txt.proc.csv`, `stream_output.txt.mem.csv` (adb pull로 가져오기)

---

## llama_test 폴더 구조

| 경로 | 설명 |
|------|------|
| `llama_test/llama3_2-1B.pte` | HF executorch-community BF16 PTE |
| `llama_test/llama3_2_instruct_cpu_xnnpack_8da4w_g64_c2048.pte` | 8da4w 양자화 PTE |
| `llama_test/Llama-3.2-1B-Instruct/original/consolidated.00.pth` | 원본 가중치 |
| `llama_test/Llama-3.2-1B-Instruct/original/params.json` | 모델 설정 |
| `llama_test/Llama-3.2-1B-Instruct/original/tokenizer.model` | 토크나이저 |

---

## llama / llama_original 폴더

| 폴더 | 용도 |
|------|------|
| `examples/models/llama/` | **수정용** — 메모리 체크 등 커스텀 C++ 코드 추가 |
| `examples/models/llama_original/` | **백업** — 원본 보관 (빌드 대상 아님) |

Qualcomm VLM 등 다른 예제는 `llama/`를 참조함. 수정은 `llama/`에서만 진행.

---

## llama 코드 수정 후 재빌드

`llama/` 내 C++ 코드(`main.cpp`, `runner/runner.cpp` 등) 수정 시, **llama runner만** 다시 빌드하면 됨.

```bash
cd /workspace/stream/executorch

cmake --build cmake-out-android/examples/models/llama -j16 --config Release
```

기기로 전송 후 실행:

```bash
cd /workspace/stream
adb push executorch/cmake-out-android/examples/models/llama/llama_main /data/local/tmp/llama_test/

adb shell "cd /data/local/tmp/llama_test && LD_LIBRARY_PATH=.:\$LD_LIBRARY_PATH ./llama_main \
  --model_path llama3_2_instruct_cpu_xnnpack_8da4w_g64_c2048.pte \
  --tokenizer_path tokenizer.model \
  --prompt '<|begin_of_text|><|start_header_id|>user<|end_header_id|>

What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

' \
  --temperature=0.6 \
  --max_new_tokens=96 \
  --warmup=1"
```

> ExecuTorch 전체 재빌드 불필요. `llama_main`만 다시 빌드하면 됨.

---

## 트러블슈팅

### `libc++_shared.so` 관련 에러

```bash
adb push $ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so /data/local/tmp/llama_test/
```

실행 시 `LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH` 유지.

### `RE2 failed to compile pattern with lookahead`

빌드 시 `-DSUPPORT_REGEX_LOOKAHEAD=ON` 추가 (Step 3.2에 포함됨).

### `--xnnpack_extended_ops` 인식 안 됨

`--xnnpack_extended_ops` (언더스코어) → `--xnnpack-extended-ops` (하이픈) 사용.

---

## 참고

- ExecuTorch Llama 공식 README: `executorch/examples/models/llama/README.md`
- Running_Llama 튜토리얼: `Executorch-Tutorials/Running_Llama/README.md`
- QNN/VLM 가이드: `prepare/how_to_install.md`
