# InternVL3-1B 텍스트 디코더 — ExecuTorch Android 실행 가이드

InternVL3-1B VLM에서 **텍스트 디코더(LLM)만** 추출하여 ExecuTorch로 Android 폰에서 실행하는 방법을 정리합니다. LLM은 Qwen2.5-0.5B 기반이므로 Qwen2.5 export 파이프라인을 사용합니다.

---

## KV cache 접근 (llama_main 경로)

stream_vlm 스타일로 `get_k_cache()`, `get_v_cache()` 사용:

```cpp
// runner: create_llama_runner()로 생성된 TextLLMRunner*
auto k_caches = runner->get_k_cache().get();
auto v_caches = runner->get_v_cache().get();

for (size_t layer = 0; layer < k_caches.size(); ++layer) {
  void* k_ptr = k_caches[layer].buffer;
  size_t k_size = k_caches[layer].size;
  void* v_ptr = v_caches[layer].buffer;
  size_t v_size = v_caches[layer].size;
  // k_ptr, v_ptr로 직접 접근
}
```

저수준 접근이 필요하면 `module()->get_planned_buffer("forward", index)` 사용.

---

## 전제 조건

- `how_to_install.md` Step 1~4 완료 (ExecuTorch 설치, 환경변수)
- Android NDK: `ANDROID_NDK_ROOT` 또는 `/opt/android-ndk-r26c`
- USB 디버깅 활성화된 Android 기기

---

## Step 1. InternVL3-1B 다운로드 (Hugging Face)

```bash
cd /workspace/stream/executorch

mkdir -p internvl3_test
huggingface-cli download OpenGVLab/InternVL3-1B \
  --local-dir internvl3_test/InternVL3-1B
```

**다운로드 결과:**
- `internvl3_test/InternVL3-1B/model.safetensors` — VLM 전체 가중치 (~1.9GB)
- `internvl3_test/InternVL3-1B/tokenizer.json` — 토크나이저
- `internvl3_test/InternVL3-1B/tokenizer_config.json`

---

## Step 2. LLM 추출 (VLM → 텍스트 디코더만)

InternVL3 VLM에서 `language_model.*` 가중치만 추출하여 HF Qwen2.5 형식으로 저장합니다.

```bash
cd /workspace/stream/executorch

python internvl3_test/extract_llm_from_internvl3.py
```

> 기본값: `--input InternVL3-1B/model.safetensors`, `--output_dir InternVL3-1B_llm_only` (스크립트 기준 상대 경로)

**추출 결과:** `internvl3_test/InternVL3-1B_llm_only/model.safetensors` (~1.2GB)

---

## Step 3. Meta 포맷 변환

ExecuTorch export용 Meta 포맷(.pth)으로 변환합니다.

```bash
cd /workspace/stream/executorch

python -m examples.models.qwen2_5.convert_weights \
  internvl3_test/InternVL3-1B_llm_only \
  internvl3_test/internvl3_1b_llm_meta.pth
```

**변환 결과:** `internvl3_test/internvl3_1b_llm_meta.pth`

---

## Step 4. PTE Export (8da4w 양자화)

InternVL3-1B는 vocab_size=151674 (Qwen2.5-0.5B 기본 151936과 다름). 전용 config 사용 필수.

```bash
cd /workspace/stream/executorch

python -m examples.models.llama.export_llama \
  -c internvl3_test/internvl3_1b_llm_meta.pth \
  -p internvl3_test/internvl3_1b_config.json \
  -d fp32 --xnnpack --xnnpack-extended-ops \
  -qmode 8da4w -G 64 \
  --max_seq_length 32768 --max_context_length 32768 \
  -kv --use_sdpa_with_kv_cache \
  --model qwen2_5_0_5b \
  -m '{"get_bos_id":151643, "get_eos_ids":[151643, 151645]}' \
  -n internvl3_test/internvl3_1b_llm_xnnpack_8da4w_32k.pte
```

**Export 결과:** `internvl3_test/internvl3_1b_llm_xnnpack_8da4w.pte` (~770MB)

> **metadata 필수**: `-m` 옵션으로 EOS 토큰(151643, 151645) 지정. 없으면 Android에서 조기 종료될 수 있음.

> `internvl3_1b_config.json`은 `vocab_size: 151674`로 설정된 전용 config. 없으면 `prepare/` 또는 `internvl3_test/`에 생성 필요.

---

## Step 5. Android 빌드

### 5.1 ExecuTorch + 라이브러리 빌드

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

### 5.2 Llama runner 빌드 (InternVL3/Qwen 토크나이저용)

**Qwen tokenizer.json 사용 시 `SUPPORT_REGEX_LOOKAHEAD=ON` 필수.**

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

**extension/module 또는 extension/llm 수정 후 재빌드 시** (llama는 별도 cmake 프로젝트라 lib/ 의존):

```bash
cd /workspace/stream/executorch
cmake --build cmake-out-android -j16 --config Release
cp cmake-out-android/extension/module/libextension_module*.a cmake-out-android/lib/
cp cmake-out-android/extension/llm/runner/libextension_llm_runner.a cmake-out-android/lib/
cmake --build cmake-out-android/examples/models/llama -j16 --config Release
```

---

## Step 6. 기기로 전송

> **stream_vlm_cpu.py 사용 시** Step 7 Option A가 push → 실행 → pull을 한 번에 수행하므로 이 단계 생략 가능.

```bash
cd /workspace/stream

adb shell mkdir -p /data/local/tmp/internvl3_test

adb push executorch/internvl3_test/internvl3_1b_llm_xnnpack_8da4w.pte /data/local/tmp/internvl3_test/
adb push executorch/internvl3_test/InternVL3-1B/tokenizer.json /data/local/tmp/internvl3_test/
adb push executorch/cmake-out-android/examples/models/llama/llama_main /data/local/tmp/internvl3_test/

adb shell "chmod +x /data/local/tmp/internvl3_test/llama_main"
```

### libc++_shared.so (필요 시)

```bash
adb push $ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so /data/local/tmp/internvl3_test/
```

---

## Step 7. 기기에서 실행

### Option A: stream_vlm_cpu.py (stream_vlm 스타일, 권장)

PC에서 한 번에 push → 실행 → pull. stream_vlm.py와 동일한 패턴.

```bash
cd /workspace/stream/executorch

python internvl3_test/stream_vlm_cpu.py \
  -s R3KYC01FW1P \
  -b cmake-out-android \
  --pte internvl3_test/internvl3_1b_llm_xnnpack_8da4w_8k.pte \
  --questions "Wrtie long korea history please" \
  --max_new_tokens 8000 \
  --save_log \
  --lazy_planned_memory \
  --ignore_eos
```

**256 토큰 prefill 측정** (`<|im_start|>` 1토큰 × 256, `--prompt`로 정확히 256토큰):

```bash
python internvl3_test/stream_vlm_cpu.py \
  -s R3KYC01FW1P \
  -b cmake-out-android \
  --pte internvl3_test/internvl3_1b_llm_xnnpack_8da4w_4k.pte \
  --prompt "$(python -c 'print("<|im_start|>" * 256)')" \
  --max_new_tokens 10 \
  --save_log \
  --lazy_planned_memory
```

`--pte` 모델 경로, `--questions`에 질문만 입력. `--model internvl3`(기본) 시 ChatML 스페셜 토큰 자동 적용. `--questions` 사용 시 Step 5.2에서 llama_main 재빌드 필요.

**Llama 모델 사용 (Instruct 템플릿):**
```bash
cd /workspace/stream/executorch


python internvl3_test/stream_vlm_cpu.py \
  -s R3KYC01FW1P \
  -b cmake-out-android \
  --model llama \
  --pte llama_test/llama3_2_instruct_cpu_xnnpack_8da4w_g64_c2048.pte \
  --tokenizer llama_test/Llama-3.2-1B-Instruct/original/tokenizer.model \
  --questions "Tell me one fun fact about penguins." \
  --max_new_tokens 2000 \
  --save_log \
  --lazy_planned_memory \
  --ignore_eos
```
`--model llama` 시 `--questions`가 Llama Instruct 템플릿으로 변환되어 `--prompt`로 전달됨. 디바이스 워크스페이스는 `/data/local/tmp/llama_test`.

옵션 예시:
```bash
# 모델명 지정 (폴더: my_save/save_log/internvl3_1b_cpu/ 또는 internvl3_1b_cpu_lazy/)
--save_log --model_name internvl3_1b

# unknown flag 에러 시 (stdout만 저장)
--save_log --no_device_save_log

# KV cache lazy 물리 할당 (8K 등 대용량 seq_len에서 load 시 OOM 방지)
--lazy_planned_memory

# Llama 모델 (Instruct 템플릿, 워크스페이스: /data/local/tmp/llama_test)
--model llama

# EOS 무시, max_new_tokens까지 생성 (말 끊김 방지)
--ignore_eos
```

**stream_vlm_cpu.py 인자:**

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `-s`, `--device` | (필수) | adb 디바이스 시리얼 (예: R3KYC01FW1P) |
| `-b`, `--build_path` | (필수) | Android 빌드 디렉토리 (예: cmake-out-android) |
| `--model` | internvl3 | 모델 타입. internvl3=ChatML, llama=Instruct 템플릿 |
| `--pte` | internvl3_1b_llm_xnnpack_8da4w.pte | PTE 파일 경로 |
| `--tokenizer` | InternVL3-1B/tokenizer.json | 토크나이저 경로 (llama 시 명시 필요) |
| `--questions` | - | 질문만 입력. model에 따라 템플릿 자동 적용 |
| `--prompt` | "Hello, who are you?" | raw 프롬프트 (--questions 미사용 시) |
| `--temperature` | 0.0 | 샘플링 온도 |
| `--max_new_tokens` | 256 | 최대 생성 토큰 수 |
| `--save_log` | false | txt, proc.csv, mem.csv, mem_merged.csv 저장 |
| `--model_name` | PTE stem | save_log 폴더명용 |
| `--no_device_save_log` | false | llama_main에 --save_log 미전달 (unknown flag 시) |
| `--warmup` | false | 측정 전 warmup 1회 |
| `--lazy_planned_memory` | false | KV cache mmap(MAP_NORESERVE) 사용 |
| `--ignore_eos` | false | EOS 토큰 무시, max_new_tokens까지 생성 |

---

### Option B: adb shell 직접 실행

```bash
adb shell "cd /data/local/tmp/internvl3_test && LD_LIBRARY_PATH=.:\$LD_LIBRARY_PATH ./llama_main \
  --model_path internvl3_1b_llm_xnnpack_8da4w.pte \
  --tokenizer_path tokenizer.json \
  --prompt \"Hello, who are you?\" \
  --seq_len 120 \
  --warmup=1"
```

### InternVL3/Qwen Instruct 템플릿 (권장)

```bash
adb shell "cd /data/local/tmp/internvl3_test && LD_LIBRARY_PATH=.:\$LD_LIBRARY_PATH ./llama_main \
  --model_path internvl3_1b_llm_xnnpack_8da4w.pte \
  --tokenizer_path tokenizer.json \
  --prompt '<|im_start|>user
Hello, who are you?<|im_end|>
<|im_start|>assistant

' \
  --temperature=0.0 \
  --max_new_tokens=256 \
  --warmup=0"
```

---

## PC에서 Python runner로 테스트

```bash
cd /workspace/stream/executorch

python -m examples.models.llama.runner.native \
  --model qwen2_5_0_5b \
  --pte internvl3_test/internvl3_1b_llm_xnnpack_8da4w.pte \
  -kv \
  --tokenizer internvl3_test/InternVL3-1B/tokenizer.json \
  --tokenizer_config internvl3_test/InternVL3-1B/tokenizer_config.json \
  --params internvl3_test/internvl3_1b_config.json \
  --prompt "<|im_start|>user
Hello, who are you?<|im_end|>
<|im_start|>assistant
" \
  --max_len 128 \
  --temperature 0 \
  --echo
```

`--echo`: stream_vlm save_log처럼 prompt+response 전체 출력 (기본값: 미사용)

`--save_log`: stream_vlm처럼 output + proc.csv (prefill/token_gen) + mem.csv (RSS) 저장

`--output_path`: save_log 시 저장 경로 (기본: llama_output.txt)

---

## internvl3_test 폴더 구조

| 경로 | 설명 |
|------|------|
| `internvl3_test/InternVL3-1B/` | 원본 VLM (model.safetensors, tokenizer.json 등) |
| `internvl3_test/InternVL3-1B_llm_only/` | 추출된 LLM (model.safetensors) |
| `internvl3_test/internvl3_1b_llm_meta.pth` | Meta 포맷 변환 결과 |
| `internvl3_test/internvl3_1b_llm_xnnpack_8da4w.pte` | 8da4w PTE |
| `internvl3_test/internvl3_1b_config.json` | vocab_size: 151674 전용 config |
| `internvl3_test/extract_llm_from_internvl3.py` | LLM 추출 스크립트 |
| `internvl3_test/stream_vlm_cpu.py` | Android 실행 스크립트 (stream_vlm 스타일) |
| `internvl3_test/merge_mem_proc.py` | mem.csv + proc.csv 병합 (phase/event, TTFT, tok/s, seq_len별 메모리) |

---

## internvl3_1b_config.json (없을 경우 생성)

```json
{
  "dim": 896,
  "ffn_dim_multiplier": 1,
  "hidden_dim": 4864,
  "n_heads": 14,
  "n_kv_heads": 2,
  "n_layers": 24,
  "norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "use_scaled_rope": false,
  "vocab_size": 151674,
  "use_hf_rope": true,
  "attention_qkv_bias": true
}
```

---

## 트러블슈팅

### `RE2 failed to compile pattern with lookahead`

Qwen tokenizer.json 사용 시 발생. 빌드 시 `-DSUPPORT_REGEX_LOOKAHEAD=ON` 추가 (Step 5.2에 포함).

### `size mismatch for tok_embeddings.weight`

InternVL3-1B vocab_size=151674, Qwen2.5-0.5B 기본=151936. `internvl3_1b_config.json` 사용 필수.

### `ERROR: unknown command line flag 'output_path'` / `'save_log'`

llama_main이 `--save_log`/`--output_path`를 지원하지 않는 구빌드일 수 있음. **Step 5.2**에서 llama_main을 재빌드한 뒤 기기로 재전송. `examples/models/llama/main.cpp`에 save_log 지원 포함. 임시로 `--no_device_save_log` 사용 시 stdout만 저장됨 (proc/mem 없음).

### `libc++_shared.so` 관련 에러

Step 6의 libc++_shared.so push 후 `LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH` 유지.

### Android에서 "Hello!"만 출력 후 조기 종료 (PC와 다름)

**원인:**
1. **PTE metadata 누락**: `get_eos_ids` 미지정 시 tokenizer fallback 사용. Step 4에서 `-m` 옵션 필수.
2. **ARM vs x86 수치 차이**: 8da4w 양자화 모델이 ARM/x86에서 argmax 결과가 달라질 수 있음 (부동소수점 누적 순서 차이).

**해결:**
1. metadata 포함해 PTE 재export 후 기기로 재전송:
   ```bash
   # Step 4 명령에 -m 추가 후 재실행
   adb push executorch/internvl3_test/internvl3_1b_llm_xnnpack_8da4w.pte /data/local/tmp/internvl3_test/
   ```
2. `--ignore_eos`로 테스트 (EOS 무시, max_new_tokens까지 생성):
   ```bash
   adb shell "cd /data/local/tmp/internvl3_test && ... ./llama_main ... --ignore_eos"
   ```
3. `--temperature=0.7` 시도 (수치 민감도 완화).
4. 입력 토큰 검증: C++ runner가 `[입력 토큰] 개수=14, IDs=[...]` 로그 출력. PC Python과 동일한지 확인.

---

## 메모리 분석 (smaps)

### 2k vs 2k_8k RSS 차이 요약

| 구분 | 2k_cpu_lazy | 2k_8k_cpu_lazy | 차이 |
|------|-------------|----------------|------|
| **총 RSS** | ~443 MB | ~489 MB | **+46 MB** |
| `[anonymous]` | ~87 MB | ~133 MB | **+47 MB** |
| `[anon:scudo:secondary]` | ~308 MB | ~308 MB | 0 |

**결론:** 47 MB 차이는 거의 전부 `[anonymous]` 영역에서 발생. 이 영역은 `mmap(MAP_ANONYMOUS | MAP_NORESERVE)`로 할당된 **KV cache planned buffer** (ExecuTorch memory planner → `extension/module/module.cpp`).

### 같은 질문인데 왜 커밋량이 다르나?

이론상 18토큰(9 프롬프트 + 9 생성)이면 같은 양만 커밋돼야 함. 가능한 원인:

1. **버퍼 레이아웃 차이**: 2k(48×2MB) vs 8k(48×8MB). 메모리 플래너가 버퍼를 묶는 방식이 다르면 stride·오프셋이 달라져, 같은 18토큰이라도 더 많은 페이지를 fault 시킴.
2. **SDPA stride**: `kStrideN`/`vStrideN`이 `max_seq_len`에 비례하면, 8k에서 같은 `n`이라도 더 먼 오프셋을 읽어 더 많은 페이지 커밋.
3. **커널/Android 동작**: prefetch·블록 단위 할당 등으로 실제 접근보다 넓은 범위가 커밋될 수 있음.

### smaps 수집·분석 절차

1. **llama_main 재빌드** (main.cpp에 smaps 덤프 포함)
2. **각 설정으로 stream_vlm_cpu 실행** (`--save_log`):
   ```bash
   python stream_vlm_cpu.py -s DEVICE -b cmake-out-android --pte .../2k.pte --lazy_planned_memory --save_log
   python stream_vlm_cpu.py -s DEVICE -b cmake-out-android --pte .../2k_8k.pte --lazy_planned_memory --save_log
   ```
3. **smaps 비교**:
   ```bash
   cd /workspace/stream
   python executorch/internvl3_test/analyze_smaps.py \
     my_save/save_log/..._2k_cpu_lazy/stream_output.txt.smaps \
     my_save/save_log/..._2k_8k_cpu_lazy/stream_output.txt.smaps --diff
   ```

### mem_merged.csv (phase/event, TTFT, tok/s, seq_len별 메모리)

`--save_log` 실행 후 `merge_mem_proc.py`가 자동 실행되어 `stream_output.txt.mem_merged.csv`가 생성됩니다.
수동 실행:

```bash
python executorch/internvl3_test/merge_mem_proc.py my_save/save_log/internvl3_1b_hybrid_seq4096_video_lazy/
```

- **mem_merged.csv**: `elapsed_s` 좌측에 `phase`, `event` 컬럼 추가. 시간순으로 V_Encode_start, V_Prefill_end, T_Decode_start 등 이벤트 표기
- **stdout**: TTFT, tok/s, seq_len(kv_pos)별 RSS MB 요약

---

## 참고

- XNNPACK 멀티모달(이미지/배치형 비디오) quick path:
  ```bash
  cd /workspace/stream/executorch

  python -m executorch.examples.models.internvl3.export_xnnpack_multimodal \
    --decoder_model internvl3_1b \
    --model_path /path/to/InternVL3-1B-hf \
    --checkpoint /path/to/internvl3_1b_meta.pth \
    --output /workspace/stream/my_save/internvl3_xnnpack_multimodal.pte \
    --max_seq_len 1024 \
    --max_context_len 1024 \
    --vision_quantize
  ```
- Android runner build:
  ```bash
  cd /workspace/stream/executorch

  cmake -S . -B cmake-out-android \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON

  cmake --build cmake-out-android --target internvl3_xnnpack_runner -j
  ```
- 이미지/배치형 비디오 실행:
  ```bash
  cd /workspace/stream/executorch

  python -m executorch.examples.models.internvl3.run_xnnpack_vlm \
    --serial DEVICE_SERIAL \
    --model_path /workspace/stream/my_save/internvl3_xnnpack_multimodal.pte \
    --runner_binary /workspace/stream/executorch/cmake-out-android/examples/models/internvl3/internvl3_xnnpack_runner \
    --image "http://images.cocodataset.org/val2017/000000039769.jpg" \
    --question "Can you describe this image?"
  ```
- 비디오는 `--image` 대신 `--video sample.mp4 --fps 1.0` 사용
- 생성물:
  - `*_artifacts/tokenizer/tokenizer.json`
  - `*_artifacts/manifest.json`

- InternVL3-1B: [OpenGVLab/InternVL3-1B](https://huggingface.co/OpenGVLab/InternVL3-1B) — LLM=Qwen2.5-0.5B, Vision=InternViT-300M
- InternVL3 텍스트 디코더만 사용 시 Vision 없이 순수 LLM으로 동작
- Llama 가이드: `prepare/how_to_llama.md`
- ExecuTorch 설치: `prepare/how_to_install.md`
