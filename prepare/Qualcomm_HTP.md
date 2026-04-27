# On-Device Streaming VLM — ExecuTorch + Qualcomm HTP 구현 노트

---

## [우리가 구현한 것] Streaming VLM

> 원본 ExecuTorch Qualcomm 예제에 **비디오 스트리밍 VLM 추론** 기능을 추가한 작업 정리.  
> 아래 섹션은 우리가 직접 만든 코드 기준. 원본 ExecuTorch README는 [아래 Overview](#overview)에서 확인.

---

### 구현 개요

```
[MP4 영상] → 1fps 샘플링 → [Vision Encoder (NPU)] → prefill_frame() → [KV Cache]
                                                                              ↓
                                                               (N초 뒤 질문 도착)
                                                                              ↓
                                                          [LLM Decoder (GPU)] → 텍스트 응답
```

- 비디오를 프레임 단위로 Vision Encoder에 통과시켜 KV Cache를 누적
- 사용자가 지정한 타임스탬프에 텍스트 질문을 던지면 누적된 KV Cache로 응답 생성
- Python 오케스트레이션(`stream_vlm.py`) + C++ 추론 바이너리(`qnn_streaming_vlm_runner`)

---

### 추가/수정한 파일

| 파일 | 변경 내용 |
|------|-----------|
| `runner/multimodal_runner/multimodal_runner.h/.cpp` | `prefill_prefix()`, `prefill_frame()` 추가 |
| `runner/client_mem.h` | KV 캐시 메모리 할당 전략 선택 (lazy mmap / eager vector) |
| `qnn_streaming_vlm_runner.cpp` | 스트리밍 추론 C++ 바이너리 신규 작성 |
| `stream_vlm.py` | Python 오케스트레이션 스크립트 신규 작성 |
| `CMakeLists.txt` | `qnn_streaming_vlm_runner` 빌드 타겟 추가 |

---

### Step 1: C++ 바이너리 빌드

ExecuTorch Android 빌드가 완료된 상태(`build-android/`)에서 아래를 실행:

```bash
cd /workspace/stream/executorch

cmake --build build-android/examples/qualcomm \
  --target qnn_streaming_vlm_runner \
  -j$(nproc)
```

빌드 결과물:
```
build-android/examples/qualcomm/oss_scripts/llama/qnn_streaming_vlm_runner
```

---

### Step 2: 모델 준비

SmolVLM 500M 또는 InternVL3 1B의 `.pte` 파일이 필요합니다.  
기존 `llama.py`로 컴파일 후 생성된 폴더(`llama_qnn_smol/` 등)를 사용합니다.

```bash
# SmolVLM 컴파일 예시 (이미 컴파일돼 있으면 생략)
python examples/qualcomm/oss_scripts/llama/llama.py \
  -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} \
  --decoder_model smolvlm_500m_instruct \
  --model_mode kv \
  --max_seq_len 1024 \
  --compile_only
```

---

### Step 3: 스트리밍 추론 실행

```bash
cd /workspace/stream/executorch

python examples/qualcomm/oss_scripts/llama/stream_vlm.py \
  -b build-android \
  -s ${SERIAL_NUM} \
  -m ${SOC_MODEL} \
  --pre_gen_pte ./llama_qnn_smol \
  --decoder_model smolvlm_500m_instruct \
  --model_mode kv \
  --video /path/to/video.mp4 \
  --questions "What is happening in the video?" \
  --timestamps 10.0 \
  --fps 1.0 \
  --seq_len 1024 \
  --lazy_kv_alloc
```

#### 실제 실행 예시 (테스트 환경: SM8750, Galaxy S25)

**InternVL3 1K (seq_len=1024)**

```bash
cd /workspace/stream/executorch && python examples/qualcomm/oss_scripts/llama/stream_vlm.py \
  -b build-android \
  -s R3KYC01FW1P \
  -m SM8750 \
  --pre_gen_pte ./llama_qnn_intern_1k_kv \
  --decoder_model internvl3_1b \
  --model_mode kv \
  --video /workspace/stream/executorch/sample.mp4 \
  --questions "What is happening in the video?" \
  --timestamps 3.0 \
  --fps 1.0 \
  --seq_len 1024 \
  --lazy_kv_alloc
```

**InternVL3 2K (seq_len=2048)**

```bash
cd /workspace/stream/executorch && python examples/qualcomm/oss_scripts/llama/stream_vlm.py \
  -b build-android \
  -s R3KYC01FW1P \
  -m SM8750 \
  --pre_gen_pte ./llama_qnn_intern_2k_kv \
  --decoder_model internvl3_1b \
  --model_mode kv \
  --video /workspace/stream/executorch/sample.mp4 \
  --questions "What is happening in the video?" \
  --timestamps 3.0 \
  --fps 1.0 \
  --seq_len 2048 \
  --lazy_kv_alloc
```

**SmolVLM (seq_len=1024)**

```bash
cd /workspace/stream/executorch && python examples/qualcomm/oss_scripts/llama/stream_vlm.py \
  -b build-android \
  -s R3KYC01FW1P \
  -m SM8750 \
  --pre_gen_pte ./llama_qnn_smol \
  --decoder_model smolvlm_500m_instruct \
  --model_mode kv \
  --video /workspace/stream/executorch/sample.mp4 \
  --questions "What is happening in the video?" \
  --timestamps 10.0 \
  --fps 1.0 \
  --seq_len 1024 \
  --lazy_kv_alloc
```

---

### CLI 인자 설명

#### `stream_vlm.py` 주요 인자

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `-b` | Android 빌드 디렉토리 | 필수 |
| `-s` | ADB 시리얼 번호 | 필수 |
| `-m` | SoC 모델명 (e.g. SM8750) | 필수 |
| `--pre_gen_pte` | 사전 컴파일된 PTE 폴더 | 필수 |
| `--decoder_model` | 모델 이름 (`smolvlm_500m_instruct`, `internvl3_1b`) | 필수 |
| `--model_mode` | 추론 모드 (`kv`, `hybrid`) | `kv` |
| `--video` | 입력 비디오 파일 경로 | 필수 |
| `--questions` | 질문 문자열 (여러 개 가능: `--questions "Q1" "Q2"`) | 필수 |
| `--timestamps` | 각 질문 타임스탬프(초) (`--timestamps 10.0 25.0`) | 필수 |
| `--fps` | 프레임 샘플링 레이트 | `1.0` |
| `--seq_len` | 생성 최대 토큰 수 | `256` |
| `--temperature` | 샘플링 온도 (0=greedy) | `0.0` |
| `--lazy_kv_alloc` | KV 캐시 lazy 물리 할당 (mmap) ↔ `--no-lazy_kv_alloc` | `True` |
| `--output` | 결과 텍스트 파일 이름 | `stream_output.txt` |

---

### KV 캐시 메모리 할당 전략

`--lazy_kv_alloc` / `--no-lazy_kv_alloc` 옵션으로 런타임에 전환 가능:

| | `--lazy_kv_alloc` (mmap) | `--no-lazy_kv_alloc` (vector) |
|--|--------------------------|-------------------------------|
| 할당 시점 | load() 시 가상 주소만 예약, write 시 물리 커밋 | load() 시 zero-init으로 전체 물리 커밋 |
| Idle RSS | **~60 MB** | ~79 MB |
| 최종 RSS | ~72 MB | ~72 MB (수렴) |
| 대용량 seq_len | **안전** (32K도 load 가능) | OOM 위험 |
| KV prefill 속도 | **~1,407 ms/frame** | ~1,448 ms/frame |
| OOM 실패 시점 | write 시 (SIGBUS) | load() 시 (명확) |

**결론**: 일반 용도에는 `--lazy_kv_alloc` 권장. 디버깅/예측 가능성 중시 시 `--no-lazy_kv_alloc`.

---

### 출력 파일

추론 완료 후 아래 파일들이 로컬에 생성됩니다:

| 파일 | 내용 |
|------|------|
| `stream_output.txt` | 모델 응답 텍스트 |
| `stream_output.txt.proc.csv` | 프레임/쿼리별 처리 통계 |
| `stream_output.txt.mem.csv` | 0.1초 간격 RSS 메모리 샘플 |

#### `proc.csv` 컬럼

```
row_type  : F(프레임) / Q(질문)
time_s    : 프레임 타임스탬프(초) 또는 질문 타임스탬프
col_a_ms  : F=encode_ms / Q=text_kv_prefill_ms
col_b_ms  : F=vision_kv_prefill_ms / Q=token_gen_ms
total_ms  : 전체 처리 시간
kv_pos    : 현재 KV 캐시 사용 위치
kv_total  : KV 캐시 총 크기
kv_used_pct: KV 캐시 사용률(%)
rss_kb    : 현재 RSS (KB)
delta_rss_kb: 직전 프레임 대비 RSS 변화
elapsed_s : 프로세스 시작(모델 로딩 2초 전)부터 경과 시간
```

#### `mem.csv` 컬럼

```
elapsed_s   : 경과 시간
rss_kb      : RSS (KB)
rss_mb      : RSS (MB)
delta_rss_kb: 직전 샘플 대비 변화량
```

---

### 실측 성능 (SM8750 / SmolVLM 500M / seq_len=1024)

| 지표 | 값 |
|------|----|
| 비전 인코더 (encode_ms) | **~29 ms / frame** |
| Vision KV prefill (vision_kv_prefill_ms) | **~1,407 ms / frame** |
| 토큰 생성 (token_gen_ms) | **~500 ms** |
| KV 토큰 소비 | **67 tokens / frame** (prefix 3 + image 64) |
| 1024 token 기준 최대 처리 프레임 | **~14 frames** |
| Idle RSS (lazy) | **60.6 MB** |
| 최종 RSS (14 frames 후) | **~72 MB** |

---

### 모델별 비교

| 모델 | tokens/frame | encode_ms | vision_kv_prefill_ms | 1024 context 최대 프레임 |
|------|-------------|-----------|---------------|--------------------------|
| SmolVLM 500M | 67 | ~29 ms | ~1,407 ms | **~14 frames** |
| InternVL3 1B | 259 | ~377 ms | ~3,460 ms | **~3 frames** |

---

## Overview

**Video Tutorial:** [Build Along: Run LLMs Locally on Qualcomm Hardware Using ExecuTorch](https://www.youtube.com/watch?v=41PKDlGM3oU)

This file provides you the instructions to run LLM Decoder model and VLM model with different parameters via Qualcomm HTP backend. We currently support the following models:
- LLM
<!-- numbered list will be automatically generated -->
 1. LLAMA2 Stories 110M
 1. LLAMA3.2 1B
 1. LLAMA3.2 3B
 1. Codegen2 1B
 1. Gemma 2B
 1. Gemma2 2B
 1. Gemma3 1B
 1. GLM 1.5B
 1. Granite3.3 2B
 1. Phi4-mini-instruct
 1. QWEN2.5 0.5B / 1.5B
 1. QWEN3 0.6B / 1.7B
 1. SmolLM2 135M
 1. SmolLM3 3B
- VLM
<!-- numbered list will be automatically generated -->
 1. SmolVLM 500M
 1. InternVL3 1B

We offer the following modes to execute the model:

- KV Cache Mode: In KV Cache mode, the model takes in a single previous token and generates the next predicted token along with its KV cache. It is efficient for generating subsequent tokens after the initial prompt.

- Hybrid Mode: Hybrid mode leverages the strengths of both AR-N model and KV cache modes to optimize token generation speed. Initially, it uses AR-N model to efficiently generate the prompt's key-value (KV) cache. Then, the mode switches to KV cache mode, which excels at generating subsequent tokens.
  - AR-N model: The auto-regression (AR) length determines the number of tokens to consume and the number of logits to produce. Use it to process the prompt and generate the key-value (kv) cache, which serves as a prompt processor in hybrid mode.
  - Prompt processing with AR-N model: 
  <figure>
    <img src="assets/PromptProcessingWithARN.png" alt="Prompt Processing With AR-N Model">
    <figcaption>Prompt processing is done using a for-loop. An N-token block is taken, and the KV cache is updated for that block. This process is repeated until all tokens are consumed, with the last block potentially requiring padding. For flexibility, the AR-N model can handle any input length less than the maximum sequence length. For TTFT, the input length (or number of blocks) will vary depending on the actual input length, rather than always being the same.
    </figcaption>
</figure>

- Lookahead Mode: Lookahead Mode introduces [lookahead decoding](https://arxiv.org/abs/2402.02057) and uses AR-N model to process prompt to enhance token generation speed. While decoding multiple tokens in a single step is infeasible, an LLM can generate multiple guess tokens in parallel. These guess tokens may fit into future parts of the generated sequence. The lookahead decoder generates and verifies these guess tokens, integrating them into the sequence if suitable. In some cases, it can obtain more than one token in a single step. Result is lossless.

## Hardware Support

We’ve validated this flow on the **Samsung Galaxy S23**, **Samsung Galaxy S24**, **Samsung Galaxy S25**, and **OnePlus 12**.  
Support on other hardware depends on the **HTP architecture (HtpArch)** and the feature set available on that version.

### HTP Minimum Version Requirements

- **LPBQ (16a4w block-wise quantization)** requires **V69 or newer**
- **Weight sharing** between prefill and decode requires **V73 or newer**
- **16-bit activations + 16-bit weights for matmul** (e.g., 16-bit KV cache) requires **V73 or newer**

### Quantization Guidance for Older Devices

For older HTP versions, you may need to adjust the quantization strategy. Recommended starting points:

- Use **16a4w** as the baseline
- Optionally apply **SpinQuant**
- Use **16a8w selectively on some layers** to further improve accuracy (mixed-precision quantization)

### Memory Limit Errors (4 GB HTP Limit)

If you encounter errors like the following, it typically means the model’s requested memory exceeds the **4 GB per-context limit** on HTP.  
To resolve this, try **increasing the sharding number** (`num_sharding`) to reduce per-shard memory usage:

```
[ERROR] [Qnn ExecuTorch]: QnnDsp <E> Failed to find available PD for contextId 1 on deviceId 0 coreId 0 with context size estimate 4025634048
[ERROR] [Qnn ExecuTorch]: QnnDsp <E> context create from binary failed on contextId 1
[ERROR] [Qnn ExecuTorch]: QnnDsp <E> Fail to create context from binary with err 1002
[ERROR] [Qnn ExecuTorch]: QnnDsp <E> Size Calculation encounter error! Doing Hard reset of reserved mem to 0.
[ERROR] [Qnn ExecuTorch]: QnnDsp <E> Failed to create context from binary with err 0x3ea
[ERROR] [Qnn ExecuTorch]: Can't create context from binary
```


## Instructions
### Note
1. For hybrid mode, the export time will be longer and can take up to 1-4 hours to complete, depending on the specific model users are exporting.
2. When exporting a hybrid mode model, memory consumption will be higher. Taking LLAMA3.2 1B as an example, please ensure the device has at least 80 GB of memory and swap space.


### Step 1: Setup
1. Follow the [tutorial](https://pytorch.org/executorch/main/getting-started-setup) to set up ExecuTorch.
2. Follow the [tutorial](https://pytorch.org/executorch/main/backends-qualcomm) to build Qualcomm AI Engine Direct Backend.
3. Please install the llm eval dependency via [examples/models/llama/install_requirements.sh](https://github.com/pytorch/executorch/blob/main/examples/models/llama/install_requirements.sh)

### Step 2: Prepare Model

#### LLAMA2
Download and prepare stories110M model

```bash
# tokenizer.model & stories110M.pt:
wget "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.pt"
wget "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model"

# tokenizer.bin:
python -m pytorch_tokenizers.tools.llama2c.convert -t tokenizer.model -o tokenizer.bin

# params.json:
echo '{"dim": 768, "multiple_of": 32, "n_heads": 12, "n_layers": 12, "norm_eps": 1e-05, "vocab_size": 32000}' > params.json
```

#### LLAMA3.2
Follow the [instructions](https://www.llama.com/) to download models.
At the end of this step, users should have the following files ready: `consolidated.00.pth`, `params.json`, and `tokenizer.model`.


### Step3: Run default examples.
#### Note:
All example scripts below use hybrid mode, which is optimized for on-device performance. However, compiling a model in hybrid mode can consume a significant amount of memory on the host machine—sometimes up to ~100 GB. If your host machine has limited memory, it is highly recommended to switch from `--model_mode hybrid` to `--model_mode kv` and remove the `--prefill_ar_len` flag.

#### LLAMA2
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --checkpoint stories110M.pt --params params.json --tokenizer_model tokenizer.model --tokenizer_bin tokenizer.bin --decoder_model stories110m --model_mode hybrid --prefill_ar_len 32 --max_seq_len 128 --prompt "Once upon a time"
```

#### LLAMA3.2 1B Instruct
Default example using hybrid mode.
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --decoder_model llama3_2-1b_instruct --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1
```

#### LLAMA3.2 3B Instruct
Default example using hybrid mode.
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --decoder_model llama3_2-3b_instruct --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1
```

#### Codegen2
Default example using kv mode.
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --decoder_model codegen2_1b --model_mode kv --max_seq_len 1024 --prompt "def hello_world():" 
```

#### Gemma 2B
Default example using hybrid mode
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --temperature 0 --model_mode hybrid --max_seq_len 1024 --prefill_ar_len 128 --decoder_model gemma-2b --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1
```

#### Gemma2 2B
Default example using hybrid mode
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --temperature 0 --model_mode hybrid --max_seq_len 1024 --prefill_ar_len 128 --decoder_model gemma2-2b --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1
```

#### Gemma3 1B
Default example using hybrid mode
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --temperature 0 --model_mode hybrid --max_seq_len 1024 --prefill_ar_len 128 --decoder_model gemma3-1b --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1
```

#### GLM 1.5B
Default example using hybrid mode
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --temperature 0 --model_mode hybrid --max_seq_len 1024 --prefill_ar_len 128 --decoder_model glm-1_5b --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1
```

#### Granite3.3 2B
Default example using hybrid mode
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --temperature 0 --model_mode hybrid --max_seq_len 1024 --prefill_ar_len 128 --decoder_model granite_3_3-2b_instruct --prompt "I would like to learn python, could you teach me with a simple example?" --eval_methods tasks_eval --task hellaswag --limit 10
```

#### Phi4-mini-instruct
Default example using hybrid mode.
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --decoder_model phi_4_mini --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1
```

#### QWEN2.5 0.5B
Default example using hybrid mode
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --temperature 0 --model_mode hybrid --max_seq_len 1024 --prefill_ar_len 128 --decoder_model qwen2_5-0_5b --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1
```

#### QWEN2.5 1.5B
Default example using hybrid mode
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --temperature 0 --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 --decoder_model qwen2_5-1_5b --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1
```

#### QWEN3 0.6B
Default example using hybrid mode
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --temperature 0 --model_mode hybrid --max_seq_len 1024 --prefill_ar_len 128 --decoder_model qwen3-0_6b --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1
```

#### QWEN3 1.7B
Default example using hybrid mode
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --temperature 0 --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 --decoder_model qwen3-1_7b --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1
```

#### SmolLM2
Default example using hybrid mode.
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --decoder_model smollm2_135m --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1
```

#### SmolLM3
Default example using hybrid mode.
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --decoder_model smollm3-3b --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1
```

## Multimodal Support

### Overview

Multimodal models extend LLM by processing multiple input modalities (vision, audio, text) simultaneously. This framework provides a unified architecture for multimodal via Qualcomm HTP backend.

**Current Support Status:**
- **Vision-Language Models (VLM)**: Fully supported
- **Audio-Language Models (ALM)**: Coming soon

### Multimodal Architecture

For general multimodal processing pipeline please refer [Multimodal Architecture](../../../../extension/llm/runner/README.md#multimodalrunner-architecture)


### Processing Pipeline

Multimodal inference follows these key stages:

1. **Modality-Specific Encoding**
   - **Vision**: Images are processed through a vision encoder to generate visual embeddings
   - **Audio**: Audio waveforms are processed through an audio encoder *(future support)*
   - **Text**: Text prompts are tokenized and embedded

2. **Embedding Fusion**
   - All modality embeddings are projected to a common embedding dimension
   - Embeddings are concatenated or fused according to the model's template
   - Special tokens are inserted to mark modality boundaries

3. **Unified Language Generation**
   - The fused embeddings are fed into the language model decoder
   - The decoder generates text autoregressively using the same execution modes as LLM models (KV Cache, Hybrid, Lookahead)

---

## Vision-Language Model (VLM) Support

Vision-Language Models (VLMs) combine computer vision and natural language processing to understand and generate text based on visual inputs. VLMs in this framework consist of:

- **[Vision Encoder](model/vision_encoder.py)**: Processes images into visual embeddings (e.g., SigLIP for SmolVLM)
  - **Projection Layer** (included in vision encoder): Aligns visual embeddings with the language model's embedding space
- **[Language Decoder](model/static_llama.py)**: Reuse static llama to generates text based on fused visual and text embeddings

### Instructions

#### SmolVLM 500M
Default example using hybrid mode.
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --decoder_model smolvlm_500m_instruct --model_mode hybrid --prefill_ar_len 16 --max_seq_len 1024 --prompt "Can you describe this image?" --image_path "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
```

#### InternVL 1B
Default example using hybrid mode.
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --decoder_model internvl3_1b --model_mode hybrid --prefill_ar_len 32 --max_seq_len 1024 --prompt "Can you describe this image?" --image_path "http://images.cocodataset.org/val2017/000000039769.jpg"
```

### Specifying Custom Image

You can specify custom image for VLM models using the `--image_path` flag:

Take a example image of Statue-of-Liberty in New York Bay
- **HTTP/HTTPS URLs**: Direct links to images on the web
  - Example: `https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg`
- **Local file paths**: Absolute or relative paths to image files on your system
  - Example: [`./examples/qualcomm/oss_scripts/llama/assets/samples/images/Statue-of-Liberty-Island-New-York-Bay.png`](assets/samples/images/Statue-of-Liberty-Island-New-York-Bay.png)

**Default behavior:**
If `--image_path` is not specified, the system will automatically use the default image URL defined in the model's configuration file (`encoder/encoder_config.py`).

#### Image Preprocessing

Each VLM model has specific preprocessing requirements defined in its configuration:

```python
# In encoder/encoder_config.py
@dataclass(init=False, frozen=True)
class SmolVLMEncoder(VisionModalityConfig):
    encoder_class = Idefics3VisionEncoder
    img_seq_len = 64
    img_resized_h = 512
    img_resized_w = 512
    img_url = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"  # Default image
    quant_recipe = SmolVLM_Encoder_QuantRecipe
```

- **img_resized_h / img_resized_w**: Target resolution for the vision encoder
- **img_seq_len**: Number of visual tokens generated by the encoder

The image is automatically:
1. Loaded from the specified URL or file path
2. Resized to the model's expected resolution and preprocessed by HuggingFace [processors](https://huggingface.co/docs/transformers/main/processors)

### Using Pre-Generated PTE Files

If you have already compiled a VLM model, you can run inference with pre-generated PTE files:

```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --decoder_model smolvlm_500m_instruct --model_mode kv --max_seq_len 1024 --prompt "Can you describe this image?" --image_path "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg" --pre_gen_pte ${FOLDER_TO_PRE_GEN_PTE}
```

### VLM Processing Details

The VLM inference pipeline consists of:

1. **Vision Encoding Phase**
   - Input image is preprocessed (resize, normalize)
   - Vision encoder generates visual embeddings: `[batch, img_seq_len, hidden_dim]`
   - Visual embeddings are projected to match the language model dimension by the modality projector

2. **Text Tokenization Phase**
   - User prompt is tokenized into text tokens
   - Text tokens are embedded: `[batch, text_seq_len, hidden_dim]`

3. **Embedding Fusion Phase**
   - Visual and text embeddings are concatenated according to the model's template
   - Special tokens (e.g., `<image>`, `<|fake_token_around_image|>`, `<fake_token_around_image>`) mark modality boundaries (see [tokenizer.py](tokenizer.py))
   
   ```python
   # Special tokens for Vision-Language Model
   VLM_SPECIAL_TOKENS = {
       "smolvlm_500m_instruct": {
           "image_token": "<image>",
           "global_img": "<global-img>",
           "fake_wrap_start": "<fake_token_around_image>",
           "fake_wrap_end": "<fake_token_around_image>",
       },
       ...
   }
   ```
   - Final fused sequence: `[batch, img_seq_len + text_seq_len, hidden_dim]`

4. **Language Generation Phase**
   - Fused embeddings are fed into the language decoder
   - Autoregressive generation produces output tokens
   - KV cache is updated for efficient subsequent token generation


### KV Cache update mechanism
We use Smart Mask mechanisms for updating the key-value (KV) cache.

#### Smart Mask mechanism:
<figure>
    <img src="assets/SmartMask.png" alt="Smart Mask mechanism">
    <figcaption>The figure illustrates how key and value caches are updated during each inference step. The Smart Mask mechanism simplifies updating tokens in the cache by modifying only the new token at the designated position. This approach is useful for shared buffers, though it does require copying data in CPU memory to update the kv cache. </figcaption>
</figure>

#### Analysis KV Cache Update Mechanism for each Layer each inference
<table>
  <tr>
    <th>Mechanism</th>
    <th colspan="2" style="text-align:center;">Time Complexity</th>
    <th colspan="2" style="text-align:center;">Space Complexity</th>
  </tr>
  <tr>
    <th></th>
    <th style="text-align:center;">K</th>
    <th style="text-align:center;">V</th>
    <th style="text-align:center;">K</th>
    <th style="text-align:center;">V</th>
  </tr>
  <tr>
    <td style="text-align:center;">Smart Mask</td>
    <td style="text-align:center;">num_head * head_dim</td>
    <td style="text-align:center;">num_head * head_dim</td>
    <td style="text-align:center;">num_head * seq_len * head_dim</td>
    <td style="text-align:center;">num_head * seq_len * head_dim</td>
  </tr>
</table>

### Additional Configs when running the script

#### Compile Only
If you would like to compile the model only, we have provided the flag `--compile_only`. Taking LLAMA3.2 as an example:
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -m ${SOC_MODEL} --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --decoder_model llama3_2 --model_mode hybrid --prefill_ar_len 32 --max_seq_len 128 --prompt "I would like to learn python, could you teach me with a simple example?" --compile_only
```

#### Pre Generated PTE
On the other hand, if you already have a pre-compiled .pte model, you can perform inference by providing the flag `--pre_gen_pte` and specifying the folder that contains the .pte model. Taking LLAMA3.2 as an example:
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --decoder_model llama3_2 --model_mode hybrid --prefill_ar_len 32 --max_seq_len 128 --prompt "I would like to learn python, could you teach me with a simple example?" --pre_gen_pte ${FOLDER_TO_PRE_GEN_PTE}
```

#### Lookahead Decoding Mode

You can choose the lookahead mode to enhance decoding speed. To use this mode, you need to specify the following parameters:
- `--ngram` (N-gram size): Represents the size of the n-grams used in the lookahead process.
- `--window` (window size): Determines how many future tokens the algorithm attempts to predict in each step.
- `--gcap` (Verification candidates): Represents the maximum number of speculations or candidate n-grams that the algorithm considers in each step for verification. It balances the trade-off between computation efficiency and exploring more possibilities.

For more details, please refer to the paper ["Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057)

```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --decoder_model llama3_2 --model_mode lookahead --prefill_ar_len 32 --max_seq_len 128 --prompt "I would like to learn python, could you teach me with a simple example?" --ngram 3 --window 2 --gcap 2
```

#### Tasks Evaluation
This script supports task evaluation and is capable of assessing evaluation scores across 3 phases: prepare_pt2e(CPU FP), convert_pt2e(CPU QDQ), QNN on device.

To evaluate the perplexity across all 3 phases, users should provide the `--eval_methods tasks_eval` flag and specify the evaluation task. Please notice when this flag is provided, the `--prompt ${PROMPT}` will be ignored.

For example, using the Qwen model and 1 wikitext sample as the evaluation task, users can assess all 3 phases perplexity score in a single run by including the appropriate configuration:
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --prompt "I would like to learn python, could you teach me with a simple example?" --temperature 0 --model_mode kv --max_seq_len 1024 --decoder_model qwen2_5-0_5b --eval_methods tasks_eval --tasks wikitext --limit 1 --verbose
```

From the example script above, 1 wikitext sample is used to evaluate all 3 phases. However, there are cases where a user may want to use one sample for quantization calibration and multiple samples for perplexity evaluation. In this case, the process should be split into two runs. In the 1st run, the model is compiled using one sample. In the 2nd run, the user can provide a different configuration for QNN device execution.
Example:
```bash
# 1st run to compile with --limit 1
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --prompt "I would like to learn python, could you teach me with a simple example?" --temperature 0 --model_mode kv --max_seq_len 1024 --decoder_model qwen2_5-0_5b --eval_methods tasks_eval --tasks wikitext --limit 1 --compile_only
```
```bash
# 2nd run to perform QNN device execution with --limit 3
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --prompt "I would like to learn python, could you teach me with a simple example?" --temperature 0 --model_mode kv --max_seq_len 1024 --decoder_model qwen2_5-0_5b --eval_methods tasks_eval --tasks wikitext --limit 3 --pre_gen_pte ${PATH_TO_ARTIFACT_IN_1ST_RUN} --quant_attrs_path ${PATH_TO_ARTIFACT_IN_1ST_RUN}/kv_llama_qnn_quant_attrs.json
```

#### Tasks quantization calibration
If `--tasks ${TASK}` is not provided, the program will use `--prompt ${PROMPT}` as the dataset for quantization calibration.
Regardless of whether `--eval_methods tasks_eval` is provided, as long as `--tasks ${TASK}` is specified, the specified tasks will be used for model quantization calibration instead of the prompt.

#### SQNR Evalution
To evaluate QNN's output logits against the golden logits from `nn.Module`, users can provide the flag `--sqnr_eval`. Please note that SQNR evaluation will only compare the logits of the user's prompt and will not compare the new tokens generated by the model.
Example:
```bash
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --prompt "I would like to learn python, could you teach me with a simple example?" --temperature 0 --model_mode kv --max_seq_len 1024 --decoder_model qwen2_5-0_5b --eval_methods sqnr_eval
```

#### Use attention sink for multi-turn conversations
Attention sink is a way to evict cache when maximum context length be reached.
There are two mainly concept for attention sink:

1. **Maintain Attention Sinks**: Always include several initial tokens as attention sinks in the kv cache.
2. **Redefine Positional Context**: Use positions relative to the cache instead of absolute positions from the original text, enhancing relevance and coherence in generated responses.

<figure>
    <img src="assets/AttentionSinkFeature.png" alt="Attention Sink Feature">
    <figcaption>This figure shows how the attention sink operates for the kv cache in LLMs when `max_context_len = 8`, `sink_size = 4`, and `eviction_batch_size = 2`. The yellow blocks represent sink tokens, blue blocks indicate the remaining tokens, and the red blocks show the newly generated token.
    </figcaption>
</figure>

This feature supports fluent multi-turn conversations and manages long-context scenarios. To enable it, set `--use_attention_sink <sink_size>,<batch_eviction_size>`.

##### Explanation of Parameters Related to Attention Sink:
1. **`--max_seq_len`**: Maximum sequence length the model can generate
2. **`--max_context_len`**: Maximum length of the model's memory/cache, including both prompt tokens and generated tokens
3. **`<sink_size>`**: Always include `sink_size` initial tokens as attention sinks in the kv cache.
4. **`<batch_eviction_size>`**: How many tokens to evict from the cache at once when the cache is full. 

Example:
```bash
# Compile llama pte file and attention sink evictor pte file with sink_size = 4 and batch_eviction_size = 64
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --decoder_model llama3_2-1b_instruct --model_mode hybrid --prefill_ar_len 128 --max_seq_len 4096 --max_context_len 1024 --prompt "I would like to learn python, could you teach me with a simple example?" --tasks wikitext --limit 1 --use_attention_sink 4,64 --compile_only
```

After running this, the `attention_sink_evictor.pte` file will be generated in the artifacts directory. This file is necessary for using the attention sink feature, as it handles removing the `eviction_batch_size` tokens from the kv cache, retaining the first `sink_size` tokens, and re-rotating the remaining tokens in the kv cache.

For multi-turn conversations or scenarios with long context using attention sink, you can set max_seq_len higher than the max_context_len used during compilation:
```bash
# Run llama with attention sink in multi-turn conversation scenario
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --decoder_model llama3_2-1b_instruct --model_mode hybrid --prefill_ar_len 128 --max_seq_len 4096 --prompt "I would like to learn python, could you teach me with a simple example?" "Could you give more difficult example in python?" "Could you add a GUI for this game?" "Could you tell me more about tkinter?" "Is possible to deploy on website?" ---pre_gen_pte ${PATH_TO_ARTIFACT_IN_1ST_RUN}  --use_attention_sink 4,64 
```

If you want to modify `sink_size` or `batch_eviction_size`, or if you have a pre-compiled llm pte file and wish to use the attention sink feature, you can recompile the `attention_sink_evictor.pte` with different attention sink config.

```bash
# Compile attention sink evictor pte file with sink_size = 4 and batch_eviction_size = 128
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --checkpoint consolidated.00.pth --params params.json --tokenizer_model tokenizer.model --decoder_model llama3_2-1b_instruct --model_mode hybrid --prefill_ar_len 128 --max_seq_len 4096 --prompt "I would like to learn python, could you teach me with a simple example?" "Could you give more difficult example in python?" "Could you add a GUI for this game?" "Could you tell me more about tkinter?" "Is possible to deploy on website?" ---pre_gen_pte ${PATH_TO_ARTIFACT_IN_1ST_RUN}  --use_attention_sink 4,128 
```

Please make sure to use the same `--max_context_len`, `--prefill_ar_len`, and `--model_mode`, etc., as those used in the LLM to ensure the kv cache shape is correct.
