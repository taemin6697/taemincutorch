# my_idea

<!-- 아이디어, 실험, 메모 등을 여기에 정리 -->

## 개요

Streaming VLM (ExecuTorch + QNN HTP) 관련 옵저베이션 및 아이디어 정리. **메인 아이디어**: NPU Vision KV 스트리밍 + CPU/GPU Reduction 파이프라인 (§6).

---

## 1. Prefill & seq_len

- **KV 모드**: ar_len=1, 토큰 1개씩 처리. 호출 횟수 = 토큰 수.
- **Hybrid 모드**: ar_len=256, 청크 단위 처리. 호출 횟수 = ceil(토큰 수 / 256).
- **공통점**: 두 모드 모두 **호출당 attention은 max_seq_len(context_len) 전체**에 대해 수행.
- **seq_len에 따라 prefill, decode 모두 증가** (2배 → 연산량 2배).

### 실측 (InternVL3-1B Hybrid)

| seq_len | vision_kv_prefill_ms (프레임당) |
|---------|-------------------------------|
| 2048    | ~175ms                        |
| 8192    | ~1130ms (~6.5×)              |

---

## 2. 논문 레퍼

| 기법 | 논문 | ExecuTorch 인용 |
|------|------|-----------------|
| **Chunked prefill** (prefill_ar_len) | SARATHI, arxiv 2308.16369 | ❌ 없음 |

- Chunked prefill: SARATHI는 GPU 기준, 우리는 NPU(HTP)에서 실행.
- ExecuTorch/Qualcomm 코드베이스에는 Hybrid/AR-N에 대한 논문 주석 없음.

---

## 3. 실행 위치 (NPU)

- Vision Encoder, LLM Prefill, LLM Decode 모두 **Qualcomm HTP(NPU)**에서 실행.
- 청크드 prefill도 NPU에서 수행.

---

## 4. 현실적으로 가능한 아이디어

1. **seq_len별 다중 그래프 불가**: NPU에 256, 512, 1024, 2048 등 그래프를 여러 개 두는 것은 **메모리 제한** 때문에 불가능하며, 로드 시간도 매우 김.

2. **270토큰 그래프 재사용**: 비전 prefill이 약 270토큰만 처리하므로, **270토큰용 연산 그래프를 미리 할당**한 뒤 **반복 사용**하는 방식이 필요함.

3. **단일 그래프 상주**: 비전 토큰을 모두 포함하는 **270토큰짜리 그래프만 NPU에 상주**시키고, 텍스트 prefill/decode도 이 그래프를 반복 사용해 처리해야 함.

4. **슬라이딩 윈도우**: 270 그래프 + eviction → 입력 길이는 무제한, 기억하는 컨텍스트는 270토큰. KV cache 270 한도.

5. **Vision 270 / Decode 4096 분리**: Vision prefill은 270 고정, Decode만 4096 사용 → **그래프 2개** 필요 → NPU 메모리 2배.

6. **Decode를 GPU로**: Vision prefill은 NPU(270 고정), **Decode는 GPU(Adreno)** → GPU는 가변 입력 지원 가능.
   - NPU: 270 그래프만 상주
   - GPU: Decode 시 가변 길이(4096 등) 처리
   - **고려사항**: KV cache NPU→GPU 전송 비용, ExecuTorch Qualcomm GPU decode 지원 여부 확인 필요.

---

## 5. NPU(270) + GPU(4096) 분리 방식

### 동작 흐름

| 시점 | 동작 |
|------|------|
| **평상시** | NPU: Vision prefill (270)만 반복 |
| **질문 들어올 때** | 1) Vision KV cache를 NPU → GPU로 한 번 전달<br>2) GPU: Text prefill + Decode (4096) |

- **모델**: 가중치 1개, **컴파일된 .pte 2개** (NPU 270, GPU 4096)
- **전송**: 질문당 1회만 (토큰마다 동기화 아님)

### 레이어별 vs 우리 방식 (HeteroLLM 대비)

| | 레이어별 (HeteroLLM) | 우리 방식 |
|---|----------------------|-----------|
| **비전 prefill 동기화** | 매 레이어 경계마다 (270스텝×31 ≈ 8,370회) | **0회** (NPU에서만) |
| **Decode 동기화** | 토큰마다 레이어 경계 (100토큰×31 ≈ 3,100회) | **1회** (질문 시 KV 전달) |
| **그래프** | 1개 | 2개 |
| **비전 KV cache** | prefill 시에도 NPU↔GPU 반복 | NPU에서만 생성, 질문 시 1회 전달 |

→ 레이어별은 **비전 KV cache를 만들 때부터** 동기화 비용 발생. 우리 방식은 비전 prefill 구간 동기화 없음.

### Zero Copy / Unified Memory

- Zero Copy라도 **동기화, 캐시 일관성, 메모리 대역폭 경쟁** 등 오버헤드 존재.
- 디코드 스텝이 짧을수록(수백 μs) 동기화 비용 비중 증가 → 동기화 횟수 최소화가 중요.

### 참고 논문

- **HeteroLLM** (arxiv 2501.14794): 레이어별 NPU-GPU 분할. unified memory, 레이어 경계마다 동기화.

---

## 6. 메인 아이디어: 스트리밍 Vision KV + Reduction 파이프라인

### Method

1. **NPU → CPU/GPU 스트리밍**: NPU가 Vision KV Prefill을 **계속 생성**하여 CPU/GPU로 전달 (헤테로지니어스). 프레임 ID와 KV cache를 함께 전송.

2. **GPU 가변 입력 + 한도 정의**: GPU는 가변 입력을 받을 수 있으나, KV cache가 너무 길면 bandwidth/메모리 한계. **LLM 입력에 따른 최대 Reduction 크기**, **사용할 KV cache 공간**을 미리 정의.

3. **병렬 Reduction**: NPU가 Vision KV Prefill을 생성하는 동안, CPU/GPU에서는 **쌓인 KV cache에 대해 Reduction** 수행.
   - 전략: 프레임별, 토큰 중복성, 모바일 최적화 등.

4. **스케줄링**:
   - **디코딩 중**: 그 시점까지 reduction된 KV cache로 decode.
   - **디코딩 안 할 때**: reduction을 계속 진행하여 안전선(메모리/bandwidth 한도) 유지.

### 평가

| 장점 | 설명 |
|------|------|
| **무한 스트리밍** | NPU가 계속 vision KV 생성, reduction으로 길이 제어 |
| **역할 분리** | NPU=생성, CPU/GPU=축적·reduction·decode |
| **안전선 유지** | reduction으로 메모리/bandwidth 한도 내 유지 |
| **디코드 시점 유연** | 질문 시점의 reduced cache 사용 |

| 과제 | 설명 |
|------|------|
| **Reduction 품질** | 압축/요약 시 정보 손실 vs 유지 trade-off |
| **Reduction 전략** | 프레임별·토큰 중복·importance 기반 등 설계 필요 |
| **동기화** | NPU→CPU/GPU 스트리밍, reduction↔decode 타이밍 |
| **모바일 최적화** | CPU/GPU에서 reduction 연산 비용 |

### 관련 기법

- **StreamingLLM** (attention sink): 최근 토큰 + sink 유지
- **TaR/VaN** (시스템 프롬프트): importance 기반 pruning
- **Hierarchical KV**: Tier1(활성) + Tier2(요약) 구조

---

## 참고

- `executorch/prefill_path_analysis.md` — Prefill 경로 상세
- `executorch/examples/qualcomm/oss_scripts/llama/README.md` — 모드 설명
