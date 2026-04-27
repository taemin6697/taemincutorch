# KV Cache Physical Commit 추적 요약

Lazy KV alloc (mmap) 모드에서 `kv_physical_committed_kb`가 어느 시점에 증가하는지 추적한 결과 요약.

---

## 1. 측정 구간 및 결과 (InternVL3 1B, 1k context)

| 구간 | kv_committed | delta |
|------|--------------|-------|
| before embedding_prefill | 0 KB | - |
| after embedding_prefill | 0 KB | 0 |
| before merge | 0 KB | 0 |
| after merge | 0 KB | 0 |
| before prefill | 0 KB | 0 |
| **prefill step iter=0** | **0 → 12,320 KB** | **+12,320** |
| prefill step iter=1 | 12,320 KB | 0 |
| after prefill | 12,320 KB | - |
| after requant | 12,320 KB | 0 |
| after rearrange_cache | 12,400 KB | +80 |
| after init_attention_mask | 12,400 KB | 0 |
| after set_outputs | 12,400 KB | 0 |
| before decode step | 12,400 KB | - |
| after decode step | 12,480 KB | +80 |

---

## 2. 핵심 결론

### 2.1 KV 전체 커밋 발생 시점

**prefill의 첫 번째 `decoder_runner_->step()` 호출**에서 전체 커밋 발생.

```
[kv_commit_verify] prefill step iter=0: before 0 KB, after 12320 KB, delta 12320 KB
[kv_commit_verify] prefill step iter=1: before 12320 KB, after 12320 KB, delta 0 KB
```

### 2.2 원인

- **실행 경로**: `decoder_runner_->step("prefill_forward", inputs_)` → `module_->execute("prefill_forward", inputs_)` → QNN `prefill_forward` 그래프 실행
- **원인**: QNN `prefill_forward` 그래프 실행 시 attention이 KV 캐시 **전체**를 읽음/쓰기
- **Runner 쪽**: `rearrange_cache`, `update_cache`는 lazy 모드에서 prefill 구간만 터치하도록 구현됨 (의도대로 동작)
- **QNN 쪽**: 그래프 실행 시 KV 버퍼 전체 접근 → mmap 페이지 전체 커밋

### 2.3 Lazy 효과 제한

- prefill 직전까지 KV 커밋 0 KB 유지 (embedding, merge는 KV 미접근)
- prefill 첫 step에서 한 번에 전체 커밋 → lazy 효과 상실
- decode step에서는 이미 커밋된 상태라 추가 증가 소량(+80 KB)

---

## 3. 코드 경로

### 3.1 prefill 실행 흐름

```
multimodal_runner.cpp
  → prompt_processor_->prefill(merged_embeddings_, ...)

multimodal_prompt_processor.cpp :: prefill()
  → for (i = 0; i < num_iters; ++i)
      decoder_runner_->step(method_name_, inputs_)   // method_name_ = "prefill_forward"
      kv_manager_->update_cache(...)

decoder_runner.cpp :: step()
  → module_->execute(method_name, inputs_)
```

### 3.2 prefill_forward 메서드

- **정의**: ExecuTorch 프로그램 메서드 (`.pte` 내 그래프)
- **역할**: LLM prefill forward (여러 토큰 한 번에 처리)
- **생성 시점**: Export 시
- **관련 파일**:
  - `decoder_constants.py`: `DECODER_GRAPH_NAMES = ["kv_forward", "prefill_forward"]`
  - `llm_wrappers.py`: prefill/decode 그래프 생성
  - `llama.py`: MultiModalManager compile

### 3.3 KV 터치 구간 (prefill 내부)

| 구간 | 터치 범위 | 커밋 영향 |
|------|-----------|-----------|
| rearrange_cache | [0, num_prompt_tokens) | 작음 |
| **decoder_runner_->step()** | **전체 (QNN 그래프)** | **큼** |
| update_cache | [pos, pos+n_update) | 작음 |

---

## 4. 추가된 측정 로그

다음 파일에 `[kv_commit_verify]` 로그 추가됨:

- `multimodal_runner.cpp`: before/after embedding_prefill, merge, prefill, requant
- `multimodal_prompt_processor.cpp`: prefill step iter별 before/after
- `token_generator.cpp`: after rearrange_cache, init_attention_mask, set_outputs, before/after decode step

---

## 5. prefill_forward 정의/생성 경로

### 5.1 Export 흐름

```
foundation/exporters/qnn.py
  → llama.py compile
  → MultiModalManager.compile()

llm_wrappers.py :: MultiModalManager
  self.decode = TextDecoder(..., Mode.DECODE)   → kv_forward
  self.prefill = TextDecoder(..., Mode.PREFILL) → prefill_forward

  graph_names = ["kv_forward", "prefill_forward"]
  edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
      module=dict(zip(graph_names, [decode.decoder, prefill.decoder])),
      ...
  )
```

### 5.2 핵심 파일

| 파일 | 역할 |
|------|------|
| `decoder_constants.py` | `DECODER_GRAPH_NAMES = ["kv_forward", "prefill_forward"]` |
| `llm_wrappers.py` | TextDecoder(Mode.PREFILL) → prefill.decoder, compile 시 prefill_forward 그래프 생성 |
| `base_component.py` | `process_model_args(..., Mode.PREFILL)`: ar_len=prefill_ar_len (16) |
| `static_llama.py` | LlamaModelWithoutEmbedding.get_example_inputs(): KV cache **전체 shape** (max_seq_len - ar_len) |

### 5.3 KV Cache Shape (Export 시)

```python
# static_llama.py:914-930 (LlamaModelWithoutEmbedding)
k_cache.append(torch.zeros(
    max_batch_size, n_kv_heads, head_dim,
    max_seq_len - ar_len,   # ← 전체 길이!
))
v_cache.append(torch.zeros(
    max_batch_size, n_kv_heads,
    max_seq_len - ar_len,   # ← 전체 길이!
    head_dim,
))
```

- prefill도 **전체 크기** KV cache를 입력으로 받음
- attention: `kh = torch.cat([k_caches, k], dim=-1)` → k_caches 전체 참조
- QNN 컴파일 시 이 그래프가 전체 버퍼 접근으로 변환됨

---

## 6. 왜 KV Cache가 점진적으로만 먹히지 않는가

### 6.1 논리적 KV 누적과 물리적 메모리 commit은 다름

- Runner 관점에서는 KV가 step마다 점진적으로 누적됨
- 하지만 QNN 그래프는 export 시점에 **고정된 전체 shape KV buffer**를 입력으로 받음
- 따라서 lazy mmap 관점에서는 "일부 토큰만 유효"하더라도, 그래프가 전체 버퍼를 읽거나 쓰면 해당 페이지들이 한 번에 commit됨

### 6.2 현재 그래프는 `cur_pos`까지만 쓰는 attention이 아님

점진적 commit이 일어나려면 prefill/decode attention이 아래 성질을 가져야 함:

- `k_cache[..., :cur_pos]`, `v_cache[..., :cur_pos]`만 읽기
- 새 토큰 결과를 `cache[..., cur_pos:cur_pos+n_new]`에만 쓰기
- 이후 연산도 현재 유효 길이까지만 shape 전파

하지만 현재 export된 그래프는 KV 전체 shape를 기준으로 trace되며, attention 내부에서도 `torch.cat([k_caches, k], dim=-1)` 같은 형태로 **전체 cache tensor**를 입력으로 받음.

### 6.3 문제는 KV buffer 자체보다 attention 서브그래프 전체

- KV만 부분 갱신한다고 해결되지 않음
- `view` / `reshape` / `transpose`
- `cat`
- `matmul` / `bmm`
- `softmax`
- cache update 관련 op

즉, `seq_len` 또는 `cur_pos`가 동적으로 반영되려면 이 경로 전체가 dynamic-friendly해야 함. 현재 QNN Llama 경로는 기본적으로 static shape 전제라, 중간의 한두 군데만 바꿔서는 전체 commit을 막기 어려움.

### 6.4 현재 측정 결과가 의미하는 것

- `rearrange_cache`, `update_cache`는 의도대로 필요한 구간만 터치함
- 그런데도 첫 `prefill_forward` 실행에서 commit이 크게 늘어남
- 따라서 병목은 Runner의 후처리가 아니라 **QNN `prefill_forward` attention 그래프의 전체-buffer 접근**으로 해석하는 것이 맞음

---

## 7. 개선 방향

1. **Export**: prefill_forward 그래프가 KV를 필요한 구간만 접근하도록 구조 변경
2. **QNN Backend**: attention op가 KV 버퍼 전체가 아닌 현재 seq_len만 접근하도록 컴파일
3. **검토 대상**:
   - `executorch/examples/qualcomm/oss_scripts/llama/wrappers/llm_wrappers.py`
   - `executorch/examples/qualcomm/oss_scripts/llama/model/static_llama.py` (get_example_inputs, attention forward)

---

## 8. 참고: 모델별 KV 크기

| 모델 | context | kv_committed (lazy) |
|------|---------|---------------------|
| InternVL3 1B | 1k | ~12,480 KB |
| InternVL3 2B | 2k | ~57,792 KB |

---

## 요약: 뭘 봐야 하나

**prefill_forward** 그래프가 KV 전체를 터치하는 문제.

1. **`static_llama.py`** (765행~, 902행~): `get_example_inputs()`가 KV cache를 **전체 shape**으로 생성 → 그래프가 전체 버퍼 입력으로 trace됨
2. **`static_llama.py`** (427행, 495행): `kh = torch.cat([k_caches, k], dim=-1)` → k_caches 전체 참조
3. **`llm_wrappers.py`** (700행~): TextDecoder(Mode.PREFILL)가 prefill.decoder를 만들어 `prefill_forward`로 export

---

## 9. 지금까지 논의한 내용 정리

### 9.1 이 문제는 prefill만의 문제가 아님

- 현재 관측은 prefill 첫 실행에서 크게 드러났지만, 본질적으로는 **KV-cache attention graph 전체의 구조 문제**
- prefill과 decode 모두 KV-cache를 읽고 쓰는 attention 경로를 사용함
- 차이는 "KV를 터치하느냐"가 아니라 **KV의 어느 범위를 터치하느냐**
- 우리가 원하는 것은 `cur_pos` 또는 유효 길이까지만 부분 접근하는 graph
- 현재 export/lowering 결과는 full-shape KV buffer를 기준으로 그래프가 형성되어, lazy mmap 입장에서는 전체 commit처럼 보일 수 있음
- decode도 같은 구조를 공유하므로, prefill이 먼저 commit을 터뜨리지 않았다면 decode 첫 실행에서도 유사한 현상이 발생할 수 있음

### 9.2 왜 prefill에서 더 크게 보였나

- prefill은 여러 토큰을 한 번에 처리하므로 문제가 더 눈에 띄게 드러남
- decode는 보통 `q_len=1`이라 동일한 구조여도 추가 commit이 작게 보일 수 있음
- 하지만 이는 decode가 안전하다는 뜻이 아니라, **prefill이 먼저 대부분 페이지를 commit한 뒤 decode가 실행되었기 때문**

### 9.3 현재 구현 구조는 무엇인가

현재 Qualcomm Llama export 경로는 논문의 chunk-sharing graph와 다르게, **통짜 decoder graph** 구조에 가깝다.

- `llm_wrappers.py`에서 export하는 decoder graph는 기본적으로 2개:
  - `kv_forward`
  - `prefill_forward`
- 둘 다 decoder 전체를 graph로 내림
- 즉 attention만 따로 분리되어 있지 않음

현재 decoder layer 내부 흐름:

- `attention_norm`
- `attention`
- residual add
- `ffn_norm`
- `feed_forward`
- residual add

즉, 지금 구조는 아래와 같음:

- shared static ops: 없음
- dynamic/bucketed attention: 없음
- 실제 구조: **attention까지 포함한 전체 decoder graph**

### 9.4 현재 attention이 full-buffer 성향을 띠는 이유

- `get_example_inputs()`가 layer별 KV cache를 `max_seq_len - ar_len`의 **전체 길이**로 생성
- attention 내부에서 `torch.cat([k_caches, k], dim=-1)` / `torch.cat([v_caches, v], dim=2)` 사용
- 이 때문에 graph tracing 시점부터 "전체 cache tensor를 입력으로 받아 결합한 뒤 attention 수행" 구조가 만들어짐
- 결과적으로 QNN lowering 시에도 부분 slice 기반 attention이 아니라, **full-cache 기반 attention 서브그래프**로 내려갈 가능성이 큼

### 9.5 왜 `torch.export.Dim("seq", ...)`만으로 해결되지 않나

- `seq_len`을 dynamic으로 두면 그 차원이 흐르는 연산 경로 전체가 dynamic-friendly해야 함
- 영향을 받는 대표 연산:
  - `view` / `reshape`
  - `transpose` / `permute`
  - `cat`
  - `matmul` / `bmm`
  - `softmax`
  - cache update 관련 op
- 즉 단순히 입력 하나만 dynamic으로 바꿔서는 부족하고, **attention 서브그래프 전체**가 dynamic shape를 받아야 함
- 현재 QNN backend에는 dynamic shape plumbing과 단순 op 테스트는 존재하지만, Llama attention 경로를 실사용 수준으로 뒷받침하는 예는 보이지 않음

### 9.6 논문은 어떻게 처리했나

참고 논문: **Fast On-device LLM Inference with NPUs** (arXiv:2407.05858v2)

이 논문은 "QNN이 true dynamic shape attention graph를 그대로 실행한다"는 방식이 아님.

핵심은 다음과 같음:

- variable-length prompt를 여러 개의 **fixed-size chunk**로 분할
- chunk 단위 graph를 미리 build / optimize
- static op는 공유하고, attention처럼 context 길이에 의존하는 op만 별도 관리

논문의 Figure 7 관점에서:

- `(a) Prompt graph`: 전체 prompt를 한 번에 처리하는 큰 graph
- `(b) Chunk graph`: prompt를 여러 고정 길이 chunk로 나눠 처리
- `(c) Chunk-sharing graph`: QKV/O/FFN 같은 static op는 공유하고, attention 같은 dynamic op만 chunk별로 다르게 처리

즉 논문은 **하나의 fully dynamic QNN graph**가 아니라:

- fixed-size chunk
- graph decomposition
- subgraph sharing

으로 dynamic prompt를 우회함.

### 9.7 우리 코드와 논문 구조의 차이

논문이 지향하는 구조:

- shared: `QKV Linear`, `O Linear`, `FFN`, `norm`
- dynamic or bucketed: `attention`

현재 우리 구조:

- `prefill_forward` / `kv_forward` 각각이 decoder 전체를 포함
- attention만 따로 분리되어 있지 않음
- 따라서 논문과 같은 chunk-sharing graph 구조가 아직 아님

### 9.8 지금 가능한 접근법

#### 접근법 A. prefill 전체 bucket graph

예:

- `prefill_forward_128`
- `prefill_forward_256`
- `prefill_forward_384`
- ...

특징:

- 전체 prefill graph를 여러 개의 static length 버전으로 export
- 현재 코드 구조를 가장 덜 깨뜨림
- 구현 난이도가 가장 낮음
- 대신 graph 수, compile 시간, artifact 크기 증가

#### 접근법 B. chunked prefill

예:

- prompt 길이 1024를 `128 x 8` chunk로 나누어 순차 실행

특징:

- chunk는 static shape
- chunk 간 KV dependency는 유지
- 논문 방향과 더 유사
- runtime orchestration과 chunk loop 제어가 필요

#### 접근법 C. attention-only bucket graph

예:

- shared static graph: `QKV Linear`, `O Linear`, `FFN`, `norm`
- dynamic graph: attention만 `128 / 256 / 384 / ...`

특징:

- 논문의 chunk-sharing graph에 가장 가까움
- 재사용성이 좋고 구조적으로 이상적
- 하지만 현재 코드 기준으로는 가장 큰 리팩터링 필요
- 중간 activation I/O, graph orchestration, runner 변경이 큼

### 9.9 현실적인 추천 순서

현재 코드베이스를 기준으로는 다음 순서가 가장 현실적이다.

1. **prefill bucket graph부터 시도**
2. 그다음 **chunked prefill**
3. 마지막으로 필요하면 **attention-only 분리**

이유:

- 먼저 적은 수정으로 commit 패턴이 실제 줄어드는지 검증 가능
- 실패 시 되돌리기 쉬움
- 처음부터 attention-only 분리로 가면 export 구조와 runtime을 동시에 크게 바꿔야 함

### 9.10 실제 수정이 필요한 곳

#### `static_llama.py`

주요 수정 후보:

- `get_example_inputs()`의 KV cache shape 정의
- attention 내부의 cache 결합 방식
- bucket/chunk별 variant 생성 로직

#### `llm_wrappers.py`

주요 수정 후보:

- 현재 `prefill_forward` / `kv_forward` 단일 graph export 구조 변경
- bucket graph 이름 및 metadata 관리
- 여러 graph를 함께 export하는 orchestration 추가

#### runner 계층

주요 수정 후보:

- 현재 position 또는 prompt length에 따라 bucket 선택
- chunk loop 수행
- graph dispatch 및 IO 연결

### 9.11 현재 결론

- 현재 병목은 Runner의 단순 메모리 복사나 cache update가 아님
- 핵심은 **export된 QNN KV-cache attention graph가 full-buffer 접근 성향을 가진다는 점**
- 따라서 근본 해결은 runner 미세 수정만으로는 어렵고, **model export 구조 자체를 바꾸는 방향**이 필요함
