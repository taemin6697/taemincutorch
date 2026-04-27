# 스트리밍 VLM RoPE 수정 및 인사이트

## 개요

스트리밍 모드(`--stream`)에서 Hybrid + `--rope` 조합 시 출력이 깨지는 문제를 수정하고, RoPE와 `copy_cache_range` 상호작용을 정리한 문서.

---

## 1. 문제 현상

| 설정 | 출력 |
|------|------|
| Hybrid + stream **without** `--rope` | 정상 (비디오 내용에 맞는 설명) |
| Hybrid + stream **with** `--rope` | 깨짐 ("A video clip" + 반복·허구 문장) |

---

## 2. 원인: RoPE + copy_cache_range 불일치

### 2.1 스트리밍에서 copy가 두 번 발생

| 단계 | copy | 설명 |
|------|------|------|
| **prefill_frame** | `[cur_pos_+prefix_len, cur_pos_+prefix_len+frame_len)` → `[cur_pos_, cur_pos_+frame_len)` | prefix 제거, FrameN만 저장 |
| **generate** | `[0, frames_len)` → `[prefix_len, prefix_len+frames_len)` | prefix 자리 확보 |

### 2.2 기존 use_decode_rope (수정 전)

```
매 prefill_frame마다:
- prefix: RoPE 0, 1, 2
- FrameN: RoPE(prefix_len + frame_idx*frame_len), ...

→ copy 후 cache 위치와 RoPE가 어긋남
→ decode 시 Q·K^T attention이 잘못됨
```

### 2.3 RoPE는 K에만 적용 (V는 무관)

| 텐서 | RoPE | 이유 |
|------|------|------|
| Q | ✅ | attention score에 위치 반영 |
| K | ✅ | Q·K^T에 위치 반영 |
| V | ❌ | 가중합에만 사용, 위치 불필요 |

---

## 3. 수정 사항

### 3.1 prefill_frame: use_decode_rope 분기 제거

**파일**: `runner/multimodal_runner/multimodal_runner.cpp`

```cpp
// 수정 전: use_decode_rope 시 decode-time position 사용 → copy와 충돌
// 수정 후: 항상 cur_pos_ + i (순차 position)
auto prefill_res = prompt_processor_->prefill(
    merged_embeddings_, cur_pos_, /*dump_logits=*/false, nullptr);
```

- `--rope` 유무와 관계없이 항상 `cur_pos_ + i` 사용
- copy 후에도 RoPE가 최종 cache 위치와 일치

### 3.2 apply_rope_rerotation_to_k_cache 추가

**파일**: `runner/kv_manager.cpp`, `runner/kv_manager.h`

- **역할**: copy 후 K cache에 RoPE 재적용 (rotate(delta))
- **수식**: `rerotation_cos = cos(δ·θ)`, `rerotation_sin = sin(δ·θ)` (δ = new_pos - old_pos)
- **현재**: `delta=0`으로 호출 → identity (cur_pos_+i면 이미 맞음)
- **향후**: 다른 prefill 전략 사용 시 `delta=prefix_len` 등으로 활용 가능

### 3.3 generate 경로: copy 직후 rerotation 호출

```cpp
kv_manager_->copy_cache_range(0, prefix_len, frames_len);
kv_manager_->apply_rope_rerotation_to_k_cache(
    prefix_len, frames_len, /*delta=*/0);
```

---

## 4. RoPE 흐름 예시 (prefix_len=3, frame_len=5, 2프레임)

### 4.1 prefill_frame 1 (cur_pos_=0)

```
입력: prefix(3) + Frame1(5)
position:  0   1   2 |  3   4   5   6   7
RoPE:    R(0) R(1) R(2) R(3) R(4) R(5) R(6) R(7)

copy: [3,8) → [0,5)
cache:  [0]   [1]   [2]   [3]   [4]
RoPE:  R(3)  R(4)  R(5)  R(6)  R(7)
```

### 4.2 prefill_frame 2 (cur_pos_=5)

```
copy: [10,15) → [5,10)
cache:  [5]   [6]   [7]   [8]   [9]
RoPE:  R(10) R(11) R(12) R(13) R(14)
```

### 4.3 generate: copy + prefix prefill

```
copy: [0,10) → [3,13)
prefix prefill at 0: R(0), R(1), R(2)
```

### 4.4 최종 decode cache layout

```
[0]   [1]   [2] | [3]   [4]   [5]   [6]   [7]   [8]   [9]   [10]  [11]  [12] | [13]  [14] ...
  prefix         |  Frame1        Frame2              |  question + assistant
R(0) R(1) R(2) R(3) R(4) R(5) R(6) R(7) R(10) R(11) R(12) R(13) R(14) R(15) R(16) ...
```

---

## 5. RoPE position 디버깅

`--dump_rope_pos PATH`로 prefill 시 모델에 전달되는 position을 덤프함.

```bash
# 배치 모드
python stream_vlm.py ... --dump_rope_pos ./rope_batch.txt  # --stream 없음

# 스트리밍 모드
python stream_vlm.py ... --stream --dump_rope_pos ./rope_stream.txt
```

덤프 파일은 `save_log` 디렉터리 또는 출력 파일과 같은 폴더에 pull됨.

덤프 형식: `prefill pos=<start> n=<count> [p0,p1,p2,...]` (한 줄 per prefill chunk)

배치와 스트리밍의 position 시퀀스가 동일한지 확인하려면 두 파일을 diff.

---

## 6. --rope 플래그

- **스트리밍 prefill_frame**: `--rope`와 무관하게 항상 `cur_pos_ + i` 사용
- **실질 효과**: 없음 (명령어에서 생략 가능)
- **배치 모드**: `--rope`가 decode-time RoPE에 영향을 줄 수 있음 (별도 경로)

---

## 7. 요약

| 항목 | 내용 |
|------|------|
| **핵심 수정** | prefill_frame에서 use_decode_rope 분기 제거, 항상 cur_pos_+i 사용 |
| **추가 인프라** | apply_rope_rerotation_to_k_cache (현재 delta=0으로 no-op) |
| **결과** | Hybrid + stream + --rope 조합에서도 정상 출력 |

---

## 참고

- `how_to_install.md` Step 8 — 스트리밍 실행 예시
- `executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_runner.cpp` — prefill_frame
- `executorch/examples/qualcomm/oss_scripts/llama/runner/kv_manager.cpp` — copy_cache_range, apply_rope_rerotation
