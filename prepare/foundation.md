# Unified VLM Foundation

## 목표

- QNN / XNNPACK 공통 artifact 규약 사용
- `manifest.json` 기반 export / run 통합
- 최종적으로는 `--backend qnn|xnnpack`를 받는 단일 C++ runner로 수렴
- 우선순위는 배치 decode loop이며, streaming은 후순위

## 현재 구현

- 공통 Python 패키지: `executorch/examples/models/foundation/`
- 공통 manifest 모듈: `executorch/examples/models/foundation/manifest.py`
- 공통 host CLI: `executorch/examples/models/foundation/cli.py`
- 공통 host launcher: `executorch/examples/models/foundation/host/launcher.py`
- 공통 C++ runner entry/scaffold:
- `executorch/examples/models/foundation/runner/xnnpack_qnn_runner.cpp`
- `executorch/examples/models/foundation/runner/xnnpack_backend.cpp`
- `executorch/examples/models/foundation/runner/qnn_backend.cpp`
- `executorch/examples/models/foundation/runner/backend.h`

## 공통 산출물 규약

```text
artifact_root/
  manifest.json
  tokenizer/
    tokenizer.json
    ...
  models/
    vision_encoder.pte
    text_embedding.pte
    text_decoder.pte
  metadata/
    export_args.json
```

실제 기존 스크립트 호환을 위해 현재는 backend별 기존 폴더 구조를 일부 유지할 수 있고,
`manifest.json`이 canonical contract 역할을 한다.

## manifest 주요 필드

```json
{
  "schema_version": 1,
  "backend": "qnn",
  "model_family": "internvl3",
  "variant": "internvl3_1b",
  "runner_type": "multimodal_split",
  "paths": {
    "artifact_root": "...",
    "vision_encoder_pte": "...",
    "text_embedding_pte": "...",
    "text_decoder_pte": "...",
    "tokenizer_path": "..."
  },
  "export": {
    "max_seq_len": 2048,
    "max_context_len": 2048
  },
  "quant": {
    "vision": "fp16",
    "decoder": "fp16",
    "embedding": "fp16"
  },
  "runtime": {
    "preferred_runner": "xnnpack_qnn_runner"
  }
}
```

## 현재 backend 연결 상태

### QNN

- foundation CLI `export --backend qnn` 으로 native export 수행 (기존 `llama.py` 엔트리 직접 호출 없음)
- `xnnpack_qnn_runner`가 QNN **split PTE 3개**를 직접 읽는 batch 경로 사용
- foundation host launcher는 `--runner_binary xnnpack_qnn_runner` 필수, fallback 없음

### XNNPACK

- foundation CLI `export --backend xnnpack` 으로 native export 수행 (기존 `export_xnnpack_multimodal` 엔트리 직접 호출 없음)
- `xnnpack_qnn_runner`가 XNNPACK **split PTE 3개**를 직접 읽는 batch 경로 사용
- foundation host launcher는 `--runner_binary xnnpack_qnn_runner` 필수
- streaming은 아직 범위 밖이며 batch decode loop 우선

## 단일 C++ runner 상태

`xnnpack_qnn_runner`는 현재 다음까지 구현됨:

- `manifest.json` 로드
- split PTE / tokenizer 경로 검증
- 단일 backend entry 확보
- XNNPACK split-PTE batch 실행 경로 연결
- QNN split-PTE batch 실행 경로 연결

아직 미완료:

- QNN/XNNPACK 공통 streaming loop 이식

## 현재 빌드 정책

### XNNPACK

- XNNPACK foundation runner는 QNN이 포함된 `build-android`를 재사용하지 않고, 별도 `build-android-xnnpack` build tree를 사용
- 이유: QNN이 섞인 build tree에서 `--backend xnnpack` 실행 시 `aten::gelu.out`, `aten::native_layer_norm.out`, `dim_order_ops::_to_dim_order_copy.out` 등 missing operator가 발생할 수 있음
- runner 경로: `executorch/build-android-xnnpack/foundation/xnnpack_qnn_runner`

### QNN

- QNN foundation runner는 Qualcomm `build.sh`가 생성한 `build-android` tree 사용
- runner 경로: `executorch/build-android/foundation/xnnpack_qnn_runner`

### 빌드 순서 요약

1. XNNPACK이면 `build-android-xnnpack`, QNN이면 `build-android` 준비
2. `executorch/examples/models/foundation/`를 같은 tree 아래에서 cmake configure/build
3. foundation CLI `export`
4. foundation CLI `run`

## 다음 구현 우선순위

1. QNN/XNNPACK 공통 streaming loop 이식
2. 실제 디바이스 빌드/실행 smoke 검증