# Foundation 실행에 필요한 파일 목록

`git clone` 한 executorch 위에 **아래 파일만 복사**하면 foundation이 동작합니다.

---

## 1. Foundation 폴더 전체 (필수)

```
executorch/examples/models/foundation/
├── __init__.py
├── cli.py
├── CMakeLists.txt
├── export.py
├── export_internvl3_all_lengths.sh
├── manifest.py
├── README.md
├── REQUIRED_FILES.md          # 이 파일
├── exporters/
│   ├── __init__.py
│   ├── qnn.py
│   └── xnnpack.py
├── host/
│   ├── __init__.py
│   ├── adb_runner.py          # QNN용 (신규)
│   ├── frame_extractor.py     # 이미지/비디오 추출 (신규)
│   └── launcher.py
└── runner/
    ├── backend.h
    ├── qnn_backend.cpp
    ├── xnnpack_backend.cpp
    └── xnnpack_qnn_runner.cpp
```

---

## 2. QNN export 시 추가로 필요한 수정 파일

Foundation QNN export는 `qualcomm.oss_scripts.llama`를 사용합니다. 아래 수정이 필요합니다.

| 파일 | 용도 |
|------|------|
| `executorch/examples/qualcomm/oss_scripts/llama/__init__.py` | InternVL3_8B `num_sharding=8` |
| `executorch/examples/qualcomm/oss_scripts/llama/wrappers/llm_wrappers.py` | Vision encoder fp16 dtype 변환 (Modality) |

---

## 2-1. QNN run 시 필요한 qualcomm runner (배치 처리)

`set_image_hidden_states_batch` 등 배치 로직은 **qualcomm runner C++**에 있습니다.  
QNN 빌드 시 foundation CMake가 아래 소스를 직접 컴파일합니다.

| 폴더/파일 | 용도 |
|-----------|------|
| `executorch/examples/qualcomm/oss_scripts/llama/runner/` | decoder, prompt, token_generator, kv_manager 등 |
| `executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/` | `set_image_hidden_states_batch`, encoder, embedding 등 |

**주의:** vanilla executorch clone에는 `examples/qualcomm/`이 없을 수 있음.  
QNN run을 쓰려면 `qualcomm/oss_scripts/llama/` 전체가 필요합니다.

---

## 3. XNNPACK export 시

`executorch.examples.models.internvl3`, `executorch.examples.models.llama` 등 기존 구조 사용.  
별도 수정 없이 동작 (foundation/xnnpack.py가 기존 모듈 import).

---

## 4. Run 전용 (export 없이 기존 artifact 사용)

**Run만** 할 경우:
- **Foundation 폴더 전체** (1번)만 있으면 됨
- `stream_vlm`, `run_xnnpack_vlm`, `qualcomm.llama` **미사용** (frame_extractor, adb_runner로 대체)

---

## 5. 빌드

Runner 바이너리(`xnnpack_qnn_runner`)는 CMake로 빌드:

- **QNN**: `executorch/build-android/foundation/`
- **XNNPACK**: `executorch/build-android-xnnpack/foundation/`

README의 빌드 절차 참고.

---

## 6. 최종 요약

### XNNPACK만 쓸 때
| 항목 | 필요 여부 |
|------|-----------|
| `foundation/` 폴더 전체 | ✅ 필수 |
| `qualcomm/` | ❌ 불필요 |
| 추가 수정 파일 | ❌ 없음 |

### QNN만 쓸 때
| 항목 | 필요 여부 |
|------|-----------|
| `foundation/` 폴더 전체 | ✅ 필수 |
| `executorch/examples/qualcomm/oss_scripts/llama/` 전체 | ✅ 필수 (runner C++, set_image_hidden_states_batch 등) |
| `llama/__init__.py` 수정 | export 시만 |
| `llama/wrappers/llm_wrappers.py` 수정 | export 시만 |

### 한 줄 정리
- **XNNPACK**: `foundation/`만 복사하면 됨
- **QNN**: `foundation/` + `qualcomm/oss_scripts/llama/` 전체 필요 (vanilla executorch에는 qualcomm 없을 수 있음 → 별도 추가 필요)
