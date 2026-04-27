# LlamaDemo Android 앱 코드 분석

> 경로: `executorch-examples/llm/android/LlamaDemo`

---

## 1. 전체 폴더 구조

```
LlamaDemo/
├── app/
│   ├── build.gradle.kts
│   └── src/main/
│       ├── java/com/example/executorchllamademo/
│       │   ├── Activities/
│       │   │   ├── MainActivity.kt              # 메인 채팅 화면
│       │   │   ├── WelcomeActivity.kt           # 시작 화면
│       │   │   ├── ModelSettingsActivity.kt     # 모델 설정 화면
│       │   │   ├── SelectPresetModelActivity.kt # 프리셋 모델 선택
│       │   │   ├── LogsActivity.kt              # 로그 화면
│       │   │   └── AppSettingsActivity.kt       # 앱 설정 화면
│       │   ├── Data Classes/
│       │   │   ├── Message.kt                   # 채팅 메시지
│       │   │   ├── MessageType.kt               # 메시지 타입 (TEXT/IMAGE/SYSTEM)
│       │   │   ├── ModuleSettings.kt            # 모델 설정 데이터
│       │   │   ├── ModelConfiguration.kt        # 모델 구성 정보
│       │   │   ├── AppSettings.kt               # 앱 전역 설정
│       │   │   ├── ModelType.kt                 # 모델 타입 enum
│       │   │   ├── BackendType.kt               # 백엔드 타입 (XNNPACK/QNN/MediaTek/Vulkan)
│       │   │   ├── AppearanceMode.kt            # 테마 모드
│       │   │   └── AppLog.kt                    # 로그 엔트리
│       │   ├── Utils/
│       │   │   ├── ModelUtils.kt                # 모델 카테고리 유틸
│       │   │   ├── PromptFormat.kt              # 프롬프트 포맷팅
│       │   │   ├── DemoSharedPreferences.kt     # SharedPreferences 래퍼
│       │   │   ├── ETLogging.kt                 # 로깅 시스템
│       │   │   ├── ETImage.kt                   # 이미지 처리
│       │   │   ├── ModelDownloadConfig.kt       # 모델 다운로드 설정
│       │   │   └── PresetConfigManager.kt       # 프리셋 설정 관리
│       │   └── ui/
│       │       ├── screens/                     # Compose 화면
│       │       │   ├── ChatScreen.kt
│       │       │   ├── WelcomeScreen.kt
│       │       │   ├── ModelSettingsScreen.kt
│       │       │   ├── SelectPresetModelScreen.kt
│       │       │   ├── LogsScreen.kt
│       │       │   └── AppSettingsScreen.kt
│       │       ├── components/                  # 재사용 컴포넌트
│       │       │   ├── MessageItem.kt
│       │       │   ├── ChatInput.kt
│       │       │   ├── SettingsRow.kt
│       │       │   └── ModelListItem.kt
│       │       ├── viewmodel/
│       │       │   ├── ChatViewModel.kt         # 핵심 ViewModel
│       │       │   ├── ModelSettingsViewModel.kt
│       │       │   ├── SelectPresetModelViewModel.kt
│       │       │   └── LogsViewModel.kt
│       │       └── theme/
│       │           └── LlamaDemoTheme.kt
│       ├── res/
│       └── assets/
│           └── preset_models.json              # 프리셋 모델 설정
├── build.gradle.kts
├── gradle.properties
└── README.md
```

---

## 2. 앱 아키텍처

**패턴: MVVM + Jetpack Compose**

```
┌─────────────────────────────────────┐
│         UI Layer (Compose)          │
│   Screen ──── Component             │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│         ViewModel Layer             │
│   ChatVM ── SettingsVM              │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│         Data Layer                  │
│   SharedPreferences ── LlmModule    │
│                       (JNI Bridge)  │
└─────────────────────────────────────┘
```

---

## 3. 주요 클래스 역할

### Activities

| 클래스 | 역할 |
|--------|------|
| `WelcomeActivity` | 시작 화면, 네비게이션 허브 |
| `MainActivity` | 메인 채팅, 이미지/카메라 권한 처리 |
| `ModelSettingsActivity` | 모델/토크나이저/백엔드 선택 |
| `SelectPresetModelActivity` | 프리셋 모델 다운로드 및 선택 |
| `LogsActivity` | 디버그 로그 표시 |
| `AppSettingsActivity` | 테마, 채팅 히스토리 설정 |

### ViewModels

| 클래스 | 역할 |
|--------|------|
| `ChatViewModel` | 모델 로드/추론, 메시지 관리, LoRA 전환 |
| `ModelSettingsViewModel` | 모델 구성, 다중 모델 관리 |
| `SelectPresetModelViewModel` | 프리셋 다운로드 및 상태 추적 |
| `LogsViewModel` | 로그 로드/저장/삭제 |

### 백엔드별 지원 현황

| 백엔드 | 모델 카테고리 | 미디어 지원 |
|--------|--------------|------------|
| XNNPACK | TEXT_MODEL, VISION_MODEL | 이미지/오디오 |
| **QNN** | **QNN_TEXT_MODEL** | **텍스트만** |
| MediaTek | MEDIATEK_TEXT_MODEL | 텍스트만 |
| Vulkan | TEXT_MODEL, VISION_MODEL | 이미지/오디오 |

---

## 4. ExecuTorch/QNN 연동 방식

### 모델 로드

```kotlin
module = LlmModule(
    modelCategory,     // QNN_TEXT_MODEL 등
    modelPath,         // .pte 파일 경로
    tokenizerPath,     // 토크나이저 파일 경로
    temperature,       // 온도 파라미터
    dataPath           // .ptd 파일 경로 (LoRA 어댑터)
)
module.load()          // JNI → ExecuTorch 런타임 초기화
```

### 이미지 Prefill (비전 모델)

```kotlin
// LLaVA-1.5: IntArray
module?.prefillImages(img.getInts(), img.width, img.height, channels)

// Gemma 3: FloatArray (정규화됨)
module?.prefillImages(img.getFloats(), img.width, img.height, channels)
```

### 텍스트 생성

```kotlin
module?.generate(
    finalPrompt,           // 포맷된 프롬프트
    appSettings.maxSeqLen, // 최대 시퀀스 길이
    this,                  // LlmCallback
    false
)

// 콜백
override fun onResult(result: String) {
    resultMessage?.appendText(result)  // 토큰 단위로 수신
}

override fun onStats(stats: String) {
    // JSON 통계: tokens/s, 추론 시간 등
}
```

### LoRA 다중 모델 로드

```kotlin
for (modelConfig in settings.models) {
    val llmModule = LlmModule(
        modelCategory,
        modelConfig.modelFilePath,      // 각 모델 PTE
        modelConfig.tokenizerFilePath,
        temperature,
        listOf(sharedDataPath) + modelConfig.adapterFilePaths
    )
    llmModule.load()
    loadedModules[modelConfig.id] = llmModule
}
```

---

## 5. 앱 실행 Flow

```
WelcomeActivity
    │
    ├── "Load local model" ──→ ModelSettingsActivity
    │                             └── 모델/토크나이저/백엔드 선택
    │                                 └── isLoadModel=true 저장
    │
    └── MainActivity (ChatViewModel)
            ├── ModuleSettings 읽기
            ├── LlmModule 생성 및 load()
            └── 사용자 입력
                    ├── 이미지 첨부 → prefillImages()
                    ├── 프롬프트 포맷팅
                    └── generate() 호출
                            └── onResult() 콜백 → UI 업데이트
```

---

## 6. 지원 모델 타입

| 모델 | 타입 | 미디어 |
|------|------|--------|
| `LLAMA_3`, `LLAMA_2` | 텍스트 전용 | - |
| `LLAVA_1_5` | 멀티모달 | 이미지 |
| `GEMMA_3` | 멀티모달 | 이미지 |
| `VOXTRAL` | 멀티모달 | 오디오 |
| `QWEN_3` | 텍스트 전용 | - |
| `LLAMA_GUARD_3` | 텍스트 전용 | - |

---

## 7. 빌드 설정 주요 사항

```kotlin
// build.gradle.kts
android {
    compileSdk = 34
    defaultConfig {
        minSdk = 28
        targetSdk = 33
    }
    buildFeatures { compose = true }
}

dependencies {
    implementation("org.pytorch:executorch-android:1.1.0")
    implementation("com.facebook.fbjni:fbjni:0.5.1")
    implementation("com.google.code.gson:gson:2.8.6")
}
```

### AndroidManifest 주요 권한

```xml
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.INTERNET" />

<!-- QNN/MediaTek 네이티브 라이브러리 -->
<uses-native-library android:name="libcdsprpc.so" android:required="false" />
```

---

## 8. KV Cache 관리 방식

KV Cache는 **Kotlin에서 직접 관리하지 않으며**, ExecuTorch 네이티브(C++) 레이어가 실제 메모리를 관리합니다.
Kotlin(`ChatViewModel`)은 제어 신호만 보냅니다.

```
┌─────────────────────────────────────────────────────┐
│              Kotlin (ChatViewModel)                 │
│                                                     │
│  module.load()          ← maxSeqLen 크기만큼 할당   │
│  module.prefillPrompt() ← 텍스트 토큰 → KV Cache    │
│  module.prefillImages() ← 이미지 → 임베딩 변환만    │
│                           (KV Cache는 generate 시)  │
│  module.generate()      ← 토큰 생성 + KV Cache 누적 │
│  module.resetContext()  ← KV Cache 초기화           │
│  module.stop()          ← 생성 중단                 │
│                                                     │
│  appSettings.maxSeqLen  ← KV Cache 크기 설정        │
└─────────────────────────────────────────────────────┘
             ↓ JNI Bridge
┌─────────────────────────────────────────────────────┐
│       ExecuTorch Native (C++)                       │
│       실제 KV Cache 메모리 관리                      │
└─────────────────────────────────────────────────────┘
```

### 메서드별 KV Cache 관계

| 메서드 | 역할 | KV Cache |
|--------|------|----------|
| `module.load()` | 모델 로드 | `maxSeqLen` 크기만큼 **할당** |
| `prefillPrompt()` | 텍스트 토큰 → Transformer 통과 | **직접 채움** |
| `prefillImages()` | 이미지 → Vision Encoder → 임베딩 추출 | **채우지 않음** |
| `generate()` | 토큰 생성 | **채우면서 진행** |
| `resetContext()` | 컨텍스트 초기화 | **전체 초기화** |

### KV Cache 생명주기

```
module.load(maxSeqLen)
    └─> KV Cache 메모리 할당 (maxSeqLen 크기)

prefillPrompt() / generate()
    └─> KV Cache 누적 (멀티턴 컨텍스트 유지)

prefillImages()
    └─> Vision Encoder(SigLIP/ViT)로 이미지 임베딩 추출
        └─> generate() 시 LLM Decoder 통과할 때 KV Cache 채워짐

채팅 히스토리 클리어 시
    └─> module.resetContext() → KV Cache 0으로 초기화

maxSeqLen 초과 시
    └─> 네이티브에서 처리 (슬라이딩 윈도우 or 오류)
```

### maxSeqLen 설정

- 앱 설정 화면(`AppSettingsScreen`)에서 사용자가 직접 설정 가능
- 기본값: `AppSettings.DEFAULT_MAX_SEQ_LEN`
- `generate()` 호출 시 매번 전달됨

```kotlin
module?.generate(finalPrompt, appSettings.maxSeqLen, this, false)
```

---

## 9. QNN 백엔드 사용 시 주의사항

- QNN 백엔드는 **텍스트 전용** (`QNN_TEXT_MODEL`)
- 이미지 입력 불가 (비전 기능 미지원)
- 디바이스에 QNN 라이브러리 푸시 필요:
  ```bash
  adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so /data/local/tmp/
  adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so /data/local/tmp/
  adb push $QNN_SDK_ROOT/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so /data/local/tmp/
  ```
- `build.gradle.kts`에서 QNN 런타임 버전이 SDK 버전과 일치해야 함:
  ```kotlin
  implementation("com.qualcomm.qti:qnn-runtime:2.37.0")
  ```
