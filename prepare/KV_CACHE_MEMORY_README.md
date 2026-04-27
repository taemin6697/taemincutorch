# KV Cache 메모리 지표 정리

이 문서는 `stream_output.txt.proc.csv`, `stream_output.txt.mem.csv`, `stream_output.txt.smaps`를 해석할 때 자주 헷갈리는 메모리 지표를 정리한 메모다.

목표는 다음 3가지다.

1. `kv_used_kb`가 무엇인지 명확히 구분한다.
2. `RSS`가 왜 오르내리는지 설명한다.
3. 논문에 어떤 메트릭을 메인으로 쓰는 것이 좋은지 정리한다.

---

## 1. 한 줄 요약

- `kv_used_kb`는 **logical KV footprint**다.
- `rss_kb`는 **프로세스 전체의 실제 resident memory**다.
- 따라서 `kv_used_kb`는 계속 증가해도 `rss_kb`는 중간에 감소할 수 있다.

---

## 2. 지표별 의미

### 2.1 `kv_used_kb`

`kv_used_kb`는 OS가 읽어준 메모리 값이 아니다.

이 값은 보통:

- 현재 `kv_pos`
- 전체 context 길이 `kv_total`
- 전체 KV cache 크기 `kv_total_kb`

를 이용해 계산한 **논리적 KV 사용량**이다.

즉 의미는:

- "현재까지 채워진 KV cache가 이론적으로 몇 KB를 차지하는가"

논문 용어로는 다음 표현이 가장 가깝다.

- `logical KV-cache footprint`
- `context memory footprint`
- `logical context memory`

### 2.2 `rss_kb`

`rss_kb`는 `/proc/self/status`의 `VmRSS` 같은 값으로, 현재 프로세스가 실제 RAM에 올려 둔 메모리 총량이다.

이 안에는 다음이 모두 섞인다.

- model weights
- activation buffer
- KV cache
- temporary scratch buffer
- allocator가 잡고 있는 메모리
- runtime/library mapping

즉 의미는:

- "지금 이 순간 프로세스가 실제로 점유하는 resident memory"

### 2.3 `PSS`

`PSS`는 shared mapping을 프로세스 간에 나눠서 계산한 실질 점유 메모리다.

공유 라이브러리나 공유 매핑이 많을 때 `RSS`보다 공정한 값이다.

### 2.4 `VmHWM`

`VmHWM`은 프로세스가 살아있는 동안 기록된 최대 RSS다.

즉:

- peak RSS
- worst-case physical memory footprint

를 볼 때 좋다.

---

## 3. `kv_used_kb`와 `rss_kb`가 왜 다르나

두 값은 아예 측정 대상이 다르다.

### `kv_used_kb`

- KV cache만 본다.
- 논리적으로 얼마나 채워졌는지 본다.
- 보통 단조 증가한다.

### `rss_kb`

- 프로세스 전체를 본다.
- 임시 버퍼, allocator, 파일 매핑, anon memory가 전부 포함된다.
- 중간에 줄어들 수 있다.

즉 아래 현상은 정상이다.

- `kv_used_kb` 증가
- `rss_kb` 소폭 감소

이건 KV가 줄었다는 뜻이 아니라, 임시 메모리나 익명 메모리 일부가 resident에서 빠졌다는 뜻이다.

---

## 4. 왜 `delta_rss_kb`가 음수가 되나

`mem.csv`의 `delta_rss_kb`가 `-128 KB`, `-256 KB`, 혹은 더 큰 음수가 되는 건 주로 다음 이유 때문이다.

1. 임시 버퍼 사용 종료
2. allocator가 일부 large allocation을 반환하거나 재정리
3. lazy allocation된 페이지 일부가 더 이상 resident가 아님
4. OS page accounting 또는 reclaim

특히 `-128 KB`, `-256 KB`는 매우 작은 페이지 단위 변화다.

- `128 KB = 32 pages` (`4 KB` 페이지 기준)
- `256 KB = 64 pages`

즉 대개는:

- 큰 메모리 구조가 무너진 것 아님
- KV cache가 줄어든 것 아님
- resident page 일부가 빠진 것

으로 해석하는 것이 맞다.

---

## 5. `anonymous private-dirty`가 뭔가

`smaps`에서 자주 보이는 `anonymous private-dirty`는 다음 뜻이다.

### `anonymous`

파일 기반 매핑이 아니다.

예:

- `malloc`
- `new`
- `mmap(MAP_ANONYMOUS)`

같은 방식으로 잡은 런타임 메모리다.

### `private`

현재 프로세스만 쓰는 메모리다.

### `dirty`

읽기만 한 페이지가 아니라, 실제로 내용을 써서 변경한 페이지다.

즉 `anonymous private-dirty`는 보통 이런 성격이다.

- heap
- scratch buffer
- intermediate activation buffer
- planner/runtime buffer
- allocator 내부 large block

따라서 `RSS` 감소가 보이면, 파일 매핑보다 이런 익명 작업 메모리 쪽이 줄었을 가능성이 더 높다.

---

## 6. `smaps`는 어디에 쓰나

`smaps`는 시간축 분석용이 아니라 **메모리 breakdown 확인용**이다.

볼 수 있는 것:

- 큰 anon mapping이 무엇인지
- `.pte` file mapping RSS가 큰지 작은지
- allocator(`scudo`)가 잡고 있는 메모리가 큰지
- `Private_Dirty`, `Anonymous`, `Rss`가 큰 매핑이 무엇인지

하지만 `smaps` 단일 스냅샷만으로는 아래는 알 수 없다.

- 정확히 어느 시점에 줄었는지
- 어떤 매핑이 그 순간 감소했는지

즉:

- `mem.csv` = 언제 줄었는지
- `proc.csv` = 어느 phase인지
- `smaps` = 무슨 종류의 메모리가 큰지

로 나눠서 봐야 한다.

---

## 7. 기존 논문들은 메모리를 어떻게 봤나

기존 KV cache 관련 논문들은 보통 `RSS` 타임라인보다 다음을 더 많이 쓴다.

### 7.1 KV footprint / context memory

예:

- `context = 2.02 GB`
- `KV cache size`
- `context memory`

이건 보통 모델 구조와 context length로 계산한 값이다.

### 7.2 Peak memory

예:

- `peak memory`
- `peak memory including model weight`
- `2.6x less peak memory`

### 7.3 Memory budget

예:

- `1GB / 2GB / 3GB` 예산에서 active context 수
- 주어진 latency 제약에서 유지 가능한 context 수

### 7.4 Traffic / transfer volume

예:

- `Flash -> DRAM (GB)`
- off-chip traffic
- swap I/O bytes

즉 기존 논문은 대체로:

- footprint
- budget
- traffic
- latency

를 같이 본다.

---

## 8. 논문에 뭘 메인으로 쓰는 게 좋은가

### 권장

논문 메인 지표는 `KV footprint` 쪽이 더 좋다.

이유:

- 모델/방법 효과를 직접 설명하기 쉬움
- seq_len, token 수, compression 효과를 명확히 보여줌
- 기존 KV cache 논문과 비교하기 쉬움

### 보조로 같이 넣을 것

- `RSS` 또는 가능하면 `PSS`
- `VmHWM`
- `TTFT`
- `decode tok/s`

즉 논문에선 다음 구성이 가장 안전하다.

1. `logical KV footprint`
2. `peak physical memory`
3. `latency / throughput`

---

## 9. 이 프로젝트에서 추천하는 메트릭

현재 로그 체계 기준 추천 순서는 아래와 같다.

### 메인

- `kv_used_kb`
- `kv_used_pct`
- `kv_total_kb`
- `TTFT`
- `decode tok/s`

### 보조

- `RSS`
- `VmHWM`
- 가능하면 `PSS`
- `MemAvailable`

### 있으면 좋은 것

- `kv_resident_kb`
- DRAM/off-chip traffic
- swap read/write bytes

---

## 10. 실전 해석 규칙

### 규칙 1

`kv_used_kb` 증가 = KV footprint 증가

### 규칙 2

`rss_kb` 감소 = KV 감소라고 해석하면 안 됨

### 규칙 3

`delta_rss_kb < 0`는 보통:

- temp buffer 종료
- allocator 정리
- page reclaim

중 하나다.

### 규칙 4

어떤 종류의 메모리가 큰지는 `smaps`로 보고,
언제 변했는지는 `mem.csv`/`proc.csv`로 본다.

---

## 11. 한 줄 결론

- `kv_used_kb`는 **logical KV footprint**
- `rss_kb`는 **physical process memory**
- 논문 메인은 **KV footprint + latency**
- 시스템 검증은 **peak RSS/PSS/VmHWM**

이 구분만 정확히 하면 로그 해석과 논문 작성이 훨씬 쉬워진다.
