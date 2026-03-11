# Foundation KV Cache Notes

foundation 경로에서 QNN과 XNNPACK은 **둘 다 KV cache를 사용**하지만,
메모리 할당 방식과 실행 shape 성격은 다릅니다.

## QNN

- 기본값은 `--no-lazy_kv_alloc` 이고, `--lazy_kv_alloc` 를 줬을 때만 lazy 모드로 동작
- `max_context_len=2048` 로 export/compile 했으면 **2048 capacity 기준 그래프/버퍼 구조**를 가짐
- 하지만 **매 step마다 2048개의 유효 토큰을 전부 계산하는 것은 아님**
- 실제로는 현재까지 채워진 KV만 의미가 있고, decode는 고정 token-generator shape로 진행
- `--lazy_kv_alloc` 사용 시 KV cache 물리 메모리는 `mmap + MAP_NORESERVE` 로 lazy commit
- 즉 **가상 주소 공간은 크게 잡아도 RSS는 쓰면서 점점 증가**할 수 있음
- `--no-lazy_kv_alloc` 사용 시 `std::vector` 기반으로 KV 전체를 즉시 물리 메모리에 잡음

## XNNPACK

- `max_context_len=2048` 로 export하면 **2048 capacity 기준 KV buffer를 모델 내부 buffer로 보유**
- 실제 추론에서 11번째 토큰을 계산할 때는 **앞의 10토큰 KV를 의미상 사용**
- 다만 메모리 구조는 export된 decoder 내부 buffer(`register_buffer`) 기준이므로,
  현재 foundation 경로에서는 QNN 같은 lazy 물리 할당은 없음
- 즉 **연산 의미상 유효 KV는 현재 길이 기준**이지만,
  **메모리 buffer capacity는 max_context_len 기준**으로 보이는 쪽에 가까움

## 한 줄 요약

- **QNN**: 정적 shape 성격이 강하고, `lazy_kv_alloc` 로 물리 메모리 commit을 늦출 수 있음
- **XNNPACK**: 연산은 현재 길이 기준으로 진행되지만, KV buffer는 현재 구조상 모델 내부에 미리 잡혀 있음
- 따라서 **"점차 차오르는 KV 메모리"** 는 현재 foundation 기준으로는 QNN lazy 쪽이 맞고,
  XNNPACK은 아직 같은 방식으로 동작하지 않음
