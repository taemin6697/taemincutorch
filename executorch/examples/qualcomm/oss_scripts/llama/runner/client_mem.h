/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/imem_alloc.h>
#include <cstdint>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace example {
/**
 * @class ClientMem
 * @brief Final class for client buffer allocation, implementing IBufferAlloc
 * interface. This is specifically designed for use cases without shared buffer.
 *
 * [메모리 할당 전략 선택]
 *
 * ── 방법 A: std::vector (원본, 즉시 전체 물리 메모리 점유) ──────────────
 *   allocated_buffers_.push_back(std::vector<std::byte>(data_size));
 *   return allocated_buffers_.back().data();
 *   → 생성자가 zero-init(memset) 수행 → 모든 물리 페이지 즉시 commit
 *   → max_seq_len=1024: ~48MB 즉시 점유 (안전)
 *   → max_seq_len=32768: ~1.6GB 즉시 점유 → load() 단계에서 OOM
 *
 * ── 방법 B: mmap+MAP_NORESERVE (현재 사용, lazy 물리 메모리 할당) ────────
 *   가상 주소만 예약, swap도 예약하지 않음
 *   실제 write가 일어나는 페이지만 그때그때 물리 메모리 할당
 *   → 프레임 처리할수록 RSS가 조금씩 증가
 *   → 대용량 max_seq_len에서도 load()는 성공
 *   → 단, 물리메모리 고갈 시 write 시점에 SIGBUS/OOM killer로 종료
 *
 * 방법 A로 되돌리려면: allocate() 안의 mmap 블록을 주석 처리하고
 *   방법 A 블록의 주석을 해제하면 됩니다.
 */
class ClientMem final : public IMemAlloc {
 public:
  /**
   * @param lazy true  → 방법 B: mmap+MAP_NORESERVE (lazy 물리 할당, 대용량 seq_len 안전)
   *             false → 방법 A: std::vector zero-init (즉시 전체 물리 점유, 예측 가능)
   */
  explicit ClientMem(bool lazy = true) : lazy_(lazy) {}
  // Disable copy constructors, r-value referencing, etc
  ClientMem(const ClientMem&) = delete;
  ClientMem& operator=(const ClientMem&) = delete;
  ClientMem(ClientMem&&) = delete;
  ClientMem& operator=(ClientMem&&) = delete;

  virtual ~ClientMem() {
    // 방법 B: mmap으로 잡은 메모리 해제
    for (auto& [ptr, size] : mmap_buffers_) {
      if (ptr != MAP_FAILED) {
        munmap(ptr, size);
      }
    }
    // 방법 A: vector 소멸자가 자동으로 해제 — 별도 처리 불필요
  };

  /**
   * @brief 설정된 전략으로 버퍼를 할당한다.
   *   lazy=true  → mmap+MAP_NORESERVE: 가상 주소만 예약, 물리 페이지는 첫 write 시 커밋
   *   lazy=false → std::vector<std::byte>: zero-init → 모든 물리 페이지 즉시 커밋
   * @param data_size 할당할 바이트 수.
   * @return 할당된 버퍼의 시작 포인터.
   */
  std::byte* allocate(size_t data_size) override {
    if (lazy_) {
      // ── 방법 B: mmap lazy commit ───────────────────────────────────────────
      void* ptr = mmap(
          nullptr,
          data_size,
          PROT_READ | PROT_WRITE,
          MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
          -1,
          0);
      mmap_buffers_.emplace_back(ptr, data_size);
      return static_cast<std::byte*>(ptr);
    } else {
      // ── 방법 A (원본): 즉시 전체 물리 메모리 점유 ──────────────────────────
      allocated_buffers_.push_back(std::vector<std::byte>(data_size));
      return allocated_buffers_.back().data();
    }
  };

  // Only used for SMART_MASK mode
  void add_memory_info(
      void* data_ptr,
      size_t data_size,
      executorch::runtime::TensorInfo tensor_info) override {};

  size_t resident_bytes(const void* data_ptr, size_t data_size) const override {
    if (!lazy_ || data_ptr == nullptr || data_size == 0) {
      return data_size;
    }

    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
      return 0;
    }

    const uintptr_t start = reinterpret_cast<uintptr_t>(data_ptr);
    const uintptr_t aligned_start = start & ~static_cast<uintptr_t>(page_size - 1);
    const uintptr_t end = start + data_size;
    const uintptr_t aligned_end =
        (end + static_cast<uintptr_t>(page_size - 1)) &
        ~static_cast<uintptr_t>(page_size - 1);
    const size_t length = aligned_end - aligned_start;
    const size_t page_count = length / static_cast<size_t>(page_size);
    if (page_count == 0) {
      return 0;
    }

    std::vector<unsigned char> residency(page_count, 0);
    if (mincore(
            reinterpret_cast<void*>(aligned_start),
            length,
            residency.data()) != 0) {
      return 0;
    }

    size_t resident_pages = 0;
    for (unsigned char value : residency) {
      resident_pages += static_cast<size_t>(value & 1U);
    }
    return resident_pages * static_cast<size_t>(page_size);
  }

 private:
  bool lazy_;

  // 방법 B용: mmap 포인터 + 크기 목록 (소멸자에서 munmap 호출)
  std::vector<std::pair<void*, size_t>> mmap_buffers_;

  // 방법 A용: vector 기반 버퍼 목록
  std::vector<std::vector<std::byte>> allocated_buffers_;
};

} // namespace example
