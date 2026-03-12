/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Multimodal runner that extends the base llama runner with vision capabilities

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/cache_utils.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/decoder_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/imem_alloc.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/kv_manager.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/embedding_processor.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/embedding_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_prompt_processor.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_token_generator.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/module/module.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace example {

// Extend DecoderModelVersion enum with multimodal models
enum MultimodalDecoderModelVersion {
  kSmolvlm = 0,
  kInternvl3,
};

enum KvBitWidth {
  kWidth8 = 8,
  kWidth16 = 16,
};

template <typename T>
class MultimodalRunner : public executorch::extension::llm::IRunner {
 public:
  struct PhaseTiming {
    long start_ms{0};
    long end_ms{0};
    long rss_kb_start{0};
    long rss_kb_end{0};
  };

  struct GeneratePhaseTimings {
    PhaseTiming embedding_and_merging;
    PhaseTiming prefill;
    PhaseTiming decode;
  };

  explicit MultimodalRunner(
      std::unique_ptr<executorch::extension::Module> module,
      std::unique_ptr<executorch::extension::Module> embedding_module,
      const std::string& decoder_model,
      const std::string& model_path,
      const std::string& tokenizer_path,
      const std::string& performance_output_path,
      const std::string& dump_logits_path,
      const float temperature = 0.8f,
      const int eval_mode = EvalMode::kHybrid,
      const bool shared_buffer = false,
      const bool lazy_kv_alloc = true,
      const int ngram = 0,
      const int window = 0,
      const int gcap = 0,
      std::unique_ptr<executorch::aten::Tensor> image_hidden_states = nullptr);

  bool is_loaded() const override;
  executorch::runtime::Error load() override;

  // Override generate to support multimodal inputs
  executorch::runtime::Error generate(
      const std::string& prompt,
      const executorch::extension::llm::GenerationConfig& config,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const executorch::llm::Stats&)> stats_callback = {})
      override;

  // Multimodal-specific generation with image embeddings
  executorch::runtime::Error generate_from_prompt_or_file(
      const std::string& prompt,
      bool tokenized_prompt,
      const executorch::extension::llm::GenerationConfig& config,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const executorch::llm::Stats&)> stats_callback = {});
  void stop() override {};
  void reset() override {};
  executorch::runtime::Result<MultimodalDecoderModelVersion>
  get_decoder_model_version();

  // Multimodal-specific method for merging embeddings
  void merge_multimodal_embeddings(
      const std::vector<uint64_t>& input_ids,
      const TensorStruct<float>& text_embeddings,
      uint64_t placeholder_token_id);

  // ── Streaming VLM API ──────────────────────────────────────────────────────
  /**
   * Prefill the conversation prefix once at startup (before any frames).
   * SmolVLM  : "<|im_start|>User:"
   * InternVL3: "<|im_start|>user:\n"
   * Must call load() first.
   */
  executorch::runtime::Error prefill_prefix();

  /**
   * Prefill a single video frame's image hidden states into the KV cache.
   * Call prefill_prefix() once before the first call to this method.
   * @param image_hidden_states Tensor [1, img_seq_len, hidden_dim] from encoder
   * Streaming: always prefix "Frame{N}: <img>", suffix "</img>\n" (HF format).
   */
  executorch::runtime::Error prefill_frame(
      const executorch::aten::Tensor& image_hidden_states);

  /**
   * Set image hidden states for batch (HF-style) prefill.
   * Concatenated [1, N*img_seq_len, hidden_dim]. Runner takes ownership of buffer.
   * Must be called before generate_from_prompt_or_file with full prompt.
   */
  void set_image_hidden_states_batch(
      std::vector<float>&& buffer,
      int64_t total_seq_len,
      int64_t hidden_dim);

  /**
   * Requantize the KV cache after all prefill_frame() calls.
   * Required in Hybrid mode before the first generate(); no-op in KV mode.
   */
  executorch::runtime::Error finalize_prefill();

  /** Current write position in KV cache (= tokens prefilled so far). */
  int64_t get_cur_pos() const {
    return cur_pos_;
  }
  /** Compiled context length of the loaded model. */
  int32_t get_context_len() const {
    return context_len_;
  }
  /** Actual allocated KV cache size in bytes (from KVManager). */
  size_t get_kv_cache_total_bytes() const {
    return kv_manager_ ? kv_manager_->total_cache_size_in_bytes() : 0;
  }
  /** Actually resident KV cache bytes currently committed in process memory. */
  size_t get_kv_cache_resident_bytes() const {
    return kv_manager_ && buffer_manager_
        ? kv_manager_->resident_cache_size_in_bytes(*buffer_manager_)
        : 0;
  }
  /** Modality placeholder token id exposed for the streaming runner. */
  uint64_t get_placeholder_token_id();

  GeneratePhaseTimings get_last_generate_phase_timings() const {
    return last_generate_phase_timings_;
  }

  /**
   * Return the exact string that prefill_prefix() tokenizes.
   * Used for kv_log to show actual inference input (no manual formatting).
   */
  std::string get_prefix_format_string() const;

  /**
   * Return the exact string that prefill_frame() tokenizes for the given frame.
   * Same format as tokenizer_->encode(prefix_str) + placeholders + encode(suffix_str).
   */
  std::string get_frame_format_string(int frame_idx, int64_t img_seq_len) const;

  /** Optional: (phase, start_ms, end_ms) for L_V_Encode/L_Decoder/L_Embedding proc.csv */
  void set_load_phase_callback(
      std::function<void(const char* phase, long start_ms, long end_ms)> cb) {
    load_phase_cb_ = std::move(cb);
  }

  /**
   * Write QNN/ExecuTorch profiling etdump to file if EventTracer is ETDumpGen.
   * No-op if path empty or event tracer not available.
   */
  void write_etdump(const std::string& path) const;

 private:
  enum EvalMode {
    kKVCached = 0,
    kHybrid,
    kLookaheadDecoding,
    kUnsupported,
  };

  // Modules
  std::unique_ptr<executorch::extension::Module> module_;
  std::unique_ptr<executorch::extension::Module> embedding_module_;

  int32_t context_len_{0};

  int ngram_{0};
  int window_{0};
  int gcap_{0};

  // Defaults to StaticCahce, indicating that the model does not use a
  // global/local architecture.
  CacheMode cache_mode_{CacheMode::StaticCahce};
  int64_t cur_pos_{0};

  std::string tokenizer_path_;
  std::string performance_output_path_;
  std::string dump_logits_path_;
  float temperature_;
  EvalMode eval_mode_;
  bool shared_buffer_;
  bool lazy_kv_alloc_;

  MultimodalDecoderModelVersion decoder_model_version_;
  std::unique_ptr<IMemAlloc> buffer_manager_;
  std::unique_ptr<KVManager<T>> kv_manager_;
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
  std::unique_ptr<DecoderRunner> decoder_runner_;
  std::unique_ptr<MultimodalPromptProcessor<T>> prompt_processor_;
  std::unique_ptr<MultimodalTokenGenerator<T>> token_generator_;
  std::unique_ptr<EmbeddingRunner> embedding_runner_;
  std::unique_ptr<EmbeddingProcessor> embedding_processor_;
  std::unique_ptr<EmbeddingProcessor> embedding_generator_;

  // Streaming state
  bool frames_prefilled_{false};
  bool kv_requantized_{false};
  int32_t prefilled_frame_count_{0}; // counts prefill_frame() calls
  bool prefix_prefilled_{false};     // true after prefill_prefix()

  // Image hidden states storage (shared_ptr for batch path from_blob)
  std::shared_ptr<executorch::aten::Tensor> image_hidden_states_;
  std::vector<float> batch_image_buffer_;  // for batch prefill, owns concat data
  std::unique_ptr<executorch::aten::TensorImpl>
      batch_image_impl_;  // for set_image_hidden_states_batch, owns TensorImpl

  // Multimodal embeddings storage
  std::vector<float> multimodal_embeddings_buffer_;
  std::vector<executorch::aten::TensorImpl::SizesType>
      multimodal_embeddings_sizes_;
  std::vector<executorch::aten::TensorImpl::DimOrderType>
      multimodal_embeddings_dim_order_;
  TensorStruct<float> merged_embeddings_;

  // scale and zero point for quantized KV cache
  std::vector<float> input_k_cache_scales_;
  std::vector<T> input_k_cache_zero_points_;
  std::vector<float> input_v_cache_scales_;
  std::vector<T> input_v_cache_zero_points_;
  std::vector<float> output_k_cache_scales_;
  std::vector<T> output_k_cache_zero_points_;
  std::vector<float> output_v_cache_scales_;
  std::vector<T> output_v_cache_zero_points_;

  // stats
  executorch::llm::Stats stats_;
  GeneratePhaseTimings last_generate_phase_timings_;

  std::function<void(const char* phase, long start_ms, long end_ms)> load_phase_cb_;
};
} // namespace example
