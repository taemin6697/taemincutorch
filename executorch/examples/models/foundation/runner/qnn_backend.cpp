/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/foundation/runner/backend.h>

#ifdef FOUNDATION_ENABLE_QNN
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/encoder.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_runner.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/module/module.h>
#endif
#include <executorch/runtime/platform/log.h>

#include <atomic>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace executorch::examples::foundation {

namespace {

#ifdef FOUNDATION_ENABLE_QNN

using executorch::extension::llm::get_rss_bytes;
using executorch::extension::llm::time_in_ms;
constexpr const char* kFoundationProcCsv = "foundation_proc.csv";

long rss_kb() {
  const size_t bytes = get_rss_bytes();
  return bytes > 0 ? static_cast<long>(bytes / 1024) : 0;
}

void write_proc_header(std::ofstream& fproc) {
  fproc << "row_type,elapsed_s_start,elapsed_s_end,rss_kb_start,rss_kb_end,"
           "col_a_ms,col_b_ms,total_ms,kv_pos,kv_total,kv_used_pct,"
           "kv_used_kb,kv_total_kb,token_idx\n";
  fproc << "# L: Loading (전체)  L_VisionLoad: 비전 인코더 load  "
           "L_DecoderLoad: 텍스트 디코더/러너 load  "
           "L_EmbeddingLoad: 텍스트 임베딩 모듈 준비\n";
  fproc << "# V_Encode: col_a=vision_encode_ms (비전 인코더만)\n";
  fproc << "# EmbeddingAndMerging: image hidden state setup + prompt build + runner prefill entry prep\n";
  fproc << "# T_Prefill: col_a=text_kv_prefill_ms  Decode: col_b=token_gen_ms\n";
  fproc << "# S: 256토큰 구간 경계  D: 토큰별 decode\n";
}

std::string frame_path(const std::string& dir, int idx) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "/frame_%04d.bin", idx);
  return dir + buf;
}

std::vector<std::string> split_questions(const std::string& questions) {
  std::vector<std::string> out;
  std::stringstream ss(questions);
  std::string token;
  while (std::getline(ss, token, ';')) {
    if (!token.empty()) {
      out.push_back(token);
    }
  }
  if (out.empty()) {
    out.push_back("Describe this image.");
  }
  return out;
}

std::string format_streaming_query(
    const std::string& question,
    const std::string& decoder_model_version) {
  if (decoder_model_version == "internvl3") {
    return question + "<|im_end|>\n<|im_start|>assistant\n";
  }
  return question;
}

std::string build_batch_prompt(
    const std::string& decoder_model_version,
    int num_frames,
    int64_t img_seq_len,
    const std::string& question) {
  if (decoder_model_version != "internvl3") {
    return question;
  }
  std::string s = "<|im_start|>user:\n";
  for (int f = 0; f < num_frames; ++f) {
    s += "Frame" + std::to_string(f + 1) + ": <img>";
    for (int64_t i = 0; i < img_seq_len; ++i) {
      s += "<IMG_CONTEXT>";
    }
    s += "</img>\n";
  }
  s += format_streaming_query(question, decoder_model_version);
  return s;
}

template <typename T>
executorch::runtime::Error run_batch_qnn(
    const ManifestData& manifest,
    const UnifiedRunConfig& config) {
  using ::executorch::extension::Module;

  const long t_run_start = time_in_ms();
  const std::string proc_csv_path = kFoundationProcCsv;
  std::ofstream fproc(proc_csv_path);
  ET_CHECK_MSG(
      fproc.is_open(),
      "Failed to open proc csv file: %s",
      proc_csv_path.c_str());
  write_proc_header(fproc);
  const long rss_load_start = rss_kb();
  const long t_load_start = t_run_start;

  std::unique_ptr<example::EncoderRunner> encoder_runner;
  long t_encoder_load_start = 0;
  long t_encoder_load_end = 0;
  long rss_encoder_before = 0;
  long rss_encoder_after = 0;
  if (config.frame_count > 0) {
    rss_encoder_before = rss_kb();
    t_encoder_load_start = time_in_ms();
    encoder_runner =
        std::make_unique<example::EncoderRunner>(manifest.paths.vision_encoder_pte);
    ET_CHECK_OK_OR_RETURN_ERROR(encoder_runner->load());
    t_encoder_load_end = time_in_ms();
    rss_encoder_after = rss_kb();
  }

  const long rss_embedding_before = rss_kb();
  const long t_embedding_load_start = time_in_ms();
  auto embedding_module = std::make_unique<Module>(
      manifest.paths.text_embedding_pte,
      Module::LoadMode::MmapUseMlockIgnoreErrors);
  const long t_embedding_load_end = time_in_ms();
  const long rss_embedding_after = rss_kb();
  const long rss_decoder_before = rss_kb();
  const long t_decoder_load_start = time_in_ms();
  auto decoder_module = std::make_unique<Module>(
      manifest.paths.text_decoder_pte,
      Module::LoadMode::MmapUseMlockIgnoreErrors);
  const long t_decoder_construct_end = time_in_ms();
  const long rss_decoder_construct_after = rss_kb();

  int64_t img_seq_len = 0;
  int64_t hidden_dim = 0;
  std::vector<float> concat_buffer;
  std::vector<std::string> vision_rows;
  if (encoder_runner) {
    for (int i = 0; i < config.frame_count; ++i) {
      const long rss_before_encode = rss_kb();
      const long t_encode_start = time_in_ms();
      auto enc = encoder_runner->encode_from_file(frame_path(config.frame_dir, i));
      ET_CHECK_OK_OR_RETURN_ERROR(enc.error());
      const long t_encode_end = time_in_ms();
      const long rss_after_encode = rss_kb();
      auto t = enc.get();
      const int64_t n = t.size(1);
      const int64_t d = t.size(2);
      if (img_seq_len == 0) {
        img_seq_len = n;
      }
      if (hidden_dim == 0) {
        hidden_dim = d;
      }
      const float* src = t.const_data_ptr<float>();
      concat_buffer.insert(concat_buffer.end(), src, src + n * d);
      std::ostringstream oss;
      oss << "V_Encode," << (t_encode_start - t_run_start) / 1000.0 << ","
          << (t_encode_end - t_run_start) / 1000.0 << ","
          << rss_before_encode << "," << rss_after_encode << ","
          << (t_encode_end - t_encode_start) << ",,"
          << (t_encode_end - t_encode_start) << ",,,,,\n";
      vision_rows.push_back(oss.str());
    }
  }

  example::MultimodalRunner<T> runner(
      std::move(decoder_module),
      std::move(embedding_module),
      "internvl3",
      manifest.paths.text_decoder_pte,
      manifest.paths.tokenizer_path,
      /*dump_logits_path=*/"",
      /*performance_output_path=*/"",
      static_cast<float>(config.temperature),
      config.eval_mode,
      /*shared_buffer=*/false,
      /*lazy_kv_alloc=*/config.lazy_kv_alloc,
      /*ngram=*/0,
      /*window=*/0,
      /*gcap=*/0,
      /*image_hidden_states=*/nullptr);

  const long t_runner_load_start = time_in_ms();
  ET_CHECK_OK_OR_RETURN_ERROR(runner.load());
  const long t_runner_load_end = time_in_ms();
  const long rss_decoder_after = rss_kb();
  const long t_load_end = t_runner_load_end;
  const long rss_load_end = rss_decoder_after;
  if (config.frame_count > 0) {
    fproc << "L_VisionLoad," << (t_encoder_load_start - t_run_start) / 1000.0 << ","
          << (t_encoder_load_end - t_run_start) / 1000.0 << ","
          << rss_encoder_before << "," << rss_encoder_after << ","
          << (t_encoder_load_end - t_encoder_load_start) << ",,"
          << (t_encoder_load_end - t_encoder_load_start) << ",,,,,\n";
  }
  fproc << "L_EmbeddingLoad," << (t_embedding_load_start - t_run_start) / 1000.0 << ","
        << (t_embedding_load_end - t_run_start) / 1000.0 << ","
        << rss_embedding_before << "," << rss_embedding_after << ","
        << (t_embedding_load_end - t_embedding_load_start) << ",,"
        << (t_embedding_load_end - t_embedding_load_start) << ",,,,,\n";
  fproc << "L_DecoderLoad," << (t_decoder_load_start - t_run_start) / 1000.0 << ","
        << (t_runner_load_end - t_run_start) / 1000.0 << ","
        << rss_decoder_before << "," << rss_decoder_after << ","
        << (t_runner_load_end - t_decoder_load_start) << ",,"
        << (t_runner_load_end - t_decoder_load_start) << ",,,,,\n";
  fproc << "L," << (t_load_start - t_run_start) / 1000.0 << ","
        << (t_load_end - t_run_start) / 1000.0 << ","
        << rss_load_start << "," << rss_load_end << ","
        << (t_load_end - t_load_start) << ",,"
        << (t_load_end - t_load_start) << ",,,,,\n";
  for (const auto& row : vision_rows) {
    fproc << row;
  }
  (void)t_decoder_construct_end;
  (void)rss_decoder_construct_after;
  if (config.frame_count > 0 && !concat_buffer.empty()) {
    runner.set_image_hidden_states_batch(
        std::move(concat_buffer),
        config.frame_count * img_seq_len,
        hidden_dim);
  }

  executorch::extension::llm::GenerationConfig gen_config{
      /*echo=*/true,
      /*ignore_eos=*/config.ignore_eos,
      /*max_new_tokens=*/-1,
      /*warming=*/false,
      /*seq_len=*/config.seq_len,
      /*temperature=*/static_cast<float>(config.temperature),
      /*num_bos=*/0,
      /*num_eos=*/0};

  std::ofstream fout(config.output_path);
  ET_CHECK_MSG(
      fout.is_open(),
      "Failed to open output file: %s",
      config.output_path.c_str());

  for (const auto& q : split_questions(config.questions)) {
    std::string full_prompt =
        build_batch_prompt("internvl3", config.frame_count, img_seq_len, q);
    std::vector<char> out_buf;
    double text_kv_prefill_ms = 0.0;
    int64_t num_prompt_tokens = 0;
    std::atomic<long> rss_at_first_token{-1};
    std::atomic<int64_t> token_count{0};
    struct SegmentEvent {
      double elapsed_s;
      long rss_kb;
      int64_t token_count;
    };
    constexpr int kSegmentSize = 256;
    std::vector<SegmentEvent> segment_events;
    std::vector<std::string> d_rows_buffer;
    auto cb = [&](const std::string& piece) {
      for (char c : piece) {
        out_buf.push_back(c);
      }
      long expected = -1;
      rss_at_first_token.compare_exchange_strong(expected, rss_kb());
      const int64_t count = token_count.fetch_add(1) + 1;
      if (count % kSegmentSize == 0) {
        segment_events.push_back(
            {static_cast<double>(time_in_ms() - t_run_start) / 1000.0, rss_kb(), count});
      }
    };
    auto stats_cb = [&](const executorch::extension::llm::Stats& s) {
      text_kv_prefill_ms = s.first_token_ms - s.inference_start_ms;
      num_prompt_tokens = s.num_prompt_tokens;
    };
    gen_config.per_token_timing_cb = [&](int64_t token_idx,
                                         int64_t kv_pos,
                                         long start_ms,
                                         long end_ms) {
      const double start_s = (start_ms - t_run_start) / 1000.0;
      const double end_s = (end_ms - t_run_start) / 1000.0;
      const long rss = rss_kb();
      const int kv_ctx = runner.get_context_len();
      const size_t kv_total_bytes = runner.get_kv_cache_total_bytes();
      const int64_t kv_total_kb = kv_total_bytes > 0
          ? static_cast<int64_t>(kv_total_bytes / 1024)
          : 0;
      const int64_t kv_used_kb = (kv_ctx > 0 && kv_total_kb > 0)
          ? (kv_pos * kv_total_kb) / kv_ctx
          : 0;
      const double kv_pct = kv_ctx > 0 ? 100.0 * kv_pos / kv_ctx : 0.0;
      std::ostringstream oss;
      oss << "D," << start_s << "," << end_s << ","
          << rss << "," << rss << ","
          << "," << (end_ms - start_ms) << "," << (end_ms - start_ms) << ","
          << kv_pos << "," << kv_ctx << "," << kv_pct << ","
          << kv_used_kb << "," << kv_total_kb << ","
          << token_idx << "\n";
      d_rows_buffer.push_back(oss.str());
    };
    const long rss_before_q = rss_kb();
    const int64_t kv_before_q = runner.get_cur_pos();
    const long t_q_start = time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(
        runner.generate_from_prompt_or_file(
            full_prompt, false, gen_config, cb, stats_cb));
    const long t_q_end = time_in_ms();
    const auto phase_timings = runner.get_last_generate_phase_timings();
    const auto& em_timing = phase_timings.embedding_and_merging;
    const auto& prefill_timing = phase_timings.prefill;
    const auto& decode_timing = phase_timings.decode;
    const bool has_internal_prefill =
        prefill_timing.start_ms > 0 && prefill_timing.end_ms >= prefill_timing.start_ms;
    const bool has_internal_decode =
        decode_timing.start_ms > 0 && decode_timing.end_ms >= decode_timing.start_ms;
    const double elapsed_prefill_start = has_internal_prefill
        ? (prefill_timing.start_ms - t_run_start) / 1000.0
        : (t_q_start - t_run_start) / 1000.0;
    const double elapsed_prefill_end = has_internal_prefill
        ? (prefill_timing.end_ms - t_run_start) / 1000.0
        : (elapsed_prefill_start + (text_kv_prefill_ms / 1000.0));
    const double elapsed_decode_start = has_internal_decode
        ? (decode_timing.start_ms - t_run_start) / 1000.0
        : elapsed_prefill_end;
    const double elapsed_decode_end = has_internal_decode
        ? (decode_timing.end_ms - t_run_start) / 1000.0
        : (t_q_end - t_run_start) / 1000.0;
    const long rss_q = rss_kb();
    const long rss_first =
        rss_at_first_token.load() >= 0 ? rss_at_first_token.load() : rss_before_q;
    const int64_t kv_after = runner.get_cur_pos();
    const int kv_ctx = runner.get_context_len();
    const double q_total_ms = t_q_end - t_q_start;
    const double q_gen_ms =
        q_total_ms > text_kv_prefill_ms ? q_total_ms - text_kv_prefill_ms : 0.0;
    const size_t kv_total_bytes_q = runner.get_kv_cache_total_bytes();
    const int64_t kv_total_kb_q = kv_total_bytes_q > 0
        ? static_cast<int64_t>(kv_total_bytes_q / 1024)
        : 0;
    const int64_t kv_prefill = kv_before_q + num_prompt_tokens;
    const int64_t kv_used_kb_prefill = (kv_ctx > 0 && kv_total_kb_q > 0)
        ? (kv_prefill * kv_total_kb_q) / kv_ctx
        : 0;
    const int64_t kv_used_kb_q = (kv_ctx > 0 && kv_total_kb_q > 0)
        ? (kv_after * kv_total_kb_q) / kv_ctx
        : 0;
    const double kv_pct_prefill =
        kv_ctx > 0 ? 100.0 * kv_prefill / kv_ctx : 0.0;
    const double kv_after_pct = kv_ctx > 0 ? 100.0 * kv_after / kv_ctx : 0.0;
    if (em_timing.start_ms > 0 && em_timing.end_ms >= em_timing.start_ms) {
      fproc << "EmbeddingAndMerging,"
            << (em_timing.start_ms - t_run_start) / 1000.0 << ","
            << (em_timing.end_ms - t_run_start) / 1000.0 << ","
            << em_timing.rss_kb_start << "," << em_timing.rss_kb_end << ","
            << (em_timing.end_ms - em_timing.start_ms) << ",,"
            << (em_timing.end_ms - em_timing.start_ms) << ",,,,,\n";
    }
    fproc << "T_Prefill," << elapsed_prefill_start << ","
          << elapsed_prefill_end << ","
          << (has_internal_prefill ? prefill_timing.rss_kb_start : rss_before_q) << ","
          << (has_internal_prefill ? prefill_timing.rss_kb_end : rss_first) << ","
          << (has_internal_prefill ? (prefill_timing.end_ms - prefill_timing.start_ms)
                                   : text_kv_prefill_ms)
          << ",,"
          << (has_internal_prefill ? (prefill_timing.end_ms - prefill_timing.start_ms)
                                   : text_kv_prefill_ms)
          << ","
          << kv_prefill << "," << kv_ctx << "," << kv_pct_prefill << ","
          << kv_used_kb_prefill << "," << kv_total_kb_q << ",\n";
    fproc << "Decode," << elapsed_decode_start << "," << elapsed_decode_end << ","
          << (has_internal_decode ? decode_timing.rss_kb_start : rss_first) << ","
          << (has_internal_decode ? decode_timing.rss_kb_end : rss_q) << ","
          << ","
          << (has_internal_decode ? (decode_timing.end_ms - decode_timing.start_ms) : q_gen_ms)
          << ","
          << (has_internal_decode ? (decode_timing.end_ms - decode_timing.start_ms) : q_gen_ms)
          << ","
          << kv_after << "," << kv_ctx << "," << kv_after_pct << ","
          << kv_used_kb_q << "," << kv_total_kb_q << ",\n";
    for (const auto& row : d_rows_buffer) {
      fproc << row;
    }
    for (const auto& seg : segment_events) {
      const int64_t seg_kv_pos = kv_prefill + seg.token_count;
      const int64_t seg_kv_used_kb = (kv_ctx > 0 && kv_total_kb_q > 0)
          ? (seg_kv_pos * kv_total_kb_q) / kv_ctx
          : 0;
      const double seg_kv_pct = kv_ctx > 0 ? 100.0 * seg_kv_pos / kv_ctx : 0.0;
      fproc << "S," << seg.elapsed_s << "," << seg.elapsed_s << ","
            << seg.rss_kb << "," << seg.rss_kb << ",,,,"
            << seg_kv_pos << "," << kv_ctx << "," << seg_kv_pct << ","
            << seg_kv_used_kb << "," << kv_total_kb_q << ",\n";
    }
    fproc.flush();

    std::string answer(out_buf.begin(), out_buf.end());
    fout << answer << "\n";
  }
  return executorch::runtime::Error::Ok;
}

class QnnBackendRunner final : public BackendRunner {
 public:
  explicit QnnBackendRunner(ManifestData manifest) : manifest_(std::move(manifest)) {}

  executorch::runtime::Error validate() override {
    if (manifest_.paths.vision_encoder_pte.empty() ||
        manifest_.paths.text_embedding_pte.empty() ||
        manifest_.paths.text_decoder_pte.empty() ||
        manifest_.paths.tokenizer_path.empty()) {
      ET_LOG(Error, "QNN manifest is missing required split-PTE paths.");
      return executorch::runtime::Error::InvalidArgument;
    }
    return executorch::runtime::Error::Ok;
  }

  executorch::runtime::Error run(const UnifiedRunConfig& config) override {
    auto decoder_module = std::make_unique<executorch::extension::Module>(
        manifest_.paths.text_decoder_pte,
        executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
    auto method_names = decoder_module->method_names();
    ET_CHECK_OK_OR_RETURN_ERROR(method_names.error());
    example::KvBitWidth kv_bitwidth = example::KvBitWidth::kWidth8;
    if (method_names->count("get_kv_io_bit_width") > 0) {
      auto kv_width_value = decoder_module->get("get_kv_io_bit_width");
      ET_CHECK_OK_OR_RETURN_ERROR(kv_width_value.error());
      kv_bitwidth = static_cast<example::KvBitWidth>(
          kv_width_value->toScalar().to<int64_t>());
    }

    if (kv_bitwidth == example::KvBitWidth::kWidth8) {
      return run_batch_qnn<uint8_t>(manifest_, config);
    }
    if (kv_bitwidth == example::KvBitWidth::kWidth16) {
      return run_batch_qnn<uint16_t>(manifest_, config);
    }
    ET_LOG(Error, "Unsupported QNN kv bit width");
    return executorch::runtime::Error::NotSupported;
  }

 private:
  ManifestData manifest_;
};

#else

class QnnBackendRunner final : public BackendRunner {
 public:
  explicit QnnBackendRunner(ManifestData manifest)
      : manifest_(std::move(manifest)) {}

  executorch::runtime::Error validate() override {
    ET_LOG(
        Error,
        "QNN backend support is not available in this xnnpack_qnn_runner build.");
    return executorch::runtime::Error::NotSupported;
  }

  executorch::runtime::Error run(const UnifiedRunConfig&) override {
    ET_LOG(
        Error,
        "QNN backend support is not available in this xnnpack_qnn_runner build.");
    return executorch::runtime::Error::NotSupported;
  }

 private:
  ManifestData manifest_;
};

#endif

} // namespace

std::unique_ptr<BackendRunner> create_qnn_backend_runner(
    const ManifestData& manifest) {
  return std::make_unique<QnnBackendRunner>(manifest);
}

} // namespace executorch::examples::foundation
