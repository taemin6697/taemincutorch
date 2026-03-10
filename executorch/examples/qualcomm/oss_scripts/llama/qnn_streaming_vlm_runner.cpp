/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file qnn_streaming_vlm_runner.cpp
 *
 * Streaming Vision-Language Model runner for Qualcomm HTP.
 *
 * Workflow
 * --------
 *  1. Sample video frames at a configurable FPS.
 *  2. For each frame: run the vision encoder → prefill_frame() into KV cache.
 *  3. At user-specified timestamps, answer the corresponding text query via
 *     generate().
 *
 * The runner is intentionally kept in KV mode (eval_mode=0) to avoid
 * multi-call requantization complexity; hybrid/lookahead modes are available
 * but require a single finalize_prefill() call before each generate().
 *
 * Expected frame binary format
 * ----------------------------
 *  Each file is a raw float32 dump of the preprocessed image tensor produced
 *  by the Python HuggingFace processor:
 *    shape = (1, C, H, W), dtype = float32, little-endian
 *  File naming: <frame_dir>/frame_NNNN.bin  (zero-padded, 4 digits)
 */

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/encoder.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_runner.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/platform/log.h>
#include <gflags/gflags.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ── Model paths ───────────────────────────────────────────────────────────────
DEFINE_string(embedding_path, "embedding.pte", "Embedding model path.");
DEFINE_string(encoder_path, "encoder.pte", "Vision encoder model path.");
DEFINE_string(decoder_path, "decoder.pte", "Decoder model path.");
DEFINE_string(tokenizer_path, "tokenizer.bin", "Tokenizer path.");
DEFINE_string(output_path, "outputs.txt", "Output text file path.");
DEFINE_string(performance_output_path, "inference_speed.txt", "Perf log path.");
DEFINE_string(dump_logits_path, "", "Optional logits dump path.");
DEFINE_string(decoder_model_version, "smolvlm", "Decoder model identifier.");

// ── Streaming inputs ──────────────────────────────────────────────────────────
DEFINE_string(frame_dir, "", "Directory containing frame_NNNN.bin files.");
DEFINE_int32(frame_count, 0, "Total number of frames to process.");
DEFINE_double(fps, 1.0, "Frame sampling rate used when extracting frames.");

// Questions and timestamps are passed as semicolon-separated strings so that
// gflags does not need special handling:
//   --questions "What is happening?;Is there water?"
//   --query_timestamps "10.0;25.5"
DEFINE_string(questions, "", "Semicolon-separated list of text queries.");
DEFINE_string(
    query_timestamps,
    "",
    "Semicolon-separated timestamps in seconds (one per question).");

// ── Generation params ─────────────────────────────────────────────────────────
DEFINE_double(temperature, 0.0, "Sampling temperature (0 = greedy).");
DEFINE_int32(seq_len, 128, "Max tokens to generate per query.");
DEFINE_int32(
    eval_mode,
    0,
    "0=KV (recommended for streaming), 1=Hybrid, 2=Lookahead.");
DEFINE_bool(shared_buffer, false, "Use shared RPC buffers.");
// KV 캐시 물리 메모리 할당 전략
//   true  (기본값) → mmap+MAP_NORESERVE: lazy 물리 커밋, 대용량 seq_len 안전
//   false          → std::vector zero-init: 로드 시 즉시 전체 물리 점유
DEFINE_bool(
    lazy_kv_alloc,
    true,
    "Lazy KV cache physical allocation via mmap+MAP_NORESERVE (true) "
    "or immediate std::vector zero-init (false).");
DEFINE_bool(
    output_minimal,
    true,
    "Output only actual input/output (KV + response), no [t=] Q: A: prefixes.");
DEFINE_bool(
    stream,
    false,
    "Streaming mode (frame-by-frame prefill). false = batch (HF-style, all frames at once).");

// ─────────────────────────────────────────────────────────────────────────────

namespace {

struct TimedQuery {
  double timestamp_sec;
  std::string question;
  bool operator>(const TimedQuery& o) const {
    return timestamp_sec > o.timestamp_sec;
  }
};

std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> out;
  std::istringstream ss(s);
  std::string token;
  while (std::getline(ss, token, delim)) {
    if (!token.empty()) {
      out.push_back(token);
    }
  }
  return out;
}

std::string frame_path(const std::string& dir, int idx) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "/frame_%04d.bin", idx);
  return dir + buf;
}


/**
 * Format a text-only query for the streaming case.
 *
 * In streaming mode the FIRST prefill_frame() already wrote
 * "<|im_start|>User:<image wrappers>..." into the KV cache.
 * The text query only needs to append the question + turn-end tokens
 * so the full KV cache sequence becomes:
 *   <|im_start|>User:<wrap><img×N><wrap>...<wrap><img×N><wrap>{question}<end>
 *   Assistant:
 */
std::string format_streaming_query(
    const std::string& question,
    const std::string& decoder_model_version) {
  if (decoder_model_version == "smolvlm") {
    // Append question directly — User: prefix was already prefilled with frame 0
    return question + "<end_of_utterance>\nAssistant:";
  } else if (decoder_model_version == "internvl3") {
    // HF chat_template: question + <|im_end|>\n<|im_start|>assistant\n
    return question + "<|im_end|>\n<|im_start|>assistant\n";
  }
  return question;
}

/** Build HF-style batch prompt: prefix + Frame1..FrameN + question + assistant */
std::string build_batch_prompt(
    const std::string& decoder_model_version,
    int num_frames,
    int64_t img_seq_len,
    const std::string& question) {
  if (decoder_model_version != "internvl3") {
    return "";  // SmolVLM batch format differs; focus on InternVL3
  }
  std::string s = "<|im_start|>user:\n";
  if (num_frames > 0 && img_seq_len > 0) {
    for (int f = 0; f < num_frames; ++f) {
      s += "Frame" + std::to_string(f + 1) + ": <img>";
      for (int64_t i = 0; i < img_seq_len; ++i) {
        s += "<IMG_CONTEXT>";
      }
      s += "</img>\n";
    }
  }
  s += format_streaming_query(question, decoder_model_version);
  return s;
}

// ── Helpers: time & memory ────────────────────────────────────────────────────

using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

static double now_ms() {
  return std::chrono::duration<double, std::milli>(
             Clock::now().time_since_epoch())
      .count();
}

/** Read process RSS from /proc/self/status (Android/Linux). Returns KB. */
static long rss_kb() {
#if defined(__linux__) || defined(__ANDROID__)
  std::ifstream f("/proc/self/status");
  if (!f.is_open()) return -1;
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("VmRSS:", 0) == 0) {
      long kb = 0;
      if (std::sscanf(line.c_str(), "VmRSS: %ld kB", &kb) == 1) return kb;
    }
  }
#endif
  return -1;
}

/** System-wide memory from /proc/meminfo (Android/Linux). */
struct MemInfo {
  long mem_total_kb = -1;
  long mem_available_kb = -1;
};
static MemInfo meminfo_kb() {
  MemInfo out;
#if defined(__linux__) || defined(__ANDROID__)
  std::ifstream f("/proc/meminfo");
  if (!f.is_open()) return out;
  std::string line;
  while (std::getline(f, line)) {
    long kb = 0;
    if (line.rfind("MemTotal:", 0) == 0 &&
        std::sscanf(line.c_str(), "MemTotal: %ld kB", &kb) == 1) {
      out.mem_total_kb = kb;
    } else if (line.rfind("MemAvailable:", 0) == 0 &&
               std::sscanf(line.c_str(), "MemAvailable: %ld kB", &kb) == 1) {
      out.mem_available_kb = kb;
    }
    if (out.mem_total_kb >= 0 && out.mem_available_kb >= 0) break;
  }
#endif
  return out;
}

/** Dump /proc/self/smaps for memory analysis. */
static void dump_smaps(const std::string& out_path) {
#if defined(__linux__) || defined(__ANDROID__)
  std::ifstream in("/proc/self/smaps");
  if (!in.is_open()) return;
  std::ofstream out(out_path);
  if (!out.is_open()) return;
  out << in.rdbuf();
  ET_LOG(Info, "smaps dumped to %s", out_path.c_str());
#endif
}

} // namespace

// ── Streaming loop ────────────────────────────────────────────────────────────

template <typename T>
void run_streaming(
    double t_run_start,
    long t_run_start_realtime,
    std::unique_ptr<example::EncoderRunner> encoder_runner,
    std::unique_ptr<executorch::extension::Module> decoder_module,
    std::unique_ptr<executorch::extension::Module> embedding_module,
    const std::priority_queue<
        TimedQuery,
        std::vector<TimedQuery>,
        std::greater<TimedQuery>>& queries_pq_in) {
  // Work on a mutable copy so we can pop
  auto queries_pq = queries_pq_in;

  // CSV 1: 프레임/질문 처리 통계 (main에서 헤더+L_Embedding+L_Decoder 이미 기록, append)
  std::string proc_csv_path = FLAGS_output_path + ".proc.csv";
  std::ofstream fproc(proc_csv_path, std::ios::app);
  std::mutex fproc_mutex;

  // CSV 2: 50ms 주기 RSS 메모리 샘플 (로딩 시점부터 측정, llama_main과 동일)
  std::string mem_csv_path = FLAGS_output_path + ".mem.csv";
  std::ofstream fmem(mem_csv_path);
  std::mutex fmem_mutex;
  fmem << "elapsed_s,rss_kb,rss_mb,delta_rss_kb,mem_total_kb,mem_available_kb\n";

  // CSV 3: 토큰별 생성 시작/종료 (생성, 종료)
  std::string tokens_csv_path = FLAGS_output_path + ".tokens.csv";
  std::ofstream ftokens(tokens_csv_path);
  std::mutex ftokens_mutex;
  if (ftokens.is_open()) {
    ftokens << "token_idx,kv_pos,elapsed_s_start,elapsed_s_end,ms\n";
  }

  long rss_load_start = rss_kb();

  std::atomic<bool> sampler_running{true};
  std::thread rss_sampler([&]() {
    long last_rss = rss_kb();
    while (sampler_running.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      if (!sampler_running.load()) break;
      double elapsed_s = (now_ms() - t_run_start) / 1000.0;
      long cur = rss_kb();
      long delta = (last_rss >= 0 && cur >= 0) ? cur - last_rss : 0;
      last_rss = cur;
      MemInfo mi = meminfo_kb();
      std::lock_guard<std::mutex> lk(fmem_mutex);
      fmem << elapsed_s << "," << cur << "," << cur / 1024.0 << "," << delta
           << "," << mi.mem_total_kb << "," << mi.mem_available_kb << "\n";
      fmem.flush();
    }
  });

  // Load encoder (L_V_Encode)
  {
    double t0 = now_ms();
    ET_CHECK_MSG(
        encoder_runner->load() == executorch::runtime::Error::Ok,
        "Failed to load vision encoder");
    double t1 = now_ms();
    long rss_enc = rss_kb();
    std::lock_guard<std::mutex> lk(fproc_mutex);
    fproc << "L_V_Encode," << (t0 - t_run_start) / 1000.0 << ","
          << (t1 - t_run_start) / 1000.0 << "," << rss_load_start << ","
          << rss_enc << "," << (t1 - t0) << ",," << (t1 - t0) << ",,,,,,,\n";
    fproc.flush();
  }

  // Create the multimodal runner (image_hidden_states starts null; will be set
  // per frame via prefill_frame())
  example::MultimodalRunner<T> runner(
      std::move(decoder_module),
      std::move(embedding_module),
      FLAGS_decoder_model_version,
      FLAGS_decoder_path,
      FLAGS_tokenizer_path,
      FLAGS_dump_logits_path,
      FLAGS_performance_output_path,
      static_cast<float>(FLAGS_temperature),
      FLAGS_eval_mode,
      FLAGS_shared_buffer,
      FLAGS_lazy_kv_alloc,
      /*ngram=*/0,
      /*window=*/0,
      /*gcap=*/0,
      /*image_hidden_states=*/nullptr);

  runner.set_load_phase_callback([&](const char* phase, long start_ms, long end_ms) {
    if (!fproc.is_open()) return;
    double start_s = (start_ms - t_run_start_realtime) / 1000.0;
    double end_s = (end_ms - t_run_start_realtime) / 1000.0;
    long rss_cur = rss_kb();
    std::lock_guard<std::mutex> lk(fproc_mutex);
    fproc << phase << "," << start_s << "," << end_s << ","
          << rss_cur << "," << rss_cur << ","
          << (end_ms - start_ms) << ",," << (end_ms - start_ms) << ",,,,,,,\n";
    fproc.flush();
  });

  ET_CHECK_MSG(
      runner.load() == executorch::runtime::Error::Ok,
      "Failed to load MultimodalRunner");

  // Prefix is prefilled with the first frame (prefill_frame includes prefix for
  // frame 0). No separate prefill_prefix() call.

  // L 행: 로딩+웜업 구간 기록 (llama_main 형식)
  {
    long rss_load_end = rss_kb();
    double elapsed_load_end = (now_ms() - t_run_start) / 1000.0;
    std::lock_guard<std::mutex> lk(fproc_mutex);
    fproc << "L,0," << elapsed_load_end << "," << rss_load_start << ","
          << rss_load_end << ",,,,,,,,\n";
    fproc.flush();
  }

  std::ofstream fout(FLAGS_output_path);
  std::vector<char> out_buf;

  executorch::extension::llm::GenerationConfig config{
      /*echo=*/true,
      /*ignore_eos=*/false,
      /*max_new_tokens=*/-1,
      /*warming=*/false,
      /*seq_len=*/FLAGS_seq_len,
      /*temperature=*/static_cast<float>(FLAGS_temperature),
      /*num_bos=*/0,
      /*num_eos=*/0};

  const double inv_fps = 1.0 / FLAGS_fps;

  // ── KV cache log: use actual inference format from runner (no manual formatting) ──
  std::string kv_log = runner.get_prefix_format_string();

  for (int frame_idx = 0; frame_idx < FLAGS_frame_count; ++frame_idx) {
    double frame_time_sec = frame_idx * inv_fps;
    long rss_frame_start = rss_kb();

    // ── Encode frame ──────────────────────────────────────────────────────────
    std::string fpath = frame_path(FLAGS_frame_dir, frame_idx);
    double t0 = now_ms();
    auto encode_result = encoder_runner->encode_from_file(fpath);
    double encode_ms = now_ms() - t0;

    if (!encode_result.ok()) {
      ET_LOG(Error, "Failed to encode frame %d (%s)", frame_idx, fpath.c_str());
      continue;
    }

    auto image_hidden_states = encode_result.get();
    int64_t img_seq_len = image_hidden_states.size(1);
    long rss_after_encode = rss_kb();
    int64_t kv_before_prefill = runner.get_cur_pos();

    double t1 = now_ms();
    ET_CHECK_MSG(
        runner.prefill_frame(image_hidden_states) ==
            executorch::runtime::Error::Ok,
        "prefill_frame failed at frame %d",
        frame_idx);
    double prefill_ms = now_ms() - t1;
    double total_frame_ms = now_ms() - t0;

    kv_log += runner.get_frame_format_string(frame_idx, img_seq_len);

    // ── Per-frame stats: V_Encode(비전 인코더만) + V_Prefill(비전→LLM KV) 분리 ──
    int64_t kv_pos = runner.get_cur_pos();
    int kv_total = runner.get_context_len();
    double kv_pct = kv_total > 0 ? 100.0 * kv_pos / kv_total : 0.0;
    long cur_rss = rss_kb();

    size_t kv_total_bytes = runner.get_kv_cache_total_bytes();
    int64_t kv_total_kb = kv_total_bytes > 0
        ? static_cast<int64_t>(kv_total_bytes / 1024)
        : (24 * 2 * 16 * 64 * 1 * kv_total) / 1024;  // fallback
    int64_t kv_used_kb_before = (kv_total > 0 && kv_total_kb > 0)
        ? (kv_before_prefill * kv_total_kb) / kv_total
        : 0;
    int64_t kv_used_kb_after = (kv_total > 0 && kv_total_kb > 0)
        ? (kv_pos * kv_total_kb) / kv_total
        : 0;
    double kv_pct_before = kv_total > 0 ? 100.0 * kv_before_prefill / kv_total : 0.0;

    double elapsed_encode_start = (t0 - t_run_start) / 1000.0;
    double elapsed_encode_end = (t1 - t_run_start) / 1000.0;
    double elapsed_prefill_start = elapsed_encode_end;
    double elapsed_prefill_end = (now_ms() - t_run_start) / 1000.0;
    {
      std::lock_guard<std::mutex> lk(fproc_mutex);
      fproc << "V_Encode," << elapsed_encode_start << "," << elapsed_encode_end << ","
            << rss_frame_start << "," << rss_after_encode << ","
            << encode_ms << ",," << encode_ms << ","
            << kv_before_prefill << "," << kv_total << "," << kv_pct_before << ","
            << kv_used_kb_before << "," << kv_total_kb << ",\n";
      fproc << "V_Prefill," << elapsed_prefill_start << "," << elapsed_prefill_end << ","
            << rss_after_encode << "," << cur_rss << ","
            << prefill_ms << ",," << prefill_ms << ","
            << kv_pos << "," << kv_total << "," << kv_pct << ","
            << kv_used_kb_after << "," << kv_total_kb << ",\n";
      fproc.flush();
    }

    long delta_rss = (rss_frame_start >= 0 && cur_rss >= 0)
                         ? cur_rss - rss_frame_start
                         : 0;
    ET_LOG(
        Info,
        "[frame %3d | t=%5.1fs] encode=%.1fms  vision_kv_prefill=%.1fms  "
        "KV=%ld/%d (%.1f%%)  RSS=%ldKB (Δ%+ldKB)",
        frame_idx,
        frame_time_sec,
        encode_ms,
        prefill_ms,
        kv_pos,
        kv_total,
        kv_pct,
        cur_rss,
        delta_rss);

    // ── Answer queries whose timestamp has arrived ────────────────────────────
    while (!queries_pq.empty() &&
           queries_pq.top().timestamp_sec <= frame_time_sec) {
      const TimedQuery& q = queries_pq.top();
      ET_LOG(
          Info,
          "Answering query at t=%.2fs: \"%s\"",
          q.timestamp_sec,
          q.question.c_str());

      out_buf.clear();

      std::atomic<long> rss_at_first_token{-1};
      double text_kv_prefill_ms = 0.0;
      int64_t num_prompt_tokens = 0;
      struct SegmentEvent {
        double elapsed_s;
        long rss_kb;
        int64_t token_count;
      };
      std::vector<SegmentEvent> segment_events;
      std::atomic<int64_t> token_count{0};
      constexpr int SEGMENT_SIZE = 256;
      std::vector<std::string> d_rows_buffer;
      std::mutex d_rows_mutex;

      auto stats_cb = [&](const executorch::extension::llm::Stats& s) {
        text_kv_prefill_ms = s.first_token_ms - s.inference_start_ms;
        num_prompt_tokens = s.num_prompt_tokens;
      };

      auto cb = [&](const std::string& piece) {
        for (char c : piece) {
          out_buf.push_back(c);
        }
        long expected = -1;
        if (rss_at_first_token.compare_exchange_strong(expected, rss_kb())) {
          // First token emitted
        }
        int64_t count = token_count.fetch_add(1) + 1;
        if (count % SEGMENT_SIZE == 0) {
          double elapsed_s = (now_ms() - t_run_start) / 1000.0;
          segment_events.push_back({elapsed_s, rss_kb(), count});
        }
      };

      config.per_token_timing_cb = [&](int64_t token_idx, int64_t kv_pos,
                                        long start_ms, long end_ms) {
        if (ftokens.is_open()) {
          double start_s = (start_ms - t_run_start_realtime) / 1000.0;
          double end_s = (end_ms - t_run_start_realtime) / 1000.0;
          std::lock_guard<std::mutex> lk(ftokens_mutex);
          ftokens << token_idx << "," << kv_pos << "," << start_s << ","
                  << end_s << "," << (end_ms - start_ms) << "\n";
          ftokens.flush();
        }
        if (fproc.is_open()) {
          double start_s = (start_ms - t_run_start_realtime) / 1000.0;
          double end_s = (end_ms - t_run_start_realtime) / 1000.0;
          long rss = rss_kb();
          int kv_ctx = runner.get_context_len();
          size_t kv_bytes = runner.get_kv_cache_total_bytes();
          int64_t kv_total_kb = kv_bytes > 0
              ? static_cast<int64_t>(kv_bytes / 1024)
              : (kv_ctx > 0 ? (24 * 2 * 16 * 64 * 1 * kv_ctx) / 1024 : 0);
          int64_t kv_used_kb = (kv_ctx > 0 && kv_total_kb > 0)
              ? (kv_pos * kv_total_kb) / kv_ctx
              : 0;
          double kv_pct = kv_ctx > 0 ? 100.0 * kv_pos / kv_ctx : 0.0;
          std::ostringstream oss;
          oss << "D," << start_s << "," << end_s << ","
              << rss << "," << rss << ","
              << "," << (end_ms - start_ms) << "," << (end_ms - start_ms) << ","
              << kv_pos << "," << kv_ctx << "," << kv_pct << ","
              << kv_used_kb << "," << kv_total_kb << ","
              << token_idx << "\n";
          std::lock_guard<std::mutex> lk(d_rows_mutex);
          d_rows_buffer.push_back(oss.str());
        }
      };

      // Format the question with the model's chat template (no image tokens –
      // image embeddings are already in the KV cache from prefill_frame()).
      std::string formatted_q =
          format_streaming_query(q.question, FLAGS_decoder_model_version);

      long rss_before_q = rss_kb();
      int64_t kv_before_q = runner.get_cur_pos();
      double t_q_start = now_ms();
      auto err = runner.generate_from_prompt_or_file(
          formatted_q, /*tokenized_prompt=*/false, config, cb, stats_cb);
      double t_q_end = now_ms();

      double q_total_ms = t_q_end - t_q_start;
      double q_gen_ms = q_total_ms - text_kv_prefill_ms;

      if (err != executorch::runtime::Error::Ok) {
        ET_LOG(Error, "generate() failed for query: %s", q.question.c_str());
      }

      std::string answer(out_buf.begin(), out_buf.end());
      long q_rss = rss_kb();
      int64_t kv_after = runner.get_cur_pos();
      int kv_ctx = runner.get_context_len();
      double kv_after_pct = kv_ctx > 0 ? 100.0 * kv_after / kv_ctx : 0.0;
      long rss_first = (rss_at_first_token >= 0) ? rss_at_first_token.load()
                                                 : rss_before_q;

      size_t kv_total_bytes_q = runner.get_kv_cache_total_bytes();
      int64_t kv_total_kb_q = kv_total_bytes_q > 0
          ? static_cast<int64_t>(kv_total_bytes_q / 1024)
          : (24 * 2 * 16 * 64 * 1 * kv_ctx) / 1024;  // fallback
      int64_t kv_prefill = kv_before_q + num_prompt_tokens;
      int64_t kv_used_kb_prefill = (kv_ctx > 0 && kv_total_kb_q > 0)
          ? (kv_prefill * kv_total_kb_q) / kv_ctx
          : 0;
      double kv_pct_prefill =
          (kv_ctx > 0) ? 100.0 * kv_prefill / kv_ctx : 0.0;
      int64_t kv_used_kb_q = (kv_ctx > 0 && kv_total_kb_q > 0)
          ? (kv_after * kv_total_kb_q) / kv_ctx
          : 0;

      double elapsed_prefill_start = (t_q_start - t_run_start) / 1000.0;
      double elapsed_prefill_end =
          elapsed_prefill_start + (text_kv_prefill_ms / 1000.0);
      double elapsed_decode_start = elapsed_prefill_end;
      double elapsed_decode_end = (t_q_end - t_run_start) / 1000.0;

      ET_LOG(
          Info,
          "[query t=%.1fs] text_kv_prefill=%.1fms  generate=%.1fms  total=%.1fms  "
          "KV=%ld/%d (%.1f%%)  RSS=%ldKB",
          q.timestamp_sec,
          text_kv_prefill_ms,
          q_gen_ms,
          q_total_ms,
          kv_after,
          kv_ctx,
          kv_after_pct,
          q_rss);

      {
        std::lock_guard<std::mutex> lk(fproc_mutex);
        fproc << "T_Prefill," << elapsed_prefill_start << ","
              << elapsed_prefill_end << ","
              << rss_before_q << "," << rss_first << ","
              << text_kv_prefill_ms << ",," << text_kv_prefill_ms << ","
              << kv_prefill << "," << kv_ctx << "," << kv_pct_prefill << ","
              << kv_used_kb_prefill << "," << kv_total_kb_q << ",\n";
        fproc << "Decode," << elapsed_decode_start << ","
              << elapsed_decode_end << ","
              << rss_first << "," << q_rss << ","
              << "," << q_gen_ms << "," << q_gen_ms << ","
              << kv_after << "," << kv_ctx << "," << kv_after_pct << ","
              << kv_used_kb_q << "," << kv_total_kb_q << ",\n";
        {
          std::lock_guard<std::mutex> lk2(d_rows_mutex);
          for (const auto& row : d_rows_buffer) {
            fproc << row;
          }
        }
        for (const auto& seg : segment_events) {
          int64_t seg_kv_pos = kv_prefill + seg.token_count;
          int64_t seg_kv_used_kb = (kv_ctx > 0 && kv_total_kb_q > 0)
              ? (seg_kv_pos * kv_total_kb_q) / kv_ctx
              : 0;
          double seg_kv_pct = (kv_ctx > 0) ? 100.0 * seg_kv_pos / kv_ctx : 0.0;
          fproc << "S," << seg.elapsed_s << "," << seg.elapsed_s << ","
                << seg.rss_kb << "," << seg.rss_kb << ",,,,"
                << seg_kv_pos << "," << kv_ctx << ","
                << seg_kv_pct << ","
                << seg_kv_used_kb << "," << kv_total_kb_q << ",\n";
        }
        fproc.flush();
      }

      // Full sequence = KV-cached frames + text query (echo) + generated answer
      if (FLAGS_output_minimal) {
        fout << kv_log << formatted_q;
        // Strip echoed query from answer to show only generated tokens
        std::string generated = answer;
        if (generated.size() >= formatted_q.size() &&
            generated.compare(0, formatted_q.size(), formatted_q) == 0) {
          generated = generated.substr(formatted_q.size());
        }
        fout << generated << "\n\n";
      } else {
        std::string generated = answer;
        if (generated.size() >= formatted_q.size() &&
            generated.compare(0, formatted_q.size(), formatted_q) == 0) {
          generated = generated.substr(formatted_q.size());
        }
        fout << "[t=" << static_cast<int>(q.timestamp_sec) << "s] Q: " << q.question << "\n";
        fout << "KV: " << kv_log << formatted_q;
        fout << "A: " << generated << "\n\n";
      }
      fout.flush();

      queries_pq.pop();
    }

    // Stop early if there's no room for the next frame + text query overhead.
    // img_seq_len + wrapper(~3) + 32 tokens margin for text query
    if (runner.get_cur_pos() + img_seq_len + 3 + 32 >= runner.get_context_len()) {
      ET_LOG(
          Info,
          "KV cache nearly full (pos=%ld). Stopping frame prefill.",
          runner.get_cur_pos());
      break;
    }
  }

  // Answer any remaining queries that didn't fire inside the frame loop
  while (!queries_pq.empty()) {
    const TimedQuery& q = queries_pq.top();
    ET_LOG(Info, "Answering remaining query: \"%s\"", q.question.c_str());

    out_buf.clear();
    std::atomic<long> rss_at_first_token2{-1};
    double text_kv_prefill_ms2 = 0.0;
    int64_t num_prompt_tokens2 = 0;
    struct SegmentEvent2 {
      double elapsed_s;
      long rss_kb;
      int64_t token_count;
    };
    std::vector<SegmentEvent2> segment_events2;
    std::atomic<int64_t> token_count2{0};
    constexpr int SEGMENT_SIZE2 = 256;
    std::vector<std::string> d_rows_buffer2;
    std::mutex d_rows_mutex2;
    auto stats_cb2 = [&](const executorch::extension::llm::Stats& s) {
      text_kv_prefill_ms2 = s.first_token_ms - s.inference_start_ms;
      num_prompt_tokens2 = s.num_prompt_tokens;
    };
    auto cb2 = [&](const std::string& piece) {
      for (char c : piece) { out_buf.push_back(c); }
      long expected = -1;
      if (rss_at_first_token2.compare_exchange_strong(expected, rss_kb())) {
        // First token emitted
      }
      int64_t count2 = token_count2.fetch_add(1) + 1;
      if (count2 % SEGMENT_SIZE2 == 0) {
        double elapsed_s2 = (now_ms() - t_run_start) / 1000.0;
        segment_events2.push_back({elapsed_s2, rss_kb(), count2});
      }
    };
    config.per_token_timing_cb = [&](int64_t token_idx, int64_t kv_pos,
                                     long start_ms, long end_ms) {
      if (ftokens.is_open()) {
        double start_s = (start_ms - t_run_start_realtime) / 1000.0;
        double end_s = (end_ms - t_run_start_realtime) / 1000.0;
        std::lock_guard<std::mutex> lk(ftokens_mutex);
        ftokens << token_idx << "," << kv_pos << "," << start_s << ","
                << end_s << "," << (end_ms - start_ms) << "\n";
        ftokens.flush();
      }
      if (fproc.is_open()) {
        double start_s = (start_ms - t_run_start_realtime) / 1000.0;
        double end_s = (end_ms - t_run_start_realtime) / 1000.0;
        long rss = rss_kb();
        int kv_ctx = runner.get_context_len();
        size_t kv_bytes = runner.get_kv_cache_total_bytes();
        int64_t kv_total_kb = kv_bytes > 0
            ? static_cast<int64_t>(kv_bytes / 1024)
            : (kv_ctx > 0 ? (24 * 2 * 16 * 64 * 1 * kv_ctx) / 1024 : 0);
        int64_t kv_used_kb = (kv_ctx > 0 && kv_total_kb > 0)
            ? (kv_pos * kv_total_kb) / kv_ctx
            : 0;
        double kv_pct = kv_ctx > 0 ? 100.0 * kv_pos / kv_ctx : 0.0;
        std::ostringstream oss;
        oss << "D," << start_s << "," << end_s << ","
            << rss << "," << rss << ","
            << "," << (end_ms - start_ms) << "," << (end_ms - start_ms) << ","
            << kv_pos << "," << kv_ctx << "," << kv_pct << ","
            << kv_used_kb << "," << kv_total_kb << ","
            << token_idx << "\n";
        std::lock_guard<std::mutex> lk(d_rows_mutex2);
        d_rows_buffer2.push_back(oss.str());
      }
    };
    std::string formatted_q =
        format_streaming_query(q.question, FLAGS_decoder_model_version);
    long rss_before_q2 = rss_kb();
    int64_t kv_before_q2 = runner.get_cur_pos();
    double t_q_start2 = now_ms();
    runner.generate_from_prompt_or_file(
        formatted_q, /*tokenized_prompt=*/false, config, cb2, stats_cb2);
    double t_q_end2 = now_ms();
    double qt2 = t_q_end2 - t_q_start2;
    double qp2 = text_kv_prefill_ms2;
    double qg2 = qt2 - qp2;
    long qr2 = rss_kb();
    int64_t kv2 = runner.get_cur_pos();
    int ctx2 = runner.get_context_len();
    double pct2 = ctx2 > 0 ? 100.0 * kv2 / ctx2 : 0.0;
    long rss_first2 =
        (rss_at_first_token2 >= 0) ? rss_at_first_token2.load() : rss_before_q2;

    int64_t kv_prefill2 = kv_before_q2 + num_prompt_tokens2;
    double kv_pct_prefill2 =
        (ctx2 > 0) ? 100.0 * kv_prefill2 / ctx2 : 0.0;

    size_t kv_total_bytes_q2 = runner.get_kv_cache_total_bytes();
    int64_t kv_total_kb_q2 = kv_total_bytes_q2 > 0
        ? static_cast<int64_t>(kv_total_bytes_q2 / 1024)
        : (24 * 2 * 16 * 64 * 1 * ctx2) / 1024;  // fallback
    int64_t kv_used_kb_prefill2 = (ctx2 > 0 && kv_total_kb_q2 > 0)
        ? (kv_prefill2 * kv_total_kb_q2) / ctx2
        : 0;
    int64_t kv_used_kb_q2 = (ctx2 > 0 && kv_total_kb_q2 > 0)
        ? (kv2 * kv_total_kb_q2) / ctx2
        : 0;
    double elapsed_prefill_start_q2 = (t_q_start2 - t_run_start) / 1000.0;
    double elapsed_prefill_end_q2 =
        elapsed_prefill_start_q2 + (text_kv_prefill_ms2 / 1000.0);
    double elapsed_decode_start_q2 = elapsed_prefill_end_q2;
    double elapsed_decode_end_q2 = (t_q_end2 - t_run_start) / 1000.0;

    ET_LOG(Info,
           "[remaining query t=%.1fs] text_kv_prefill=%.1fms  generate=%.1fms  KV=%ld/%d (%.1f%%)",
           q.timestamp_sec, qp2, qg2, kv2, ctx2, pct2);
    {
      std::lock_guard<std::mutex> lk(fproc_mutex);
      fproc << "T_Prefill," << elapsed_prefill_start_q2 << ","
            << elapsed_prefill_end_q2 << ","
            << rss_before_q2 << "," << rss_first2 << ","
            << text_kv_prefill_ms2 << ",," << text_kv_prefill_ms2 << ","
            << kv_prefill2 << "," << ctx2 << "," << kv_pct_prefill2 << ","
            << kv_used_kb_prefill2 << "," << kv_total_kb_q2 << ",\n";
      fproc << "Decode," << elapsed_decode_start_q2 << ","
            << elapsed_decode_end_q2 << ","
            << rss_first2 << "," << qr2 << ","
            << "," << qg2 << "," << qg2 << ","
            << kv2 << "," << ctx2 << "," << pct2 << ","
            << kv_used_kb_q2 << "," << kv_total_kb_q2 << ",\n";
      {
        std::lock_guard<std::mutex> lk2(d_rows_mutex2);
        for (const auto& row : d_rows_buffer2) {
          fproc << row;
        }
      }
      for (const auto& seg : segment_events2) {
        int64_t seg_kv_pos2 = kv_prefill2 + seg.token_count;
        int64_t seg_kv_used_kb2 = (ctx2 > 0 && kv_total_kb_q2 > 0)
            ? (seg_kv_pos2 * kv_total_kb_q2) / ctx2
            : 0;
        double seg_kv_pct2 = (ctx2 > 0) ? 100.0 * seg_kv_pos2 / ctx2 : 0.0;
        fproc << "S," << seg.elapsed_s << "," << seg.elapsed_s << ","
              << seg.rss_kb << "," << seg.rss_kb << ",,,,"
              << seg_kv_pos2 << "," << ctx2 << ","
              << seg_kv_pct2 << ","
              << seg_kv_used_kb2 << "," << kv_total_kb_q2 << ",\n";
      }
      fproc.flush();
    }
    std::string answer2(out_buf.begin(), out_buf.end());
    std::string formatted_q2 =
        format_streaming_query(q.question, FLAGS_decoder_model_version);
    if (FLAGS_output_minimal) {
      fout << kv_log << formatted_q2;
      std::string generated2 = answer2;
      if (generated2.size() >= formatted_q2.size() &&
          generated2.compare(0, formatted_q2.size(), formatted_q2) == 0) {
        generated2 = generated2.substr(formatted_q2.size());
      }
      fout << generated2 << "\n\n";
    } else {
      std::string generated2 = answer2;
      if (generated2.size() >= formatted_q2.size() &&
          generated2.compare(0, formatted_q2.size(), formatted_q2) == 0) {
        generated2 = generated2.substr(formatted_q2.size());
      }
      fout << "[end] Q: " << q.question << "\n";
      fout << "KV: " << kv_log << formatted_q2;
      fout << "A: " << generated2 << "\n\n";
    }
    queries_pq.pop();
  }

  // 인퍼런스 종료 후 5초간 mem.csv 샘플링 유지 (llama_main과 동일)
  std::this_thread::sleep_for(std::chrono::seconds(5));
  sampler_running.store(false);
  rss_sampler.join();

  dump_smaps(FLAGS_output_path + ".smaps");

  fout.close();
  fproc.close();
  fmem.close();
  double elapsed_s = (now_ms() - t_run_start) / 1000.0;
  ET_LOG(
      Info,
      "Streaming VLM done in %.1fs. Output: %s  Proc: %s  Mem: %s  Smaps: %s.smaps",
      elapsed_s,
      FLAGS_output_path.c_str(),
      proc_csv_path.c_str(),
      mem_csv_path.c_str(),
      FLAGS_output_path.c_str());
}

// ── Batch (HF-style) loop: all frames at once, single prefill ──────────────────

template <typename T>
void run_batch(
    double t_run_start,
    long t_run_start_realtime,
    std::unique_ptr<example::EncoderRunner> encoder_runner,
    std::unique_ptr<executorch::extension::Module> decoder_module,
    std::unique_ptr<executorch::extension::Module> embedding_module,
    const std::vector<std::pair<double, std::string>>& questions) {
  std::string proc_csv_path = FLAGS_output_path + ".proc.csv";
  std::ofstream fproc(proc_csv_path, std::ios::app);
  std::mutex fproc_mutex;

  std::string mem_csv_path = FLAGS_output_path + ".mem.csv";
  std::ofstream fmem(mem_csv_path);
  std::mutex fmem_mutex;
  fmem << "elapsed_s,rss_kb,rss_mb,delta_rss_kb,mem_total_kb,mem_available_kb\n";

  std::string tokens_csv_path = FLAGS_output_path + ".tokens.csv";
  std::ofstream ftokens(tokens_csv_path);
  std::mutex ftokens_mutex;
  if (ftokens.is_open()) {
    ftokens << "token_idx,kv_pos,elapsed_s_start,elapsed_s_end,ms\n";
  }

  long rss_load_start = rss_kb();
  std::atomic<bool> sampler_running{true};
  std::thread rss_sampler([&]() {
    long last_rss = rss_kb();
    while (sampler_running.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      if (!sampler_running.load()) break;
      double elapsed_s = (now_ms() - t_run_start) / 1000.0;
      long cur = rss_kb();
      long delta = (last_rss >= 0 && cur >= 0) ? cur - last_rss : 0;
      last_rss = cur;
      MemInfo mi = meminfo_kb();
      std::lock_guard<std::mutex> lk(fmem_mutex);
      fmem << elapsed_s << "," << cur << "," << cur / 1024.0 << "," << delta
           << "," << mi.mem_total_kb << "," << mi.mem_available_kb << "\n";
      fmem.flush();
    }
  });

  double t_encode_start = now_ms();
  ET_CHECK_MSG(
      encoder_runner->load() == executorch::runtime::Error::Ok,
      "Failed to load vision encoder");

  // Encode all frames and concatenate
  std::vector<float> concat_buffer;
  int64_t img_seq_len = 0;
  int64_t hidden_dim = 0;
  struct BatchFrameEncodeStat {
    double start_ms;
    double end_ms;
    long rss_start;
    long rss_end;
  };
  std::vector<BatchFrameEncodeStat> frame_encode_stats;

  for (int i = 0; i < FLAGS_frame_count; ++i) {
    long rss_frame_start = rss_kb();
    double t0 = now_ms();
    std::string fpath = frame_path(FLAGS_frame_dir, i);
    auto enc = encoder_runner->encode_from_file(fpath);
    ET_CHECK_MSG(enc.ok(), "Failed to encode frame %d", i);
    double t1 = now_ms();
    long rss_frame_end = rss_kb();
    frame_encode_stats.push_back({t0, t1, rss_frame_start, rss_frame_end});
    auto t = enc.get();
    int64_t n = t.size(1);
    int64_t d = t.size(2);
    if (img_seq_len == 0) img_seq_len = n;
    if (hidden_dim == 0) hidden_dim = d;
    const float* src = t.const_data_ptr<float>();
    concat_buffer.insert(
        concat_buffer.end(), src, src + n * d);
  }
  double t_encode_end = now_ms();
  long rss_after_encode = rss_kb();
  if (fproc.is_open()) {
    double elapsed_s = (t_encode_end - t_run_start) / 1000.0;
    double encode_ms = t_encode_end - t_encode_start;
    fproc << "L_V_Encode,0," << elapsed_s << ","
          << rss_load_start << "," << rss_after_encode << ","
          << encode_ms << ",," << encode_ms << ",,,,,,,\n";
    fproc.flush();
  }

  example::MultimodalRunner<T> runner(
      std::move(decoder_module),
      std::move(embedding_module),
      FLAGS_decoder_model_version,
      FLAGS_decoder_path,
      FLAGS_tokenizer_path,
      FLAGS_dump_logits_path,
      FLAGS_performance_output_path,
      static_cast<float>(FLAGS_temperature),
      FLAGS_eval_mode,
      FLAGS_shared_buffer,
      FLAGS_lazy_kv_alloc,
      /*ngram=*/0,
      /*window=*/0,
      /*gcap=*/0,
      /*image_hidden_states=*/nullptr);

  ET_CHECK_MSG(
      runner.load() == executorch::runtime::Error::Ok,
      "Failed to load MultimodalRunner");

  double t_load_end = now_ms();
  long rss_load_end = rss_kb();
  if (fproc.is_open()) {
    double elapsed_s = (t_load_end - t_run_start) / 1000.0;
    fproc << "L,0," << elapsed_s << "," << rss_load_start << ","
          << rss_load_end << ",,,,,,,,\n";
    fproc.flush();
  }

  if (FLAGS_frame_count > 0 && !concat_buffer.empty()) {
    runner.set_image_hidden_states_batch(
        std::move(concat_buffer),
        FLAGS_frame_count * img_seq_len,
        hidden_dim);
  }

  std::ofstream fout(FLAGS_output_path);
  executorch::extension::llm::GenerationConfig config{
      /*echo=*/true,
      /*ignore_eos=*/false,
      /*max_new_tokens=*/-1,
      /*warming=*/false,
      /*seq_len=*/FLAGS_seq_len,
      /*temperature=*/static_cast<float>(FLAGS_temperature),
      /*num_bos=*/0,
      /*num_eos=*/0};

  std::string kv_log = runner.get_prefix_format_string();
  for (int frame_idx = 0; frame_idx < FLAGS_frame_count; ++frame_idx) {
    kv_log += runner.get_frame_format_string(frame_idx, img_seq_len);
  }

  if (FLAGS_frame_count > 0) {
    int kv_total = runner.get_context_len();
    size_t kv_total_bytes = runner.get_kv_cache_total_bytes();
    int64_t kv_total_kb = kv_total_bytes > 0
        ? static_cast<int64_t>(kv_total_bytes / 1024)
        : (24 * 2 * 16 * 64 * 1 * kv_total) / 1024;
    for (const auto& stat : frame_encode_stats) {
      double elapsed_start = (stat.start_ms - t_run_start) / 1000.0;
      double elapsed_end = (stat.end_ms - t_run_start) / 1000.0;
      double encode_ms = stat.end_ms - stat.start_ms;
      fproc << "V_Encode," << elapsed_start << "," << elapsed_end << ","
            << stat.rss_start << "," << stat.rss_end << ","
            << encode_ms << ",," << encode_ms << ","
            << 0 << "," << kv_total << "," << 0 << ","
            << 0 << "," << kv_total_kb << ",\n";
    }
    fproc.flush();
  }

  for (const auto& [ts, q] : questions) {
    std::string full_prompt = build_batch_prompt(
        FLAGS_decoder_model_version,
        FLAGS_frame_count,
        img_seq_len,
        q);
    ET_CHECK_MSG(!full_prompt.empty(), "Failed to build batch prompt");
    std::string formatted_q =
        format_streaming_query(q, FLAGS_decoder_model_version);

    std::vector<char> out_buf;
    std::atomic<long> rss_at_first_token{-1};
    double text_kv_prefill_ms = 0.0;
    int64_t num_prompt_tokens = 0;
    struct SegmentEvent {
      double elapsed_s;
      long rss_kb;
      int64_t token_count;
    };
    std::vector<SegmentEvent> segment_events;
    std::atomic<int64_t> token_count{0};
    constexpr int SEGMENT_SIZE = 256;
    std::vector<std::string> d_rows_buffer;
    std::mutex d_rows_mutex;

    auto stats_cb = [&](const executorch::extension::llm::Stats& s) {
      text_kv_prefill_ms = s.first_token_ms - s.inference_start_ms;
      num_prompt_tokens = s.num_prompt_tokens;
    };
    auto cb = [&](const std::string& piece) {
      for (char c : piece) out_buf.push_back(c);
      long expected = -1;
      if (rss_at_first_token.compare_exchange_strong(expected, rss_kb())) {
      }
      int64_t count = token_count.fetch_add(1) + 1;
      if (count % SEGMENT_SIZE == 0) {
        double elapsed_s = (now_ms() - t_run_start) / 1000.0;
        segment_events.push_back({elapsed_s, rss_kb(), count});
      }
    };

    config.per_token_timing_cb = [&](int64_t token_idx, int64_t kv_pos,
                                     long start_ms, long end_ms) {
      if (ftokens.is_open()) {
        double start_s = (start_ms - t_run_start_realtime) / 1000.0;
        double end_s = (end_ms - t_run_start_realtime) / 1000.0;
        std::lock_guard<std::mutex> lk(ftokens_mutex);
        ftokens << token_idx << "," << kv_pos << "," << start_s << ","
                << end_s << "," << (end_ms - start_ms) << "\n";
        ftokens.flush();
      }
      if (fproc.is_open()) {
        double start_s = (start_ms - t_run_start_realtime) / 1000.0;
        double end_s = (end_ms - t_run_start_realtime) / 1000.0;
        long rss = rss_kb();
        int kv_ctx = runner.get_context_len();
        size_t kv_bytes = runner.get_kv_cache_total_bytes();
        int64_t kv_total_kb = kv_bytes > 0
            ? static_cast<int64_t>(kv_bytes / 1024)
            : (kv_ctx > 0 ? (24 * 2 * 16 * 64 * 1 * kv_ctx) / 1024 : 0);
        int64_t kv_used_kb = (kv_ctx > 0 && kv_total_kb > 0)
            ? (kv_pos * kv_total_kb) / kv_ctx
            : 0;
        double kv_pct = kv_ctx > 0 ? 100.0 * kv_pos / kv_ctx : 0.0;
        std::ostringstream oss;
        oss << "D," << start_s << "," << end_s << ","
            << rss << "," << rss << ","
            << "," << (end_ms - start_ms) << "," << (end_ms - start_ms) << ","
            << kv_pos << "," << kv_ctx << "," << kv_pct << ","
            << kv_used_kb << "," << kv_total_kb << ","
            << token_idx << "\n";
        std::lock_guard<std::mutex> lk(d_rows_mutex);
        d_rows_buffer.push_back(oss.str());
      }
    };

    long rss_before_q = rss_kb();
    int64_t kv_before_q = runner.get_cur_pos();
    double t_q_start = now_ms();
    auto err = runner.generate_from_prompt_or_file(
        full_prompt, /*tokenized_prompt=*/false, config, cb, stats_cb);
    double t_q_end = now_ms();
    ET_CHECK_MSG(err == executorch::runtime::Error::Ok, "generate failed");

    if (fproc.is_open()) {
      double elapsed_prefill_start = (t_q_start - t_run_start) / 1000.0;
      double elapsed_prefill_end =
          elapsed_prefill_start + (text_kv_prefill_ms / 1000.0);
      double elapsed_decode_start = elapsed_prefill_end;
      double elapsed_decode_end = (t_q_end - t_run_start) / 1000.0;
      long rss_q = rss_kb();
      long rss_first = (rss_at_first_token >= 0) ? rss_at_first_token.load()
                                                 : rss_before_q;
      int64_t kv_after = runner.get_cur_pos();
      int kv_ctx = runner.get_context_len();
      double kv_after_pct = kv_ctx > 0 ? 100.0 * kv_after / kv_ctx : 0.0;
      double q_total_ms = t_q_end - t_q_start;
      double q_gen_ms = q_total_ms - text_kv_prefill_ms;
      size_t kv_total_bytes_q = runner.get_kv_cache_total_bytes();
      int64_t kv_total_kb_q = kv_total_bytes_q > 0
          ? static_cast<int64_t>(kv_total_bytes_q / 1024)
          : (24 * 2 * 16 * 64 * 1 * kv_ctx) / 1024;
      int64_t kv_prefill = kv_before_q + num_prompt_tokens;
      int64_t kv_used_kb_prefill = (kv_ctx > 0 && kv_total_kb_q > 0)
          ? (kv_prefill * kv_total_kb_q) / kv_ctx
          : 0;
      double kv_pct_prefill =
          (kv_ctx > 0) ? 100.0 * kv_prefill / kv_ctx : 0.0;
      int64_t kv_used_kb_q = (kv_ctx > 0 && kv_total_kb_q > 0)
          ? (kv_after * kv_total_kb_q) / kv_ctx
          : 0;

      fproc << "T_Prefill," << elapsed_prefill_start << ","
            << elapsed_prefill_end << ","
            << rss_before_q << "," << rss_first << ","
            << text_kv_prefill_ms << ",," << text_kv_prefill_ms << ","
            << kv_prefill << "," << kv_ctx << "," << kv_pct_prefill << ","
            << kv_used_kb_prefill << "," << kv_total_kb_q << ","
            << "\n";
      fproc << "Decode," << elapsed_decode_start << "," << elapsed_decode_end
            << ","
            << rss_first << "," << rss_q << ","
            << "," << q_gen_ms << "," << q_gen_ms << ","
            << kv_after << "," << kv_ctx << "," << kv_after_pct << ","
            << kv_used_kb_q << "," << kv_total_kb_q << ","
            << "\n";
      {
        std::lock_guard<std::mutex> lk(d_rows_mutex);
        for (const auto& row : d_rows_buffer) {
          fproc << row;
        }
      }
      for (const auto& seg : segment_events) {
        int64_t seg_kv_pos = kv_prefill + seg.token_count;
        int64_t seg_kv_used_kb = (kv_ctx > 0 && kv_total_kb_q > 0)
            ? (seg_kv_pos * kv_total_kb_q) / kv_ctx
            : 0;
        double seg_kv_pct = (kv_ctx > 0) ? 100.0 * seg_kv_pos / kv_ctx : 0.0;
        fproc << "S," << seg.elapsed_s << "," << seg.elapsed_s << ","
              << seg.rss_kb << "," << seg.rss_kb << ",,,,"
              << seg_kv_pos << "," << kv_ctx << ","
              << seg_kv_pct << ","
              << seg_kv_used_kb << "," << kv_total_kb_q << ",\n";
      }
      fproc.flush();
    }

    std::string answer(out_buf.begin(), out_buf.end());
    // echo=true이므로 answer = full_prompt + generated_tokens. 중복 방지.
    std::string generated;
    if (answer.size() >= full_prompt.size() &&
        answer.compare(0, full_prompt.size(), full_prompt) == 0) {
      generated = answer.substr(full_prompt.size());
    } else {
      generated = answer;
    }

    if (FLAGS_output_minimal) {
      fout << kv_log << formatted_q;
      fout << generated << "\n\n";
    } else {
      fout << "[t=" << static_cast<int>(ts) << "s] Q: " << q << "\n";
      fout << "KV: " << kv_log << formatted_q;
      fout << "A: " << generated << "\n\n";
    }
    fout.flush();
  }

  fout.close();
  fproc.close();
  sampler_running.store(false);
  if (rss_sampler.joinable()) {
    rss_sampler.join();
  }
  fmem.close();
  ftokens.close();
  double elapsed_s = (now_ms() - t_run_start) / 1000.0;
  ET_LOG(Info, "Batch VLM done in %.1fs. Output: %s", elapsed_s, FLAGS_output_path.c_str());
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  ET_CHECK_MSG(
      FLAGS_frame_count >= 0,
      "--frame_count must be >= 0 (0 = text-only, no vision input)");

  auto question_list = split(FLAGS_questions, ';');
  auto timestamp_strs = split(FLAGS_query_timestamps, ';');
  ET_CHECK_MSG(
      question_list.size() == timestamp_strs.size(),
      "Number of questions (%zu) must match number of timestamps (%zu)",
      question_list.size(),
      timestamp_strs.size());

  std::priority_queue<
      TimedQuery,
      std::vector<TimedQuery>,
      std::greater<TimedQuery>>
      queries_pq;
  for (size_t i = 0; i < question_list.size(); ++i) {
    queries_pq.push({std::stod(timestamp_strs[i]), question_list[i]});
  }

  ET_LOG(Info, "lazy_kv_alloc=%s", FLAGS_lazy_kv_alloc ? "true" : "false");
  ET_LOG(Info, "Encoder:   %s", FLAGS_encoder_path.c_str());
  ET_LOG(Info, "Decoder:   %s", FLAGS_decoder_path.c_str());
  ET_LOG(Info, "Embedding: %s", FLAGS_embedding_path.c_str());
  ET_LOG(Info, "Frame dir: %s  count=%d  fps=%.2f",
         FLAGS_frame_dir.c_str(), FLAGS_frame_count, FLAGS_fps);

  double t_run_start = now_ms();
  long t_run_start_realtime = executorch::extension::llm::time_in_ms();
  std::string proc_csv_path = FLAGS_output_path + ".proc.csv";
  std::unique_ptr<executorch::extension::Module> embedding_module;
  std::unique_ptr<executorch::extension::Module> decoder_module;
  {
    std::ofstream fproc(proc_csv_path);
    fproc << "row_type,elapsed_s_start,elapsed_s_end,rss_kb_start,rss_kb_end,"
             "col_a_ms,col_b_ms,total_ms,kv_pos,kv_total,kv_used_pct,"
             "kv_used_kb,kv_total_kb,token_idx\n";
    fproc << "# L: Loading (전체)  L_V_Encode: 비전 인코더  L_Decoder: 텍스트 디코더  "
             "L_Embedding: 텍스트 임베딩\n";
    fproc << "# V_Encode: col_a=vision_encode_ms (비전 인코더만)\n";
    fproc << "# V_Prefill: col_a=vision_kv_prefill_ms (비전→LLM KV)\n";
    fproc << "# T_Prefill: col_a=text_kv_prefill_ms  Decode: col_b=token_gen_ms\n";
    fproc << "# S: 256토큰 구간 경계  D: 토큰별 decode\n";

    embedding_module = std::make_unique<executorch::extension::Module>(
        FLAGS_embedding_path,
        executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
    decoder_module = std::make_unique<executorch::extension::Module>(
        FLAGS_decoder_path,
        executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
    fproc.flush();
  }

  auto encoder_runner =
      std::make_unique<example::EncoderRunner>(FLAGS_encoder_path);

  example::KvBitWidth kv_bitwidth = example::KvBitWidth::kWidth8;
  if (decoder_module->method_names()->count("get_kv_io_bit_width") > 0) {
    kv_bitwidth = static_cast<example::KvBitWidth>(
        decoder_module->get("get_kv_io_bit_width")
            .get()
            .toScalar()
            .to<int64_t>());
  }

  std::vector<std::pair<double, std::string>> questions_vec;
  while (!queries_pq.empty()) {
    auto q = queries_pq.top();
    questions_vec.push_back({q.timestamp_sec, q.question});
    queries_pq.pop();
  }

  if (FLAGS_stream) {
    std::priority_queue<TimedQuery, std::vector<TimedQuery>, std::greater<TimedQuery>>
        pq_restored;
    for (const auto& [ts, q] : questions_vec) {
      pq_restored.push({ts, q});
    }
    if (kv_bitwidth == example::KvBitWidth::kWidth8) {
      run_streaming<uint8_t>(
          t_run_start, t_run_start_realtime,
          std::move(encoder_runner),
          std::move(decoder_module),
          std::move(embedding_module),
          pq_restored);
    } else {
      run_streaming<uint16_t>(
          t_run_start, t_run_start_realtime,
          std::move(encoder_runner),
          std::move(decoder_module),
          std::move(embedding_module),
          pq_restored);
    }
  } else {
    if (kv_bitwidth == example::KvBitWidth::kWidth8) {
      run_batch<uint8_t>(
          t_run_start, t_run_start_realtime,
          std::move(encoder_runner),
          std::move(decoder_module),
          std::move(embedding_module),
          questions_vec);
    } else {
      run_batch<uint16_t>(
          t_run_start, t_run_start_realtime,
          std::move(encoder_runner),
          std::move(decoder_module),
          std::move(embedding_module),
          questions_vec);
    }
  }

  return 0;
}