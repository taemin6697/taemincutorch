/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/foundation/runner/backend.h>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/llm/sampler/util.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/log.h>

#include <cstring>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace executorch::examples::foundation {

namespace {

using ::executorch::extension::llm::load_tokenizer;
using ::executorch::extension::llm::logits_to_token;
using ::executorch::extension::llm::populate_start_pos_or_cache_position;
using ::executorch::extension::llm::get_rss_bytes;
using ::executorch::extension::llm::time_in_ms;
using ::executorch::extension::clone_tensor_ptr;
using ::executorch::extension::from_blob;
using ::executorch::extension::make_tensor_ptr;
using ::executorch::extension::Module;

constexpr int32_t kInternVL3ImageSeqLen = 256;
constexpr const char* kInternVL3ImageToken = "<IMG_CONTEXT>";
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
           "L_DecoderLoad: 텍스트 디코더 load  "
           "L_EmbeddingLoad: 토크나이저/텍스트 임베딩 load\n";
  fproc << "# V_Encode: col_a=vision_encode_ms (비전 인코더만)\n";
  fproc << "# T_Prefill: col_a=text_kv_prefill_ms  Decode: col_b=token_gen_ms\n";
  fproc << "# S: 256토큰 구간 경계  D: 토큰별 decode\n";
}

std::string frame_path(const std::string& dir, int idx) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "/frame_%04d.bin", idx);
  return dir + buf;
}

::executorch::extension::TensorPtr load_preprocessed_frame(
    const std::string& path,
    int32_t image_size,
    int32_t channels = 3) {
  std::ifstream input(path, std::ios::binary);
  ET_CHECK_MSG(input.is_open(), "Failed to open frame bin: %s", path.c_str());

  input.seekg(0, std::ios::end);
  const auto num_bytes = input.tellg();
  input.seekg(0, std::ios::beg);

  const size_t expected_floats =
      static_cast<size_t>(channels) * image_size * image_size;
  const size_t expected_bytes = expected_floats * sizeof(float);
  ET_CHECK_MSG(
      static_cast<size_t>(num_bytes) == expected_bytes,
      "Unexpected frame size for %s (expected %zu bytes, got %lld)",
      path.c_str(),
      expected_bytes,
      static_cast<long long>(num_bytes));

  std::vector<float> data(expected_floats);
  input.read(reinterpret_cast<char*>(data.data()), expected_bytes);
  ET_CHECK_MSG(input.good(), "Failed to read frame data: %s", path.c_str());
  return make_tensor_ptr(
      std::vector<executorch::aten::SizesType>{1, channels, image_size, image_size},
      std::move(data));
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

std::string build_full_prompt_text(int32_t frame_count, const std::string& question) {
  std::string s = "<|im_start|>user:\n";
  for (int32_t idx = 0; idx < frame_count; ++idx) {
    const int32_t frame_num = idx + 1;
    s += "Frame" + std::to_string(frame_num) + ": <img>";
    for (int32_t i = 0; i < kInternVL3ImageSeqLen; ++i) {
      s += kInternVL3ImageToken;
    }
    s += "</img>\n";
  }
  s += question + "<|im_end|>\n<|im_start|>assistant\n";
  return s;
}

uint64_t placeholder_token_id(tokenizers::Tokenizer* tokenizer) {
  auto tk = tokenizer->encode(kInternVL3ImageToken, 0, 0);
  ET_CHECK_MSG(tk.ok(), "Failed to encode placeholder token");
  ET_CHECK_MSG(!tk->empty(), "Placeholder token encoding is empty");
  return tk->at(0);
}

std::unordered_set<uint64_t> stop_token_ids(tokenizers::Tokenizer* tokenizer) {
  std::unordered_set<uint64_t> ids;
  const auto eos = tokenizer->eos_tok();
  if (eos >= 0) {
    ids.insert(static_cast<uint64_t>(eos));
  }
  for (const char* token : {"<|im_end|>", "<|end_of_text|>"}) {
    auto encoded = tokenizer->encode(token, 0, 0);
    if (!encoded.ok()) {
      continue;
    }
    for (auto id : *encoded) {
      ids.insert(id);
    }
  }
  return ids;
}

bool contains_stop_marker(const std::string& piece) {
  return piece.find("<|im_end|>") != std::string::npos ||
      piece.find("<|end_of_text|>") != std::string::npos;
}

template <typename T>
void merge_image_features_typed(
    std::vector<T>& merged,
    const executorch::aten::Tensor& text_embeddings,
    const std::vector<executorch::aten::Tensor>& image_tensors,
    const std::vector<uint64_t>& input_ids,
    uint64_t image_placeholder_id) {
  const auto* text_ptr = text_embeddings.const_data_ptr<T>();
  const int64_t num_tokens = text_embeddings.size(1);
  const int64_t hidden_dim = text_embeddings.size(2);
  merged.assign(
      text_ptr,
      text_ptr + static_cast<size_t>(num_tokens * hidden_dim));

  std::vector<size_t> placeholder_positions;
  for (size_t i = 0; i < input_ids.size(); ++i) {
    if (input_ids[i] == image_placeholder_id) {
      placeholder_positions.push_back(i);
    }
  }

  size_t image_token_offset = 0;
  for (const auto& image_tensor : image_tensors) {
    ET_CHECK_MSG(
        image_tensor.scalar_type() == text_embeddings.scalar_type(),
        "Image hidden state dtype must match text embedding dtype");
    const auto* image_ptr = image_tensor.const_data_ptr<T>();
    const int64_t image_seq_len = image_tensor.size(1);
    for (int64_t i = 0; i < image_seq_len; ++i) {
      const size_t pos = placeholder_positions.at(image_token_offset + i);
      std::memcpy(
          merged.data() + pos * hidden_dim,
          image_ptr + i * hidden_dim,
          static_cast<size_t>(hidden_dim) * sizeof(T));
    }
    image_token_offset += static_cast<size_t>(image_seq_len);
  }
}

::executorch::extension::TensorPtr build_merged_embeddings(
    const executorch::aten::Tensor& text_embeddings,
    const std::vector<executorch::aten::Tensor>& image_tensors,
    const std::vector<uint64_t>& input_ids,
    uint64_t image_placeholder_id) {
  const auto sizes = std::vector<executorch::aten::SizesType>{
      1,
      static_cast<executorch::aten::SizesType>(text_embeddings.size(1)),
      static_cast<executorch::aten::SizesType>(text_embeddings.size(2))};
  switch (text_embeddings.scalar_type()) {
    case executorch::aten::ScalarType::Float: {
      std::vector<float> merged;
      merge_image_features_typed<float>(
          merged, text_embeddings, image_tensors, input_ids, image_placeholder_id);
      return make_tensor_ptr(std::move(sizes), std::move(merged));
    }
    case executorch::aten::ScalarType::Half: {
      std::vector<executorch::aten::Half> merged;
      merge_image_features_typed<executorch::aten::Half>(
          merged, text_embeddings, image_tensors, input_ids, image_placeholder_id);
      return make_tensor_ptr(
          std::move(sizes),
          std::move(merged),
          {},
          {},
          executorch::aten::ScalarType::Half);
    }
    case executorch::aten::ScalarType::BFloat16: {
      std::vector<executorch::aten::BFloat16> merged;
      merge_image_features_typed<executorch::aten::BFloat16>(
          merged, text_embeddings, image_tensors, input_ids, image_placeholder_id);
      return make_tensor_ptr(
          std::move(sizes),
          std::move(merged),
          {},
          {},
          executorch::aten::ScalarType::BFloat16);
    }
    default:
      ET_CHECK_MSG(false, "Unsupported embedding dtype for XNNPACK split backend");
  }
}

executorch::runtime::Result<executorch::aten::Tensor> run_token_embedding(
    Module& embedding_module,
    const std::vector<uint64_t>& tokens) {
  auto token_tensor = from_blob(
      const_cast<uint64_t*>(tokens.data()),
      {1, static_cast<executorch::aten::SizesType>(tokens.size())},
      executorch::aten::ScalarType::Long);
  auto outputs = embedding_module.execute("forward", token_tensor);
  if (!outputs.ok()) {
    return outputs.error();
  }
  return outputs->at(0).toTensor();
}

executorch::runtime::Result<executorch::aten::Tensor> run_decoder_forward(
    Module& decoder_module,
    const executorch::runtime::EValue& embeddings,
    int64_t& start_pos,
    int seq_len) {
  std::vector<int64_t> cache_positions;
  auto cache_position_tensor =
      populate_start_pos_or_cache_position(&decoder_module, start_pos, cache_positions, seq_len);
  if (!cache_position_tensor.ok()) {
    return cache_position_tensor.error();
  }
  auto outputs =
      decoder_module.execute("forward", {embeddings, *cache_position_tensor.get()});
  if (!outputs.ok()) {
    return outputs.error();
  }
  return outputs->at(0).toTensor();
}

class XnnpackBackendRunner final : public BackendRunner {
 public:
  explicit XnnpackBackendRunner(ManifestData manifest)
      : manifest_(std::move(manifest)) {}

  executorch::runtime::Error validate() override {
    if (manifest_.paths.tokenizer_path.empty()) {
      ET_LOG(Error, "Missing tokenizer_path in manifest");
      return executorch::runtime::Error::InvalidArgument;
    }
    if (manifest_.paths.vision_encoder_pte.empty() ||
        manifest_.paths.text_embedding_pte.empty() ||
        manifest_.paths.text_decoder_pte.empty()) {
      ET_LOG(Error, "XNNPACK manifest requires split-PTE paths.");
      return executorch::runtime::Error::NotSupported;
    }
    return executorch::runtime::Error::Ok;
  }

  executorch::runtime::Error run(const UnifiedRunConfig& config) override {
    ET_CHECK_MSG(!config.frame_dir.empty(), "--frame_dir is required.");
    ET_CHECK_MSG(config.frame_count > 0, "--frame_count must be > 0.");
    const long t_run_start = time_in_ms();
    std::ofstream fproc(kFoundationProcCsv);
    ET_CHECK_MSG(
        fproc.is_open(),
        "Failed to open proc csv file: %s",
        kFoundationProcCsv);
    write_proc_header(fproc);
    const long t_load_start = t_run_start;
    const long rss_load_start = rss_kb();

    const long rss_embedding_before = rss_kb();
    const long t_embedding_load_start = time_in_ms();
    auto tokenizer = load_tokenizer(manifest_.paths.tokenizer_path);
    ET_CHECK_MSG(
        tokenizer != nullptr,
        "Failed to load tokenizer: %s",
        manifest_.paths.tokenizer_path.c_str());
    const long t_embedding_load_end = time_in_ms();
    const long rss_embedding_after = rss_kb();
    const auto eos_token_id = static_cast<uint64_t>(tokenizer->eos_tok());
    const auto image_placeholder_id = placeholder_token_id(tokenizer.get());
    const auto stop_ids = stop_token_ids(tokenizer.get());

    const long rss_vision_before = rss_kb();
    const long t_vision_load_start = time_in_ms();
    Module vision_module(
        manifest_.paths.vision_encoder_pte,
        Module::LoadMode::MmapUseMlockIgnoreErrors);
    ET_CHECK_OK_OR_RETURN_ERROR(vision_module.load_method("forward"));
    const long t_vision_load_end = time_in_ms();
    const long rss_vision_after = rss_kb();

    const long rss_embedding_module_before = rss_kb();
    const long t_embedding_module_start = time_in_ms();
    Module embedding_module(
        manifest_.paths.text_embedding_pte,
        Module::LoadMode::MmapUseMlockIgnoreErrors);
    ET_CHECK_OK_OR_RETURN_ERROR(embedding_module.load_method("forward"));
    const long t_embedding_module_end = time_in_ms();
    const long rss_embedding_module_after = rss_kb();

    const long rss_decoder_before = rss_kb();
    const long t_decoder_load_start = time_in_ms();
    {
      Module decoder_probe(
          manifest_.paths.text_decoder_pte,
          Module::LoadMode::MmapUseMlockIgnoreErrors);
      ET_CHECK_OK_OR_RETURN_ERROR(decoder_probe.load_method("forward"));
    }
    const long t_decoder_load_end = time_in_ms();
    const long rss_decoder_after = rss_kb();
    const long t_load_end = t_decoder_load_end;
    const long rss_load_end = rss_decoder_after;
    fproc << "L_VisionLoad," << (t_vision_load_start - t_run_start) / 1000.0 << ","
          << (t_vision_load_end - t_run_start) / 1000.0 << ","
          << rss_vision_before << "," << rss_vision_after << ","
          << (t_vision_load_end - t_vision_load_start) << ",,"
          << (t_vision_load_end - t_vision_load_start) << ",,,,,\n";
    fproc << "L_EmbeddingLoad," << (t_embedding_load_start - t_run_start) / 1000.0 << ","
          << (t_embedding_module_end - t_run_start) / 1000.0 << ","
          << rss_embedding_before << "," << rss_embedding_module_after << ","
          << (t_embedding_module_end - t_embedding_load_start) << ",,"
          << (t_embedding_module_end - t_embedding_load_start) << ",,,,,\n";
    fproc << "L_DecoderLoad," << (t_decoder_load_start - t_run_start) / 1000.0 << ","
          << (t_decoder_load_end - t_run_start) / 1000.0 << ","
          << rss_decoder_before << "," << rss_decoder_after << ","
          << (t_decoder_load_end - t_decoder_load_start) << ",,"
          << (t_decoder_load_end - t_decoder_load_start) << ",,,,,\n";
    fproc << "L," << (t_load_start - t_run_start) / 1000.0 << ","
          << (t_load_end - t_run_start) / 1000.0 << ","
          << rss_load_start << "," << rss_load_end << ","
          << (t_load_end - t_load_start) << ",,"
          << (t_load_end - t_load_start) << ",,,,,\n";

    std::vector<executorch::aten::Tensor> image_tensors;
    image_tensors.reserve(config.frame_count);
    for (int idx = 0; idx < config.frame_count; ++idx) {
      const long rss_before_encode = rss_kb();
      const long t_encode_start = time_in_ms();
      auto frame_tensor = load_preprocessed_frame(frame_path(config.frame_dir, idx), 448);
      // fp16 vision encoder expects Half; frame .bin is always float32
      auto frame_fp16 =
          clone_tensor_ptr(*frame_tensor, executorch::aten::ScalarType::Half);
      auto outputs = vision_module.execute("forward", frame_fp16);
      ET_CHECK_OK_OR_RETURN_ERROR(outputs.error());
      const long t_encode_end = time_in_ms();
      const long rss_after_encode = rss_kb();
      image_tensors.push_back(outputs->at(0).toTensor());
      fproc << "V_Encode," << (t_encode_start - t_run_start) / 1000.0 << ","
            << (t_encode_end - t_run_start) / 1000.0 << ","
            << rss_before_encode << "," << rss_after_encode << ","
            << (t_encode_end - t_encode_start) << ",,"
            << (t_encode_end - t_encode_start) << ",,,,,\n";
    }

    std::ostringstream final_output;
    for (const auto& question : split_questions(config.questions)) {
      std::string full_prompt = build_full_prompt_text(config.frame_count, question);
      auto encoded = tokenizer->encode(full_prompt, 0, 0);
      ET_CHECK_MSG(encoded.ok(), "Failed to encode prompt for question");
      std::vector<uint64_t> prompt_tokens = std::move(*encoded);
      final_output << full_prompt;

      auto text_embeddings_res = run_token_embedding(embedding_module, prompt_tokens);
      ET_CHECK_OK_OR_RETURN_ERROR(text_embeddings_res.error());
      auto text_embeddings = text_embeddings_res.get();

      auto merged_embeddings = build_merged_embeddings(
          text_embeddings, image_tensors, prompt_tokens, image_placeholder_id);
      int64_t start_pos = 0;
      const int64_t kv_total =
          static_cast<int64_t>(prompt_tokens.size()) + static_cast<int64_t>(config.seq_len);

      Module decoder_module(
          manifest_.paths.text_decoder_pte,
          Module::LoadMode::MmapUseMlockIgnoreErrors);
      ET_CHECK_OK_OR_RETURN_ERROR(decoder_module.load_method("forward"));

      const long rss_before_q = rss_kb();
      const long t_prefill_start = time_in_ms();
      auto logits_res = run_decoder_forward(
          decoder_module,
          *merged_embeddings,
          start_pos,
          static_cast<int>(prompt_tokens.size()));
      ET_CHECK_OK_OR_RETURN_ERROR(logits_res.error());
      const long t_prefill_end = time_in_ms();
      auto logits = logits_res.get();
      start_pos += static_cast<int64_t>(prompt_tokens.size());
      uint64_t cur_token =
          static_cast<uint64_t>(logits_to_token(logits, static_cast<float>(config.temperature)));
      uint64_t prev_token = cur_token;
      int64_t kv_pos = static_cast<int64_t>(prompt_tokens.size());
      std::ostringstream answer;
      std::vector<std::string> d_rows_buffer;
      struct SegmentEvent {
        double elapsed_s;
        long rss_kb_value;
        int64_t token_count;
      };
      std::vector<SegmentEvent> segment_events;
      constexpr int kSegmentSize = 256;
      int64_t emitted_tokens = 0;
      int64_t timed_token_idx = 0;
      long rss_first = rss_before_q;
      bool saw_first_token = false;
      const long t_decode_start = t_prefill_end;

      auto first_piece = tokenizer->decode(prev_token, cur_token);
      if (first_piece.ok()) {
        rss_first = rss_kb();
        saw_first_token = true;
        answer << *first_piece;
        emitted_tokens += 1;
        if (contains_stop_marker(*first_piece)) {
          goto finish_question;
        }
      }
      if (cur_token == eos_token_id || stop_ids.count(cur_token) > 0) {
        goto finish_question;
      }

      for (int i = 1; i < config.seq_len; ++i) {
        std::vector<uint64_t> next_token{cur_token};
        const long t_token_start = time_in_ms();
        auto next_emb_res = run_token_embedding(embedding_module, next_token);
        ET_CHECK_OK_OR_RETURN_ERROR(next_emb_res.error());
        auto next_emb = next_emb_res.get();

        start_pos = kv_pos;
        auto next_logits_res =
            run_decoder_forward(decoder_module, next_emb, start_pos, 1);
        ET_CHECK_OK_OR_RETURN_ERROR(next_logits_res.error());
        auto next_logits = next_logits_res.get();
        const uint64_t next_generated_token = static_cast<uint64_t>(
            logits_to_token(next_logits, static_cast<float>(config.temperature)));
        const long t_token_end = time_in_ms();

        prev_token = cur_token;
        cur_token = next_generated_token;
        kv_pos += 1;

        auto decode_piece = tokenizer->decode(prev_token, cur_token);
        if (decode_piece.ok()) {
          answer << *decode_piece;
          const double start_s = (t_token_start - t_run_start) / 1000.0;
          const double end_s = (t_token_end - t_run_start) / 1000.0;
          const double token_ms = static_cast<double>(t_token_end - t_token_start);
          const long rss_now = rss_kb();
          const double kv_pct = kv_total > 0 ? 100.0 * kv_pos / kv_total : 0.0;
          std::ostringstream drow;
          drow << "D," << start_s << "," << end_s << ","
               << rss_now << "," << rss_now << ","
               << "," << token_ms << "," << token_ms << ","
               << kv_pos << "," << kv_total << "," << kv_pct << ","
               << 0 << "," << 0 << ","
               << timed_token_idx << "\n";
          d_rows_buffer.push_back(drow.str());
          timed_token_idx += 1;
          emitted_tokens += 1;
          if (emitted_tokens % kSegmentSize == 0) {
            segment_events.push_back(
                {static_cast<double>(t_token_end - t_run_start) / 1000.0, rss_now, emitted_tokens});
          }
          if (contains_stop_marker(*decode_piece)) {
            break;
          }
        }
        if (cur_token == eos_token_id || stop_ids.count(cur_token) > 0) {
          break;
        }
      }

finish_question:

      const long t_q_end = time_in_ms();
      const double text_kv_prefill_ms = static_cast<double>(t_prefill_end - t_prefill_start);
      const double q_gen_ms = static_cast<double>(t_q_end - t_decode_start);
      const double elapsed_prefill_start = (t_prefill_start - t_run_start) / 1000.0;
      const double elapsed_prefill_end = (t_prefill_end - t_run_start) / 1000.0;
      const double elapsed_decode_end = (t_q_end - t_run_start) / 1000.0;
      const long rss_q = rss_kb();
      const int64_t kv_prefill = static_cast<int64_t>(prompt_tokens.size());
      const int64_t kv_after = kv_prefill + emitted_tokens;
      const double kv_pct_prefill = kv_total > 0 ? 100.0 * kv_prefill / kv_total : 0.0;
      const double kv_after_pct = kv_total > 0 ? 100.0 * kv_after / kv_total : 0.0;
      fproc << "T_Prefill," << elapsed_prefill_start << ","
            << elapsed_prefill_end << ","
            << rss_before_q << "," << rss_first << ","
            << text_kv_prefill_ms << ",," << text_kv_prefill_ms << ","
            << kv_prefill << "," << kv_total << "," << kv_pct_prefill << ","
            << 0 << "," << 0 << ",\n";
      fproc << "Decode," << elapsed_prefill_end << "," << elapsed_decode_end << ","
            << rss_first << "," << rss_q << ","
            << "," << q_gen_ms << "," << q_gen_ms << ","
            << kv_after << "," << kv_total << "," << kv_after_pct << ","
            << 0 << "," << 0 << ",\n";
      for (const auto& row : d_rows_buffer) {
        fproc << row;
      }
      for (const auto& seg : segment_events) {
        const int64_t seg_kv_pos = kv_prefill + seg.token_count;
        const double seg_kv_pct = kv_total > 0 ? 100.0 * seg_kv_pos / kv_total : 0.0;
        fproc << "S," << seg.elapsed_s << "," << seg.elapsed_s << ","
              << seg.rss_kb_value << "," << seg.rss_kb_value << ",,,,"
              << seg_kv_pos << "," << kv_total << "," << seg_kv_pct << ","
              << 0 << "," << 0 << ",\n";
      }
      fproc.flush();

      final_output << answer.str();
      if (!question.empty()) {
        final_output << "\n";
      }
    }

    if (!config.output_path.empty()) {
      std::ofstream out(config.output_path);
      ET_CHECK_MSG(out.is_open(), "Failed to open output file: %s", config.output_path.c_str());
      out << final_output.str();
    } else {
      std::fwrite(final_output.str().data(), 1, final_output.str().size(), stdout);
    }
    return executorch::runtime::Error::Ok;
  }

 private:
  ManifestData manifest_;
};

} // namespace

std::unique_ptr<BackendRunner> create_xnnpack_backend_runner(
    const ManifestData& manifest) {
  return std::make_unique<XnnpackBackendRunner>(manifest);
}

} // namespace executorch::examples::foundation
