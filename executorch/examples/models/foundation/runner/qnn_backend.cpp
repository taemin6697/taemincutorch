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
#include <executorch/extension/module/module.h>
#endif
#include <executorch/runtime/platform/log.h>

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

  std::unique_ptr<example::EncoderRunner> encoder_runner;
  if (config.frame_count > 0) {
    encoder_runner =
        std::make_unique<example::EncoderRunner>(manifest.paths.vision_encoder_pte);
    ET_CHECK_OK_OR_RETURN_ERROR(encoder_runner->load());
  }

  auto embedding_module = std::make_unique<Module>(
      manifest.paths.text_embedding_pte,
      Module::LoadMode::MmapUseMlockIgnoreErrors);
  auto decoder_module = std::make_unique<Module>(
      manifest.paths.text_decoder_pte,
      Module::LoadMode::MmapUseMlockIgnoreErrors);

  int64_t img_seq_len = 0;
  int64_t hidden_dim = 0;
  std::vector<float> concat_buffer;
  if (encoder_runner) {
    for (int i = 0; i < config.frame_count; ++i) {
      auto enc = encoder_runner->encode_from_file(frame_path(config.frame_dir, i));
      ET_CHECK_OK_OR_RETURN_ERROR(enc.error());
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
      /*lazy_kv_alloc=*/true,
      /*ngram=*/0,
      /*window=*/0,
      /*gcap=*/0,
      /*image_hidden_states=*/nullptr);

  ET_CHECK_OK_OR_RETURN_ERROR(runner.load());
  if (config.frame_count > 0 && !concat_buffer.empty()) {
    runner.set_image_hidden_states_batch(
        std::move(concat_buffer),
        config.frame_count * img_seq_len,
        hidden_dim);
  }

  executorch::extension::llm::GenerationConfig gen_config{
      /*echo=*/false,
      /*ignore_eos=*/false,
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
    auto cb = [&](const std::string& piece) {
      for (char c : piece) {
        out_buf.push_back(c);
      }
    };
    ET_CHECK_OK_OR_RETURN_ERROR(
        runner.generate_from_prompt_or_file(full_prompt, false, gen_config, cb));

    std::string answer(out_buf.begin(), out_buf.end());
    std::string generated;
    if (answer.size() >= full_prompt.size() &&
        answer.compare(0, full_prompt.size(), full_prompt) == 0) {
      generated = answer.substr(full_prompt.size());
    } else {
      generated = answer;
    }
    fout << generated << "\n";
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
