/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file xnnpack_qnn_runner.cpp
 *
 * Standalone VLM runner for XNNPACK and QNN backends.
 * Same interface and capability level as qnn_multimodal_runner.cpp.
 * Supports SmolVLM 500M, InternVL3 1B/2B/8B.
 */

#include <executorch/examples/models/foundation/runner/backend.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/log.h>
#include <gflags/gflags.h>

#include <filesystem>
#include <fstream>
#include <vector>

// Backend selection
DEFINE_string(backend, "qnn", "Backend: xnnpack or qnn.");

// Model paths (same as qnn_multimodal_runner)
DEFINE_string(embedding_path, "embedding.pte", "Path to embedding model.");
DEFINE_string(encoder_path, "encoder.pte", "Path to vision encoder model.");
DEFINE_string(decoder_path, "decoder.pte", "Path to decoder model.");
DEFINE_string(tokenizer_path, "tokenizer.bin", "Tokenizer path.");

// Output
DEFINE_string(output_path, "outputs.txt", "Output file path.");
DEFINE_string(performance_output_path, "inference_speed.txt", "Inference speed log.");
DEFINE_string(dump_logits_path, "", "Dump logits path (KV mode only).");

// Model config
DEFINE_string(decoder_model_version, "internvl3", "Decoder model version.");
DEFINE_string(prompt, "Describe this image:", "Text prompt.");
DEFINE_string(tokenized_prompt, "", "Alternative: tokenized prompt file.");
DEFINE_string(image_path, "", "Path to preprocessed image .bin or frame directory.");
DEFINE_int32(frame_count, 1, "Number of frames to use (1 for single image).");
DEFINE_string(system_prompt, "", "System prompt.");

// Generation
DEFINE_double(temperature, 0.0f, "Sampling temperature.");
DEFINE_int32(seq_len, 128, "Max tokens to generate.");
DEFINE_int32(eval_mode, 1, "0=KV, 1=Hybrid, 2=Lookahead.");
DEFINE_bool(shared_buffer, false, "Use shared buffers (QNN).");

// Lookahead
DEFINE_int32(ngram, 0, "Lookahead ngram size.");
DEFINE_int32(window, 0, "Lookahead window.");
DEFINE_int32(gcap, 0, "Lookahead gcap.");

DEFINE_int32(num_iters, 1, "Number of iterations.");

std::vector<std::string> CollectPrompts(int argc, char** argv) {
  std::vector<std::string> prompts;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--prompt" && i + 1 < argc) {
      prompts.push_back(argv[i + 1]);
      i++;
    }
  }
  return prompts;
}

#ifdef FOUNDATION_ENABLE_QNN
#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/encoder.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_runner.h>
#include <executorch/extension/module/module.h>

struct SpecialTokens {
  std::string image_token;
  std::string global_img;
  std::string fake_wrap_start;
  std::string fake_wrap_end;
};

SpecialTokens get_special_tokens(
    example::MultimodalDecoderModelVersion decoder_model_version) {
  SpecialTokens tokens;
  switch (decoder_model_version) {
    case example::MultimodalDecoderModelVersion::kSmolvlm:
      tokens.image_token = "<image>";
      tokens.global_img = "<global-img>";
      tokens.fake_wrap_start = "<fake_token_around_image>";
      tokens.fake_wrap_end = "<fake_token_around_image>";
      break;
    case example::MultimodalDecoderModelVersion::kInternvl3:
      tokens.image_token = "<IMG_CONTEXT>";
      tokens.global_img = "";
      tokens.fake_wrap_start = "<img>";
      tokens.fake_wrap_end = "</img>";
      break;
    default:
      break;
  }
  return tokens;
}

std::string prepare_multimodal_prompt(
    const std::string& prompt,
    int image_seq_len,
    const SpecialTokens& specials) {
  std::string image_prompt = specials.fake_wrap_start + specials.global_img;
  for (int i = 0; i < image_seq_len; ++i) {
    image_prompt += specials.image_token;
  }
  image_prompt += specials.fake_wrap_end;

  std::string expanded = prompt;
  size_t pos = 0;
  while ((pos = expanded.find(specials.image_token, pos)) != std::string::npos) {
    expanded.replace(pos, specials.image_token.size(), image_prompt);
    pos += image_prompt.size();
  }
  return expanded;
}

std::string get_formatted_prompt(
    const std::string& prompt,
    const std::string& system_prompt,
    example::MultimodalDecoderModelVersion decoder_model_version,
    int32_t img_seq_len) {
  SpecialTokens specials = get_special_tokens(decoder_model_version);
  std::string formatted_prompt;

  switch (decoder_model_version) {
    case example::MultimodalDecoderModelVersion::kSmolvlm:
      if (!system_prompt.empty()) {
        formatted_prompt.append(
            "<|start_header_id|>system<|end_header_id|>\n\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|eot_id|>");
      }
      formatted_prompt.append("<|im_start|>User:");
      formatted_prompt.append(specials.image_token);
      formatted_prompt.append(prompt);
      formatted_prompt.append("<end_of_utterance>\nAssistant:");
      break;
    case example::MultimodalDecoderModelVersion::kInternvl3:
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|im_start|>system<|im_end|>\n\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|im_end|>");
      }
      formatted_prompt.append("<|im_start|>user:\n");
      formatted_prompt.append(specials.image_token);
      formatted_prompt.append("\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<|im_end|>assistant\n");
      break;
    default:
      ET_CHECK_MSG(false, "Unsupported VLM version");
  }

  return prepare_multimodal_prompt(formatted_prompt, img_seq_len, specials);
}

template <typename T>
void run_qnn_multimodal(
    std::unique_ptr<example::EncoderRunner> encoder_runner,
    std::unique_ptr<executorch::extension::Module> module,
    std::unique_ptr<executorch::extension::Module> embedding,
    std::vector<std::string>& prompts) {
  bool use_tokenized_prompt =
      gflags::GetCommandLineFlagInfoOrDie("tokenized_prompt").is_default
          ? false
          : true;

  if (encoder_runner->load() != executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to load encoder");
    return;
  }

  auto encode_result =
      encoder_runner->encode_from_file(FLAGS_image_path.c_str());
  if (!encode_result.ok()) {
    ET_LOG(Error, "Failed to encode image");
    return;
  }

  auto image_hidden_states = encode_result.get();

  example::MultimodalRunner<T> runner(
      std::move(module),
      std::move(embedding),
      FLAGS_decoder_model_version.c_str(),
      FLAGS_decoder_path.c_str(),
      FLAGS_tokenizer_path.c_str(),
      FLAGS_dump_logits_path.c_str(),
      FLAGS_performance_output_path.c_str(),
      static_cast<float>(FLAGS_temperature),
      FLAGS_eval_mode,
      FLAGS_shared_buffer,
      /*lazy_kv_alloc=*/true,
      FLAGS_ngram,
      FLAGS_window,
      FLAGS_gcap,
      std::make_unique<executorch::aten::Tensor>(image_hidden_states));

  auto decoder_model_version = runner.get_decoder_model_version();
  std::vector<char> buf;
  buf.reserve(5 * FLAGS_seq_len);
  std::ofstream fout(FLAGS_output_path.c_str());

  auto callback = [&](const std::string& piece) {
    for (const char c : piece) {
      buf.push_back(c);
    }
  };

  executorch::extension::llm::GenerationConfig config{
      true, false, -1, false, FLAGS_seq_len,
      static_cast<float>(FLAGS_temperature), 0, 0};

  int32_t img_seq_len = encoder_runner->get_image_seq_len();

  if (use_tokenized_prompt) {
    runner.generate_from_prompt_or_file(
        FLAGS_tokenizer_path.c_str(), use_tokenized_prompt, config, callback);
  } else {
    for (int i = 0; i < FLAGS_num_iters; i++) {
      for (const auto& prompt : prompts) {
        std::string formatted_prompt = get_formatted_prompt(
            prompt, FLAGS_system_prompt, decoder_model_version.get(), img_seq_len);
        runner.generate_from_prompt_or_file(
            formatted_prompt.c_str(), use_tokenized_prompt, config, callback);
      }
    }
  }
  fout.write(buf.data(), buf.size());
  fout.close();
}
#endif

void run_xnnpack_backend(
    const std::string& encoder_path,
    const std::string& embedding_path,
    const std::string& decoder_path,
    const std::string& tokenizer_path,
    const std::string& image_path,
    const std::vector<std::string>& prompts) {
  using namespace executorch::examples::foundation;

  std::string frame_dir;
  {
    std::filesystem::path p(image_path);
    if (std::filesystem::is_directory(p)) {
      frame_dir = p.string();
    } else {
      ET_CHECK_MSG(
          std::filesystem::exists(p),
          "Image path does not exist: %s",
          image_path.c_str());
      frame_dir = p.parent_path().string();
      std::string expected = frame_dir + "/frame_0000.bin";
      if (p.string() != expected) {
        std::string tmp = std::filesystem::temp_directory_path().string() +
            "/xnnpack_qnn_runner_frame";
        std::filesystem::create_directories(tmp);
        std::filesystem::copy_file(
            p, tmp + "/frame_0000.bin",
            std::filesystem::copy_options::overwrite_existing);
        frame_dir = tmp;
      }
    }
    if (frame_dir.empty()) {
      frame_dir = ".";
    }
  }

  ManifestData manifest;
  manifest.backend = "xnnpack";
  manifest.paths.vision_encoder_pte = encoder_path;
  manifest.paths.text_embedding_pte = embedding_path;
  manifest.paths.text_decoder_pte = decoder_path;
  manifest.paths.tokenizer_path = tokenizer_path;

  UnifiedRunConfig config;
  config.frame_dir = frame_dir;
  config.frame_count = FLAGS_frame_count > 0 ? FLAGS_frame_count : 1;
  config.questions = prompts.empty() ? "Describe this image." : prompts[0];
  for (size_t i = 1; i < prompts.size(); ++i) {
    config.questions += ";" + prompts[i];
  }
  config.seq_len = FLAGS_seq_len;
  config.temperature = static_cast<float>(FLAGS_temperature);
  config.output_path = FLAGS_output_path;

  auto runner = create_xnnpack_backend_runner(manifest);
  ET_CHECK_MSG(runner != nullptr, "Failed to create XNNPACK runner.");
  ET_CHECK_MSG(
      runner->validate() == executorch::runtime::Error::Ok,
      "XNNPACK runner validate failed.");

  auto err = runner->run(config);
  ET_CHECK_MSG(
      err == executorch::runtime::Error::Ok,
      "XNNPACK run failed: %d",
      static_cast<int>(err));
}

int main(int argc, char** argv) {
  std::vector<std::string> prompts = CollectPrompts(argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (!gflags::GetCommandLineFlagInfoOrDie("prompt").is_default &&
      !gflags::GetCommandLineFlagInfoOrDie("tokenized_prompt").is_default) {
    ET_CHECK_MSG(false, "Provide prompt or tokenized_prompt, not both.");
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("dump_logits_path").is_default &&
      FLAGS_eval_mode != 0) {
    ET_CHECK_MSG(false, "dump_logits only supported in KV mode.");
  }

  ET_LOG(Info, "Backend: %s", FLAGS_backend.c_str());
  ET_LOG(Info, "Encoder: %s", FLAGS_encoder_path.c_str());
  ET_LOG(Info, "Embedding: %s", FLAGS_embedding_path.c_str());
  ET_LOG(Info, "Decoder: %s", FLAGS_decoder_path.c_str());

  if (FLAGS_backend == "xnnpack") {
    std::string image_path = FLAGS_image_path;
    if (image_path.empty()) {
      ET_LOG(Error, "--image_path required for XNNPACK (preprocessed .bin)");
      return 1;
    }
    run_xnnpack_backend(
        FLAGS_encoder_path,
        FLAGS_embedding_path,
        FLAGS_decoder_path,
        FLAGS_tokenizer_path,
        image_path,
        prompts);
    return 0;
  }

#ifdef FOUNDATION_ENABLE_QNN
  if (FLAGS_backend == "qnn") {
    if (FLAGS_image_path.empty()) {
      ET_LOG(Error, "--image_path required for QNN (preprocessed .bin)");
      return 1;
    }

    const int frame_count =
        FLAGS_frame_count > 0 ? FLAGS_frame_count : 1;

    if (frame_count > 1) {
      std::string frame_dir = FLAGS_image_path;
      if (frame_dir != "." && !std::filesystem::is_directory(frame_dir)) {
        std::filesystem::path p(FLAGS_image_path);
        frame_dir = p.parent_path().string();
        if (frame_dir.empty()) {
          frame_dir = ".";
        }
      }

      executorch::examples::foundation::ManifestData manifest;
      manifest.backend = "qnn";
      manifest.paths.vision_encoder_pte = FLAGS_encoder_path;
      manifest.paths.text_embedding_pte = FLAGS_embedding_path;
      manifest.paths.text_decoder_pte = FLAGS_decoder_path;
      manifest.paths.tokenizer_path = FLAGS_tokenizer_path;

      executorch::examples::foundation::UnifiedRunConfig config;
      config.frame_dir = frame_dir;
      config.frame_count = frame_count;
      config.questions =
          prompts.empty() ? "Describe this image." : prompts[0];
      for (size_t i = 1; i < prompts.size(); ++i) {
        config.questions += ";" + prompts[i];
      }
      config.seq_len = FLAGS_seq_len;
      config.temperature = static_cast<float>(FLAGS_temperature);
      config.eval_mode = FLAGS_eval_mode;
      config.output_path = FLAGS_output_path;

      auto runner =
          executorch::examples::foundation::create_qnn_backend_runner(manifest);
      ET_CHECK_MSG(
          runner->validate() == executorch::runtime::Error::Ok,
          "QNN backend validate failed.");
      auto err = runner->run(config);
      ET_CHECK_MSG(
          err == executorch::runtime::Error::Ok,
          "QNN run failed: %d",
          static_cast<int>(err));
      return 0;
    }

    auto encoder_runner =
        std::make_unique<example::EncoderRunner>(FLAGS_encoder_path.c_str());
    auto embedding = std::make_unique<executorch::extension::Module>(
        FLAGS_embedding_path.c_str(),
        executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
    auto module = std::make_unique<executorch::extension::Module>(
        FLAGS_decoder_path.c_str(),
        executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);

    example::KvBitWidth kv_bitwidth = example::KvBitWidth::kWidth8;
    if (module->method_names()->count("get_kv_io_bit_width") > 0) {
      kv_bitwidth = static_cast<example::KvBitWidth>(
          module->get("get_kv_io_bit_width").get().toScalar().to<int64_t>());
    }

    if (kv_bitwidth == example::KvBitWidth::kWidth8) {
      run_qnn_multimodal<uint8_t>(
          std::move(encoder_runner), std::move(module), std::move(embedding),
          prompts);
    } else if (kv_bitwidth == example::KvBitWidth::kWidth16) {
      run_qnn_multimodal<uint16_t>(
          std::move(encoder_runner), std::move(module), std::move(embedding),
          prompts);
    } else {
      ET_CHECK_MSG(
          false,
          "Unsupported kv bitwidth: %ld",
          static_cast<int64_t>(kv_bitwidth));
    }
    return 0;
  }
#endif

  ET_LOG(Error, "Unsupported backend: %s", FLAGS_backend.c_str());
  return 1;
}
