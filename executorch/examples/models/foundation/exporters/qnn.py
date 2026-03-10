# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Foundation-native QNN export. Uses compile/_write_foundation_manifest from llama as libs."""

from __future__ import annotations

import argparse
from pathlib import Path


def export_qnn(args: argparse.Namespace) -> int:
    """Foundation-native QNN export. Uses compile/_write_foundation_manifest as libs."""
    if not args.build_path or not args.device or not args.model:
        raise SystemExit(
            "QNN export에는 --build_path (-b), --device (-s), --model (-m) 이 필요합니다."
        )

    from executorch.examples.qualcomm.oss_scripts.llama import SUPPORTED_LLM_MODELS
    from executorch.examples.qualcomm.oss_scripts.llama.dataset import DatasetBuilder
    from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
        ATTENTION_SINK_EVICTOR,
        AUDIO_ENCODER,
        TEXT_DECODER,
        TEXT_EMBEDDING,
        TEXT_ENCODER,
        VISION_ENCODER,
    )
    from executorch.examples.qualcomm.oss_scripts.llama.tokenizer import TokenizerWrapper
    from executorch.examples.qualcomm.oss_scripts.llama.wrappers import MultiModalManager

    if args.decoder_model not in SUPPORTED_LLM_MODELS:
        raise SystemExit(
            f"Unsupported decoder_model for QNN: {args.decoder_model}. "
            f"Use one of {list(SUPPORTED_LLM_MODELS.keys())}."
        )

    decoder_model_config = SUPPORTED_LLM_MODELS[args.decoder_model]
    # llama.py: max_context_len defaults to max_seq_len; must be >= prefill_ar_len
    max_context_len = getattr(args, "max_context_len", None) or args.max_seq_len
    if max_context_len < args.max_seq_len:
        raise SystemExit(
            f"max_context_len ({max_context_len}) >= max_seq_len ({args.max_seq_len}) 필요"
        )

    if max_context_len < args.prefill_ar_len:
        raise SystemExit(
            f"max_context_len ({max_context_len}) >= prefill_ar_len ({args.prefill_ar_len}) 필요"
        )

    model_mode = getattr(args, "model_mode", "hybrid")
    if model_mode == "kv":
        pte_filename = "kv_llama_qnn"
    elif model_mode == "hybrid":
        pte_filename = "hybrid_llama_qnn"
    elif model_mode == "lookahead":
        pte_filename = "lookahead_llama_qnn"
    else:
        raise SystemExit(f"Unknown model_mode: {model_mode}")

    # MultiModalManager expects all modalities; unused ones (audio/text encoder) get placeholder
    if "internvl" in args.decoder_model.lower():
        pte_filenames = {
            AUDIO_ENCODER: f"{AUDIO_ENCODER}_qnn",  # unused for InternVL3
            TEXT_ENCODER: f"{TEXT_ENCODER}_qnn",  # unused for InternVL3
            TEXT_DECODER: pte_filename,
            VISION_ENCODER: f"{VISION_ENCODER}_qnn",
            TEXT_EMBEDDING: f"{TEXT_EMBEDDING}_qnn",
        }
    else:
        pte_filenames = {
            AUDIO_ENCODER: f"{AUDIO_ENCODER}_qnn",
            TEXT_ENCODER: f"{TEXT_ENCODER}_qnn",
            TEXT_DECODER: pte_filename,
            VISION_ENCODER: f"{VISION_ENCODER}_qnn",
            TEXT_EMBEDDING: f"{TEXT_EMBEDDING}_qnn",
        }

    qnn_args = argparse.Namespace(
        artifact=str(Path(args.artifact_root).resolve()),
        decoder_model=args.decoder_model,
        model_mode=model_mode,
        prefill_ar_len=args.prefill_ar_len,
        max_seq_len=args.max_seq_len,
        max_context_len=max_context_len,
        dtype_override=args.dtype,
        vision_quant=getattr(args, "vision_quant", "fp16"),
        decoder_quant=getattr(args, "decoder_quant", "fp16"),
        embedding_quant=getattr(args, "embedding_quant", "fp16"),
        embedding_quantize=(
            args.embedding_quant if args.embedding_quant != "fp16" else None
        ),
        model=args.model,
        build_path=args.build_path,
        device=args.device,
        enable_x86_64=getattr(args, "enable_x86_64", False),
        prompt=args.prompts or ["Can you describe this image?"],
        system_prompt=getattr(args, "system_prompt", ""),
        tokenizer_model=getattr(args, "tokenizer_model", None),
        tokenizer_bin=getattr(args, "tokenizer_bin", None),
        image_path=getattr(args, "image_path", None),
        params=getattr(args, "params", None),
        checkpoint=getattr(args, "checkpoint", None),
        model_path=getattr(args, "model_path", None),
        window=getattr(args, "window", 8),
        ngram=getattr(args, "ngram", 5),
        gcap=getattr(args, "gcap", 8),
        verbose=False,
        pre_gen_pte=None,
        compile_only=True,
        eval_methods=["prompt_eval"],
        tasks=None,
        limit=1,
        ip=None,
        port=-1,
    )
    tokenizer_wrapper = TokenizerWrapper(qnn_args, decoder_model_config)
    runtime_tokenizer_path, tokenizer, chat_template = (
        tokenizer_wrapper.get_runtime_tokenizer(
            qnn_args.tokenizer_model, qnn_args.tokenizer_bin
        )
    )

    dataset_builder = DatasetBuilder(qnn_args, decoder_model_config, tokenizer_wrapper)
    calibration_data = dataset_builder.prepare_calibration_dataset(
        qnn_args.prompt, chat_template
    )

    from executorch.examples.qualcomm.oss_scripts.llama.llama import (
        compile as qnn_compile,
        _write_foundation_manifest,
    )

    text_decoder_pte_path = f"{qnn_args.artifact}/{pte_filenames[TEXT_DECODER]}.pte"
    encoder_pte_path = f"{qnn_args.artifact}/{pte_filenames[VISION_ENCODER]}.pte"
    text_embedding_pte_path = f"{qnn_args.artifact}/{pte_filenames[TEXT_EMBEDDING]}.pte"

    qnn_compile(
        qnn_args,
        decoder_model_config,
        pte_filenames,
        tokenizer,
        calibration_data,
    )

    _write_foundation_manifest(
        qnn_args,
        runtime_tokenizer_path,
        text_decoder_pte_path,
        encoder_pte_path,
        text_embedding_pte_path,
    )

    return 0
