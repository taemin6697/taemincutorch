# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Frame extraction for images and videos. Foundation-internal, no external deps."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_MODEL_CONFIGS = {
    "smolvlm_500m_instruct": (
        "HuggingFaceM4/SmolVLM-500M-Instruct",
        512,
        512,
        "<image>",
    ),
    "internvl3_1b": ("OpenGVLab/InternVL3-1B-hf", 448, 448, "<IMG_CONTEXT>"),
    "internvl3_2b": ("OpenGVLab/InternVL3-2B-hf", 448, 448, "<IMG_CONTEXT>"),
    "internvl3_8b": ("OpenGVLab/InternVL3-8B-hf", 448, 448, "<IMG_CONTEXT>"),
}


def _save_frame_simple(bgr, bin_path: str) -> None:
    """HF 프로세서 없을 때 ImageNet 정규화 fallback."""
    import numpy as np

    import cv2

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    arr = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[None]  # (1, 3, H, W)
    arr.tofile(bin_path)


def extract_image(image_path: str, out_dir: str | Path, model_name: str) -> int:
    """
    단일 이미지(URL 또는 로컬 경로)를 전처리해 out_dir/frame_0000.bin 으로 저장.
    반환값: 1 (프레임 1개)
    """
    try:
        import cv2
    except ImportError:
        sys.exit("[foundation] opencv-python 이 필요합니다:  pip install opencv-python")

    hf_processor = None
    img_h, img_w = 512, 512
    image_token = "<image>"
    cfg = _MODEL_CONFIGS.get(model_name)
    if cfg is not None:
        hf_model_id, img_h, img_w, image_token = cfg
        try:
            from transformers import AutoProcessor

            hf_processor = AutoProcessor.from_pretrained(hf_model_id)
        except Exception:
            pass

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / "frame_0000.bin"

    import numpy as np

    if image_path.startswith(("http://", "https://")):
        from transformers.image_utils import load_image

        pil_img = load_image(image_path)
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        bgr = cv2.imread(image_path)
        if bgr is None:
            sys.exit(f"[foundation] 이미지를 열 수 없습니다: {image_path}")

    if hf_processor is not None:
        from PIL import Image

        rgb_pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        try:
            inputs = hf_processor(
                text=image_token,
                images=[rgb_pil],
                return_tensors="pt",
                crop_to_patches=False,
                size={"height": img_h, "width": img_w},
            )
        except TypeError:
            inputs = hf_processor(
                text=image_token,
                images=[rgb_pil],
                return_tensors="pt",
            )
        inputs["pixel_values"].numpy().tofile(str(bin_path))
    else:
        _save_frame_simple(bgr, str(bin_path))

    return 1


def extract_frames(
    video_path: str,
    fps: float,
    out_dir: str | Path,
    model_name: str,
    max_frames: int | None = None,
) -> int:
    """
    비디오에서 fps 단위로 프레임을 추출해 out_dir/frame_NNNN.bin 으로 저장.
    max_frames 지정 시 해당 수만큼만 추출.
    반환값: 저장된 프레임 수
    """
    try:
        import cv2
    except ImportError:
        sys.exit("[foundation] opencv-python 이 필요합니다:  pip install opencv-python")

    hf_processor = None
    img_h, img_w = 512, 512
    image_token = "<image>"
    cfg = _MODEL_CONFIGS.get(model_name)
    if cfg is not None:
        hf_model_id, img_h, img_w, image_token = cfg
        try:
            from transformers import AutoProcessor

            hf_processor = AutoProcessor.from_pretrained(hf_model_id)
        except Exception:
            pass

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[foundation] 비디오를 열 수 없습니다: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, round(native_fps / fps))

    frame_count = 0
    native_idx = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if native_idx % frame_interval == 0:
            bin_path = out_dir / f"frame_{frame_count:04d}.bin"

            if hf_processor is not None:
                from PIL import Image

                rgb_pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                try:
                    inputs = hf_processor(
                        text=image_token,
                        images=[rgb_pil],
                        return_tensors="pt",
                        crop_to_patches=False,
                        size={"height": img_h, "width": img_w},
                    )
                except TypeError:
                    inputs = hf_processor(
                        text=image_token,
                        images=[rgb_pil],
                        return_tensors="pt",
                    )
                inputs["pixel_values"].numpy().tofile(str(bin_path))
            else:
                _save_frame_simple(bgr, str(bin_path))

            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break

        native_idx += 1

    cap.release()
    return frame_count
