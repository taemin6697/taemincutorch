"""
stream_vlm.py  –  Vision-Language Model inference (QNN HTP)
===========================================================

입력: 비디오 / 이미지 / 텍스트 전용 (둘 중 하나만, 둘 다 없으면 텍스트 전용)

모드:
  - 배치(HF) 모드 (기본): 전체(또는 --decode_after_frames N개) 프레임 prefill 후 질문 처리
  - 스트리밍 모드 (--stream): 프레임별 prefill, 타임스탬프 시점에 질문 처리

실행 예시:
    # 비디오
    python stream_vlm.py -b build-android -s DEVICE -m SM8750 \\
        --pre_gen_pte ./save_model/llama_qnn_intern_1k_kv \\
        --decoder_model internvl3_1b --video sample.mp4 \\
        --questions "What is happening?" --timestamps 3.0

    # 이미지 (1프레임)
    python stream_vlm.py -b build-android -s DEVICE -m SM8750 \\
        --pre_gen_pte ./save_model/llama_qnn_intern_1k_kv \\
        --decoder_model internvl3_1b --image "https://example.com/img.jpg" \\
        --questions "Describe this image" --timestamps 0.0

    # 텍스트 전용
    python stream_vlm.py -b build-android -s DEVICE -m SM8750 \\
        --pre_gen_pte ./save_model/llama_qnn_intern_1k_kv \\
        --decoder_model internvl3_1b \\
        --questions "Hello" --timestamps 0.0

파이프라인
----------
1. 비디오: OpenCV로 프레임 샘플링 | 이미지: 1프레임 | 텍스트: 프레임 없음
2. HuggingFace 프로세서로 전처리 → frame_NNNN.bin 저장 (비전 입력 시)
3. adb push  : 프레임 바이너리 + .pte + C++ 바이너리 → 디바이스
4. adb shell : qnn_streaming_vlm_runner 실행
5. adb pull  : 결과 텍스트 파일 → 호스트
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# ── HF 프로세서 설정 ───────────────────────────────────────────────────────────

_MODEL_CONFIGS = {
    #  (repo_id, img_h, img_w, image_token_for_processor)
    "smolvlm_500m_instruct": (
        "HuggingFaceM4/SmolVLM-500M-Instruct",
        512, 512,
        "<image>",
    ),
    "internvl3_1b": (
        "OpenGVLab/InternVL3-1B-hf",
        448, 448,
        "<IMG_CONTEXT>",
    ),
    "internvl3_2b": (
        "OpenGVLab/InternVL3-2B-hf",
        448, 448,
        "<IMG_CONTEXT>",
    ),
    "internvl3_8b": (
        "OpenGVLab/InternVL3-8B-hf",
        448, 448,
        "<IMG_CONTEXT>",
    ),
}

# C++ MultimodalRunner 가 받아들이는 짧은 이름 (multimodal_runner.cpp 기준)
_CPP_MODEL_VERSION = {
    "smolvlm_500m_instruct": "smolvlm",
    "internvl3_1b": "internvl3",
    "internvl3_2b": "internvl3",
    "internvl3_8b": "internvl3",
}


# ── 프레임 추출 ───────────────────────────────────────────────────────────────

def _save_frame_simple(bgr, bin_path: str):
    """HF 프로세서 없을 때 ImageNet 정규화 fallback."""
    import numpy as np
    import cv2
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    arr = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[None]       # (1, 3, H, W)
    arr.tofile(bin_path)


def extract_image(image_path: str, out_dir: str, model_name: str) -> int:
    """
    단일 이미지(URL 또는 로컬 경로)를 전처리해 out_dir/frame_0000.bin 으로 저장.
    반환값: 1 (프레임 1개)
    """
    try:
        import cv2
    except ImportError:
        sys.exit("[stream_vlm] opencv-python 이 필요합니다:  pip install opencv-python")

    hf_processor = None
    img_h, img_w = 512, 512
    image_token = "<image>"
    cfg = _MODEL_CONFIGS.get(model_name)
    if cfg is not None:
        hf_model_id, img_h, img_w, image_token = cfg
        try:
            from transformers import AutoProcessor
            hf_processor = AutoProcessor.from_pretrained(hf_model_id)
            print(f"[stream_vlm] HF processor 로드: {hf_model_id}")
        except Exception as e:
            print(f"[stream_vlm] HF processor 로드 실패 ({e}), fallback 정규화 사용")

    bin_path = os.path.join(out_dir, "frame_0000.bin")

    # URL 또는 로컬 경로에서 이미지 로드
    import numpy as np
    if image_path.startswith(("http://", "https://")):
        from transformers.image_utils import load_image
        pil_img = load_image(image_path)
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        bgr = cv2.imread(image_path)
        if bgr is None:
            sys.exit(f"[stream_vlm] 이미지를 열 수 없습니다: {image_path}")

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
        inputs["pixel_values"].numpy().tofile(bin_path)
    else:
        _save_frame_simple(bgr, bin_path)

    print(f"[stream_vlm] 이미지 1프레임 → {bin_path}")
    return 1


def extract_frames(
    video_path: str,
    fps: float,
    out_dir: str,
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
        sys.exit("[stream_vlm] opencv-python 이 필요합니다:  pip install opencv-python")

    # HuggingFace 프로세서 로드 시도
    hf_processor = None
    img_h, img_w = 512, 512
    image_token = "<image>"
    cfg = _MODEL_CONFIGS.get(model_name)
    if cfg is not None:
        hf_model_id, img_h, img_w, image_token = cfg
        try:
            from transformers import AutoProcessor
            # dataset.py와 동일하게 trust_remote_code 없이 로드
            hf_processor = AutoProcessor.from_pretrained(hf_model_id)
            print(f"[stream_vlm] HF processor 로드: {hf_model_id}")
        except Exception as e:
            print(f"[stream_vlm] HF processor 로드 실패 ({e}), fallback 정규화 사용")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[stream_vlm] 비디오를 열 수 없습니다: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, round(native_fps / fps))
    print(
        f"[stream_vlm] 비디오 FPS={native_fps:.1f}, "
        f"샘플링 간격={frame_interval}프레임 → ~{fps:.2f} FPS"
    )

    frame_count = 0
    native_idx  = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if native_idx % frame_interval == 0:
            bin_path = os.path.join(out_dir, f"frame_{frame_count:04d}.bin")

            if hf_processor is not None:
                from PIL import Image
                rgb_pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                # dataset.py와 동일한 방식: 모델별 image_token, size 지정, crop_to_patches=False
                try:
                    inputs = hf_processor(
                        text=image_token,
                        images=[rgb_pil],
                        return_tensors="pt",
                        crop_to_patches=False,
                        size={"height": img_h, "width": img_w},
                    )
                except TypeError:
                    # crop_to_patches / size 미지원 processor fallback
                    inputs = hf_processor(
                        text=image_token,
                        images=[rgb_pil],
                        return_tensors="pt",
                    )
                inputs["pixel_values"].numpy().tofile(bin_path)
            else:
                _save_frame_simple(bgr, bin_path)

            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break

        native_idx += 1

    cap.release()
    print(f"[stream_vlm] {frame_count}개 프레임 → {out_dir}")
    return frame_count


# ── ADB 헬퍼 ──────────────────────────────────────────────────────────────────

class ADBRunner:
    """
    llama.py / utils.py 의 SimpleADB 와 동일한 adb push/shell/pull 패턴.
    """

    DEVICE_WORKSPACE = "/data/local/tmp/stream_vlm"

    def __init__(self, build_path: str, device_id: str, qnn_sdk: str, soc_model: str):
        self.build_path  = build_path
        self.device_id   = device_id
        self.qnn_sdk     = qnn_sdk
        self.soc_model   = soc_model
        self.target      = "aarch64-android"
        self._htp_arch   = self._get_htp_arch()

    def _get_htp_arch(self) -> str:
        """SoC → HTP arch 버전."""
        mapping = {
            "SM8750": "79", "SM8650": "75", "SM8550": "73",
            "SM8450": "69", "SM8350": "68",
        }
        return mapping.get(self.soc_model, "73")

    def _adb(self, cmd: list, capture: bool = False) -> str:
        full = ["adb", "-s", self.device_id] + cmd
        if capture:
            r = subprocess.run(full, capture_output=True, text=True)
            return r.stdout + r.stderr
        subprocess.run(full, check=False)
        return ""

    def setup_workspace(self):
        self._adb(["shell", f"rm -rf {self.DEVICE_WORKSPACE}"])
        self._adb(["shell", f"mkdir -p {self.DEVICE_WORKSPACE}"])

    def push_qnn_libs(self):
        """QNN .so 라이브러리 푸시 (llama.py / SimpleADB 와 동일 목록)."""
        arch = self._htp_arch
        libs = [
            f"{self.qnn_sdk}/lib/{self.target}/libQnnHtp.so",
            f"{self.qnn_sdk}/lib/hexagon-v{arch}/unsigned/libQnnHtpV{arch}Skel.so",
            f"{self.qnn_sdk}/lib/{self.target}/libQnnHtpV{arch}Stub.so",
            f"{self.qnn_sdk}/lib/{self.target}/libQnnHtpPrepare.so",
            f"{self.qnn_sdk}/lib/{self.target}/libQnnSystem.so",
            f"{self.build_path}/backends/qualcomm/libqnn_executorch_backend.so",
            f"{self.qnn_sdk}/lib/{self.target}/libQnnModelDlc.so",
        ]
        for lib in libs:
            if os.path.exists(lib):
                self._adb(["push", lib, self.DEVICE_WORKSPACE])
            else:
                print(f"[stream_vlm] 경고: 라이브러리 없음 {lib}")

    def push_file(self, local_path: str):
        self._adb(["push", local_path, self.DEVICE_WORKSPACE])

    def push_dir_files(self, local_dir: str, glob: str = "*.bin"):
        """로컬 디렉토리의 파일들을 디바이스로 푸시."""
        import fnmatch
        for fname in sorted(os.listdir(local_dir)):
            if fnmatch.fnmatch(fname, glob):
                self._adb(["push", os.path.join(local_dir, fname), self.DEVICE_WORKSPACE])

    def execute(self, cmd: str) -> str:
        """adb shell 에서 커맨드 실행 후 stdout 반환."""
        full_cmd = f"cd {self.DEVICE_WORKSPACE} && {cmd}"
        return self._adb(["shell", full_cmd], capture=True)

    def pull(self, device_path: str, local_path: str):
        self._adb(["pull", device_path, local_path])


# ── 메인 스트리밍 파이프라인 ──────────────────────────────────────────────────


def _make_param_folder_name(args) -> str:
    """파라미터 기반 폴더명 생성 (save_log 하위용)."""
    input_type = "video" if args.video else ("image" if args.image else "text")
    run_type = "stream" if args.stream else "batch"
    lazy = "lazy" if args.lazy_kv_alloc else "nolazy"
    fps_tag = str(args.fps).replace(".", "p")
    frame_tag = (
        f"frames{args.decode_after_frames}"
        if args.decode_after_frames is not None
        else "framesall"
    )
    parts = [
        args.decoder_model,
        args.model_mode,
        run_type,
        f"seq{args.seq_len}",
        input_type,
        f"fps{fps_tag}",
        frame_tag,
        f"eval{args.eval_mode}",
        lazy,
    ]
    # 파일명에 쓸 수 없는 문자 제거
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in "_".join(parts))
    return safe


def run(args):
    # --save_log: my_save/save_log/파라미터폴더/ 하위에 txt, csv 저장
    if getattr(args, "save_log", False):
        if getattr(args, "save_log_dir", None):
            base_dir = Path(args.save_log_dir).resolve()
        else:
            stream_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
            base_dir = stream_root / "my_save" / "save_log"
        param_folder = _make_param_folder_name(args)
        out_dir = base_dir / param_folder
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / "stream_output.txt")
        print(f"[stream_vlm] save_log: {args.output}")

    pte_dir   = Path(args.pre_gen_pte).resolve()
    build_dir = Path(args.build_path).resolve()

    # llama.py 가 생성하는 실제 파일명 (pte_filenames 딕셔너리 기준)
    # TEXT_DECODER  : kv_llama_qnn.pte  (model_mode=kv 기준)
    # VISION_ENCODER: vision_encoder_qnn.pte
    # TEXT_EMBEDDING: text_embedding_qnn.pte
    # tokenizer     : tokenizer.json (HuggingFace) 또는 tokenizer.bin
    mode_prefix = {
        "kv": "kv",
        "hybrid": "hybrid",
        "lookahead": "lookahead",
    }.get(args.model_mode, "kv")

    encoder_pte   = pte_dir / "vision_encoder_qnn.pte"
    decoder_pte   = pte_dir / f"{mode_prefix}_llama_qnn.pte"
    embedding_pte = pte_dir / "text_embedding_qnn.pte"

    # 토크나이저: HuggingFace(.json) 우선, 없으면 llama2c(.bin)
    tokenizer_bin = pte_dir / "tokenizer.json"
    if not tokenizer_bin.exists():
        tokenizer_bin = pte_dir / "tokenizer.bin"

    for f in [encoder_pte, decoder_pte, embedding_pte, tokenizer_bin]:
        if not f.exists():
            sys.exit(f"[stream_vlm] 파일 없음: {f}")

    # C++ 스트리밍 바이너리 (build.sh 와 동일하게 build-android/examples/qualcomm/... 에 생성됨)
    runner_bin = build_dir / "examples/qualcomm/oss_scripts/llama/qnn_streaming_vlm_runner"
    if not runner_bin.exists():
        sys.exit(f"[stream_vlm] 바이너리 없음: {runner_bin}\n빌드 후 재시도하세요.")

    qnn_sdk = os.environ.get("QNN_SDK_ROOT", "")
    if not qnn_sdk:
        sys.exit("[stream_vlm] 환경변수 QNN_SDK_ROOT 가 설정되지 않았습니다.")

    # ── 1. 프레임 추출 (PC) ──────────────────────────────────────────────────
    frame_dir = tempfile.mkdtemp(prefix="stream_vlm_frames_")
    try:
        if args.video:
            extracted_count = extract_frames(
                args.video, args.fps, frame_dir, args.decoder_model
            )
            if extracted_count == 0:
                sys.exit(f"[stream_vlm] 프레임을 추출할 수 없습니다: {args.video}")
            # 배치 모드: decode_after_frames로 사용할 프레임 수 제한
            frame_count = extracted_count
            if not args.stream and args.decode_after_frames is not None:
                frame_count = min(extracted_count, args.decode_after_frames)
            video_duration = frame_count / args.fps
            if args.stream:
                timestamps = [min(t, video_duration) for t in args.timestamps]
            else:
                timestamps = [(frame_count - 1) / args.fps] * len(args.questions)
            num_frames_to_push = extracted_count
        elif args.image:
            frame_count = extract_image(args.image, frame_dir, args.decoder_model)
            num_frames_to_push = frame_count
            if args.stream:
                timestamps = [min(t, 1.0) for t in args.timestamps]
            else:
                timestamps = [0.0] * len(args.questions)
        else:
            # 텍스트 전용: 프레임 없음
            frame_count = 0
            num_frames_to_push = 0
            timestamps = [0.0] * len(args.questions)

        # ── 2. adb push ──────────────────────────────────────────────────────
        adb = ADBRunner(
            build_path=str(build_dir),
            device_id=args.device,
            qnn_sdk=qnn_sdk,
            soc_model=args.model,
        )

        print("[stream_vlm] 디바이스 워크스페이스 초기화...")
        adb.setup_workspace()

        print("[stream_vlm] QNN 라이브러리 푸시...")
        adb.push_qnn_libs()

        print("[stream_vlm] 모델 파일 푸시...")
        for f in [encoder_pte, decoder_pte, embedding_pte, tokenizer_bin]:
            adb.push_file(str(f))

        if num_frames_to_push > 0:
            msg = f"프레임 {num_frames_to_push}개 푸시"
            if frame_count != num_frames_to_push:
                msg += f" (러너: {frame_count}개 사용)"
            print(f"[stream_vlm] {msg}...")
            adb.push_dir_files(frame_dir, "*.bin")

        print("[stream_vlm] 스트리밍 러너 바이너리 푸시...")
        adb.push_file(str(runner_bin))

        # ── 3. adb shell 실행 ────────────────────────────────────────────────
        questions_arg  = ";".join(args.questions)
        timestamps_arg = ";".join(str(t) for t in timestamps)

        device_output = f"{ADBRunner.DEVICE_WORKSPACE}/stream_output.txt"

        runner_cmd = " ".join([
            f"chmod +x ./qnn_streaming_vlm_runner &&",
            f"./qnn_streaming_vlm_runner",
            f"--encoder_path={encoder_pte.name}",
            f"--decoder_path={decoder_pte.name}",
            f"--embedding_path={embedding_pte.name}",
            f"--tokenizer_path={tokenizer_bin.name}",
            f"--decoder_model_version={_CPP_MODEL_VERSION[args.decoder_model]}",
            f"--frame_dir=.",
            f"--frame_count={frame_count}",
            f"--fps={args.fps}",
            f'--questions="{questions_arg}"',
            f'--query_timestamps="{timestamps_arg}"',
            f"--seq_len={args.seq_len}",
            f"--temperature={args.temperature}",
            f"--eval_mode={args.eval_mode}",
            f"--{'lazy_kv_alloc' if args.lazy_kv_alloc else 'nolazy_kv_alloc'}",
            f"--{'stream' if args.stream else 'nostream'}",
            f"--output_path=stream_output.txt",
            f"--performance_output_path=inference_speed.txt",
            *(["--nooutput_minimal"] if getattr(args, "output_verbose", False) else ["--output_minimal"]),
        ])

        print("[stream_vlm] 디바이스에서 추론 실행 중...")
        output = adb.execute(runner_cmd)
        if output:
            print(output)

        # ── 4. adb pull ──────────────────────────────────────────────────────
        local_output = args.output
        adb.pull(device_output, local_output)
        print(f"[stream_vlm] 결과 저장: {local_output}")

        # Pull proc CSV (프레임/질문 처리 통계)
        local_proc = local_output + ".proc.csv"
        try:
            adb.pull(device_output + ".proc.csv", local_proc)
            print(f"[stream_vlm] 처리 통계: {local_proc}")
        except Exception:
            local_proc = None

        # Pull mem CSV (50ms RSS 샘플)
        local_mem = local_output + ".mem.csv"
        try:
            adb.pull(device_output + ".mem.csv", local_mem)
            print(f"[stream_vlm] 메모리 통계: {local_mem}")
        except Exception:
            local_mem = None

        # Pull tokens CSV (토큰별 생성 시작/종료)
        local_tokens = local_output + ".tokens.csv"
        try:
            adb.pull(device_output + ".tokens.csv", local_tokens)
            print(f"[stream_vlm] 토큰 타이밍: {local_tokens}")
        except Exception:
            local_tokens = None

        # Pull smaps (메모리 분석용)
        local_smaps = local_output + ".smaps"
        try:
            adb.pull(device_output + ".smaps", local_smaps)
            print(f"[stream_vlm] smaps 덤프: {local_smaps}")
        except Exception:
            local_smaps = None

        # mem + proc 병합 (phase/event, TTFT, tok/s, seq_len별 메모리)
        if local_proc and local_mem and os.path.isfile(local_proc) and os.path.isfile(local_mem):
            try:
                executorch_root = Path(__file__).resolve().parents[4]
                merge_script = executorch_root / "internvl3_test" / "merge_mem_proc.py"
                if merge_script.exists():
                    log_dir = str(Path(local_output).parent)
                    merge_args = [sys.executable, str(merge_script), log_dir]
                    if os.path.isfile(local_output + ".tokens.csv"):
                        merge_args.append("--include-tokens")
                    subprocess.run(
                        merge_args,
                        check=False,
                        cwd=str(executorch_root.parent),
                    )
            except Exception:
                pass

        # 결과 출력
        results = _parse_output(local_output)
        print("\n── 결과 ─────────────────────────────────────────────────────────")
        for ts in sorted(results):
            label = f"t={ts:.1f}s" if ts >= 0 else "end"
            entry = results[ts]
            if isinstance(entry, dict):
                print(f"[{label}]  A: {entry['answer']}")
                if entry.get("kv"):
                    kv = entry["kv"]
                    print(f"         KV({len(kv)}chars): {kv[:120]}{'...' if len(kv)>120 else ''}")
            else:
                print(f"[{label}]  {entry}")
        print("─────────────────────────────────────────────────────────────────\n")

        # 통계 출력
        if local_proc and os.path.isfile(local_proc):
            _print_proc_stats(local_proc, mem_path=local_mem or "")
        if local_mem and os.path.isfile(local_mem):
            _print_mem_stats(local_mem)

    finally:
        if not args.keep_frames:
            shutil.rmtree(frame_dir, ignore_errors=True)


def _load_mem_index(mem_path: str) -> list:
    """mem.csv를 로드해 [(elapsed_s, rss_mb), ...] 반환."""
    import csv
    rows = []
    try:
        with open(mem_path, encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    rows.append((float(r["elapsed_s"]), float(r["rss_mb"])))
                except Exception:
                    pass
    except Exception:
        pass
    return rows


def _rss_at(mem_rows: list, elapsed_s: float) -> str:
    """elapsed_s와 가장 가까운 mem.csv 샘플의 RSS(MB) 문자열 반환."""
    if not mem_rows:
        return "   N/A"
    closest = min(mem_rows, key=lambda r: abs(r[0] - elapsed_s))
    return f"{closest[1]:>6.1f}MB"


def _print_proc_stats(proc_path: str, mem_path: str = "") -> None:
    """proc.csv: 프레임 prefill + 질문 처리 통계 출력.

    Columns: row_type, time_s, col_a_ms, col_b_ms, total_ms,
             kv_pos, kv_total, kv_used_pct, rss_kb, delta_rss_kb, elapsed_s
    row_type F → col_a=encode_ms, col_b=vision_kv_prefill_ms
    row_type Q → col_a=text_kv_prefill_ms, col_b=token_gen_ms
    elapsed_s: mem.csv 기준 벽시계 시간 (모델 로딩 시점 = 0s)
    """
    mem_rows = _load_mem_index(mem_path) if mem_path else []

    frame_rows, query_rows = [], []
    try:
        with open(proc_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("#") or line.startswith("row_type"):
                    continue
                parts = line.rstrip("\n").split(",")
                if len(parts) < 10:
                    continue
                if parts[0] == "F":
                    frame_rows.append(parts)
                elif parts[0] == "Q":
                    query_rows.append(parts)
    except Exception as e:
        print(f"[proc_stats] 읽기 오류: {e}")
        return

    # ── 프레임 prefill ───────────────────────────────────────────────────────
    if frame_rows:
        W = 115
        print("\n" + "─" * W)
        print("  프레임 prefill 통계  (elapsed_s = 모델 로딩 시작 기준 벽시계)")
        print("─" * W)
        print(f"  {'영상(s)':>7}  {'elapsed_s':>9}  {'인코딩(ms)':>10}  "
              f"{'vision_kv_prefill(ms)':>20}  {'합계(ms)':>9}  "
              f"{'KV pos/total':>13}  {'KV%':>5}  "
              f"{'proc RSS':>9}  {'mem RSS':>8}  {'ΔRSS(MB)':>9}")
        print("  " + "─" * (W - 2))
        enc_l, prf_l, tot_l = [], [], []
        for p in frame_rows:
            try:
                vt  = float(p[1])
                enc = float(p[2]); prf = float(p[3]); tot = float(p[4])
                kp  = int(p[5]);   kt  = int(p[6]);   kpct = float(p[7])
                rss = int(p[8]);   drss = int(p[9]) if p[9] else 0
                elapsed = float(p[10]) if len(p) > 10 and p[10] else -1.0
                enc_l.append(enc); prf_l.append(prf); tot_l.append(tot)
                kv_str = f"{kp}/{kt}"
                mem_rss = _rss_at(mem_rows, elapsed) if elapsed >= 0 else "   N/A"
                elapsed_str = f"{elapsed:>9.1f}" if elapsed >= 0 else "      N/A"
                print(f"  {vt:>7.1f}  {elapsed_str}  {enc:>10.1f}  "
                      f"{prf:>20.1f}  {tot:>9.1f}  "
                      f"{kv_str:>13}  {kpct:>5.1f}  "
                      f"{rss/1024:>9.1f}  {mem_rss}  {drss/1024:>+9.1f}")
            except Exception:
                pass
        if enc_l:
            n = len(enc_l)
            print("  " + "─" * (W - 2))
            print(f"  {'평균':>7}  {'':>9}  {sum(enc_l)/n:>10.1f}  "
                  f"{sum(prf_l)/n:>20.1f}  {sum(tot_l)/n:>9.1f}")
        print("─" * W)

    # ── 질문 처리 ────────────────────────────────────────────────────────────
    if query_rows:
        W = 115
        print("\n" + "─" * W)
        print("  질문 처리 통계 — text_kv_prefill vs 토큰 생성  (elapsed_s = 모델 로딩 기준)")
        print("─" * W)
        print(f"  {'질문t(s)':>8}  {'elapsed_s':>9}  {'text_kv_prefill(ms)':>20}  "
              f"{'토큰생성(ms)':>13}  {'합계(ms)':>9}  "
              f"{'KV pos/total':>13}  {'KV%':>5}  {'proc RSS':>9}  {'mem RSS':>8}")
        print("  " + "─" * (W - 2))
        for p in query_rows:
            try:
                qt   = float(p[1]); qpre = float(p[2]); qgen = float(p[3]); qtot = float(p[4])
                kp   = int(p[5]);   kt   = int(p[6]);   kpct = float(p[7]); rss  = int(p[8])
                elapsed = float(p[10]) if len(p) > 10 and p[10] else -1.0
                kv_str = f"{kp}/{kt}"
                mem_rss = _rss_at(mem_rows, elapsed) if elapsed >= 0 else "   N/A"
                elapsed_str = f"{elapsed:>9.1f}" if elapsed >= 0 else "      N/A"
                print(f"  {qt:>8.1f}  {elapsed_str}  {qpre:>20.1f}  "
                      f"{qgen:>13.1f}  {qtot:>9.1f}  "
                      f"{kv_str:>13}  {kpct:>5.1f}  {rss/1024:>9.1f}  {mem_rss}")
            except Exception:
                pass
        print("─" * W)


def _print_mem_stats(mem_path: str) -> None:
    """mem.csv: 0.1초 주기 RSS 메모리 샘플 출력.

    Columns: elapsed_s, rss_kb, rss_mb, delta_rss_kb
    행이 많으므로 최소/최대/평균 요약 + 스파크라인 출력.
    """
    import csv

    rows = []
    try:
        with open(mem_path, encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    rows.append((
                        float(r["elapsed_s"]),
                        float(r["rss_mb"]),
                        int(r["delta_rss_kb"]),
                    ))
                except Exception:
                    pass
    except Exception as e:
        print(f"[mem_stats] 읽기 오류: {e}")
        return

    if not rows:
        return

    rss_vals = [r[1] for r in rows]
    mn, mx, avg = min(rss_vals), max(rss_vals), sum(rss_vals) / len(rss_vals)

    W = 80
    print("\n" + "─" * W)
    print("  메모리 샘플 요약 (0.1초 간격, mem.csv)")
    print("─" * W)
    print(f"  샘플 수: {len(rows)}개  |  "
          f"최소: {mn:.1f}MB  |  최대: {mx:.1f}MB  |  평균: {avg:.1f}MB  |  "
          f"변동폭: {mx - mn:.1f}MB")
    print()

    # 스파크라인: 1초 단위로 평균 내어 그래프 표시
    from collections import defaultdict
    buckets = defaultdict(list)
    for elapsed, rss_mb, _ in rows:
        buckets[int(elapsed)].append(rss_mb)
    sorted_buckets = sorted(buckets.items())

    bar_chars = " ▁▂▃▄▅▆▇█"
    if mx > mn:
        spark = "".join(
            bar_chars[int((sum(v) / len(v) - mn) / (mx - mn) * 8)]
            for _, v in sorted_buckets
        )
    else:
        spark = "─" * len(sorted_buckets)
    print(f"  RSS 추이 ({sorted_buckets[0][0]}s ~ {sorted_buckets[-1][0]}s, 1칸=1초):")
    print(f"  [{mn:.0f}MB] {spark} [{mx:.0f}MB]")
    print()

    # 전체 행 출력 (1초 단위 대표값만)
    print(f"  {'경과(s)':>8}  {'RSS(MB)':>8}  {'ΔRSS(KB)':>10}  {'바 그래프 (1칸≈1MB, 기준=최소값)':}")
    print("  " + "─" * (W - 2))
    for sec, vals in sorted_buckets:
        avg_mb = sum(vals) / len(vals)
        # delta: 이 버킷의 마지막 - 처음 sample delta 합
        delta_kb = sum(
            r[2] for r in rows if int(r[0]) == sec
        )
        bar_len = max(0, int(avg_mb - mn))
        bar = "█" * min(bar_len, 40)
        sign = "+" if delta_kb >= 0 else ""
        print(f"  {sec:>8}  {avg_mb:>8.1f}  {sign}{delta_kb:>+9}  {bar}")
    print("─" * W)


def _parse_output(output_path: str) -> dict:
    """Parse output file with format:
    [t=Xs] Q: ...
    KV: ...               (may span many lines)
    A: ...                (answer line)
    """
    results = {}
    if not os.path.isfile(output_path):
        return results
    with open(output_path, encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Split on blank-line separated blocks
    blocks = content.strip().split("\n\n")
    for block in blocks:
        lines = block.splitlines()
        if not lines:
            continue
        header = lines[0].strip()
        if not (header.startswith("[t=") or header.startswith("[end]")):
            continue
        try:
            t = float(header[3:header.index("s]")])
        except Exception:
            t = -1.0

        # KV spans multiple lines until "A: "; A: may span to end of block
        kv_parts = []
        answer_parts = []
        in_kv = False
        in_a = False
        for l in lines[1:]:
            if l.startswith("KV: "):
                in_kv = True
                in_a = False
                kv_parts.append(l[4:])
            elif in_kv and l.startswith("A: "):
                in_kv = False
                in_a = True
                answer_parts.append(l[3:])
            elif in_a:
                answer_parts.append(l)
            elif in_kv:
                kv_parts.append(l)

        results[t] = {"kv": "\n".join(kv_parts), "answer": "\n".join(answer_parts).strip()}
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Streaming VLM on-device inference (llama.py 스타일)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # llama.py 와 동일한 플래그 이름 사용
    parser.add_argument("-b", "--build_path", required=True,
                        help="빌드 디렉토리 (예: build-android)")
    parser.add_argument("-s", "--device", required=True,
                        help="adb 디바이스 시리얼 번호")
    parser.add_argument("-m", "--model", required=True,
                        help="SoC 모델 (예: SM8750)")
    parser.add_argument("--pre_gen_pte", required=True,
                        help=".pte 파일이 들어있는 폴더 (llama.py 로 생성된 것)")
    parser.add_argument("--decoder_model", default="smolvlm_500m_instruct",
                        choices=list(_MODEL_CONFIGS.keys()))
    parser.add_argument("--model_mode", default="kv",
                        choices=["kv", "hybrid", "lookahead"],
                        help="llama.py 컴파일 시 사용한 모드 (기본: kv)")

    # 스트리밍 전용 (video / image 둘 중 하나만, 둘 다 없으면 텍스트 전용)
    parser.add_argument("--video", default=None,
                        help="입력 비디오 파일 경로 (.mp4 등). 비디오 있으면 비디오 사용")
    parser.add_argument("--image", default=None,
                        help="입력 이미지 경로 또는 URL. 비디오 없으면 이미지 사용 (1프레임)")
    parser.add_argument("--questions", nargs="+", required=True,
                        help="질문 리스트 (각각 따옴표로 감싸기)")
    parser.add_argument("--timestamps", nargs="+", type=float, default=None,
                        help="각 질문에 대응하는 비디오 시작 후 초(sec). --decode_after_frames 사용 시 무시")
    parser.add_argument("--stream", action="store_true",
                        help="스트리밍 모드: 프레임별 prefill 후 타임스탬프 시점에 질문 처리. 기본은 배치(HF) 모드")
    parser.add_argument("--decode_after_frames", type=int, default=None,
                        help="스트리밍: N프레임 직후 디코딩 (--timestamps 대체). 배치: 첫 N프레임만 사용 (미지정 시 전체)")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="프레임 샘플링 레이트 (기본 1.0)")

    # 생성 파라미터
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--eval_mode", type=int, default=0,
                        help="0=KV(권장), 1=Hybrid, 2=Lookahead")

    # KV 캐시 메모리 할당 전략
    parser.add_argument(
        "--lazy_kv_alloc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="KV 캐시 lazy 물리 할당(mmap, 기본값) vs 즉시 전체 할당(std::vector). "
             "--lazy_kv_alloc (기본) or --no-lazy_kv_alloc"
    )

    # 기타
    parser.add_argument("--output", default="stream_output.txt",
                        help="출력 파일 경로. --save_log 사용 시 무시됨")
    parser.add_argument("--save_log", action="store_true",
                        help="my_save/save_log/파라미터폴더/ 하위에 txt, csv 저장")
    parser.add_argument("--save_log_dir", default=None,
                        help="save_log 저장 기준 경로 (기본: workspace/my_save/save_log)")

    parser.add_argument("--output_verbose", action="store_true",
                        help="상세 로그 ([t=] Q: KV: A: 접두사 포함). 기본은 실제 입출력만 출력")
    parser.add_argument("--keep_frames", action="store_true",
                        help="임시 프레임 파일 삭제 안 함")

    args = parser.parse_args()

    if args.stream:
        # 스트리밍 모드: 프레임별 prefill, 타임스탬프 시점에 질문 처리
        if args.decode_after_frames is not None:
            if args.decode_after_frames < 1:
                parser.error("--decode_after_frames 는 1 이상이어야 합니다.")
            args.timestamps = [(args.decode_after_frames - 1) / args.fps] * len(args.questions)
            print(f"[stream_vlm] --stream: decode_after_frames={args.decode_after_frames} → timestamps={args.timestamps}")
        elif args.timestamps is None:
            parser.error("--stream 모드에서는 --timestamps 또는 --decode_after_frames를 지정하세요.")
    else:
        # 배치(HF) 모드: 모든 프레임 prefill 후 마지막에 질문 처리 (timestamps는 run()에서 설정)
        if args.decode_after_frames is not None and args.decode_after_frames < 1:
            parser.error("--decode_after_frames 는 1 이상이어야 합니다.")
        if args.decode_after_frames is not None:
            print(f"[stream_vlm] 배치 모드: decode_after_frames={args.decode_after_frames} → 첫 N프레임만 사용")
        else:
            print("[stream_vlm] 배치 모드: 전체 프레임 사용")

    if args.stream and args.timestamps is not None and len(args.questions) != len(args.timestamps):
        parser.error(
            f"--questions({len(args.questions)}개)와 "
            f"--timestamps({len(args.timestamps)}개) 개수가 다릅니다."
        )

    # video / image 상호배타: 둘 다 있으면 에러, 둘 다 없으면 텍스트 전용
    if args.video and args.image:
        parser.error("--video 와 --image 를 동시에 지정할 수 없습니다. 둘 중 하나만 사용하세요.")
    if not args.video and not args.image:
        print("[stream_vlm] 비디오/이미지 없음 → 텍스트 전용 모드")

    run(args)


if __name__ == "__main__":
    main()
