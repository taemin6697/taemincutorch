from __future__ import annotations

import os
import subprocess
import tempfile
import shlex
from pathlib import Path

from executorch.examples.models.foundation.manifest import (
    FoundationManifest,
    load_manifest,
)


def _run(cmd: list[str]) -> int:
    return subprocess.run(cmd, check=True).returncode


def _artifact_root(manifest: FoundationManifest) -> Path:
    return Path(manifest.paths["artifact_root"]).resolve()


def _adb_base(serial: str | None) -> list[str]:
    cmd = ["adb"]
    if serial:
        cmd.extend(["-s", serial])
    return cmd


def _run_unified_xnnpack(
    manifest: FoundationManifest,
    manifest_path: Path,
    *,
    runner_binary: str,
    device: str | None,
    image: str | None,
    video: str | None,
    questions: list[str],
    timestamps: list[float] | None,
    seq_len: int | None,
    temperature: float | None,
    eval_mode: int = 0,
    save_log: bool = False,
) -> int:
    from executorch.examples.models.internvl3.run_xnnpack_vlm import (
        extract_frames,
        extract_image,
    )

    adb = _adb_base(device)
    remote_root = "/data/local/tmp/foundation_runner"
    remote_frames = f"{remote_root}/frames"
    remote_runner = f"{remote_root}/xnnpack_qnn_runner"
    remote_output = f"{remote_root}/foundation_output.txt"

    with tempfile.TemporaryDirectory(prefix="foundation_xnnpack_") as tmpdir:
        tmpdir = Path(tmpdir)
        frame_dir = tmpdir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

        if image:
            frame_count = extract_image(image, frame_dir, manifest.variant)
        elif video:
            frame_count = extract_frames(video, 1.0, frame_dir, manifest.variant)
        else:
            raise SystemExit("XNNPACK unified run requires --image or --video")

        device_manifest = manifest.resolve_paths(manifest_path.parent)
        rel_paths = {}
        for key, value in device_manifest.paths.items():
            if key == "artifact_root":
                rel_paths[key] = "."
            elif value:
                rel_paths[key] = Path(value).name
        device_manifest.paths = rel_paths

        _run(adb + ["shell", "mkdir", "-p", remote_root, remote_frames])
        _run(adb + ["push", str(runner_binary), remote_runner])
        # xnnpack_qnn_runner links to libqnn_executorch_backend.so; push it for load
        qnn_lib = Path(runner_binary).resolve().parent.parent / "backends/qualcomm/libqnn_executorch_backend.so"
        if qnn_lib.exists():
            _run(adb + ["push", str(qnn_lib), remote_root])

        for key in (
            "vision_encoder_pte",
            "text_embedding_pte",
            "text_decoder_pte",
            "tokenizer_path",
        ):
            path = manifest.paths.get(key)
            if path:
                _run(adb + ["push", str(path), f"{remote_root}/{Path(path).name}"])

        # adb push dir dest → dest/dir/ 생성. 내용만 푸시하려면 dir/. 사용
        _run(adb + ["push", str(frame_dir) + "/.", remote_frames])
        _run(adb + ["shell", "chmod", "+x", remote_runner])

        encoder_path = device_manifest.paths.get("vision_encoder_pte", "encoder.pte")
        embedding_path = device_manifest.paths.get("text_embedding_pte", "embedding.pte")
        decoder_path = device_manifest.paths.get("text_decoder_pte", "decoder.pte")
        tokenizer_path = device_manifest.paths.get("tokenizer_path", "tokenizer.bin")

        args = [
            "--backend=xnnpack",
            f"--encoder_path={encoder_path}",
            f"--embedding_path={embedding_path}",
            f"--decoder_path={decoder_path}",
            f"--tokenizer_path={tokenizer_path}",
            f"--image_path={remote_frames}",
            f"--seq_len={seq_len or 128}",
            f"--temperature={temperature if temperature is not None else 0.0}",
            f"--eval_mode={eval_mode}",
            f"--output_path=foundation_output.txt",
        ]
        for q in questions:
            args.extend(["--prompt", shlex.quote(q)])
        if save_log:
            args.append("--save_log")
        shell_cmd = f"cd {remote_root} && export LD_LIBRARY_PATH=. && ./xnnpack_qnn_runner " + " ".join(args)
        _run(adb + ["shell", shell_cmd])
        return _run(adb + ["pull", remote_output, str(Path.cwd() / "foundation_output.txt")])


def _run_unified_qnn(
    manifest: FoundationManifest,
    manifest_path: Path,
    *,
    runner_binary: str,
    build_path: str,
    device: str,
    model: str,
    image: str | None,
    video: str | None,
    questions: list[str],
    timestamps: list[float] | None,
    seq_len: int | None,
    temperature: float | None,
    eval_mode: int = 0,
    save_log: bool = False,
) -> int:
    from executorch.examples.qualcomm.oss_scripts.llama.stream_vlm import (
        ADBRunner,
        extract_frames,
        extract_image,
    )

    qnn_sdk = os.environ.get("QNN_SDK_ROOT", "")
    if not qnn_sdk:
        raise SystemExit("QNN 실행에는 환경변수 QNN_SDK_ROOT 가 필요합니다.")

    with tempfile.TemporaryDirectory(prefix="foundation_qnn_") as tmpdir:
        tmpdir = Path(tmpdir)
        frame_dir = tmpdir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

        if image:
            frame_count = extract_image(image, str(frame_dir), manifest.variant)
        elif video:
            frame_count = extract_frames(video, 1.0, str(frame_dir), manifest.variant)
        else:
            frame_count = 0

        device_manifest = manifest.resolve_paths(manifest_path.parent)
        rel_paths = {}
        for key, value in device_manifest.paths.items():
            if key == "artifact_root":
                rel_paths[key] = "."
            elif value:
                rel_paths[key] = Path(value).name
        device_manifest.paths = rel_paths

        adb = ADBRunner(
            build_path=str(build_path),
            device_id=device,
            qnn_sdk=qnn_sdk,
            soc_model=model,
        )
        adb.setup_workspace()
        adb.push_qnn_libs()
        adb.push_file(str(runner_binary))
        for key in (
            "vision_encoder_pte",
            "text_embedding_pte",
            "text_decoder_pte",
            "tokenizer_path",
        ):
            path = manifest.paths.get(key)
            if path:
                adb.push_file(str(path))
        if frame_count > 0:
            adb.push_dir_files(str(frame_dir), "*.bin")

        encoder_path = device_manifest.paths.get("vision_encoder_pte", "encoder.pte")
        embedding_path = device_manifest.paths.get("text_embedding_pte", "embedding.pte")
        decoder_path = device_manifest.paths.get("text_decoder_pte", "decoder.pte")
        tokenizer_path = device_manifest.paths.get("tokenizer_path", "tokenizer.bin")
        image_path = "frame_0000.bin" if frame_count > 0 else ""
        if not image_path:
            raise SystemExit("QNN foundation 실행에는 --image 또는 --video 가 필요합니다.")

        device_output = f"{ADBRunner.DEVICE_WORKSPACE}/foundation_output.txt"
        cmd = [
            "chmod +x ./xnnpack_qnn_runner &&",
            "export LD_LIBRARY_PATH=. &&",
            "./xnnpack_qnn_runner",
            "--backend=qnn",
            f"--encoder_path={encoder_path}",
            f"--embedding_path={embedding_path}",
            f"--decoder_path={decoder_path}",
            f"--tokenizer_path={tokenizer_path}",
            f"--image_path={image_path}",
            f"--seq_len={seq_len or manifest.export.get('max_seq_len', 1024)}",
            f"--temperature={temperature if temperature is not None else 0.0}",
            f"--eval_mode={eval_mode}",
            "--output_path=foundation_output.txt",
        ]
        for q in questions:
            cmd.extend(["--prompt", q])
        if save_log:
            cmd.append("--save_log")
        runner_out = adb.execute(" ".join(cmd))
        if runner_out:
            print(runner_out)
        try:
            return _run(["adb", "-s", device, "pull", device_output, str(Path.cwd() / "foundation_output.txt")])
        except subprocess.CalledProcessError:
            print(
                "\n[foundation] 러너가 foundation_output.txt 를 생성하지 못했습니다. "
                "위 디바이스 출력을 확인하세요."
            )
            raise


def run_with_manifest(
    manifest_path: Path,
    *,
    build_path: str | None = None,
    device: str | None = None,
    model: str | None = None,
    image: str | None = None,
    video: str | None = None,
    questions: list[str] | None = None,
    timestamps: list[float] | None = None,
    seq_len: int | None = None,
    temperature: float | None = None,
    eval_mode: int = 0,
    save_log: bool = False,
    stream: bool = False,
    runner_binary: str | None = None,
) -> int:
    manifest = load_manifest(Path(manifest_path))
    artifact_root = _artifact_root(manifest)
    questions = questions or ["Describe this image."]
    if eval_mode is None:
        eval_mode = 1 if manifest.backend == "qnn" else 0

    if manifest.backend == "qnn":
        if not build_path or not device or not model:
            raise SystemExit("QNN 실행에는 --build_path, --device, --model 이 필요합니다.")
        if not runner_binary or Path(runner_binary).name != "xnnpack_qnn_runner":
            raise SystemExit(
                "QNN foundation 실행에는 --runner_binary xnnpack_qnn_runner 가 필요합니다."
            )
        return _run_unified_qnn(
            manifest,
            Path(manifest_path),
            runner_binary=runner_binary,
            build_path=build_path,
            device=device,
            model=model,
            image=image,
            video=video,
            questions=questions,
            timestamps=timestamps,
            seq_len=seq_len,
            temperature=temperature,
            eval_mode=eval_mode,
            save_log=save_log,
        )

    if manifest.backend == "xnnpack":
        if not runner_binary:
            raise SystemExit("XNNPACK foundation 실행에는 --runner_binary xnnpack_qnn_runner 가 필요합니다.")
        if Path(runner_binary).name != "xnnpack_qnn_runner":
            raise SystemExit("XNNPACK foundation 실행은 xnnpack_qnn_runner 만 지원합니다.")
        return _run_unified_xnnpack(
            manifest,
            Path(manifest_path),
            runner_binary=runner_binary,
            device=device,
            image=image,
            video=video,
            questions=questions,
            timestamps=timestamps,
            seq_len=seq_len,
            temperature=temperature,
            eval_mode=eval_mode,
            save_log=save_log,
        )

    raise SystemExit(f"지원하지 않는 backend: {manifest.backend}")
