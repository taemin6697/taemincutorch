from __future__ import annotations

import os
import re
import subprocess
import tempfile
import shlex
import time
from pathlib import Path

from executorch.examples.models.foundation.manifest import (
    FoundationManifest,
    load_manifest,
)


def _run(cmd: list[str]) -> int:
    return subprocess.run(cmd, check=True).returncode


def _run_capture(cmd: list[str], *, check: bool = True) -> str:
    proc = subprocess.run(cmd, check=check, capture_output=True, text=True)
    return (proc.stdout or "") + (proc.stderr or "")


def _artifact_root(manifest: FoundationManifest) -> Path:
    return Path(manifest.paths["artifact_root"]).resolve()


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _eval_mode_name(eval_mode: int) -> str:
    return {0: "kv", 1: "hybrid", 2: "lookahead"}.get(eval_mode, f"eval{eval_mode}")


def _save_log_dir(
    manifest: FoundationManifest,
    *,
    image: str | None,
    video: str | None,
    frame_count: int,
    seq_len: int | None,
    eval_mode: int,
    stream: bool,
    lazy_kv_alloc: bool,
    ignore_eos: bool = False,
) -> Path:
    input_type = "video" if video else ("image" if image else "text")
    run_type = "stream" if stream else "batch"
    fps_tag = "1p0"
    lazy_tag = "lazy" if lazy_kv_alloc else "nolazy"
    ignore_eos_tag = "ignore_eos" if ignore_eos else "noignoreeos"
    if manifest.variant.startswith(f"{manifest.model_family}_"):
        model_tag = manifest.variant
    else:
        model_tag = f"{manifest.model_family}_{manifest.variant}"
    seq_tag = f"seq{seq_len or manifest.export.get('max_seq_len') or manifest.export.get('max_context_len') or 1024}"
    frame_tag = f"frames{frame_count}"
    folder_name = "_".join(
        [
            model_tag,
            _eval_mode_name(eval_mode),
            run_type,
            seq_tag,
            input_type,
            f"fps{fps_tag}",
            frame_tag,
            f"eval{eval_mode}",
            lazy_tag,
            ignore_eos_tag,
        ]
    )
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in folder_name)
    backend_dir = "qnn" if manifest.backend == "qnn" else "cpu"
    out_dir = _workspace_root() / "my_save" / "save_log" / backend_dir / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _adb_base(serial: str | None) -> list[str]:
    cmd = ["adb"]
    if serial:
        cmd.extend(["-s", serial])
    return cmd


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _adb_shell_capture(adb: list[str], command: str, *, check: bool = True) -> str:
    return _run_capture(adb + ["shell", command], check=check)


def _capture_meminfo(adb: list[str], output_path: Path) -> None:
    _write_text(output_path, _adb_shell_capture(adb, "cat /proc/meminfo", check=False))


def _capture_process_memory_texts(adb: list[str], pid: int) -> tuple[str, str]:
    dumpsys = _adb_shell_capture(adb, f"dumpsys meminfo {pid}", check=False).strip()
    smaps_rollup = _adb_shell_capture(adb, f"cat /proc/{pid}/smaps_rollup", check=False).strip()
    return dumpsys, smaps_rollup


def _is_process_alive(adb: list[str], pid: int) -> bool:
    out = _adb_shell_capture(
        adb,
        f"if [ -d /proc/{pid} ]; then echo alive; else echo dead; fi",
        check=False,
    )
    return "alive" in out


def _capture_process_memory(adb: list[str], pid: int, out_dir: Path) -> None:
    dumpsys, smaps_rollup = _capture_process_memory_texts(adb, pid)
    if not dumpsys:
        dumpsys = f"[foundation] dumpsys meminfo unavailable for pid {pid}\n"
    if not smaps_rollup:
        smaps_rollup = f"[foundation] smaps_rollup unavailable for pid {pid}\n"
    _write_text(out_dir / "android_dumpsys_meminfo.txt", dumpsys + "\n")
    _write_text(out_dir / "android_smaps_rollup.txt", smaps_rollup + "\n")


def _ensure_live_snapshot_files(out_dir: Path, pid: int) -> None:
    dumpsys_path = out_dir / "android_dumpsys_meminfo.txt"
    smaps_path = out_dir / "android_smaps_rollup.txt"
    if not dumpsys_path.exists():
        _write_text(dumpsys_path, f"[foundation] dumpsys meminfo unavailable for pid {pid}\n")
    if not smaps_path.exists():
        _write_text(smaps_path, f"[foundation] smaps_rollup unavailable for pid {pid}\n")


def _extract_named_kb_value(text: str, name: str) -> int | None:
    match = re.search(rf"^{re.escape(name)}:\s*(\d+)\s*kB$", text, flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def _extract_dumpsys_summary_value(text: str, label: str) -> int | None:
    match = re.search(rf"{re.escape(label)}:\s*(\d+)", text)
    return int(match.group(1)) if match else None


def _init_memory_timeline(path: Path) -> None:
    _write_text(
        path,
        "elapsed_s,phase,pid_alive,dumpsys_total_pss_kb,dumpsys_total_rss_kb,"
        "dumpsys_total_swap_kb,smaps_rss_kb,smaps_pss_kb,smaps_private_dirty_kb,"
        "smaps_shared_clean_kb,mem_available_kb,cached_kb,dma_heap_pool_kb,"
        "gpu_total_kb,kgsl_shmem_usage_kb,self_rss_kb,kv_physical_committed_kb,"
        "kv_total_kb\n",
    )


def _append_memory_timeline_sample(
    timeline_path: Path,
    *,
    elapsed_s: float,
    phase: str,
    pid_alive: bool,
    dumpsys: str,
    smaps_rollup: str,
    meminfo: str,
) -> None:
    row = [
        f"{elapsed_s:.3f}",
        phase,
        "1" if pid_alive else "0",
        str(_extract_dumpsys_summary_value(dumpsys, "TOTAL PSS") or ""),
        str(_extract_dumpsys_summary_value(dumpsys, "TOTAL RSS") or ""),
        str(_extract_dumpsys_summary_value(dumpsys, "TOTAL SWAP (KB)") or ""),
        str(_extract_named_kb_value(smaps_rollup, "Rss") or ""),
        str(_extract_named_kb_value(smaps_rollup, "Pss") or ""),
        str(_extract_named_kb_value(smaps_rollup, "Private_Dirty") or ""),
        str(_extract_named_kb_value(smaps_rollup, "Shared_Clean") or ""),
        str(_extract_named_kb_value(meminfo, "MemAvailable") or ""),
        str(_extract_named_kb_value(meminfo, "Cached") or ""),
        str(_extract_named_kb_value(meminfo, "DmaHeapPool") or ""),
        str(_extract_named_kb_value(meminfo, "GpuTotal") or ""),
        str(_extract_named_kb_value(meminfo, "KgslShmemUsage") or ""),
        "",  # self_rss_kb (ADB는 프로세스 내부 아님)
        "",  # kv_physical_committed_kb
        "",  # kv_total_kb
    ]
    with timeline_path.open("a", encoding="utf-8") as f:
        f.write(",".join(row) + "\n")


def _merge_memory_timeline(
    launcher_timeline_path: Path,
    runner_timeline_path: Path,
    output_path: Path,
) -> None:
    """prelaunch(launcher) + runner + postrun(launcher) 병합. runner 시작 -2초 ~ 종료 +2초."""
    import csv

    launcher_rows: list[dict[str, str]] = []
    with launcher_timeline_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            launcher_rows.append(row)

    prelaunch = [r for r in launcher_rows if float(r.get("elapsed_s", 0)) < 0]
    postrun = [r for r in launcher_rows if (r.get("phase") or "").strip() == "postrun"]

    runner_rows: list[dict[str, str]] = []
    if runner_timeline_path.exists():
        with runner_timeline_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or fieldnames
            for row in reader:
                runner_rows.append(row)

    # runner 없으면 launcher running 사용
    running = (
        runner_rows
        if runner_rows
        else [r for r in launcher_rows if (r.get("phase") or "").strip() == "running"]
    )
    merged = prelaunch + running + postrun
    if not merged:
        return
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)


def _capture_prelaunch_timeline(
    adb: list[str],
    timeline_path: Path,
    *,
    baseline_window_s: float = 2.0,
    poll_interval_s: float = 0.02,
) -> None:
    start = time.monotonic()
    next_sample_at = start
    while True:
        now = time.monotonic()
        if now < next_sample_at:
            time.sleep(next_sample_at - now)
            continue
        elapsed_s = now - start - baseline_window_s
        meminfo = _adb_shell_capture(adb, "cat /proc/meminfo", check=False)
        _append_memory_timeline_sample(
            timeline_path,
            elapsed_s=elapsed_s,
            phase="prelaunch",
            pid_alive=False,
            dumpsys="",
            smaps_rollup="",
            meminfo=meminfo,
        )
        next_sample_at += poll_interval_s
        if now - start >= baseline_window_s:
            break


def _start_remote_runner(
    adb: list[str],
    remote_root: str,
    runner_cmd: str,
) -> int:
    remote_launch = (
        f"cd {shlex.quote(remote_root)} && "
        "rm -f foundation_output.txt foundation_proc.csv android_memory_timeline.csv runner_stdout.txt runner.pid && "
        "if command -v nohup >/dev/null 2>&1; then "
        f"  nohup sh -c {shlex.quote(runner_cmd)} > runner_stdout.txt 2>&1 < /dev/null & "
        "else "
        f"  sh -c {shlex.quote(runner_cmd)} > runner_stdout.txt 2>&1 < /dev/null & "
        "fi; "
        "pid=$!; echo $pid > runner.pid; echo $pid"
    )
    pid_text = _adb_shell_capture(adb, remote_launch, check=False).strip()
    pid_line = pid_text.splitlines()[-1].strip() if pid_text else ""
    if not pid_line.isdigit():
        raise RuntimeError(f"Failed to start runner on device. pid output: {pid_text}")
    return int(pid_line)


def _wait_for_process_exit(
    adb: list[str],
    pid: int,
    out_dir: Path,
    timeline_path: Path,
    *,
    poll_interval_s: float = 0.02,
    detailed_interval_s: float = 0.5,
    postrun_window_s: float = 2.0,
) -> bool:
    live_snapshot_taken = False
    started_at = time.monotonic()
    last_detailed_at = -1e18
    while _is_process_alive(adb, pid):
        sample_started_at = time.monotonic()
        if not live_snapshot_taken:
            _capture_process_memory(adb, pid, out_dir)
            live_snapshot_taken = True
        elapsed_s = sample_started_at - started_at
        if elapsed_s - last_detailed_at >= detailed_interval_s:
            dumpsys, smaps_rollup = _capture_process_memory_texts(adb, pid)
            last_detailed_at = elapsed_s
        else:
            dumpsys, smaps_rollup = "", ""
        meminfo = _adb_shell_capture(adb, "cat /proc/meminfo", check=False)
        _append_memory_timeline_sample(
            timeline_path,
            elapsed_s=elapsed_s,
            phase="running",
            pid_alive=True,
            dumpsys=dumpsys,
            smaps_rollup=smaps_rollup,
            meminfo=meminfo,
        )
        sleep_s = poll_interval_s - (time.monotonic() - sample_started_at)
        if sleep_s > 0:
            time.sleep(sleep_s)
    postrun_start = time.monotonic()
    next_sample_at = postrun_start
    while True:
        now = time.monotonic()
        if now < next_sample_at:
            time.sleep(next_sample_at - now)
            continue
        _append_memory_timeline_sample(
            timeline_path,
            elapsed_s=now - started_at,
            phase="postrun",
            pid_alive=False,
            dumpsys="",
            smaps_rollup="",
            meminfo=_adb_shell_capture(adb, "cat /proc/meminfo", check=False),
        )
        next_sample_at += poll_interval_s
        if now - postrun_start >= postrun_window_s:
            break
    return live_snapshot_taken


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
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    decode_after_frames: int | None = None,
    eval_mode: int = 0,
    lazy_kv_alloc: bool = False,
    ignore_eos: bool = False,
    save_log: bool = False,
    stream: bool = False,
) -> int:
    from executorch.examples.models.foundation.host.frame_extractor import (
        extract_frames,
        extract_image,
    )
    from executorch.examples.models.foundation.host.memory_plot import (
        generate_memory_timeline_plot,
    )

    adb = _adb_base(device)
    remote_root = "/data/local/tmp/foundation_runner"
    remote_frames = f"{remote_root}/frames"
    remote_runner = f"{remote_root}/xnnpack_qnn_runner"
    remote_output = f"{remote_root}/foundation_output.txt"
    remote_timeline = f"{remote_root}/android_memory_timeline.csv"

    with tempfile.TemporaryDirectory(prefix="foundation_xnnpack_") as tmpdir:
        tmpdir = Path(tmpdir)
        frame_dir = tmpdir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

        if image:
            frame_count = extract_image(image, frame_dir, manifest.variant)
        elif video:
            frame_count = extract_frames(
                video,
                1.0,
                frame_dir,
                manifest.variant,
                max_frames=decode_after_frames,
            )
        else:
            frame_count = 0

        out_dir = (
            _save_log_dir(
                manifest,
                image=image,
                video=video,
                frame_count=frame_count,
                seq_len=seq_len,
                eval_mode=eval_mode,
                stream=stream,
                lazy_kv_alloc=lazy_kv_alloc,
                ignore_eos=ignore_eos,
            )
            if save_log
            else Path.cwd()
        )
        local_output = out_dir / "foundation_output.txt"
        local_proc = out_dir / "foundation_proc.csv"
        local_runner_stdout = out_dir / "device_runner_stdout.txt"
        local_meminfo_before = out_dir / "android_proc_meminfo_before.txt"
        local_meminfo_after = out_dir / "android_proc_meminfo_after.txt"
        local_pid = out_dir / "android_runner_pid.txt"
        local_timeline = out_dir / "android_memory_timeline.csv"

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
            f"--frame_count={frame_count}",
            f"--seq_len={seq_len or manifest.export.get('max_seq_len') or manifest.export.get('max_context_len') or 1024}",
            f"--temperature={temperature if temperature is not None else 0.0}",
            f"--eval_mode={eval_mode}",
            f"--output_path=foundation_output.txt",
        ]
        if max_new_tokens is not None and max_new_tokens >= 0:
            args.append(f"--max_new_tokens={max_new_tokens}")
        if ignore_eos:
            args.append("--ignore_eos")
        for q in questions:
            args.extend(["--prompt", shlex.quote(q)])
        remote_proc = f"{remote_root}/foundation_proc.csv"
        runner_cmd = "export LD_LIBRARY_PATH=. && ./xnnpack_qnn_runner " + " ".join(args)
        _capture_meminfo(adb, local_meminfo_before)
        _init_memory_timeline(local_timeline)
        _capture_prelaunch_timeline(adb, local_timeline, baseline_window_s=2.0, poll_interval_s=0.1)
        pid = _start_remote_runner(adb, remote_root, runner_cmd)
        _write_text(local_pid, f"{pid}\n")
        _wait_for_process_exit(adb, pid, out_dir, local_timeline)
        _ensure_live_snapshot_files(out_dir, pid)
        _capture_meminfo(adb, local_meminfo_after)
        runner_stdout = _adb_shell_capture(adb, f"cd {remote_root} && cat runner_stdout.txt", check=False)
        _write_text(local_runner_stdout, runner_stdout)
        if runner_stdout.strip():
            print(runner_stdout)
        _run(adb + ["pull", remote_output, str(local_output)])
        rc = _run(adb + ["pull", remote_proc, str(local_proc)])
        runner_timeline = out_dir / "android_memory_timeline_runner.csv"
        _run_capture(adb + ["pull", remote_timeline, str(runner_timeline)], check=False)
        _merge_memory_timeline(local_timeline, runner_timeline, local_timeline)
        if save_log:
            generate_memory_timeline_plot(out_dir)
        return rc


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
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    decode_after_frames: int | None = None,
    eval_mode: int = 0,
    lazy_kv_alloc: bool = False,
    ignore_eos: bool = False,
    save_log: bool = False,
    stream: bool = False,
) -> int:
    from executorch.examples.models.foundation.host.adb_runner import ADBRunner
    from executorch.examples.models.foundation.host.frame_extractor import (
        extract_frames,
        extract_image,
    )
    from executorch.examples.models.foundation.host.memory_plot import (
        generate_memory_timeline_plot,
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
            frame_count = extract_frames(
                video,
                1.0,
                str(frame_dir),
                manifest.variant,
                max_frames=decode_after_frames,
            )
        else:
            frame_count = 0

        out_dir = (
            _save_log_dir(
                manifest,
                image=image,
                video=video,
                frame_count=frame_count,
                seq_len=seq_len,
                eval_mode=eval_mode,
                stream=stream,
                lazy_kv_alloc=lazy_kv_alloc,
                ignore_eos=ignore_eos,
            )
            if save_log
            else Path.cwd()
        )
        local_output = out_dir / "foundation_output.txt"
        local_proc = out_dir / "foundation_proc.csv"
        local_runner_stdout = out_dir / "device_runner_stdout.txt"
        local_meminfo_before = out_dir / "android_proc_meminfo_before.txt"
        local_meminfo_after = out_dir / "android_proc_meminfo_after.txt"
        local_pid = out_dir / "android_runner_pid.txt"
        local_timeline = out_dir / "android_memory_timeline.csv"

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
        adb_cmd = _adb_base(device)
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
        if frame_count > 1:
            image_path = "."
        elif frame_count > 0:
            image_path = "frame_0000.bin"
        else:
            image_path = "."
        # frame_count=0: text-only mode (--image/--video 없이 --questions만)

        device_output = f"{ADBRunner.DEVICE_WORKSPACE}/foundation_output.txt"
        device_proc = f"{ADBRunner.DEVICE_WORKSPACE}/foundation_proc.csv"
        device_timeline = f"{ADBRunner.DEVICE_WORKSPACE}/android_memory_timeline.csv"
        device_etdump = f"{ADBRunner.DEVICE_WORKSPACE}/foundation_qnn_profiling.etdp"
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
            f"--frame_count={frame_count}",
            f"--seq_len={seq_len or manifest.export.get('max_seq_len') or manifest.export.get('max_context_len') or 1024}",
            f"--temperature={temperature if temperature is not None else 0.0}",
            f"--eval_mode={eval_mode}",
            f"--{'lazy_kv_alloc' if lazy_kv_alloc else 'nolazy_kv_alloc'}",
            f"--output_path={ADBRunner.DEVICE_WORKSPACE}/foundation_output.txt",
            f"--proc_path={ADBRunner.DEVICE_WORKSPACE}/foundation_proc.csv",
        ]
        if max_new_tokens is not None and max_new_tokens >= 0:
            cmd.append(f"--max_new_tokens={max_new_tokens}")
        if save_log:
            cmd.append("--etdump_path=foundation_qnn_profiling.etdp")
        if ignore_eos:
            cmd.append("--ignore_eos")
        for q in questions:
            cmd.extend(["--prompt", shlex.quote(q)])
        runner_cmd = " ".join(cmd)
        _capture_meminfo(adb_cmd, local_meminfo_before)
        _init_memory_timeline(local_timeline)
        _capture_prelaunch_timeline(adb_cmd, local_timeline, baseline_window_s=2.0, poll_interval_s=0.1)
        pid = _start_remote_runner(adb_cmd, ADBRunner.DEVICE_WORKSPACE, runner_cmd)
        _write_text(local_pid, f"{pid}\n")
        _wait_for_process_exit(adb_cmd, pid, out_dir, local_timeline)
        _ensure_live_snapshot_files(out_dir, pid)
        _capture_meminfo(adb_cmd, local_meminfo_after)
        runner_out = _adb_shell_capture(
            adb_cmd,
            f"cd {ADBRunner.DEVICE_WORKSPACE} && cat runner_stdout.txt",
            check=False,
        )
        _write_text(local_runner_stdout, runner_out)
        if runner_out.strip():
            print(runner_out)
        try:
            _run(["adb", "-s", device, "pull", device_output, str(local_output)])
            rc = _run(["adb", "-s", device, "pull", device_proc, str(local_proc)])
            runner_timeline = out_dir / "android_memory_timeline_runner.csv"
            _run_capture(
                ["adb", "-s", device, "pull", device_timeline, str(runner_timeline)],
                check=False,
            )
            _merge_memory_timeline(local_timeline, runner_timeline, local_timeline)
            if save_log:
                local_etdump = out_dir / "foundation_qnn_profiling.etdp"
                _run_capture(
                    ["adb", "-s", device, "pull", device_etdump, str(local_etdump)],
                    check=False,
                )
                generate_memory_timeline_plot(out_dir)
            return rc
        except subprocess.CalledProcessError:
            print(
                "\n[foundation] 러너가 foundation_output.txt 또는 foundation_proc.csv 를 생성하지 못했습니다. "
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
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    decode_after_frames: int | None = None,
    eval_mode: int = 0,
    save_log: bool = False,
    stream: bool = False,
    runner_binary: str | None = None,
    lazy_kv_alloc: bool = False,
    ignore_eos: bool = False,
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
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            decode_after_frames=decode_after_frames,
            eval_mode=eval_mode,
            lazy_kv_alloc=lazy_kv_alloc,
            ignore_eos=ignore_eos,
            save_log=save_log,
            stream=stream,
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
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            decode_after_frames=decode_after_frames,
            eval_mode=eval_mode,
            lazy_kv_alloc=lazy_kv_alloc,
            ignore_eos=ignore_eos,
            save_log=save_log,
            stream=stream,
        )

    raise SystemExit(f"지원하지 않는 backend: {manifest.backend}")
