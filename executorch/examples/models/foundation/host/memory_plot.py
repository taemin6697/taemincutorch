from __future__ import annotations

import csv
import math
from pathlib import Path


def _float_or_nan(value: str | None) -> float:
    value = (value or "").strip()
    return float(value) if value else math.nan


def generate_memory_timeline_plot(out_dir: Path) -> Path | None:
    timeline_csv = out_dir / "android_memory_timeline.csv"
    proc_csv = out_dir / "foundation_proc.csv"
    output_png = out_dir / "memory_timeline_plot.png"

    if not timeline_csv.exists() or not proc_csv.exists():
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - best effort plotting
        print(f"[foundation] 경고: matplotlib 로 plot 생성 실패: {exc}")
        return None

    timeline_rows: list[dict[str, float | str]] = []
    with timeline_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timeline_rows.append(
                {
                    "elapsed_s": _float_or_nan(row.get("elapsed_s")),
                    "phase": (row.get("phase") or "").strip(),
                    "mem_available_mb": _float_or_nan(row.get("mem_available_kb")) / 1024.0,
                    "dumpsys_total_rss_mb": _float_or_nan(row.get("dumpsys_total_rss_kb")) / 1024.0,
                    "smaps_rss_mb": _float_or_nan(row.get("smaps_rss_kb")) / 1024.0,
                    "kv_physical_committed_mb": _float_or_nan(row.get("kv_physical_committed_kb")) / 1024.0,
                    "dma_heap_pool_mb": _float_or_nan(row.get("dma_heap_pool_kb")) / 1024.0,
                }
            )

    proc_rows: list[tuple[str, float, float, dict[str, str]]] = []
    with proc_csv.open(encoding="utf-8") as f:
        header = None
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            if raw.startswith("row_type,"):
                header = raw.split(",")
                continue
            parts = raw.split(",")
            extra = {}
            if header and len(header) == len(parts):
                for i, h in enumerate(header):
                    extra[h.strip()] = parts[i].strip() if i < len(parts) else ""
            proc_rows.append((parts[0], float(parts[1]), float(parts[2]), extra))

    phase_colors = {
        "L": "#6c5ce7",
        "L_VisionLoad": "#8e44ad",
        "L_EmbeddingLoad": "#9b59b6",
        "L_DecoderLoad": "#a569bd",
        "V_Encode": "#00b894",
        "EmbeddingAndMerging": "#fdcb6e",
        "T_Prefill": "#e17055",
        "Decode": "#d63031",
        "D": "#ff7675",
    }

    fig, ax = plt.subplots(figsize=(19, 8), dpi=160)

    xs = [float(r["elapsed_s"]) for r in timeline_rows]
    mem_available = [float(r["mem_available_mb"]) for r in timeline_rows]
    dumpsys_rss = [float(r["dumpsys_total_rss_mb"]) for r in timeline_rows]
    smaps_rss = [float(r["smaps_rss_mb"]) for r in timeline_rows]
    kv_physical_committed = [float(r["kv_physical_committed_mb"]) for r in timeline_rows]
    dma_heap_pool = [float(r["dma_heap_pool_mb"]) for r in timeline_rows]

    ax.plot(
        xs,
        mem_available,
        color="#0984e3",
        linewidth=2.4,
        marker="o",
        markersize=2.6,
        label="MemAvailable (MB)",
    )
    if any(not math.isnan(v) for v in dumpsys_rss):
        ax.plot(
            xs,
            dumpsys_rss,
            color="#636e72",
            linewidth=1.1,
            alpha=0.75,
            label="dumpsys TOTAL RSS (MB)",
        )
    if any(not math.isnan(v) for v in smaps_rss):
        ax.plot(
            xs,
            smaps_rss,
            color="#b2bec3",
            linewidth=1.1,
            alpha=0.9,
            label="smaps RSS (MB)",
        )
    if any(not math.isnan(v) and v > 0 for v in kv_physical_committed):
        ax.plot(
            xs,
            kv_physical_committed,
            color="#00cec9",
            linewidth=1.4,
            alpha=0.9,
            label="KV physical committed (MB)",
        )
    if any(not math.isnan(v) and v > 0 for v in dma_heap_pool):
        ax.plot(
            xs,
            dma_heap_pool,
            color="#e67e22",
            linewidth=1.2,
            alpha=0.85,
            label="DmaHeapPool (MB)",
        )

    label_y_axes = 0.045

    prelaunch_rows = [r for r in timeline_rows if r["phase"] == "prelaunch"]
    if prelaunch_rows:
        prelaunch_start = float(prelaunch_rows[0]["elapsed_s"])
        prelaunch_end = float(prelaunch_rows[-1]["elapsed_s"])
        ax.axvspan(prelaunch_start, prelaunch_end, color="#dfe6e9", alpha=0.35)
        ax.text(
            (prelaunch_start + prelaunch_end) / 2.0,
            label_y_axes,
            "prelaunch",
            fontsize=9,
            ha="center",
            va="bottom",
            color="#2d3436",
            transform=ax.get_xaxis_transform(),
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#636e72", "alpha": 0.8},
        )

    major_types = ["L", "V_Encode", "EmbeddingAndMerging", "T_Prefill", "Decode"]
    phase_count: dict[str, int] = {}
    ymin, ymax = ax.get_ylim()
    for row_type, start, end, _ in proc_rows:
        if row_type not in major_types:
            continue
        phase_count[row_type] = phase_count.get(row_type, 0) + 1
        if row_type == "V_Encode":
            label = f"{row_type}{phase_count[row_type]}"
        elif row_type == "EmbeddingAndMerging":
            label = "EM"
        else:
            label = row_type
        color = phase_colors.get(row_type, "#2d3436")
        ax.axvline(start, color=color, linestyle="--", linewidth=1.1, alpha=0.95)
        ax.axvline(end, color=color, linestyle="--", linewidth=1.1, alpha=0.95)
        ax.axvspan(start, end, color=color, alpha=0.06)
        ax.text(
            (start + end) / 2.0,
            label_y_axes,
            label,
            fontsize=9,
            ha="center",
            va="bottom",
            color=color,
            transform=ax.get_xaxis_transform(),
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": color, "alpha": 0.75},
        )

    d_rows = [(start, end) for row_type, start, end, _ in proc_rows if row_type == "D"]
    if d_rows:
        for x, label in ((d_rows[0][0], "D start"), (d_rows[-1][1], "D end")):
            ax.axvline(x, color=phase_colors["D"], linestyle=":", linewidth=1.1, alpha=0.95)
            ax.text(
                x,
                label_y_axes,
                label,
                fontsize=8,
                rotation=90,
                ha="left",
                va="bottom",
                color=phase_colors["D"],
                transform=ax.get_xaxis_transform(),
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": phase_colors["D"],
                    "alpha": 0.75,
                },
            )

    ax.axvline(0.0, color="#2d3436", linestyle="-", linewidth=1.2, alpha=0.9)
    ax.text(
        0.0,
        label_y_axes,
        "runner start",
        fontsize=8,
        rotation=90,
        ha="left",
        va="bottom",
        color="#2d3436",
        transform=ax.get_xaxis_transform(),
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "#2d3436", "alpha": 0.75},
    )

    ax.set_title("Android Memory Timeline vs Foundation Events")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Memory (MB)")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        fontsize=8,
        framealpha=0.9,
        handlelength=2.0,
        labelspacing=0.35,
        borderpad=0.4,
    )
    ax.set_xlim(left=min(xs) if xs else -2.0, right=max(xs) if xs else 1.0)

    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)

    # Decode speed vs kv_pos (X축=kv_pos, Y축=tokens/sec)
    decode_speed_png = out_dir / "decode_speed_plot.png"
    d_rows_full = [(r[1], r[2], r[3]) for r in proc_rows if r[0] == "D"]
    if d_rows_full:
        kv_pos_list: list[float] = []
        tokens_per_sec: list[float] = []
        for start, end, extra in d_rows_full:
            dt = end - start
            if dt > 0:
                kv = _float_or_nan(extra.get("kv_pos"))
                if not math.isnan(kv):
                    kv_pos_list.append(kv)
                    tokens_per_sec.append(1.0 / dt)
        if kv_pos_list:
            fig2, ax2 = plt.subplots(figsize=(14, 5), dpi=120)
            ax2.plot(kv_pos_list, tokens_per_sec, color="#d63031", linewidth=1.2, alpha=0.9)
            ax2.set_title("Decode Speed vs KV Position (tokens/sec)")
            ax2.set_xlabel("kv_pos (after token generation)")
            ax2.set_ylabel("Tokens/sec")
            ax2.grid(True, linestyle=":", alpha=0.35)
            fig2.tight_layout()
            fig2.savefig(decode_speed_png, bbox_inches="tight")
            plt.close(fig2)

    return output_png
