"""共用計時 / 系統資訊工具。

給 scripts/train.py、scripts/evaluate.py、scripts/predict.py 用，
方便對比不同硬體（特別是 Orin）的效能。

用法：
    from woundseg.utils.timing import print_system_info, EpochTimer, format_seconds

    sys_info = print_system_info()           # 印硬體資訊並回傳 dict
    timer = EpochTimer()                     # 給 run_training 當 callback
    run_training(cfg, epoch_callback=timer)
    print(timer.summary_text())              # 印總結
    summary_dict = timer.summary()           # 拿 dict 寫 JSON
"""
from __future__ import annotations

import platform
import statistics
import time
from datetime import datetime
from os import cpu_count
from typing import Any

import torch


def print_system_info() -> dict[str, Any]:
    """印硬體 / 軟體版本，回傳 dict 方便寫 JSON。

    Orin 上會印 GPU name 含 'Orin' 字樣、CUDA 版本、PyTorch 版本。
    """
    info: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": cpu_count(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }
    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["gpu_backend"] = "CUDA"
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        info["gpu_total_memory_GB"] = round(props.total_memory / 1024**3, 2)
        info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
    elif torch.backends.mps.is_available():
        info["device"] = "mps"
        info["gpu_backend"] = "Apple Metal Performance Shaders"
        info["gpu_name"] = "Apple Silicon GPU"
    else:
        info["device"] = "CPU"
        info["gpu_backend"] = None
        info["gpu_name"] = None

    print("=" * 60)
    print("System Info")
    print("=" * 60)
    for k, v in info.items():
        print(f"  {k:<28}: {v}")
    print()
    return info


def format_seconds(s: float) -> str:
    """把秒數變人類可讀。例：3725.5 → '1h 02m 05.5s'"""
    if s < 60:
        return f"{s:.2f}s"
    if s < 3600:
        m, sec = divmod(s, 60)
        return f"{int(m)}m {sec:.1f}s"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h)}h {int(m):02d}m {sec:.1f}s"


def sync_cuda() -> None:
    """確保 GPU 上的非同步工作完成，才能準確計時。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class EpochTimer:
    """給 run_training 當 epoch_callback。

    需要的介面：
        on_epoch_start()
        on_epoch_end(epoch: int, metrics: dict | None = None)
    """

    def __init__(self):
        self.epochs: list[dict] = []
        self._t_start: float | None = None
        self._t_overall_start = time.perf_counter()

    # ------------ callback API ----------------
    def on_epoch_start(self) -> None:
        sync_cuda()
        self._t_start = time.perf_counter()

    def on_epoch_end(self, epoch: int, metrics: dict | None = None) -> None:
        if self._t_start is None:
            return
        sync_cuda()
        elapsed = time.perf_counter() - self._t_start
        entry: dict = {"epoch": epoch, "seconds": round(elapsed, 3)}
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    entry[k] = float(v)
        self.epochs.append(entry)
        print(f"  [timing] epoch {epoch}: {format_seconds(elapsed)}")
        self._t_start = None

    # ------------ summary ---------------------
    def total_seconds(self) -> float:
        return time.perf_counter() - self._t_overall_start

    def summary(self) -> dict:
        secs = [e["seconds"] for e in self.epochs]
        out: dict = {
            "total_seconds": round(self.total_seconds(), 3),
            "total_formatted": format_seconds(self.total_seconds()),
            "num_epochs": len(secs),
        }
        if secs:
            out["avg_epoch_seconds"] = round(statistics.mean(secs), 3)
            out["min_epoch_seconds"] = round(min(secs), 3)
            out["max_epoch_seconds"] = round(max(secs), 3)
            out["median_epoch_seconds"] = round(statistics.median(secs), 3)
            if len(secs) > 1:
                out["stdev_epoch_seconds"] = round(statistics.stdev(secs), 3)
        return out

    def summary_text(self) -> str:
        s = self.summary()
        lines = [
            "=" * 60,
            "Training timing summary",
            "=" * 60,
            f"  總時間          : {s['total_formatted']}  ({s['total_seconds']} s)",
            f"  epoch 數        : {s['num_epochs']}",
        ]
        if s.get("avg_epoch_seconds") is not None:
            lines.extend([
                f"  平均 / epoch    : {format_seconds(s['avg_epoch_seconds'])}",
                f"  最快 epoch      : {format_seconds(s['min_epoch_seconds'])}",
                f"  最慢 epoch      : {format_seconds(s['max_epoch_seconds'])}",
            ])
        return "\n".join(lines)


class LatencyMeter:
    """給 inference / evaluate 量單張延遲用。

    用法：
        meter = LatencyMeter()
        for x in batch:
            with meter:
                y = model(x)
        print(meter.summary_text())
    """

    def __init__(self):
        self.times: list[float] = []
        self._t0: float | None = None

    def __enter__(self):
        sync_cuda()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        sync_cuda()
        if self._t0 is not None:
            self.times.append(time.perf_counter() - self._t0)
        self._t0 = None

    def summary(self) -> dict:
        if not self.times:
            return {"n_samples": 0}
        total = sum(self.times)
        return {
            "n_samples": len(self.times),
            "total_seconds": round(total, 3),
            "total_formatted": format_seconds(total),
            "mean_ms": round(statistics.mean(self.times) * 1000, 2),
            "median_ms": round(statistics.median(self.times) * 1000, 2),
            "min_ms": round(min(self.times) * 1000, 2),
            "max_ms": round(max(self.times) * 1000, 2),
            "stdev_ms": (round(statistics.stdev(self.times) * 1000, 2)
                         if len(self.times) > 1 else 0.0),
            "fps": round(len(self.times) / total, 2) if total > 0 else 0.0,
        }

    def summary_text(self) -> str:
        s = self.summary()
        if s["n_samples"] == 0:
            return "  [timing] 沒有記錄到樣本"
        return "\n".join([
            "=" * 60,
            "Latency summary",
            "=" * 60,
            f"  樣本數          : {s['n_samples']}",
            f"  總時間          : {s['total_formatted']}  ({s['total_seconds']} s)",
            f"  平均延遲        : {s['mean_ms']} ms",
            f"  median 延遲     : {s['median_ms']} ms",
            f"  最快 / 最慢     : {s['min_ms']} / {s['max_ms']} ms",
            f"  標準差          : {s['stdev_ms']} ms",
            f"  FPS             : {s['fps']}",
        ])
