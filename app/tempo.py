"""从对齐结果的 matched_beats 估计局部节奏曲线（tempo map）。

matched_beats 是按参考拍索引排序的一一对应匹配序列。相邻匹配之间：

* 正常情况下参考索引与演奏索引都各推进 1，演奏时间差即一拍时长，
  ``bpm = 60 / Δperf`` 就是局部瞬时速度；
* 漏拍会使一次匹配间隔跨过多个参考拍，演奏时间差近似翻倍，瞬时 BPM
  掉到一半左右；
* 插入拍（演奏多出的音未参与匹配）一般不改变匹配间隔；只有极端错配
  使相邻匹配的演奏侧只推进半个拍长时，瞬时 BPM 才会翻倍。

单条间隔的瞬时 BPM 容易被漏拍 / 插入拍污染，因此对每个间隔取居中
中位数窗口（窗口长度取不大于 ``window_beats`` 的奇数）得到稳健的
局部 BPM；原始瞬时 BPM 同时满足以下两条才视为有效：

1. 落在 ``[min_bpm, max_bpm]`` 范围内；
2. 与窗口中位数的相对偏差不超过 ``outlier_tolerance``（默认 20%）。

其余间隔记入 ``ignored_intervals`` 并在 ``ignored_indices`` 中给出其
间隔索引（按参考拍顺序、0 基）。有效间隔再按局部 BPM 压缩：相邻有效
间隔的平滑 BPM 与所在事件锚点 BPM 的相对差不超过
``compression_tolerance``（默认 2%）时归并为同一 tempo event，
事件按参考拍排序，给出起拍、BPM 与覆盖拍数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

# 原始瞬时 BPM 相对窗口中位数的最大允许偏差（超出判为漏拍 / 插入拍异常）
OUTLIER_TOLERANCE = 0.20
# 相邻间隔平滑 BPM 相对事件锚点的最大偏差（超出则另起一个 tempo event）
COMPRESSION_TOLERANCE = 0.02

DEFAULT_WINDOW_BEATS = 5
DEFAULT_MIN_BPM = 40.0
DEFAULT_MAX_BPM = 300.0


class TempoError(ValueError):
    """无法从对齐结果构建节奏曲线（如匹配拍不足）。"""


@dataclass(frozen=True)
class TempoParams:
    """节奏曲线估计参数。

    window_beats: 中位数窗口包含的间隔（拍）数上限；实际窗口取不大于
        该值的最大奇数（如 5→5、4→3、1→1），居中截取、边界处收窄。
    min_bpm / max_bpm: 合法瞬时 BPM 区间，超出的间隔直接忽略。
    """

    window_beats: int = DEFAULT_WINDOW_BEATS
    min_bpm: float = DEFAULT_MIN_BPM
    max_bpm: float = DEFAULT_MAX_BPM

    @property
    def window_size(self) -> int:
        """实际使用的奇数窗口长度。"""
        return self.window_beats if self.window_beats % 2 == 1 else self.window_beats - 1

    def validate(self) -> None:
        if self.window_beats < 1:
            raise TempoError("window_beats 必须 >= 1")
        if self.min_bpm <= 0 or self.max_bpm <= 0:
            raise TempoError("min_bpm / max_bpm 必须为正数")
        if self.min_bpm >= self.max_bpm:
            raise TempoError("min_bpm 必须小于 max_bpm")


def _median(values: Sequence[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _round_bpm(value: float) -> float:
    return round(value, 2)


def build_tempo_map(
    result: dict[str, Any],
    params: TempoParams | None = None,
) -> dict[str, Any]:
    """根据对齐结果构建按参考拍排序的 tempo events。

    ``result`` 为 ``align_beats`` 返回的结果字典（至少包含
    ``matched_beats``），也接受持久化后从 SQLite 读回的结构。
    """
    params = params or TempoParams()
    params.validate()

    beats = result.get("matched_beats") or []
    if len(beats) < 2:
        raise TempoError(
            f"匹配拍仅 {len(beats)} 个（至少需要 2 个），无法估计节奏曲线"
        )

    interval_count = len(beats) - 1

    # 1) 相邻匹配的原始瞬时 BPM 与跨度信息
    raw_bpms: list[float | None] = []
    ref_spans: list[int] = []
    perf_spans: list[int] = []
    for k in range(interval_count):
        a, b = beats[k], beats[k + 1]
        dt = b["perf_time"] - a["perf_time"]
        ref_spans.append(b["ref_index"] - a["ref_index"])
        perf_spans.append(b["perf_index"] - a["perf_index"])
        raw_bpms.append(60.0 / dt if dt > 0 else None)

    # 2) 居中中位数窗口（实际窗口取奇数，边界处收窄；非有限 BPM 不参与）
    half = params.window_size // 2
    median_bpms: list[float | None] = []
    for k in range(interval_count):
        lo = max(0, k - half)
        hi = min(interval_count, k + half + 1)
        window = [v for v in raw_bpms[lo:hi] if v is not None and math.isfinite(v)]
        median_bpms.append(_median(window) if window else None)

    # 3) 逐条判定异常间隔（范围 + 相对中位数偏差）
    ignored: list[dict[str, Any]] = []
    kept: list[int] = []
    for k, raw in enumerate(raw_bpms):
        reasons: list[str] = []
        if raw is None or not math.isfinite(raw):
            reasons.append("non_positive_interval")
        else:
            if raw < params.min_bpm:
                reasons.append("below_min_bpm")
            if raw > params.max_bpm:
                reasons.append("above_max_bpm")
            med = median_bpms[k]
            if (
                med is not None
                and med > 0
                and abs(raw - med) / med > OUTLIER_TOLERANCE
            ):
                reasons.append("outlier")
        if reasons:
            a, b = beats[k], beats[k + 1]
            ignored.append({
                "interval_index": k,
                "ref_start_index": a["ref_index"],
                "ref_end_index": b["ref_index"],
                "ref_start_time": a["ref_time"],
                "ref_end_time": b["ref_time"],
                "ref_span": ref_spans[k],
                "perf_span": perf_spans[k],
                "raw_bpm": _round_bpm(raw) if raw is not None else None,
                "median_bpm": _round_bpm(median_bpms[k]) if median_bpms[k] is not None else None,
                "reasons": reasons,
            })
        else:
            kept.append(k)

    # 4) 有效间隔压缩为按参考拍排序的 tempo events。
    #    事件 = 索引连续（中间没有被忽略间隔）且平滑 BPM 稳定的最大间隔串；
    #    被忽略间隔两侧即使 BPM 相同也不合并——该间隔的速度未知。串内以首个
    #    间隔的平滑 BPM 为锚点，相对偏差超过 COMPRESSION_TOLERANCE 时另起事件。
    events: list[dict[str, Any]] = []
    run: list[int] = []
    anchor: float | None = None

    def flush_run(indices: Sequence[int]) -> None:
        if not indices:
            return
        start, end = beats[indices[0]], beats[indices[-1] + 1]
        bpm = _median([median_bpms[i] for i in indices])  # 全部非 None
        events.append({
            "event_index": len(events),
            "ref_start_index": start["ref_index"],
            "ref_end_index": end["ref_index"],
            "ref_start_time": start["ref_time"],
            "ref_end_time": end["ref_time"],
            "perf_start_time": start["perf_time"],
            "bpm": _round_bpm(bpm),
            "beat_count": len(indices) + 1,
            "interval_count": len(indices),
        })

    for k in kept:
        bpm = median_bpms[k]
        gap = bool(run) and k != run[-1] + 1
        if (
            not run
            or gap
            or anchor is None
            or bpm is None
            or abs(bpm - anchor) / anchor > COMPRESSION_TOLERANCE
        ):
            flush_run(run)
            run = []
            anchor = bpm
        run.append(k)
    flush_run(run)

    return {
        "parameters": {
            "window_beats": params.window_beats,
            "window_size": params.window_size,
            "min_bpm": params.min_bpm,
            "max_bpm": params.max_bpm,
            "outlier_tolerance": OUTLIER_TOLERANCE,
            "compression_tolerance": COMPRESSION_TOLERANCE,
        },
        "matched_beat_count": len(beats),
        "interval_count": interval_count,
        "event_count": len(events),
        "ignored_count": len(ignored),
        "events": events,
        "ignored_indices": [item["interval_index"] for item in ignored],
        "ignored_intervals": ignored,
    }
