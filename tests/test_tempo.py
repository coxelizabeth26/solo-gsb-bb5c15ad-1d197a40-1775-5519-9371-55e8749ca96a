"""节奏曲线（tempo map）估计的单元测试。"""

import unittest

from app.alignment import AlignmentParams, align_beats
from app.tempo import (
    TempoError,
    TempoParams,
    build_tempo_map,
)


def _matched_from_times(ref_times, perf_times, ref_offset=0, perf_offset=0):
    """直接用等长时间序列构造 matched_beats（视为全部一一匹配）。"""
    return [
        {
            "ref_index": ref_offset + i,
            "perf_index": perf_offset + i,
            "ref_time": round(r, 6),
            "perf_time": round(p, 6),
            "deviation": round(p - r, 6),
            "status": "on_time",
        }
        for i, (r, p) in enumerate(zip(ref_times, perf_times))
    ]


def _result_with(matched):
    return {"matched_beats": matched}


class TestBuildTempoMap(unittest.TestCase):
    def test_constant_tempo_single_event(self):
        # 120 BPM：拍间隔 0.5s，共 9 个间隔
        beats = _matched_from_times([0.5 * i for i in range(10)],
                                    [0.5 * i for i in range(10)])
        out = build_tempo_map(_result_with(beats))
        self.assertEqual(out["matched_beat_count"], 10)
        self.assertEqual(out["interval_count"], 9)
        self.assertEqual(out["ignored_indices"], [])
        self.assertEqual(len(out["events"]), 1)
        ev = out["events"][0]
        self.assertEqual(ev["ref_start_index"], 0)
        self.assertEqual(ev["ref_end_index"], 9)
        self.assertEqual(ev["beat_count"], 10)
        self.assertEqual(ev["interval_count"], 9)
        self.assertAlmostEqual(ev["bpm"], 120.0, places=2)
        self.assertEqual(out["parameters"]["window_size"], 5)

    def test_missed_beat_interval_ignored(self):
        # ref 节拍间隔 0.5s；演奏漏掉一拍，DTW 后匹配在索引 1 处跨两拍，
        # 该间隔 dt≈1.0s ⇒ 60 BPM，应被中位窗口识别为异常并忽略
        ref = [0.5 * i for i in range(9)]
        perf = [0.0, 0.5, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        aligned = align_beats(ref, perf, AlignmentParams(band_width=2))
        out = build_tempo_map(aligned)
        ignored = out["ignored_indices"]
        self.assertTrue(ignored, "漏拍造成的倍间隔应被忽略")
        # 被忽略的正是 (ref 1 -> ref 3) 那一条相邻匹配间隔
        flagged = {
            (item["ref_start_index"], item["ref_end_index"])
            for item in out["ignored_intervals"]
        }
        self.assertIn((1, 3), flagged)
        detail = next(
            item for item in out["ignored_intervals"]
            if (item["ref_start_index"], item["ref_end_index"]) == (1, 3)
        )
        self.assertEqual(detail["ref_span"], 2)
        self.assertIn("outlier", detail["reasons"])
        self.assertAlmostEqual(detail["raw_bpm"], 60.0, places=2)
        # 其余有效间隔仍为 120 BPM；被忽略间隔把事件切成两段
        self.assertEqual(out["event_count"], 2)
        for ev in out["events"]:
            self.assertAlmostEqual(ev["bpm"], 120.0, places=2)
            self.assertEqual(ev["beat_count"], ev["interval_count"] + 1)
        self.assertEqual(
            [(ev["ref_start_index"], ev["ref_end_index"]) for ev in out["events"]],
            [(0, 1), (3, 8)],
        )
        # 事件按参考拍排序
        starts = [ev["ref_start_index"] for ev in out["events"]]
        self.assertEqual(starts, sorted(starts))

    def test_inserted_beat_kept_when_matches_undisturbed(self):
        # 插入音在两个匹配拍之间：匹配间隔本身仍为 0.5s，不应误忽略
        ref = [0.0, 0.5, 1.0, 1.5, 2.0]
        perf = [0.0, 0.5, 0.75, 1.0, 1.5, 2.0]
        aligned = align_beats(ref, perf, AlignmentParams(band_width=2))
        out = build_tempo_map(aligned)
        self.assertEqual(out["ignored_indices"], [])
        self.assertEqual(len(out["events"]), 1)
        self.assertAlmostEqual(out["events"][0]["bpm"], 120.0, places=2)

    def test_tempo_change_split_events(self):
        # 前 6 拍 120 BPM（0.5s，0..2.5），之后 60 BPM（1.0s，3.5..6.5）
        times = [0.5 * i for i in range(6)] + [2.5 + i for i in range(1, 5)]
        beats = _matched_from_times(times, times)
        out = build_tempo_map(_result_with(beats))
        self.assertEqual(len(out["events"]), 2)
        self.assertAlmostEqual(out["events"][0]["bpm"], 120.0, places=1)
        self.assertAlmostEqual(out["events"][1]["bpm"], 60.0, places=1)
        # 事件按参考拍排序；中位数窗口把过渡间隔（间隔 5=60BPM）归入
        # 后一事件，两个事件共享边界拍 5（间隔层面仍首尾相接）
        self.assertEqual(out["events"][0]["ref_end_index"], 5)
        self.assertEqual(out["events"][1]["ref_start_index"], 5)
        end_before = out["events"][0]["ref_start_index"] + out["events"][0]["interval_count"]
        self.assertEqual(end_before, out["events"][1]["ref_start_index"])
        total_intervals = sum(ev["interval_count"] for ev in out["events"])
        self.assertEqual(total_intervals, out["interval_count"] - out["ignored_count"])

    def test_bpm_range_filter(self):
        beats = _matched_from_times([0.5 * i for i in range(8)],
                                    [0.5 * i for i in range(8)])
        # 120 BPM 落在 [140, 200] 外：全部间隔被范围过滤，事件为空
        out = build_tempo_map(
            _result_with(beats), TempoParams(min_bpm=140, max_bpm=200)
        )
        self.assertEqual(out["events"], [])
        self.assertEqual(out["event_count"], 0)
        self.assertEqual(len(out["ignored_indices"]), 7)
        self.assertTrue(
            all("below_min_bpm" in i["reasons"] for i in out["ignored_intervals"])
        )

    def test_window_beats_even_uses_odd_window(self):
        params = TempoParams(window_beats=4)
        self.assertEqual(params.window_size, 3)
        beats = _matched_from_times([0.5 * i for i in range(6)],
                                    [0.5 * i for i in range(6)])
        out = build_tempo_map(_result_with(beats), params)
        self.assertEqual(out["parameters"]["window_size"], 3)
        self.assertEqual(out["parameters"]["window_beats"], 4)

    def test_window_beats_one(self):
        # 窗口=1 时中位数就是原始值，没有邻居可抑制，漏拍间隔不会被标 outlier
        ref = [0.5 * i for i in range(7)]
        perf = [0.0, 0.5, 1.5, 2.0, 2.5, 3.0]
        aligned = align_beats(ref, perf, AlignmentParams(band_width=2))
        out = build_tempo_map(aligned, TempoParams(window_beats=1))
        self.assertTrue(all("outlier" not in i["reasons"])
                        for i in out["ignored_intervals"])

    def test_non_positive_interval_reported(self):
        # 演奏时间非递增（异常数据）：非正间隔被标记且不致崩
        beats = _matched_from_times(
            [0.0, 0.5, 1.0, 1.5],
            [0.0, 0.5, 0.5, 1.5],
        )
        out = build_tempo_map(_result_with(beats))
        reasons = {i["interval_index"]: i["reasons"] for i in out["ignored_intervals"]}
        self.assertIn(1, reasons)
        self.assertIn("non_positive_interval", reasons[1])
        self.assertIsNone(
            next(i for i in out["ignored_intervals"] if i["interval_index"] == 1)["raw_bpm"]
        )

    def test_too_few_matched_beats_raises(self):
        with self.assertRaises(TempoError):
            build_tempo_map(_result_with([]))
        with self.assertRaises(TempoError):
            build_tempo_map(_result_with(_matched_from_times([0.0], [0.0])))

    def test_invalid_params_raise(self):
        beats = _matched_from_times([0.5 * i for i in range(4)],
                                    [0.5 * i for i in range(4)])
        result = _result_with(beats)
        with self.assertRaises(TempoError):
            TempoParams(window_beats=0).validate()
        with self.assertRaises(TempoError):
            TempoParams(min_bpm=200, max_bpm=100).validate()
        with self.assertRaises(TempoError):
            TempoParams(min_bpm=-1).validate()
        with self.assertRaises(TempoError):
            build_tempo_map(result, TempoParams(min_bpm=200, max_bpm=100))


if __name__ == "__main__":
    unittest.main()
