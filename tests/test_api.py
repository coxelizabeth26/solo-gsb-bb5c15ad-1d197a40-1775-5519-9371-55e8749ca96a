"""FastAPI 接口集成测试：使用临时 SQLite 文件。"""

import os
import tempfile
import unittest

# 必须在导入 app.* 之前设置数据库路径（app.database 在导入时读取该变量）
_TMPDIR = tempfile.mkdtemp(prefix="beat-align-test-")
os.environ["ALIGN_DB_PATH"] = os.path.join(_TMPDIR, "test.db")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

REF = [0.0, 0.5, 1.0, 1.5, 2.0]
PERF = [0.0, 0.48, 1.52, 2.0]  # 漏第 3 拍(1.0)，且有轻微抢/拖


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._cm = TestClient(app)
        cls.client = cls._cm.__enter__()
        cls.addClassCleanup(cls._cm.__exit__, None, None, None)

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})

    def test_full_flow(self):
        # 1. 创建
        r = self.client.post("/jobs", json={
            "name": "take-01",
            "reference_beats": REF,
            "performance_onsets": PERF,
            "params": {"band_width": 2},
        })
        self.assertEqual(r.status_code, 201, r.text)
        job = r.json()
        self.assertEqual(job["status"], "pending")
        job_id = job["id"]

        # 2. 对齐
        r = self.client.post(f"/jobs/{job_id}/align")
        self.assertEqual(r.status_code, 200, r.text)
        done = r.json()
        self.assertEqual(done["status"], "done")
        self.assertEqual(done["result"]["statistics"]["missed_count"], 1)
        self.assertEqual(done["result"]["statistics"]["matched_count"], 4)

        # 3. 查询
        r = self.client.get(f"/jobs/{job_id}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "done")

        r = self.client.get(f"/jobs/{job_id}/result")
        self.assertEqual(r.status_code, 200)
        self.assertIn("matched_beats", r.json()["result"])

        r = self.client.get("/jobs")
        self.assertEqual(r.status_code, 200)
        self.assertGreaterEqual(len(r.json()), 1)

        # 4. 重复对齐不重算；force=true 可重算
        r = self.client.post(f"/jobs/{job_id}/align")
        self.assertEqual(r.status_code, 200)
        r = self.client.post(f"/jobs/{job_id}/align?force=true")
        self.assertEqual(r.status_code, 200)

    def test_404(self):
        r = self.client.get("/jobs/999999")
        self.assertEqual(r.status_code, 404)
        r = self.client.post("/jobs/999999/align")
        self.assertEqual(r.status_code, 404)

    def test_result_before_align_conflicts(self):
        r = self.client.post("/jobs", json={
            "reference_beats": REF, "performance_onsets": REF,
        })
        job_id = r.json()["id"]
        r = self.client.get(f"/jobs/{job_id}/result")
        self.assertEqual(r.status_code, 409)

    def test_band_too_narrow_rejected(self):
        r = self.client.post("/jobs", json={
            "reference_beats": [0.0, 0.5, 1.0, 1.5],
            "performance_onsets": [0.0],
            "params": {"band_width": 1},
        })
        self.assertEqual(r.status_code, 400)

    def test_non_monotonic_rejected(self):
        r = self.client.post("/jobs", json={
            "reference_beats": [0.0, 0.6, 0.5],
            "performance_onsets": [0.0, 0.5],
        })
        self.assertEqual(r.status_code, 422)

    def test_anchor_full_flow(self):
        # 自由对齐会漏掉 ref[1]、ref[2]；锚点强制 (1,1)、(2,2) 匹配
        body = {
            "name": "anchor-take",
            "reference_beats": [0.0, 1.0, 2.0, 3.0],
            "performance_onsets": [0.0, 0.1, 2.9],
            "params": {"band_width": 3},
            "anchors": [
                {"ref_index": 1, "perf_index": 1},
                {"ref_index": 2, "perf_index": 2},
            ],
        }
        r = self.client.post("/jobs", json=body)
        self.assertEqual(r.status_code, 201, r.text)
        job = r.json()
        # 任务查询包含提交的锚点（SQLite 持久化往返）
        self.assertEqual(job["anchors"], body["anchors"])
        job_id = job["id"]

        r = self.client.get(f"/jobs/{job_id}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["anchors"], body["anchors"])

        r = self.client.post(f"/jobs/{job_id}/align")
        self.assertEqual(r.status_code, 200, r.text)
        result = r.json()["result"]
        pairs = [(b["ref_index"], b["perf_index"]) for b in result["matched_beats"]]
        self.assertEqual(pairs, [(0, 0), (1, 1), (2, 2)])
        # result 中包含已采用锚点
        self.assertEqual(
            [(a["ref_index"], a["perf_index"]) for a in result["anchors"]],
            [(1, 1), (2, 2)],
        )
        self.assertEqual(result["anchors"][0]["deviation"], -0.9)

        # /result 端点同样可查到已采用锚点
        r = self.client.get(f"/jobs/{job_id}/result")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()["result"]["anchors"]), 2)

        # 重新查询确认结果已从 SQLite 读出
        r = self.client.get(f"/jobs/{job_id}")
        self.assertEqual(len(r.json()["result"]["anchors"]), 2)

    def test_anchor_skipped_within_segment(self):
        # 锚点之间仍允许漏拍
        r = self.client.post("/jobs", json={
            "reference_beats": [0.0, 0.5, 1.0, 1.5, 2.0],
            "performance_onsets": [0.0, 0.5, 1.5, 2.0],
            "params": {"band_width": 3},
            "anchors": [{"ref_index": 0, "perf_index": 0},
                        {"ref_index": 4, "perf_index": 3}],
        })
        self.assertEqual(r.status_code, 201, r.text)
        job_id = r.json()["id"]
        r = self.client.post(f"/jobs/{job_id}/align")
        self.assertEqual(r.status_code, 200, r.text)
        result = r.json()["result"]
        self.assertEqual([b["ref_index"] for b in result["missed_beats"]], [2])

    def test_anchor_out_of_range_400(self):
        r = self.client.post("/jobs", json={
            "reference_beats": [0.0, 0.5, 1.0],
            "performance_onsets": [0.0, 0.5, 1.0],
            "params": {"band_width": 1},
            "anchors": [{"ref_index": 3, "perf_index": 0}],
        })
        self.assertEqual(r.status_code, 400)
        self.assertIn("越界", r.json()["detail"])

    def test_anchor_not_increasing_400(self):
        r = self.client.post("/jobs", json={
            "reference_beats": [0.0, 0.5, 1.0, 1.5],
            "performance_onsets": [0.0, 0.5, 1.0, 1.5],
            "params": {"band_width": 2},
            "anchors": [{"ref_index": 1, "perf_index": 1},
                        {"ref_index": 1, "perf_index": 2}],
        })
        self.assertEqual(r.status_code, 400)
        self.assertIn("严格递增", r.json()["detail"])

    def test_anchor_band_unreachable_400(self):
        r = self.client.post("/jobs", json={
            "reference_beats": [0.0, 0.5, 1.0, 1.5],
            "performance_onsets": [0.0, 0.5, 1.0, 1.5],
            "params": {"band_width": 1},
            "anchors": [{"ref_index": 3, "perf_index": 0}],
        })
        self.assertEqual(r.status_code, 400)
        self.assertIn("带宽", r.json()["detail"])

    def test_anchor_negative_index_400(self):
        r = self.client.post("/jobs", json={
            "reference_beats": [0.0, 0.5, 1.0],
            "performance_onsets": [0.0, 0.5, 1.0],
            "anchors": [{"ref_index": -1, "perf_index": 0}],
        })
        self.assertEqual(r.status_code, 400)
        self.assertIn("不能为负", r.json()["detail"])

    def test_job_without_anchors_compatible(self):
        # 未提供 anchors 的任务：输入侧返回 []，结果侧返回 []
        r = self.client.post("/jobs", json={
            "reference_beats": [0.0, 0.5, 1.0],
            "performance_onsets": [0.0, 0.5, 1.0],
        })
        self.assertEqual(r.status_code, 201, r.text)
        job_id = r.json()["id"]
        self.assertEqual(r.json()["anchors"], [])
        r = self.client.post(f"/jobs/{job_id}/align")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["result"]["anchors"], [])

    def test_tempo_map_full_flow(self):
        # 120 BPM 的稳定片段 + 一处漏拍（1.0s），漏拍间隔应被忽略
        r = self.client.post("/jobs", json={
            "name": "tempo-take",
            "reference_beats": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "performance_onsets": [0.0, 0.5, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "params": {"band_width": 2},
        })
        job_id = r.json()["id"]

        # 未完成 → 409
        r = self.client.get(f"/jobs/{job_id}/tempo-map")
        self.assertEqual(r.status_code, 409, r.text)

        r = self.client.post(f"/jobs/{job_id}/align")
        self.assertEqual(r.status_code, 200, r.text)

        r = self.client.get(f"/jobs/{job_id}/tempo-map")
        self.assertEqual(r.status_code, 200, r.text)
        tm = r.json()
        self.assertEqual(tm["matched_beat_count"], 8)
        self.assertEqual(tm["interval_count"], 7)
        # 漏拍造成的 60 BPM 间隔被忽略
        self.assertEqual(tm["ignored_count"], 1)
        self.assertEqual(len(tm["ignored_indices"]), 1)
        detail = tm["ignored_intervals"][0]
        self.assertEqual(
            (detail["ref_start_index"], detail["ref_end_index"]), (1, 3)
        )
        self.assertEqual(detail["ref_span"], 2)
        self.assertIn("outlier", detail["reasons"])
        self.assertAlmostEqual(detail["raw_bpm"], 60.0, places=2)
        # 稳定区间为 120 BPM
        self.assertGreaterEqual(tm["event_count"], 1)
        for ev in tm["events"]:
            self.assertAlmostEqual(ev["bpm"], 120.0, places=2)
            self.assertEqual(
                ev["beat_count"],
                ev["ref_end_index"] - ev["ref_start_index"] + 1,
            )
        # 事件按参考拍排序
        starts = [ev["ref_start_index"] for ev in tm["events"]]
        self.assertEqual(starts, sorted(starts))
        # 回显实际使用的估计参数
        self.assertEqual(tm["parameters"]["window_beats"], 5)
        self.assertEqual(tm["parameters"]["window_size"], 5)

    def test_tempo_map_query_params(self):
        r = self.client.post("/jobs", json={
            "reference_beats": [0.5 * i for i in range(8)],
            "performance_onsets": [0.5 * i for i in range(8)],
        })
        job_id = r.json()["id"]
        self.client.post(f"/jobs/{job_id}/align")

        # window_beats 为偶数时实际窗口取奇数（4→3）
        r = self.client.get(f"/jobs/{job_id}/tempo-map?window_beats=4")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["parameters"]["window_size"], 3)

        # 自定义 BPM 范围（120 落在 [140,200] 外，全部忽略）
        r = self.client.get(f"/jobs/{job_id}/tempo-map?min_bpm=140&max_bpm=200")
        self.assertEqual(r.status_code, 200, r.text)
        tm = r.json()
        self.assertEqual(tm["events"], [])
        self.assertEqual(len(tm["ignored_indices"]), 7)
        self.assertTrue(
            all("below_min_bpm" in i["reasons"] for i in tm["ignored_intervals"])
        )

        # min_bpm >= max_bpm → 400
        r = self.client.get(f"/jobs/{job_id}/tempo-map?min_bpm=300&max_bpm=100")
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("min_bpm", r.json()["detail"])

        # window_beats=0 / 负数 → 422（Query 约束）
        r = self.client.get(f"/jobs/{job_id}/tempo-map?window_beats=0")
        self.assertEqual(r.status_code, 422)
        r = self.client.get(f"/jobs/{job_id}/tempo-map?min_bpm=-5")
        self.assertEqual(r.status_code, 422)

    def test_tempo_map_404(self):
        r = self.client.get("/jobs/999999/tempo-map")
        self.assertEqual(r.status_code, 404)

    def test_tempo_map_too_few_matched_beats_conflicts(self):
        # 单参考拍 + 单演奏音：只能产生 1 个匹配，无法估计间隔 → 409
        r = self.client.post("/jobs", json={
            "reference_beats": [0.0],
            "performance_onsets": [0.0],
            "params": {"band_width": 1},
        })
        job_id = r.json()["id"]
        r = self.client.post(f"/jobs/{job_id}/align")
        self.assertEqual(r.status_code, 200, r.text)
        r = self.client.get(f"/jobs/{job_id}/tempo-map")
        self.assertEqual(r.status_code, 409, r.text)
        self.assertIn("匹配拍", r.json()["detail"])

    def test_tempo_map_failed_job_conflicts(self):
        # 直接把任务置为 error：/tempo-map 应返回 409 并提示上次对齐失败
        import sqlite3
        db_path = os.environ["ALIGN_DB_PATH"]
        r = self.client.post("/jobs", json={
            "reference_beats": [0.0, 0.5],
            "performance_onsets": [0.0, 0.5],
        })
        job_id = r.json()["id"]
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "UPDATE jobs SET status='error', error=? WHERE id=?",
                ("boom", job_id),
            )
            conn.commit()
        finally:
            conn.close()
        r = self.client.get(f"/jobs/{job_id}/tempo-map")
        self.assertEqual(r.status_code, 409)
        self.assertIn("boom", r.json()["detail"])

    def test_legacy_db_without_anchors_column(self):
        # 模拟旧版数据库（jobs 表没有 anchors 列）：迁移后旧行可读、新任务可建
        import sqlite3
        from app import database
        db_path = os.environ["ALIGN_DB_PATH"]

        # 测试库：丢弃当前表并按旧结构重建，写入一行旧数据
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("DROP TABLE IF EXISTS jobs")
            conn.commit()
        finally:
            conn.close()
        conn = sqlite3.connect(db_path)
        try:
            conn.executescript(
                """
                CREATE TABLE jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    reference_beats TEXT NOT NULL,
                    performance_onsets TEXT NOT NULL,
                    params TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    result TEXT,
                    created_at TEXT NOT NULL,
                    aligned_at TEXT
                );
                INSERT INTO jobs (name, reference_beats, performance_onsets,
                                  params, status, created_at)
                VALUES ('old', '[0, 1]', '[0, 1]', '{}', 'pending',
                        '2026-01-01T00:00:00+00:00');
                """
            )
            conn.commit()
        finally:
            conn.close()

        database.init_db()  # 触发幂等迁移
        # 旧行可读，anchors 兜底为空列表
        r = self.client.get("/jobs")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(any(j["name"] == "old" for j in r.json()))

        # 迁移后仍可正常创建带 / 不带锚点的新任务
        r = self.client.post("/jobs", json={
            "reference_beats": [0.0, 1.0],
            "performance_onsets": [0.0, 1.0],
            "anchors": [{"ref_index": 0, "perf_index": 0}],
        })
        self.assertEqual(r.status_code, 201, r.text)
        self.assertEqual(r.json()["anchors"], [{"ref_index": 0, "perf_index": 0}])


if __name__ == "__main__":
    unittest.main()
