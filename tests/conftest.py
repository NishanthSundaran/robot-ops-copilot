"""Tests run without Ollama, GPU or the corpus: a tiny index is written to a temp dir before copilot imports."""

import json
import os
import pathlib
import sqlite3
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
TMP = pathlib.Path(tempfile.mkdtemp())
os.environ["COPILOT_DATA"] = str(TMP)
IDX = TMP / "index"
IDX.mkdir(parents=True)

(IDX / "errors.json").write_text(json.dumps({
    "codes": [{"code": "C153A1", "title": "Detected by the Shoulder joint.", "group": "C153",
               "group_title": "Position deviates from path", "page": 34,
               "explanation": "Robot could not follow the path", "suggestion": "Check payload"}],
    "groups": [{"code": "C153", "title": "Position deviates from path", "page": 34, "notes": []}]}))
(IDX / "params.json").write_text(json.dumps([
    {"file": "safety_monitor.yaml", "name": "safety_monitor.max_guide_speed", "value": "0.25", "comment": "m/s"},
    {"file": "safety_monitor.yaml", "name": "safety_monitor.max_guide_force", "value": "150.0", "comment": "N"},
    {"file": "joint_limits.yaml", "name": "joint_limits.elbow_joint.max_velocity", "value": "3.14", "comment": ""}]))
con = sqlite3.connect(IDX / "trials.sqlite")
con.execute("CREATE TABLE trials (trial_id, total_time_s, grasp_success, peak_force_N)")
con.executemany("INSERT INTO trials VALUES (?,?,?,?)",
                [("20260421_150000", "40.0", "True", "10.0"), ("20260421_160000", "50.0", "False", "30.0")])
con.commit()
con.close()
