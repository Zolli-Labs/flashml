# tests/test_run_json.py
"""The run.json viewer contract (viewer_v1) + async submit/wait.

run.json is the SDK↔viewer handshake: submit() writes it atomically after
every state change so a viewer (Task 6) can render a live run from disk.
"""

from __future__ import annotations

import json
import sys
import textwrap
import time  # noqa: F401  (kept per the Task 3 brief's verbatim test header)


def _write_script(tmp_path, body: str) -> str:
    # Duplicated from test_sdk_submit (5 lines) instead of shared: these two
    # SDK test modules are independent, and a one-file helper import would
    # couple them for no real gain — the brief calls the duplication fine.
    src = tmp_path / "userproj"
    src.mkdir(exist_ok=True)
    (src / "train.py").write_text(textwrap.dedent(body))
    return str(src)


def test_run_json_written_and_versioned(tmp_path):
    import flashruntime as flash

    src = _write_script(tmp_path, "import json; json.dump({}, open('metrics.json','w'))")
    run = flash.submit(
        flash.CommandWorkload(command=f"{sys.executable} train.py", source={"path": src}),
        output_dir=tmp_path / "o",
    )
    doc = json.loads((tmp_path / "o" / "run.json").read_text())
    assert doc["contract"] == "viewer_v1"
    assert doc["state"] == "SUCCEEDED"
    assert doc["attempts"][0]["state"] == "SUCCEEDED"
    assert any(e["type"] == "LAUNCH_STARTED" for e in doc["events"])


def test_async_submit_wait(tmp_path):
    import flashruntime as flash

    src = _write_script(tmp_path, "import time; time.sleep(1)")
    run = flash.submit(
        flash.CommandWorkload(command=f"{sys.executable} train.py", source={"path": src}),
        output_dir=tmp_path / "o",
        wait=False,
    )
    assert run.state.value in ("PENDING", "RUNNING")  # returned before finish
    assert run.wait(timeout=30).value == "SUCCEEDED"
    assert run.state.value == "SUCCEEDED"
