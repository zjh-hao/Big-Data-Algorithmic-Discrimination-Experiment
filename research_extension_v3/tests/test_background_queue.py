import json
import os
from pathlib import Path
from research_extension_v3 import background_queue as queue

def test_exclusive_lock_is_released(tmp_path):
    first=queue.acquire_lock(tmp_path)
    assert first is not None
    assert queue.acquire_lock(tmp_path) is None
    first.close()
    again=queue.acquire_lock(tmp_path)
    assert again is not None
    again.close()

def test_completed_tasks_resume_without_duplicate_execution(tmp_path,monkeypatch):
    marker=tmp_path/'counter.txt'
    code=f"from pathlib import Path; p=Path({str(marker)!r}); p.write_text(p.read_text()+'x' if p.exists() else 'x')"
    monkeypatch.setattr(queue,'task_plan',lambda:[('one',['-c',code]),('two',['-c',code])])
    original_sleep=queue.time.sleep
    monkeypatch.setattr(queue.time,'sleep',lambda seconds:original_sleep(.01))
    assert queue.run_queue(tmp_path)==0
    assert marker.read_text()=='xx'
    assert queue.run_queue(tmp_path)==0
    assert marker.read_text()=='xx'
    assert json.loads((tmp_path/'queue.json').read_text())['status']=='COMPLETE'

def test_failure_is_bounded_and_requires_attention(tmp_path,monkeypatch):
    monkeypatch.setattr(queue,'task_plan',lambda:[('failure',['-c','raise SystemExit(3)'])])
    original_sleep=queue.time.sleep
    monkeypatch.setattr(queue.time,'sleep',lambda seconds:original_sleep(.01))
    assert queue.run_queue(tmp_path)==2
    state=json.loads((tmp_path/'queue.json').read_text())
    assert state['status']=='NEEDS_ATTENTION'
    assert state['tasks'][0]['attempts']==2
    assert queue.run_queue(tmp_path)==2
    assert json.loads((tmp_path/'queue.json').read_text())['tasks'][0]['attempts']==2

def test_pid_probe_detects_current_process():
    assert queue.process_alive(os.getpid())
    assert not queue.process_alive(4294967294)
