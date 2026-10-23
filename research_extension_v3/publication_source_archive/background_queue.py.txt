"""Finite, resumable local experiment queue with an OS-released singleton lock."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import HERE, ROOT, dump, now

QUEUE = HERE/'continuation_queue'
RUN = 'research_extension_v3/results/run_002'

def task_plan():
    return [
        ('baseline_integrity',['-m','research_extension_v3.verify','--run-dir','research_extension_v3/results/run_001','--require-complete']),
        ('expanded_lightgbm',['-m','research_extension_v3.followup','--phase','search_lightgbm','--run',RUN]),
        ('expanded_tabm',['-m','research_extension_v3.followup','--phase','search_tabm','--run',RUN]),
        ('three_seed_models',['-m','research_extension_v3.followup','--phase','fit_models','--run',RUN]),
        ('strong_model_bias_stress',['-m','research_extension_v3.followup','--phase','bias_stress','--run',RUN]),
        ('calibration_ablation',['-m','research_extension_v3.followup_analysis','--phase','calibration_ablation','--run',RUN]),
        ('grid_and_capital_sensitivity',['-m','research_extension_v3.followup_analysis','--phase','grid_capital','--run',RUN]),
        ('business_fairlearn',['-m','research_extension_v3.fairlearn_business','--run',RUN]),
        ('paired_intervals',['-m','research_extension_v3.inference','--run',RUN,'--repetitions','1000']),
        ('scientific_figures',['-m','research_extension_v3.report','--run-dir',RUN]),
        ('independent_verification',['-m','research_extension_v3.verify','--run-dir',RUN,'--require-complete']),
        ('followup_comparison',['-m','research_extension_v3.followup_analysis','--phase','finalize','--run',RUN]),
        ('provenance',['-m','research_extension_v3.provenance','--run',RUN]),
    ]

def initialize(folder=QUEUE):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    path=folder/'queue.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    state=dict(version=1,status='PENDING',created_utc=now(),worker_pid=None,
               objective='Complete the finite follow-up experiments, then stop; preserve run_001.',
               run_directory=str(ROOT/RUN),max_attempts=2,
               tasks=[dict(id=name,args=args,status='PENDING',attempts=0) for name,args in task_plan()])
    dump(path,state);return state

def acquire_lock(folder):
    import msvcrt
    path=Path(folder)/'worker.lock'
    stream=path.open('a+b')
    if path.stat().st_size==0:stream.write(b'1');stream.flush()
    stream.seek(0)
    try:msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:
        stream.close();return None
    return stream

def run_queue(folder=QUEUE):
    folder=Path(folder);state=initialize(folder)
    lock=acquire_lock(folder)
    if lock is None:
        print('Queue worker already running; no duplicate launched.',flush=True);return 0
    try:
        # Recover interrupted child states once the exclusive worker lock is held.
        for task in state['tasks']:
            if task['status']=='RUNNING':
                old_pid=task.get('pid')
                if old_pid and process_alive(old_pid):
                    state.update(status='WAITING_FOR_EXISTING_CHILD',worker_pid=os.getpid(),existing_child_pid=old_pid,updated_utc=now())
                    dump(folder/'queue.json',state)
                    while process_alive(old_pid):time.sleep(5)
                task['status']='PENDING'
        state.update(status='RUNNING',worker_pid=os.getpid(),updated_utc=now())
        dump(folder/'queue.json',state)
        environment=dict(os.environ,PYTHONUTF8='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        for task in state['tasks']:
            if task['status']=='PASS':continue
            if task['attempts']>=state['max_attempts']:
                state.update(status='NEEDS_ATTENTION',current_task=task['id'],updated_utc=now())
                dump(folder/'queue.json',state);return 2
            while task['attempts']<state['max_attempts']:
                task.update(status='RUNNING',attempts=task['attempts']+1,started_utc=now())
                log=folder/(task['id']+'.log')
                with log.open('ab',buffering=0) as output:
                    output.write(f'\n[{now()}] attempt {task["attempts"]}\n'.encode())
                    process=subprocess.Popen([sys.executable,*task['args']],cwd=ROOT,env=environment,
                        stdout=output,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW)
                    task.update(pid=process.pid,log=str(log))
                    state.update(current_task=task['id'],updated_utc=now())
                    dump(folder/'queue.json',state)
                    while process.poll() is None:
                        state['updated_utc']=now();dump(folder/'queue.json',state);time.sleep(5)
                    task.update(exit_code=process.returncode,finished_utc=now(),
                                status='PASS' if process.returncode==0 else 'FAIL')
                state['updated_utc']=now();dump(folder/'queue.json',state)
                if task['status']=='PASS':break
                time.sleep(10)
            if task['status']!='PASS':
                state.update(status='NEEDS_ATTENTION',updated_utc=now());dump(folder/'queue.json',state);return 2
        state.update(status='COMPLETE',current_task=None,completed_utc=now(),updated_utc=now())
        dump(folder/'queue.json',state)
        print('All queued experiments completed.',flush=True);return 0
    finally:lock.close()

def process_alive(pid):
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.windll.kernel32
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    process=kernel.OpenProcess(0x1000,False,int(pid))
    if not process:return False
    code=wintypes.DWORD()
    okay=kernel.GetExitCodeProcess(process,ctypes.byref(code))
    kernel.CloseHandle(process)
    return bool(okay and code.value==259)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--initialize',action='store_true');parser.add_argument('--status',action='store_true')
    args=parser.parse_args()
    if args.initialize or args.status:
        state=initialize();print(json.dumps(state,ensure_ascii=False,indent=2));return
    sys.exit(run_queue())

if __name__=='__main__':main()
