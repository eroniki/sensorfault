"""scripts/stage1_run.sh must record the trainer's own pid (so `kill -TERM $(cat pid)` reaches it), pass TRAIN_ARGS, and refuse a busy GPU."""
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env(tmp_path, free_mib):
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    (fake / "nvidia-smi").write_text(f"#!/bin/sh\necho {free_mib}\n")
    (fake / "nvidia-smi").chmod(0o755)
    py = tmp_path / "fakepy.sh"
    py.write_text('#!/bin/sh\necho "ARGS: $@" >> "$STAGE1_WORK/args.txt"\nsleep 60\n')
    py.chmod(0o755)
    e = dict(os.environ, PATH=f"{fake}:{os.environ['PATH']}", PY=str(py), STAGE1_WORK=str(tmp_path / "work"), SDXL_DIR="x", NO_TMUX="1",
             TRAIN_ARGS="--stop-at 7")
    return e


def test_pid_file_is_the_trainer_and_sigterm_reaches_it(tmp_path):
    e = _env(tmp_path, 24000)
    r = subprocess.run(["bash", str(ROOT / "scripts" / "stage1_run.sh")], env=e, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    pidf = tmp_path / "work" / "pid"
    for _ in range(50):
        if pidf.exists() and (tmp_path / "work" / "args.txt").exists():
            break
        time.sleep(0.2)
    pid = int(pidf.read_text())
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode().replace("\0", " ")
    assert "fakepy.sh" in cmdline, cmdline              # the pid is the (fake) python, not a wrapper shell
    assert "--stop-at 7" in (tmp_path / "work" / "args.txt").read_text()
    os.kill(pid, signal.SIGTERM)
    time.sleep(0.5)
    assert not Path(f"/proc/{pid}").exists() or Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z"


def test_refuses_to_start_on_a_busy_gpu(tmp_path):
    e = _env(tmp_path, 4000)
    r = subprocess.run(["bash", str(ROOT / "scripts" / "stage1_run.sh")], env=e, capture_output=True, text=True, timeout=60)
    assert r.returncode == 3 and "another heavy job" in r.stdout
    assert not (tmp_path / "work" / "pid").exists()
