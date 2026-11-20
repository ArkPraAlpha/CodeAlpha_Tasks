"""Checkpoints: save the result of every expensive step so a stopped run can resume.

Each finished step writes  <output-dir>/checkpoints/<step>.joblib  and is ticked in
checkpoints/state.json. When the same command is run again, ticked steps are loaded
instead of recomputed. If the settings (input file, options, ...) change, old
checkpoints are discarded automatically so results never get mixed.
"""
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import joblib


class Checkpoint:
    def __init__(self, out_dir, fingerprint, fresh=False):
        self.dir = Path(out_dir) / "checkpoints"
        self.state_file = self.dir / "state.json"
        self.fingerprint = fingerprint
        if fresh and self.dir.exists():
            shutil.rmtree(self.dir)
            print("  --fresh: old checkpoints deleted, starting from scratch")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state = self._read()
        if self.state.get("fingerprint") != fingerprint:
            if self.state.get("steps"):
                print("  Settings changed since the last run -> old checkpoints discarded")
                shutil.rmtree(self.dir)
                self.dir.mkdir(parents=True, exist_ok=True)
            self.state = {"fingerprint": fingerprint, "steps": {}}
            self._write()

    def _read(self):
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _write(self):
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        os.replace(tmp, self.state_file)  # atomic: a crash cannot corrupt the state file

    def done(self, step):
        return step in self.state["steps"] and (self.dir / f"{step}.joblib").exists()

    def save(self, step, obj):
        tmp = self.dir / f"{step}.tmp"
        joblib.dump(obj, tmp)
        os.replace(tmp, self.dir / f"{step}.joblib")
        self.state["steps"][step] = datetime.now().isoformat(timespec="seconds")
        self._write()

    def load(self, step):
        return joblib.load(self.dir / f"{step}.joblib")

    def run(self, step, title, compute):
        """Load the step from its checkpoint, or compute and save it."""
        if self.done(step):
            print(f"  [x] {title}  (loaded from checkpoint {self.state['steps'][step]})")
            return self.load(step)
        print(f"  [ ] {title} ...")
        result = compute()
        self.save(step, result)
        print(f"  [x] {title}  (saved)")
        return result

    def show(self, steps):
        """Print the checklist."""
        print("Progress checklist:")
        for key, title in steps:
            print(f"  [{'x' if self.done(key) else ' '}] {title}")
        print()

    def chunk_dir(self, name):
        path = self.dir / name
        path.mkdir(exist_ok=True)
        return path
