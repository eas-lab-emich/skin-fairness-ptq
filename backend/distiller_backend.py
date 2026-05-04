import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


class DistillerBackend:
    METRICS_RE = re.compile(r"==> Top1:\s*([0-9.]+)\s+Top5:\s*([0-9.]+)\s+Loss:\s*([0-9.]+)")

    def __init__(self, data_dir, checkpoint, arch, eval_batches, output_dir, python_bin=None, distiller_dir=None,
                 backend="subprocess"):
        self.data_dir = str(Path(data_dir).resolve())
        self.checkpoint = str(Path(checkpoint).resolve())
        self.arch = arch
        self.eval_batches = int(eval_batches)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        root = Path(__file__).resolve().parents[1]
        self.distiller_dir = Path(distiller_dir).resolve() if distiller_dir else (root / "distiller").resolve()
        self.example_dir = self.distiller_dir / "examples" / "classifier_compression"
        self.python_bin = python_bin or str((root / ".venv" / "bin" / "python").resolve())
        self.backend = backend
        self.effective_backend = "subprocess"

        self.cache = {}
        self.eval_count = 0

    def load_cache(self, cache_path):
        path = Path(cache_path)
        if path.exists():
            self.cache = json.loads(path.read_text())

    def save_cache(self, cache_path):
        Path(cache_path).write_text(json.dumps(self.cache, indent=2) + "\n")

    def evaluate_baseline(self):
        payload = {
            "mode": "baseline",
            "arch": self.arch,
            "checkpoint": self.checkpoint,
            "eval_batches": self.eval_batches,
            "backend": self.backend,
        }
        key = self._cache_key(payload)
        if key in self.cache:
            return self.cache[key]

        cmd = [
            self.python_bin,
            "compress_classifier.py",
            "--arch",
            self.arch,
            self.data_dir,
            "--resume",
            self.checkpoint,
            "--evaluate",
            "--deterministic",
            "-j",
            "1",
            "-p",
            "100",
        ]
        if self.eval_batches > 0:
            cmd.extend(["--eval-batches", str(self.eval_batches)])

        base = self._run_and_parse(cmd)
        result = {
            "top1": base["top1"],
            "top5": base["top5"],
            "loss": base["loss"],
            "command": self._format_command(cmd),
        }
        self.cache[key] = result
        self.eval_count += 1
        return result

    def evaluate_quantized(self, assignment_wts=None, assignment_acts=None, bits_acts=8, default_bits_wts=8):
        normalized_wts = {k: int(v) for k, v in sorted((assignment_wts or {}).items())}
        normalized_acts = {k: int(v) for k, v in sorted((assignment_acts or {}).items())}
        payload = {
            "mode": "quantized",
            "arch": self.arch,
            "checkpoint": self.checkpoint,
            "eval_batches": self.eval_batches,
            "bits_acts": int(bits_acts),
            "default_bits_wts": int(default_bits_wts),
            "assignment_wts": normalized_wts,
            "assignment_acts": normalized_acts,
            "backend": self.backend,
        }
        key = self._cache_key(payload)
        if key in self.cache:
            return self.cache[key]

        assign_wts_path = self.output_dir / f"assignment_wts_{key[:12]}.json"
        assign_acts_path = self.output_dir / f"assignment_acts_{key[:12]}.json"
        assign_wts_path.write_text(json.dumps(normalized_wts, indent=2) + "\n")
        assign_acts_path.write_text(json.dumps(normalized_acts, indent=2) + "\n")

        cmd = [
            self.python_bin,
            "compress_classifier.py",
            "--arch",
            self.arch,
            self.data_dir,
            "--resume",
            self.checkpoint,
            "--evaluate",
            "--quantize",
            "--q-bits-acts",
            str(bits_acts),
            "--q-bits-wts",
            str(default_bits_wts),
            "--q-bits-wts-map",
            str(assign_wts_path),
            "--q-bits-acts-map",
            str(assign_acts_path),
            "--deterministic",
            "-j",
            "1",
            "-p",
            "100",
        ]
        if self.eval_batches > 0:
            cmd.extend(["--eval-batches", str(self.eval_batches)])

        base = self._run_and_parse(cmd)
        result = {
            "top1": base["top1"],
            "top5": base["top5"],
            "loss": base["loss"],
            "command": self._format_command(cmd),
            "assignment_wts_path": str(assign_wts_path),
            "assignment_acts_path": str(assign_acts_path),
        }
        self.cache[key] = result
        self.eval_count += 1
        return result

    def _run_and_parse(self, cmd):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.distiller_dir)
        proc = subprocess.run(
            cmd,
            cwd=str(self.example_dir),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0:
            raise RuntimeError(f"Distiller eval failed (exit={proc.returncode})\nCommand: {self._format_command(cmd)}\n{output}")

        matches = self.METRICS_RE.findall(output)
        if not matches:
            raise RuntimeError(f"Could not parse metrics from output\nCommand: {self._format_command(cmd)}\n{output}")

        top1, top5, loss = matches[-1]
        return {
            "top1": float(top1),
            "top5": float(top5),
            "loss": float(loss),
        }

    @staticmethod
    def _cache_key(payload):
        s = json.dumps(payload, sort_keys=True)
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    @staticmethod
    def _format_command(cmd):
        return " ".join(cmd)
