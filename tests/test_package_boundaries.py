"""服务导入不触发设备依赖，源码指纹覆盖拆包后的计算路径。"""
import subprocess
import sys

from laya.runtime import fingerprints


def test_service_and_cli_import_without_compute_dependencies():
    subprocess.run([sys.executable, "-c", """
import importlib.abc
import sys
class NoCompute(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'triton', 'transformers'}:
            raise AssertionError(f'服务进程导入了计算依赖: {fullname}')
sys.meta_path.insert(0, NoCompute())
from laya.api.app import create_app
from laya.commands.app import app
from laya.engine.core import Engine
from laya.engine.worker import main
"""], check=True)


def test_source_fingerprint_tracks_split_files(tmp_path, monkeypatch):
    root = tmp_path / "laya"
    (root / "runtime").mkdir(parents=True)
    monkeypatch.setattr(fingerprints, "__file__", str(root / "runtime/fingerprints.py"))
    paths = ["preprocessing.py", "model.py"]
    for name in paths:
        (root / name).write_text(name)
    original = fingerprints.source_fingerprint(paths)
    assert fingerprints.source_fingerprint(paths[::-1]) == original
    for name in paths:
        (root / name).write_text("changed")
        assert fingerprints.source_fingerprint(paths) != original
        (root / name).write_text(name)


def test_runtime_fingerprints_distinguish_execution_paths():
    values = {fingerprints.runtime_source_fingerprint(runner, quantized)
              for runner in ("eager", "cuda-graph") for quantized in (False, True)}
    assert len(values) == 4
