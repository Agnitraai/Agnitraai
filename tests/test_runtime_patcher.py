import importlib.util

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from agnitra.core.kernel import KernelGenerator
from agnitra.core.runtime import (
    FXNodePatch,
    ForwardHookPatch,
    RuntimePatcher,
)


def _load_run_kernel(module_path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load kernel module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return getattr(module, "run_kernel")


class _AddNet(nn.Module):
    def forward(self, x, y):  # type: ignore[override]
        return x + y


def test_fx_patch_applies_kernel_and_respects_copy_semantics():
    patcher = RuntimePatcher()
    module = _AddNet()
    x = torch.ones(4)
    y = torch.arange(4, dtype=torch.float32)
    baseline = module(x, y)

    patch = FXNodePatch(
        name="shifted-add",
        target="operator.add",
        kernel=lambda a, b: a + b + 1.0,
        metadata={"test": "fx"},
    )

    report = patcher.patch(module, fx_patches=[patch], copy_module=True)
    patched = report.module

    assert torch.allclose(patched(x, y), baseline + 1.0)
    assert torch.allclose(module(x, y), baseline)

    assert report.applied
    log = report.applied[0]
    assert log.name == "shifted-add"
    assert log.strategy == "fx"
    assert log.metadata["test"] == "fx"
    assert log.matched, "expected at least one node to be patched"


def test_fx_patch_validator_triggers_fallback():
    patcher = RuntimePatcher()
    module = _AddNet()
    x = torch.randn(4)
    y = torch.randn(4)
    baseline = module(x, y)

    def validator(result, args, kwargs):  # noqa: D401 - short helper
        return False

    patch = FXNodePatch(
        name="always-invalid",
        target="operator.add",
        kernel=lambda a, b: a - b,
        validator=validator,
    )

    report = patcher.patch(module, fx_patches=[patch], copy_module=True)
    patched = report.module
    assert torch.allclose(patched(x, y), baseline)


class _LinearNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        nn.init.eye_(self.linear.weight)

    def forward(self, x):  # type: ignore[override]
        return self.linear(x)


def test_forward_hook_patch_with_fallback():
    patcher = RuntimePatcher()
    module = _LinearNet()
    x = torch.ones(1, 4)
    baseline = module(x)

    def validator(result, *_):
        return torch.isfinite(result).all()

    patch = ForwardHookPatch(
        name="nan-guard",
        module_path="linear",
        kernel=lambda mod, _inputs, output: output * float("nan"),
        validator=validator,
    )

    report = patcher.patch(module, hook_patches=[patch])
    patched_out = module(x)

    assert torch.allclose(patched_out, baseline)
    assert report.applied
    assert report.applied[0].strategy == "hook"
    assert report.applied[0].matched == ("linear",)


def test_runtime_patcher_with_generated_kernel(tmp_path):
    generator = KernelGenerator(output_dir=tmp_path)
    result = generator.generate("runtime-test", validate=False)
    run_kernel = _load_run_kernel(result.module_path)

    patcher = RuntimePatcher()
    module = _AddNet()
    x = torch.randn(8)
    y = torch.randn(8)

    patch = FXNodePatch(
        name="generated-kernel",
        target="operator.add",
        kernel=run_kernel,
        metadata={"kernel_file": result.module_path.name},
    )

    report = patcher.patch(module, fx_patches=[patch], copy_module=True)
    patched = report.module

    assert report.applied
    assert report.applied[0].metadata["kernel_file"] == result.module_path.name
    assert torch.allclose(patched(x, y), module(x, y))


def test_runtime_patcher_with_generated_kernel_and_hook(tmp_path):
    generator = KernelGenerator(output_dir=tmp_path)
    result = generator.generate("runtime-hook", validate=False)
    run_kernel = _load_run_kernel(result.module_path)

    class AddReluNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.relu = nn.ReLU()

        def forward(self, x, y):  # type: ignore[override]
            return self.relu(x + y)

    module = AddReluNet()
    patcher = RuntimePatcher()

    fx_patch = FXNodePatch(
        name="generated-kernel",
        target="operator.add",
        kernel=run_kernel,
    )
    hook_patch = ForwardHookPatch(
        name="relu-halver",
        module_path="relu",
        kernel=lambda mod, inputs, output: output * 0.5,
        validator=lambda result, *_: torch.isfinite(result).all(),
    )

    x = torch.rand(4)
    y = torch.rand(4)
    baseline = module(x, y)
    report = patcher.patch(
        module,
        fx_patches=[fx_patch],
        hook_patches=[hook_patch],
        copy_module=True,
    )
    patched = report.module

    assert len(report.applied) == 2
    assert torch.allclose(patched(x, y), baseline * 0.5)
