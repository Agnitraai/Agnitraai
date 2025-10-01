import pytest

torch = pytest.importorskip("torch")
from torch import nn

from agnitra.core.runtime import (
    FXNodePatch,
    ForwardHookPatch,
    RuntimePatcher,
)


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
