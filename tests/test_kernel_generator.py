import importlib.util
from pathlib import Path

import pytest
import torch

from agnitra.core.kernel import KernelGenerator


@pytest.fixture
def generator(tmp_path: Path) -> KernelGenerator:
    return KernelGenerator(output_dir=tmp_path)


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_generate_vector_add_kernel(generator: KernelGenerator):
    result = generator.generate(
        {"template": "vector_add", "parameters": {"BLOCK_SIZE": 64}},
        module_name="vector_add_test",
    )
    assert result.module_path.exists()
    content = result.module_path.read_text()
    assert "BLOCK_SIZE = 64" in content
    assert "def vector_add_test" in content
    assert result.validation is not None
    assert result.validation.status == "passed"

    module = _load_module(result.module_path)
    x = torch.randn(128)
    y = torch.randn(128)
    torch.testing.assert_close(module.run_kernel(x, y), x + y)


@pytest.mark.parametrize("template", ["vector_add", "matrix_multiply", "layer_norm"])
def test_template_roundtrip(template: str, generator: KernelGenerator):
    result = generator.generate({"template": template})
    assert result.module_path.exists()
    assert result.validation is not None
    assert result.validation.status in {"passed", "skipped"}


def test_generate_from_policy(generator: KernelGenerator):
    result = generator.generate("Policy<tiling>")
    assert result.module_path.exists()
    assert result.validation is not None
    assert result.validation.status in {"passed", "skipped"}
    assert result.module_path.name.endswith(".py")


def test_unknown_template(generator: KernelGenerator):
    with pytest.raises(KeyError):
        generator.generate({"template": "unknown"})
