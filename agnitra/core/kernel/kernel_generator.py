"""Utilities for programmatically generating Triton kernels.

The generator consumes lightweight templates with ``{{PLACEHOLDER}}`` markers
and produces standalone Python modules that expose a ``run_kernel`` helper. The
helper prefers executing the Triton kernel but falls back to a pure PyTorch
implementation so validation can happen even when Triton is not installed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

LOGGER = logging.getLogger(__name__)

try:  # Optional dependency: validation falls back when torch unavailable
    import torch
except Exception:  # pragma: no cover - torch not installed
    torch = None  # type: ignore[assignment]

TorchTensor = Any


@dataclass(frozen=True)
class KernelTestCase:
    """Callable factory for validation inputs and expected outputs."""

    input_fn: Callable[[], Sequence[TorchTensor]]
    reference_fn: Callable[[Sequence[TorchTensor], Mapping[str, Any]], TorchTensor]
    atol: float = 1e-4
    rtol: float = 1e-4


@dataclass(frozen=True)
class KernelTemplate:
    """Template metadata describing the code and validation strategy."""

    name: str
    description: str
    source: str
    defaults: Mapping[str, Any] = field(default_factory=dict)
    test_cases: Sequence[KernelTestCase] = field(default_factory=tuple)


@dataclass(frozen=True)
class KernelValidationResult:
    """Outcome of executing validation inputs against a generated kernel."""

    status: str
    details: str
    failures: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class KernelGenerationResult:
    """Summary returned after rendering and (optionally) validating a kernel."""

    template: str
    parameters: Mapping[str, Any]
    module_path: Path
    validation: Optional[KernelValidationResult]

    def as_dict(self) -> Dict[str, Any]:
        """Return a serializable view of the generation metadata."""

        payload: Dict[str, Any] = {
            "template": self.template,
            "module_path": str(self.module_path),
            "parameters": dict(self.parameters),
        }
        if self.validation is not None:
            payload["validation"] = {
                "status": self.validation.status,
                "details": self.validation.details,
                "failures": list(self.validation.failures),
            }
        return payload


class KernelGenerator:
    """Render Triton kernel templates and run lightweight validation."""

    def __init__(
        self,
        *,
        output_dir: Optional[Path] = None,
        templates: Optional[Mapping[str, KernelTemplate]] = None,
    ) -> None:
        self.output_dir = Path(output_dir or Path.cwd() / '.agnitra' / 'kernels')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._templates: Dict[str, KernelTemplate] = dict(templates or _default_templates())

    # ------------------------------------------------------------------
    # Template management helpers
    # ------------------------------------------------------------------
    def available_templates(self) -> Sequence[str]:
        """Return the names of the registered templates."""

        return sorted(self._templates)

    def describe_template(self, name: str) -> KernelTemplate:
        """Return the template metadata by name."""

        if name not in self._templates:
            raise KeyError(
                f"Unknown kernel template '{name}'. Available: {', '.join(self.available_templates())}"
            )
        return self._templates[name]

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------
    def generate(
        self,
        request: str | Mapping[str, Any],
        *,
        validate: bool = True,
        module_name: Optional[str] = None,
    ) -> KernelGenerationResult:
        """Generate a kernel either from a policy string or explicit spec.

        Parameters
        ----------
        request:
            Either a policy string (used to derive template parameters) or a
            mapping with ``template`` and ``parameters`` keys.
        validate:
            Whether to execute the template's test cases using the generated
            module. Validation is skipped automatically if PyTorch is missing
            or if the template does not define test cases.
        module_name:
            Optional override for the output module file name (without
            extension). When omitted, the name is derived from the template and
            policy hash to keep file paths stable.
        """

        if isinstance(request, str):
            template_name = "vector_add"
            inferred_parameters = self._parameters_from_policy(request)
        else:
            request = dict(request)
            try:
                template_name = request["template"]
            except KeyError as exc:  # pragma: no cover - defensive programming
                raise KeyError("Explicit requests must include a 'template' key") from exc
            inferred_parameters = dict(request.get("parameters", {}))

        template = self.describe_template(template_name)
        merged_parameters = {**template.defaults, **inferred_parameters}

        resolved_module_name = module_name or _derive_module_name(
            template.name, repr(sorted(merged_parameters.items()))
        )
        if "KERNEL_NAME" not in inferred_parameters:
            merged_parameters["KERNEL_NAME"] = resolved_module_name

        rendered_source = self._render_template(template.source, merged_parameters)
        module_path = self.output_dir / f"{resolved_module_name}.py"
        if module_path.exists() and module_path.read_text() == rendered_source:
            LOGGER.debug("Kernel unchanged: %s", module_path)
        else:
            module_path.write_text(rendered_source)
            LOGGER.info("Wrote kernel module: %s", module_path)

        validation = (
            self._validate_module(
                module_path, resolved_module_name, template.test_cases, merged_parameters
            )
            if validate
            else None
        )

        return KernelGenerationResult(
            template=template.name,
            parameters=merged_parameters,
            module_path=module_path,
            validation=validation,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _parameters_from_policy(self, policy: str) -> Dict[str, Any]:
        """Translate a policy string into concrete parameters."""

        block_sizes = (32, 64, 128, 256)
        digest = int(hashlib.sha1(policy.encode("utf-8")).hexdigest(), 16)
        block_size = block_sizes[digest % len(block_sizes)]
        return {"BLOCK_SIZE": block_size}

    def _render_template(self, source: str, parameters: Mapping[str, Any]) -> str:
        rendered = source
        for key, value in parameters.items():
            rendered = rendered.replace(f"{{{{{key}}}}}", _format_value(value))
        remaining = re.findall(r"{{\s*([A-Z0-9_]+)\s*}}", rendered)
        if remaining:
            raise ValueError(f"Unresolved placeholders: {', '.join(sorted(set(remaining)))}")
        return rendered

    def _validate_module(
        self,
        module_path: Path,
        module_name: str,
        test_cases: Sequence[KernelTestCase],
        parameters: Mapping[str, Any],
    ) -> KernelValidationResult:
        if not test_cases:
            return KernelValidationResult(status="skipped", details="No validation cases supplied.")
        if torch is None:
            return KernelValidationResult(
                status="skipped",
                details="PyTorch not available; validation disabled.",
            )

        module = _import_module_from_path(module_path, module_name)
        run_fn = getattr(module, "run_kernel", None)
        if run_fn is None:
            return KernelValidationResult(
                status="failed",
                details="Generated module does not expose a run_kernel function.",
            )

        failures: list[str] = []
        for idx, case in enumerate(test_cases, start=1):
            raw_inputs = tuple(case.input_fn())
            reference_inputs = tuple(_clone_tensor(t) for t in raw_inputs)
            expected = case.reference_fn(reference_inputs, parameters)
            execution_inputs = tuple(_clone_tensor(t) for t in raw_inputs)
            try:
                result = run_fn(*execution_inputs)
            except Exception as exc:  # pragma: no cover - protective wrapper
                failures.append(f"case {idx}: execution raised {exc!r}")
                continue

            if not _allclose_tensors(result, expected, case.atol, case.rtol):
                diff = _max_diff(result, expected)
                failures.append(
                    f"case {idx}: output mismatch (max_diff={diff})"
                )

        if failures:
            return KernelValidationResult(
                status="failed",
                details=f"{len(failures)} validation case(s) failed.",
                failures=tuple(failures),
            )
        return KernelValidationResult(
            status="passed", details=f"Validated {len(test_cases)} case(s)."
        )


# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------

def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        inner = ", ".join(_format_value(v) for v in value)
        return f"[{inner}]"
    if isinstance(value, dict):
        items = ", ".join(f"{_format_value(k)}: {_format_value(v)}" for k, v in value.items())
        return f"{{{items}}}"
    if isinstance(value, str):
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            return value
        return repr(value)
    return repr(value)


def _derive_module_name(template: str, seed: str) -> str:
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]
    sanitized = re.sub(r"[^0-9a-zA-Z_]", "_", template)
    return f"{sanitized}_kernel_{digest}".lower()


def _import_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clone_tensor(value: TorchTensor) -> TorchTensor:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.clone()
    return value


def _allclose_tensors(a: TorchTensor, b: TorchTensor, atol: float, rtol: float) -> bool:
    if torch is None:
        return False
    a_tensor = a if isinstance(a, torch.Tensor) else torch.as_tensor(a)
    b_tensor = b if isinstance(b, torch.Tensor) else torch.as_tensor(b)
    if a_tensor.shape != b_tensor.shape:
        return False
    return torch.allclose(a_tensor, b_tensor, atol=atol, rtol=rtol)


def _max_diff(a: TorchTensor, b: TorchTensor) -> float:
    if torch is None:
        return float("nan")
    a_tensor = a if isinstance(a, torch.Tensor) else torch.as_tensor(a)
    b_tensor = b if isinstance(b, torch.Tensor) else torch.as_tensor(b)
    diff = (a_tensor - b_tensor).abs()
    return float(diff.max().detach().cpu().item())


# ----------------------------------------------------------------------
# Built-in template registry
# ----------------------------------------------------------------------


def _default_templates() -> Dict[str, KernelTemplate]:
    return {
        "vector_add": KernelTemplate(
            name="vector_add",
            description="Elementwise vector addition with configurable block size.",
            source=_vector_add_template(),
            defaults={"BLOCK_SIZE": 128, "KERNEL_NAME": "vector_add_kernel"},
            test_cases=(
                KernelTestCase(_vector_add_inputs, _vector_add_reference),
            ),
        ),
        "matrix_multiply": KernelTemplate(
            name="matrix_multiply",
            description="Tiled matrix multiplication with fallback torch path.",
            source=_matmul_template(),
            defaults={
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_M": 8,
                "KERNEL_NAME": "matmul_kernel",
            },
            test_cases=(
                KernelTestCase(_matmul_inputs, _matmul_reference),
            ),
        ),
        "layer_norm": KernelTemplate(
            name="layer_norm",
            description="Row-wise layer normalization with configurable epsilon.",
            source=_layer_norm_template(),
            defaults={"BLOCK_SIZE": 128, "EPSILON": 1e-5, "KERNEL_NAME": "layer_norm_kernel"},
            test_cases=(
                KernelTestCase(_layer_norm_inputs, _layer_norm_reference),
            ),
        ),
    }


def _vector_add_template() -> str:
    return textwrap.dedent(
        '''
        """Autogenerated Triton kernel for vector addition."""

        from __future__ import annotations

        import torch

        try:
            import triton
            import triton.language as tl
        except ImportError:  # pragma: no cover - optional dependency
            triton = None
            tl = None


        BLOCK_SIZE = {{BLOCK_SIZE}}


        def _ceil_div(x: int, y: int) -> int:
            return (x + y - 1) // y


        if triton is not None:

            @triton.jit
            def {{KERNEL_NAME}}(x_ptr, y_ptr, output_ptr, n_elements, *, BLOCK_SIZE: tl.constexpr):
                pid = tl.program_id(axis=0)
                start = pid * BLOCK_SIZE
                offsets = start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
                tl.store(output_ptr + offsets, x + y, mask=mask)


        def run_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Compute ``x + y`` via Triton when available else Torch fallback."""

            if x.shape != y.shape:
                raise ValueError("Inputs must share the same shape")
            if x.dim() != 1:
                raise ValueError("Only 1D tensors are supported")
            x = x.to(dtype=torch.float32)
            y = y.to(dtype=torch.float32)

            if triton is None:
                return x + y

            n_elements = x.numel()
            output = torch.empty_like(x)
            grid = (_ceil_div(n_elements, BLOCK_SIZE),)
            {{KERNEL_NAME}}[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
            return output
        '''
    ).strip()


def _matmul_template() -> str:
    return textwrap.dedent(
        '''
        """Autogenerated Triton kernel for tiled matrix multiplication."""

        from __future__ import annotations

        import torch

        try:
            import triton
            import triton.language as tl
        except ImportError:  # pragma: no cover - optional dependency
            triton = None
            tl = None


        BLOCK_M = {{BLOCK_M}}
        BLOCK_N = {{BLOCK_N}}
        BLOCK_K = {{BLOCK_K}}
        GROUP_M = {{GROUP_M}}


        if triton is not None:

            @triton.jit
            def {{KERNEL_NAME}}(
                a_ptr,
                b_ptr,
                c_ptr,
                M,
                N,
                K,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                stride_cm,
                stride_cn,
                *,
                BLOCK_M: tl.constexpr,
                BLOCK_N: tl.constexpr,
                BLOCK_K: tl.constexpr,
                GROUP_M: tl.constexpr,
            ):
                pid = tl.program_id(axis=0)
                grid_m = tl.cdiv(M, BLOCK_M)
                grid_n = tl.cdiv(N, BLOCK_N)
                group_id = pid // (GROUP_M * grid_n)
                group_count = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
                pid_m = group_id * GROUP_M + pid % group_count
                pid_n = (pid // group_count) % grid_n

                offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                offs_k = tl.arange(0, BLOCK_K)

                a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
                b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

                accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for k in range(0, K, BLOCK_K):
                    a_block = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
                    b_block = tl.load(b_ptrs, mask=(offs_bn[None, :] < N) & (offs_k[:, None] + k < K), other=0.0)
                    accumulator += tl.dot(a_block, b_block)
                    a_ptrs += BLOCK_K * stride_ak
                    b_ptrs += BLOCK_K * stride_bk

                c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
                tl.store(c_ptrs, accumulator, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))


        def run_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """Multiply matrices using Triton when available else Torch fallback."""

            if a.dim() != 2 or b.dim() != 2:
                raise ValueError("Inputs must be 2D matrices")
            if a.size(1) != b.size(0):
                raise ValueError("Incompatible matrix dimensions")
            a = a.to(dtype=torch.float32)
            b = b.to(dtype=torch.float32)

            if triton is None:
                return a @ b

            M, K = a.shape
            _, N = b.shape
            c = torch.empty((M, N), device=a.device, dtype=a.dtype)
            grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
            {{KERNEL_NAME}}[
                grid
            ](
                a,
                b,
                c,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
                GROUP_M=GROUP_M,
            )
            return c
        '''
    ).strip()


def _layer_norm_template() -> str:
    return textwrap.dedent(
        '''
        """Autogenerated Triton kernel for row-wise layer normalization."""

        from __future__ import annotations

        import torch

        try:
            import triton
            import triton.language as tl
        except ImportError:  # pragma: no cover - optional dependency
            triton = None
            tl = None


        BLOCK_SIZE = {{BLOCK_SIZE}}
        EPSILON = {{EPSILON}}


        if triton is not None:

            @triton.jit
            def {{KERNEL_NAME}}(x_ptr, gamma_ptr, beta_ptr, output_ptr, stride, hidden_size, *, BLOCK_SIZE: tl.constexpr):
                row = tl.program_id(axis=0)
                offsets = row * stride + tl.arange(0, BLOCK_SIZE)
                mask = offsets < (row + 1) * stride

                x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                mean = tl.sum(x, axis=0) / hidden_size
                centered = x - mean
                variance = tl.sum(centered * centered, axis=0) / hidden_size
                inv_std = tl.rsqrt(variance + EPSILON)
                gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0)
                beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)
                normalized = centered * inv_std * gamma + beta
                tl.store(output_ptr + offsets, normalized, mask=mask)


        def run_kernel(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
            """Apply layer normalization with learned scale and bias."""

            if x.dim() != 2:
                raise ValueError("Input must be a 2D tensor [batch, hidden]")
            batch, hidden = x.shape
            if gamma.shape != (hidden,) or beta.shape != (hidden,):
                raise ValueError("Gamma/Beta shapes must match the hidden dimension")

            if triton is None:
                return torch.nn.functional.layer_norm(x, (hidden,), weight=gamma, bias=beta, eps=EPSILON)

            output = torch.empty_like(x)
            grid = (batch,)
            {{KERNEL_NAME}}[grid](
                x,
                gamma,
                beta,
                output,
                x.stride(0),
                hidden,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            return output
        '''
    ).strip()


# ----------------------------------------------------------------------
# Built-in validation helpers
# ----------------------------------------------------------------------

def _vector_add_inputs() -> Sequence[TorchTensor]:
    if torch is None:
        raise RuntimeError("PyTorch is required for validation")
    torch.manual_seed(0)
    size = 256
    x = torch.randn(size, dtype=torch.float32)
    y = torch.randn(size, dtype=torch.float32)
    return (x, y)


def _vector_add_reference(
    inputs: Sequence[TorchTensor], _: Mapping[str, Any]
) -> TorchTensor:
    x, y = inputs
    return x + y


def _matmul_inputs() -> Sequence[TorchTensor]:
    if torch is None:
        raise RuntimeError("PyTorch is required for validation")
    torch.manual_seed(1)
    a = torch.randn(64, 48, dtype=torch.float32)
    b = torch.randn(48, 32, dtype=torch.float32)
    return (a, b)


def _matmul_reference(
    inputs: Sequence[TorchTensor], _: Mapping[str, Any]
) -> TorchTensor:
    a, b = inputs
    return a @ b


def _layer_norm_inputs() -> Sequence[TorchTensor]:
    if torch is None:
        raise RuntimeError("PyTorch is required for validation")
    torch.manual_seed(2)
    x = torch.randn(4, 128, dtype=torch.float32)
    gamma = torch.randn(128, dtype=torch.float32)
    beta = torch.randn(128, dtype=torch.float32)
    return (x, gamma, beta)


def _layer_norm_reference(
    inputs: Sequence[TorchTensor], parameters: Mapping[str, Any]
) -> TorchTensor:
    x, gamma, beta = inputs
    eps = float(parameters.get("EPSILON", 1e-5))
    return torch.nn.functional.layer_norm(
        x, (x.size(1),), weight=gamma, bias=beta, eps=eps
    )


__all__ = [
    "KernelGenerator",
    "KernelTemplate",
    "KernelTestCase",
    "KernelGenerationResult",
    "KernelValidationResult",
]
