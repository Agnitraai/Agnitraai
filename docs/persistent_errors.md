# Persistent Notebook Errors

## Triton ``NameError`` During Kernel Generation

**Symptom**
- Running the vector-add demo in `agnitra_enhanced_demo.ipynb` shows:
  - ``CompilationError`` from Triton complaining it "Cannot access global variable BLOCK_SIZE".
  - The generated module lives under `.agnitra/notebook-kernels/notebook_vector_add.py`.

**Root cause**
- The notebook kernel imported `KernelGenerator` before the fix that replaced the global ``BLOCK_SIZE`` constant with ``DEFAULT_BLOCK_SIZE``.
- Python keeps the old module in memory, so regenerating the kernel still writes the stale source and Triton rejects it.

**Resolution**
- Restart the notebook kernel, or reload the module explicitly:
  ```python
  import importlib
  import agnitra.core.kernel.kernel_generator as kernel_gen
  importlib.reload(kernel_gen)
  ```
- Then rerun the generation cell. The regenerated file contains `DEFAULT_BLOCK_SIZE` and validation passes.
- If issues persist, remove the cached kernel directory (`.agnitra/notebook-kernels/`) before rerunning.

**Status**
- Fixed in source by commit "Fix Triton vector add block size constant"; only notebook sessions started before the fix need the restart/reload.
