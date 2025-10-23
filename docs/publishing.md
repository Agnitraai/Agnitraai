# Package Publishing

Guidelines for shipping the Agnitra SDKs to PyPI and npm.

## Python (PyPI)

1. Bump the version in `pyproject.toml` (and update release notes if applicable).
2. Verify the workspace is clean, then run the test suite:
   ```bash
   pytest -q
   ```
3. Build the wheel and source distribution:
   ```bash
   python -m build
   ```
4. Inspect `dist/` to confirm `agnitra-<version>.tar.gz` and `.whl` look correct and include `LICENSE`, templates, and `py.typed`.
5. Upload to PyPI (or TestPyPI when rehearsing):
   ```bash
   twine upload dist/*
   ```
6. Tag the release in git and push the tag so CI/CD workflows can publish changelog artifacts.

## JavaScript / TypeScript (npm)

1. Enter the SDK workspace and install dependencies if needed:
   ```bash
   cd js
   npm install
   ```
2. Bump the version (`npm version patch|minor|major`) to sync with the PyPI release when applicable.
3. Run the type checker (and any integration checks):
   ```bash
   npm run lint
   ```
4. The `prepare` script builds `dist/` automatically. To preview the package contents before publishing run:
   ```bash
   npm pack
   ```
5. Publish the package:
   ```bash
   npm publish
   ```
6. Push the updated tag and commit so the monorepo tracks the published version.

Both packages share the Apache-2.0 license. Remember to export `AGNITRA_API_KEY` and `AGNITRA_API_BASE_URL` in your shell when validating end-to-end flows against the hosted control plane.
