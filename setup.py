"""Legacy setup.py shim reading metadata from pyproject.toml."""

from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - older Python fallback
    import tomli as tomllib  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent
PYPROJECT = ROOT / "pyproject.toml"
README = ROOT / "README.md"

with PYPROJECT.open("rb") as fh:
    config = tomllib.load(fh)
project = config.get("project", {})

scripts = project.get("scripts", {}) or {}
console_scripts = [f"{name}={target}" for name, target in scripts.items()]
entry_points = {"console_scripts": console_scripts} if console_scripts else {}

optional_deps = project.get("optional-dependencies", {}) or {}

long_description = README.read_text(encoding="utf-8") if README.exists() else ""

authors = project.get("authors") or []
author = None
author_email = None
if authors:
    first = authors[0]
    author = first.get("name")
    author_email = first.get("email")

license_field = project.get("license") or {}
if isinstance(license_field, dict):
    license_name = license_field.get("text") or license_field.get("file") or ""
else:
    license_name = str(license_field)

keywords = project.get("keywords") or []
if isinstance(keywords, (list, tuple)):
    keywords_str = ", ".join(str(item) for item in keywords)
else:
    keywords_str = str(keywords)

urls = project.get("urls") or {}

setup(
    name=project.get("name", "agnitra"),
    version=project.get("version", "0.0.0"),
    description=project.get("description", ""),
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=project.get("requires-python", ">=3.8"),
    packages=find_packages(include=("agnitra*", "cli*")),
    extras_require=optional_deps,
    entry_points=entry_points,
    author=author,
    author_email=author_email,
    license=license_name,
    keywords=keywords_str,
    url=urls.get("Homepage") or urls.get("homepage"),
    project_urls=urls,
    classifiers=project.get("classifiers", []),
    license_files=("LICENSE",),
)
