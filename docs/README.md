# SpecForge Documentation

We recommend new contributors to start from writing documentation, which helps you quickly understand the SpecForge codebase.
Most documentation files are located under the `docs/` folder.

## Docs workflow

### Install dependencies

```bash
pip install -r requirements.txt
```

### Build and preview

Documentation sources are Markdown and reStructuredText. If you add a page,
include it in `index.rst` or the relevant nested toctree.

Build the site and preview it with live reload:

```bash
make html
make serve
PORT=8080 make serve
```

Run repository formatting checks before submitting changes:

```bash
pre-commit run --all-files
```

Use relative links for repository documentation and follow the existing pages
for command and configuration examples.
