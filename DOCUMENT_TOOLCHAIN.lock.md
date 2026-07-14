# Document artifact toolchain lock

Level A manuscript artifacts were verified with the following command-line tools:

| Tool | Required version | Purpose |
|---|---:|---|
| Pandoc | 3.10 | Markdown-to-LaTeX conversion |
| Tectonic | 0.16.9 | LaTeX-to-PDF compilation |
| Poppler | 26.05.0 | PDF metadata and page rendering during visual QA |

For the supported Conda-compatible workflow, install the document builders with:

```bash
micromamba install -c conda-forge pandoc=3.10 tectonic=0.16.9 poppler=26.05.0
```

`scripts/reproduce_artifacts.py` refuses a different Pandoc or Tectonic version. Poppler is used
for independent PDF inspection rather than for generation. The supplied `environment.lock.yml`
contains exact `linux-64` build strings for the Python experiment environment; users on other
platforms must create a platform-specific environment that preserves the documented top-level
versions.
