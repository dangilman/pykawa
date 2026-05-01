# pykawa

![PyPI version](https://img.shields.io/pypi/v/pykawa.svg)

Exact solutioins for the phase shifts the scattering cross section for a Yukawa potential.

* [GitHub](https://github.com/dangilman/pykawa/) | [PyPI](https://pypi.org/project/pykawa/) | [Documentation](https://dangilman.github.io/pykawa/)
* Created by [Daniel Gilman](https://audrey.feldroy.com/) | GitHub [@dangilman](https://github.com/dangilman) | PyPI [@gilmanda](https://pypi.org/user/gilmanda/)
* MIT License

## Features

* TODO

## Documentation

Documentation is built with [Zensical](https://zensical.org/) and deployed to GitHub Pages.

* **Live site:** https://dangilman.github.io/pykawa/
* **Preview locally:** `just docs-serve` (serves at http://localhost:8000)
* **Build:** `just docs-build`

API documentation is auto-generated from docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

Docs deploy automatically on push to `main` via GitHub Actions. To enable this, go to your repo's Settings > Pages and set the source to **GitHub Actions**.

## Development

To set up for local development:

```bash
# Clone your fork
git clone git@github.com:your_username/pykawa.git
cd pykawa

# Install in editable mode with live updates
uv tool install --editable .
```

This installs the CLI globally but with live updates - any changes you make to the source code are immediately available when you run `pykawa`.

Run tests:

```bash
uv run pytest
```

Run quality checks (format, lint, type check, test):

```bash
just qa
```

## Author

pykawa was created in 2026 by Daniel Gilman.

Built with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
# pykawa
# pykawa
