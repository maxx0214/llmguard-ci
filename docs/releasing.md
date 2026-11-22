# Releasing LLMGuard CI

Build on the approved release branch. The package metadata uses `0.3.0`
and the Git tag uses `v0.3.0`. Obtain approval before uploading to either
package index, pushing to a public GitHub repository, or creating a release.
Use the same reviewed artifacts throughout.

## Local checks

From the repository root, install development tooling and use a new output
directory for this build. Run the same checks on Python 3.11, 3.12, and 3.13.

```powershell
python -m pip install -e ".[dev]" "langchain-core>=0.3.86,<1"
python -m pytest -q
python -m build --outdir outputs/v0.3.0
python -m twine check --strict "outputs/v0.3.0/*"
python scripts/check_release.py outputs/v0.3.0
Get-FileHash outputs/v0.3.0/*
```

Review the committed tree and both archive member lists before publication.
Keep raw evaluation data, credentials, cache files, model outputs, and
private integration code in local storage. The published sample report
comes from synthetic cases in the Quick Start.

### License and dependencies

The project uses MIT. The wheel contains project code and its HTML template.
The sdist also includes project documentation, tests, public benchmark
scripts, and synthetic examples. Neither distribution bundles dependency
packages, model weights, or benchmark datasets.

Direct dependency metadata reviewed for this release reports permissive
licenses. MIT covers PyYAML, Pydantic, pytest, and build. BSD-family licenses
cover HTTPX, python-dotenv, NumPy, SciPy, scikit-learn, and Jinja2.
Apache-2.0 covers the OpenAI SDK, datasets, PyArrow, rank-bm25, transformers,
and Twine. The optional LangChain core compatibility test uses MIT-licensed
`langchain-core`. Extras are installed separately under their respective
licenses. Review separate benchmark tool and dataset licenses when
redistributing any of those materials.

## TestPyPI after approval

Create a TestPyPI account and use its API token through Twine's prompt or
secure credential storage. Use `__token__` as the username. Confirm the
project name is accepted before proceeding to production.

```powershell
$wheel = 'outputs/v0.3.0/llmguard_ci-0.3.0-py3-none-any.whl'
$sdist = 'outputs/v0.3.0/llmguard_ci-0.3.0.tar.gz'
python -m twine upload --repository testpypi $wheel $sdist
python -m venv .venv/testpypi-v030
& .venv/testpypi-v030/Scripts/python.exe -m pip install --index-url https://test.pypi.org/simple/ --no-deps --only-binary=:all: "llmguard-ci==0.3.0"
& .venv/testpypi-v030/Scripts/python.exe examples/quickstart.py
```

Check the expected `WARN` in the Quick Start and verify the installed version
with `& .venv/testpypi-v030/Scripts/python.exe -m pip show llmguard-ci`.
Use the TestPyPI index only for the test package. The core has no dependencies.
[TestPyPI documentation](https://packaging.python.org/en/latest/guides/using-testpypi/)

## PyPI and GitHub after approval

Confirm TestPyPI passed, verify the SHA-256 hashes, and use a separate
production API token. The private repository history includes a local
integration runner. For the first public GitHub release, export only
the reviewed source tree with `git archive --format=zip --output=outputs/llmguard-source.zip HEAD`.
Extract it into a fresh directory, initialize a new Git repository on
`release/0.3.0`, and commit the extracted files. Set `origin` to the intended
public repository after approval and push the reviewed branch. Confirm that
GitHub CI is green for that commit before tagging.

Prepare release notes from the `0.3.0` changelog section and the verified
artifact hashes. Run the following commands from the reviewed public source
commit, with `$wheel` and `$sdist` pointing to the checked local artifacts.

```powershell
python -m twine upload --repository pypi $wheel $sdist
git tag -a v0.3.0 -m "LLMGuard CI 0.3.0"
git push origin v0.3.0
gh release create v0.3.0 $wheel $sdist --verify-tag --draft --title "LLMGuard CI v0.3.0" --notes-file $releaseNotes
```

Set `$releaseNotes` to the local release-notes file before creating the
draft. Verify the tag and attachments in the draft, then publish it from
GitHub. Finally, install `llmguard-ci==0.3.0` from PyPI in a fresh environment
and repeat the Quick Start. [PyPI packaging guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/) · [GitHub release guide](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
