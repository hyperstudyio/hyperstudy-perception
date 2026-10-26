# Releasing hyperstudy-perception

A release is a version of the package on [PyPI](https://pypi.org/project/hyperstudy-perception/). The annotator app
installs the newest one compatible with it, and so does `uv tool install hyperstudy-perception`.

## Cutting a release

1. Merge a pull request that sets `version` in `orchestrator/pyproject.toml` (for example `0.2.0`). Before 1.0, a
   minor version (0.**2**.0) may change the `--json` protocol or the bundle format; a patch version (0.2.**1**) may
   not, because the app takes patch versions without asking.
2. Tag that commit of main and push the tag:

   ```sh
   git switch main && git pull
   git tag v0.2.0 && git push origin v0.2.0
   ```

3. **Release** (`.github/workflows/release.yml`) does the rest:
   - It checks that the tag is on main and matches the version.
   - It builds the package (`scripts/build-dist.sh`).
   - It installs and runs it as the app does (`scripts/smoke-package.sh`).
   - It drafts a GitHub release with the wheel and the sdist attached. The notes list the pull requests merged since
     the last release.
4. Read the draft and edit the notes if needed. Then **publish** it. Publishing is the approval: **Publish**
   (`.github/workflows/publish.yml`) uploads those same files to PyPI. It then opens a pull request on hyperstudy-docs
   that adds the notes to the `docs/experimenters/hyperstudy-perception.md` page. Merge that pull request to put them
   on the docs site.

A version on PyPI cannot be replaced. To fix a bad release, release the next patch version.

## One-time setup

These need the account owners, not the workflows:

- **The PyPI publisher.** On pypi.org, under *Your projects → Publishing → Add a new pending publisher*, enter:
  - PyPI project name: `hyperstudy-perception`
  - Owner: `hyperstudyio`
  - Repository name: `hyperstudy-perception`
  - Workflow name: `publish.yml`
  - Environment name: `pypi`

  The first publish creates the project. No PyPI token is stored anywhere.
- **The `pypi` environment** in this repository's settings. Optionally, add yourself as a required reviewer.
- **`DOCS_REPO_TOKEN`**, a repository secret: a fine-grained token for `hyperstudyio/hyperstudy-docs` only, with
  *Contents* and *Pull requests* read and write.
- **The docs page** `docs/experimenters/hyperstudy-perception.md` in hyperstudy-docs, with a
  `<!-- RELEASE_NOTES_START -->` line where entries go, newest first.
