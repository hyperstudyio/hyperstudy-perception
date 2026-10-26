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
   (`.github/workflows/publish.yml`) uploads those same files to PyPI. It then adds the notes to the
   `docs/experimenters/hyperstudy-perception.md` page of hyperstudy-docs, as hyperstudy-bridge does for its page.

A version on PyPI cannot be replaced. To fix a bad release, release the next patch version.

## One-time setup

These need the account owners, not the workflows. `CLAUDE_CODE_OAUTH_TOKEN` (pull request reviews) and
`DOCS_REPO_TOKEN` are hyperstudyio organization secrets, so they already reach this repository. Build provenance
(release.yml attests its files; publish.yml verifies them before uploading) needs nothing set up in a public repository.

- **The PyPI publisher.** On pypi.org, under *Your projects → Publishing → Add a new pending publisher*, enter:
  - PyPI project name: `hyperstudy-perception`
  - Owner: `hyperstudyio`
  - Repository name: `hyperstudy-perception`
  - Workflow name: `publish.yml`
  - Environment name: `pypi`

  The first publish creates the project. No PyPI token is stored anywhere.
- **The `pypi` environment** in this repository's settings. Optionally, add yourself as a required reviewer.
- **`DOCS_REPO_TOKEN`**, an organization secret shared with hyperstudy-bridge: a fine-grained token with *Contents*
  read and write on `hyperstudyio/hyperstudy-docs`.
- **The docs page** `docs/experimenters/hyperstudy-perception.md` in hyperstudy-docs, with a
  `<!-- RELEASE_NOTES_START -->` line where entries go, newest first.
