# Contributing
This document gathers information on how to contribute to the alphaDIA project.

## Release process
### Tagging of changes
In order to have release notes automatically generated, changes need to be tagged with labels.
The following labels are used (should be safe-explanatory):
`breaking-change`, `bug`, `enhancement`.

### Release a new version
Note: Releases need to be done from the `main` branch.

1. Bump the version locally to (e.g. to `X.Y.Z`) and merge the change to `main`.
2. Create a new draft release on GitHub using the
[Create Draft Release](https://github.com/MannLabs/alphadia/actions/workflows/create_release.yml) workflow.
You need to specify the commit to release, and the release tag (e.g. `vX.Y.Z`).
3. Test the release manually.
4. Add release notes and publish the release on GitHub.
5. Run the [Publish on PyPi](https://github.com/MannLabs/alphadia/actions/workflows/publish_on_pypi.yml) workflow,
specifying the release tag (e.g. `vX.Y.Z`).
6. Run the [Publish Docker Image](https://github.com/MannLabs/alphadia/actions/workflows/publish_docker_image.yml) workflow,
specifying the release tag (e.g. `vX.Y.Z`).


## Notes for developers
### pre-commit hooks
It is highly recommended to use the provided pre-commit hooks, as the CI pipeline enforces all checks therein to
pass in order to merge a branch.

The hooks need to be installed once by
```bash
pre-commit install
```
You can run the checks yourself using:
```bash
pre-commit run --all-files
```
Make sure you use the same version of pre-commit as defined in `requirements_development.txt`.
