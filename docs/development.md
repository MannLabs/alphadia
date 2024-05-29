# Development

## Release a new version

1. Bump the version locally to (e.g. to `X.Y.Z`) and create a PR with the version change.
2. After the PR is merged, create a new draft release on GitHub using the
[Create Draft Release](https://github.com/MannLabs/alphadia/actions/workflows/create_release.yml) workflow.
You need to specifcy the commit to release, and the release tag (e.g. `vX.Y.Z`).
3. Test the release manually.
4. Add release notes and publish the release on GitHub.
5. Run the [Publish on PyPi](https://github.com/MannLabs/alphadia/actions/workflows/publish_on_pypi.yml) workflow,
specifying the release tag (e.g. `vX.Y.Z`).
6. Run the [Publish Docker Image](https://github.com/MannLabs/alphadia/actions/workflows/publish_docker_image.yml) workflow,
specifying the release tag (e.g. `vX.Y.Z`).
