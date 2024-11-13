# Developer Guide
This document gathers information on how to develop and contribute to the alphaDIA project.

## Release process

### Tagging of changes
In order to have release notes automatically generated, changes need to be tagged with labels.
The following labels are used (should be safe-explanatory):
`breaking-change`, `bug`, `enhancement`.

### Release a new version
This package uses a shared release process defined in the
[alphashared](https://github.com/MannLabs/alphashared) repository. Please see the instructions
[there](https://github.com/MannLabs/alphashared/blob/reusable-release-workflow/.github/workflows/README.md#release-a-new-version)


## Notes for developers
### Debugging
A good start for debugging is this notebook: `nvs/debug/debug_lvl1.ipynb`

##### Debug e2e tests with PyCharm
1. Create a "Run/Debug configuration" with
 - "module": `alphadia.cli`
 - "script parameters": `--config /abs/path/to/tests/e2e_tests/basic/config.yaml`
 - "working directory": `/abs/path/to/tests/e2e_tests`
2. Uncomment the lines following the `uncomment for debugging` comment in `alphadia/cli.py`.
3. Run the configuration.

##### Debug e2e tests with VsCode
1. Create the following debug configuration (`launch.json`):
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Current File with Arguments",
            "type": "debugpy",
            "request": "launch",
            "cwd": "/abs/path/to/tests/e2e_tests",
            "program": "../../alphadia/cli.py",
            "console": "integratedTerminal",
            "args": [
                "--config", "/abs/path/to/tests/e2e_tests/basic/config.yaml"
            ]
        }
    ]
}
```
2. Uncomment the lines following the `uncomment for debugging` comment in `alphadia/cli.py`.
3. Run the configuration.


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
