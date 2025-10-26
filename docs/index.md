
# AlphaDIA Documentation
![GitHub Release](https://img.shields.io/github/v/release/mannlabs/alphadia?logoColor=green&color=brightgreen)
![Versions](https://img.shields.io/badge/python-3.10_%7C_3.11_%7C_3.12-brightgreen)
![License](https://img.shields.io/badge/License-Apache-brightgreen)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/mannlabs/alphadia/e2e_testing.yml?branch=main&label=E2E%20Tests)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/mannlabs/alphadia/pip_installation.yml?branch=main&label=Unit%20Tests)
![Docs](https://readthedocs.org/projects/alphadia/badge/?version=latest)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/mannlabs/alphadia/publish_docker_image.yml?branch=main&label=Deploy%20Docker)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/mannlabs/alphadia/publish_on_pypi.yml?branch=main&label=Deploy%20PyPi)
![Coverage](https://github.com/MannLabs/alphadia/raw/main/coverage.svg)
![Github](https://img.shields.io/github/stars/mannlabs/alphadia?style=social)

Open-source DIA search engine built with the alphaX ecosystem. Built with [alpharaw](https://github.com/MannLabs/alpharaw) and [alphatims](https://github.com/MannLabs/alphatims) for raw file acces. Spectral libraries are predicted with [peptdeep](https://github.com/MannLabs/alphapeptdeep) and managed by [alphabase](https://github.com/MannLabs/alphabase). Quantification is powered by [directLFQ](https://github.com/MannLabs/directLFQ).

**Features**
- Empirical library and fully predicted library search
- End-to-end transfer learning for custom RT, mobility, and MS2 models
- Label free quantification
- DIA multiplexing

We support the following vendor and processing modes:

| Platform              | Empirical lib | Predicted lib |
| :---------------- | :------: | :----: |
| Thermo .raw |   ✅   | ✅ |
| Sciex .wiff |   ✅   | ✅ |
| Bruker .d |  ✅   | ⚠️ |

:::{admonition} Predicted libraries with Bruker .d data
:class: warning
Although search is possible, alphaDIA's feature-free search takes a long time with fully predicted libraries. We are still evaluating how to better support fully predicted libraries.
:::

**Manuscript**
> **AlphaDIA enables DIA transfer learning for feature-free proteomics**<br>
> Georg Wallmann, Patricia Skowronek, Vincenth Brennsteiner, Mikhail Lebedev, Marvin Thielert, Sophia Steigerwald, Mohamed Kotb, Oscar Despard, Tim Heymann, Xie-Xuan Zhou, Maximilian T. Strauss, Constantin Ammar, Sander Willems, Magnus Schwörer, Wen-Feng Zeng & Matthias Mann
> [Nature Biotechnology (2025)](https://doi.org/10.1038/s41587-025-02791-w)


:::{card} Installation
:link: installation.html

Install alphaDIA on your system to run your own DIA searches.
:::

:::{card} Quickstart
:link: quickstart.html

Introduction to your first DIA search with alphaDIA.
:::

```{toctree}
:hidden:

🔧 Installation<installation>
🚀 Quickstart<quickstart>
📚 User Guides<guides>
📖 Methods<methods>
 🛠️ Developer guide<developer_guide>
```

```{toctree}
:caption: Development
:hidden:

modules
```
