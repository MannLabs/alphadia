# Installation

AlphaDIA can be installed on Windows, macOS and Linux. Please choose the preferred installation:

* [**One-click GUI installation:**](#one-click-gui-installation) Choose this installation if you only want the GUI and/or keep things as simple as possible.

* [**Pip installation**](#pip-installation)This version allows you to use alphaDIA as a package within your conda environment. You will only have access to the search engine backend and the command line but not the GUI.

* [**Developer installation:**](#developer-installation) This installation allows to modify alphaDIA's source code directly. Generally, the developer version of alphaDIA outperforms the precompiled versions which makes this the installation of choice for high-throughput experiments.

* [**Docker installation:**](#docker-installation) Choose this for running alphaDIA in a Docker container, which is useful if you want to run it in a cloud environment.

* [**Slurm installation:**](#slurm-cluster-installation) Choose this for running alphaDIA on a research cluster with Slurm.

## One-click GUI installation
Currently available for **MacOS** and **Windows**.
You can download the latest release of alphaDIA [here](https://github.com/Mannlabs/alphadia/releases/latest).

* **Windows:** Download the latest `win-x64` build. Save it and double click it to install. If you receive a warning during installation click *Run anyway*.
* **MacOS:** Download the latest `darwin-arm64` build. Please note that alphaDIA currently requires an ARM based M1/2/3 processor for the one-click installer. Save the installer and open the parent folder in Finder. Right-click and select *open*. If you receive a warning during installation click *Open*. If you want to use `.raw` files on Thermo instruments alphaRaw is required, which depends on Mono. A detailed guide to installing alphaRaw with mono can be found [here](https://github.com/MannLabs/alpharaw#installation).

As of now, **Linux** users need follow the steps for the
[developer installation](#developer-installation) in order to use the GUI.

## Pip installation
If you want to use alphaDIA as a python library (e.g. for importing it into Jupyter notebooks) or only use the command-line interface,
you can install alphaDIA via `pip`.

### 1. Prerequisites
Please make sure you have a valid installation of conda or miniconda.
We recommend setting up miniconda as described on their [website](https://docs.conda.io/projects/miniconda/en/latest/).

If you want to use `.raw` files on Thermo instruments alphaRaw is required, which depends on Mono.
A detailed guide to installing alphaRaw with mono can be found [here](https://github.com/MannLabs/alpharaw#installation), a short version is given
[below](#2-installing-mono).

### 2. Setting up the environment

It is highly recommended to use a conda environment for the installation of alphaDIA.
This ensures that all dependencies are installed correctly and do not interfere with other packages.
```bash
conda create -n alphadia python=3.11 -y
conda activate alphadia
```

### 3. Installing alphaDIA
Finally, alphaDIA and all its dependencies can be installed by
```bash
pip install "alphadia[stable]"
```
We strongly recommend using the `stable` version, which has all dependencies fixed,
for reasons of reproducibility and integrity.

Alternatively, use
`pip install alphadia`, which comes with less version constraints. This is not recommended, but may be useful to avoid
version clashes if alphaDIA is imported as a library into a defined python requirement.
Note however, that this "loose" version might be affected e.g. by breaking changes of third-party dependencies.

Finally, run `alphadia -v` to check if the installation was successful;
`alphadia -h` will give you a list of command-line options.


## Developer installation
AlphaDIA can be installed in editable (i.e. developer) mode. This allows to fully customize the software and even modify the source code to your specific needs. When an editable Python package is installed, its source code is stored in a location of your choice.

Make sure you have a conda environment and Mono installed for reading `.raw` files as described [here](https://github.com/MannLabs/alpharaw#installation).

See also the [developer guide](contributing.md) for more information on how to contribute to alphaDIA.

### 1. Setting up the repository

Navigate to a folder where you would like to install alphaDIA and
 download the alphaDIA repository. This creates a subfolder `alphadia` in your current directory
```bash
cd ~/work/search_engines
git clone git@github.com:MannLabs/alphadia.git
cd alphadia
```

Optionally, to get the latest features, switch to the `development` branch and pull the most recent version
```bash
git switch development
git pull
```

### 2. Installation: backend
Set up a conda environment and activate it as described [here](#2-setting-up-the-environment).

Use pip to install alphaDIA, using the `development` tag to install additional packages required for development only
```bash
pip install -e ".[stable,development]"
```
If you need less strict versions for third-party dependencies, use `pip install -e ".[development]"`.

Note: by using the editable flag `-e`, all modifications to the alphaDIA source code folder are directly reflected when running alphaDIA. Note that the alphaDIA folder cannot be moved and/or renamed if an editable version is installed.


### 3. Installation: GUI (optional)

If you want to use or extend the GUI, please install NodeJS as described on their [website](https://nodejs.org/en/download).

Install all frontend packages using npm
```bash
cd gui
npm install
```
Note that this gui install only works if the conda environment is called "`alphadia`",
unless the "envName" variables in `profile.js` are adjusted.

The GUI can then be started by typing
```bash
npm run dev
```

## Docker Installation
The containerized version can be used to run alphaDIA e.g. on cloud platforms.
It can be used to run alphaDIA for multiple input files, as well as for single files only
(trivial parallelization on computing clusters).

Note that this container is not optimized neither for performance nor size yet, and does not run on Apple Silicon chips
(M1/M2/M3) due to problems with mono. Also, it currently relies on the input files being organized
in a specific folder structure.

### 1. Setting up Docker
Install the latest version of docker (https://docs.docker.com/engine/install/).

### 2. Prepare folder structure
Set up your data to match the expected folder structure:
- Create a folder and store its name in a variable, e.g. `DATA_FOLDER=/home/username/data; mkdir -p $DATA_FOLDER`
- In this folder, create 4 subfolders:
  - `library`: put your library file here, make it writeable for any user (`chmod o+rw *`)
  - `raw`: put your raw data here
  - `output`: make this folder writeable for any user: `chmod -R o+rwx output` (this is where the output files will be stored)
  - `config`: create a file named `config.yaml` here, with the following content:
```yaml
library: /app/data/library/LIBRARY_FILE.hdf
raw_path_list:
  - /app/data/raw/RAW_FILE_1.raw
  - /app/data/raw/RAW_FILE_2.raw
  - ...
output_directory: /app/data/output
```
  Substitute `LIBRARY_FILE` and `RAW_FILE` with your respective file names, but preserve the `/app/data/../` prefix.
  The rest of the config values are taken from `default.yaml`, unless you overwrite any value from there
  in your `config.yaml`.

### 3. Start the container
```bash
docker run -v $DATA_FOLDER:/app/data/ mannlabs/alphadia:latest
```
After initial download of the container, alphaDIA will start running immediately. Alternatively, you can run an interactive session with
`docker run -v $DATA_FOLDER:/app/data/ -it mannlabs/alphadia:latest bash`

### 4. (Advanced) Build the image yourself
If you want to build the image yourself, you can do so by
```bash
docker build -t alphadia-docker .
```
and run it with
```bash
docker run -v $DATA_FOLDER:/app/data/ --rm alphadia-docker
```

## Slurm cluster Installation

### 1. Set up environment
Check if conda is available on your cluster. If not, install it, or, if provided, load the corresponding module, e.g.
`module load anaconda/3/2023.03`.

Create and activate a new conda environment
```bash
conda create -n alphadia python=3.11 -y
conda activate alphadia
```
### 2. Installing mono
Install mono to support reading proprietary vendor formats like Thermo `.raw` files.

Please make sure you include the conda-forge channel
```bash
conda install python=3.11 -c conda-forge
```
Then install mono by
```bash
conda install mono -c conda-forge
```

Make sure mono is installed by running
```bash
mono --version
```

Make sure the output looks something like this:
```
Mono JIT compiler version 6.12.0.90 (tarball Fri Mar  5 04:37:13 UTC 2021)
Copyright (C) 2002-2014 Novell, Inc, Xamarin Inc and Contributors. www.mono-project.com
	TLS:           __thread
	SIGSEGV:       altstack
	Notifications: epoll
	Architecture:  amd64
	Disabled:      none
	Misc:          softdebug
	Interpreter:   yes
	LLVM:          supported, not enabled.
	Suspend:       hybrid
	GC:            sgen (concurrent by default)
```


### 3. Installing alphaDIA

Install alphaDIA using pip:
```bash
pip install "alphadia[stable]"
```
Afterwards, verify the alphaDIA installation by running:
`alphadia --version` which should output the current version.

### Notes on running alphaDIA as part of automated workflows
AlphaDIA is designed to be run in a headless mode. In case of an error, a nonzero exit code is returned.
A exit code of 127 indicates that there was an unknown error. All other nonzero exit codes pertain to
'business errors', i.e. those caused most likely by user input (data and/or configuration).

Further details on such errors can be found in the `events.jsonl` file in the `.progress` folder(s) of the output directory.
