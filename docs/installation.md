# Installation

## Developer installation

AlphaDIA can be installed in editable (i.e. developer) mode. This allows to fully customize the software and even modify the source code to your specific needs. When an editable Python package is installed, its source code is stored in a location of your choice.

Make sure to first read the prerequisites section [here](../README.md#1-prerequisites)  to have all necessary software
(conda, mono) installed.

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

Use pip to install alphaDIA, using the `development` tag to install additional packages required for development only
```bash
pip install -e ".[stable,development]"
```
If you need less strict versions for third-party dependencies, use
`pip install -e ".[development]"`, but make sure to read the corresponding caveats [here](../README.md#3-installation).

Note: by using the editable flag `-e`, all modifications to the [alphaDIA source code folder](alphadia ) are directly reflected when running alphaDIA. Note that the alphaDIA folder cannot be moved and/or renamed if an editable version is installed.


### 3. Installation: GUI (optional)

If you want to use or extend the GUI, please install NodeJS as described on their  [website](https://nodejs.org/en/download).

Install all frontend packages using npm
```bash
cd gui
npm install
```

The GUI can then be started by typing
```bash
npm run dev
```

## Use the dockerized version
The containerized version can be used to run alphaDIA e.g. on cloud platforms.
It can be used to run alphaDIA for multiple input files, as well as for single files only
(trivial parallelization on computing clusters).

Note that this container is not optimized for performance yet, and does not run on Apple Silicone chips
(M1/M2/M3) due to problems with mono. Also, it currently relies on the input files being organized
in a specific folder structure.

1) Install the latest version of docker (https://docs.docker.com/engine/install/).
2) Set up your data to match the expected folder structure:
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

3) Start the container
```bash
docker run -v $DATA_FOLDER:/app/data/ mannlabs/alphadia:latest
```
After initial download of the container, alphaDIA will start running immediately. Alternatively, you can run an interactive session with
`docker run -v $DATA_FOLDER:/app/data/ -it mannlabs/alphadia:latest bash`

### Build the image yourself (advanced users)
If you want to build the image yourself, you can do so by
```bash
docker build -t alphadia-docker .
```
and run it with
```bash
docker run -v $DATA_FOLDER:/app/data/ --rm alphadia-docker
```

## Installing alphaDIA on a SLURM cluster

### 1. Prerequisites
Check the prerequisites section [here](../README.md#1-prerequisites).

### 2. Set up environment
Create and activate a conda environment as described [here](../README.md#2-setting-up-the-environment).

### 3. Installing mono
Install mono to support reading proprietary vendor formats like Thermo `.raw` files.

Please make sure you include the conda-forge channel
```bash
conda install python=3.9 -c conda-forge
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


### 4. Installing alphaDIA

Install alphaDIA using pip as described [here](../README.md#3-installation).

Afterwards, verify the alphaDIA installation by running:
`alphadia -h`
which should give an output like this ```1.5.5```.

#### Obsolete once available on pip:
As long as there is no public release yet, we can't use `pip install alphadia`.
Below you can find hosted `.tar.gz` versions for now.
- [alphadia-v1.5.4](https://datashare.biochem.mpg.de/s/Llz4lEJhQacZWGr/download)
- [alphadia-v1.5.5](https://datashare.biochem.mpg.de/s/nryp3IUrVs9jucg/download)

Copy the link for the most recent version and download it using wget.
```bash
wget https://datashare.biochem.mpg.de/s/nryp3IUrVs9jucg/download -O alphadia.tar.gz
```

Untar the file
```bash
tar -xf ./alphadia.tar.gz
```

You should get a folder named `alphadia-x.x.x`

Install alphaDIA using pip, this might take some time.
```bash
pip install -e ./alphadia-1.5.5
```
