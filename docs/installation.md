# Installation

## Install alphaDIA on a SLURM cluster

### 1. Prerequisites
Please make sure that conda is available and custom environments can be created.

### 2. Setting up the conda environment
First we will create a new conda environment and install python 3.11. Depending on the cluster a lower python version might be needed.
```bash
conda create -n alphadia 
conda activate alphadia
```
Please make sure you include the conda-forge channel
```bash
conda install python=3.11 -c conda-forge
```

### 3. Installing mono
We will next install mono to support reading proprietary vendor formats like Thermo `.raw` files.
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
Next we will need to install alphaDIA. As there is no public release yet, we can't use `pip install alphadia`.
Below you can find hosted `.tar.gz` versions for now.
- [alphadia-v1.5.4](https://datashare.biochem.mpg.de/s/Llz4lEJhQacZWGr/download)
- [alphadia-v1.5.5](https://datashare.biochem.mpg.de/s/nryp3IUrVs9jucg/download)

Navigate to your home directory or a directory where you have write and execute permissions.
```bash
cd ~
```

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

Verify the alphaDIA installation by running:
```bash
alphadia -h
```

You should get an ouptut like this:
```
usage: alphadia [-h] [--version] [--output [OUTPUT]] [--file FILE] [--directory DIRECTORY] [--regex [REGEX]] [--library [LIBRARY]] [--fasta FASTA] [--config [CONFIG]] [--wsl] [--config-dict [CONFIG_DICT]]

Search DIA experiments with alphaDIA

options:
  -h, --help            show this help message and exit
  --version, -v         Print version and exit
  --output [OUTPUT], -o [OUTPUT]
                        Output directory
  --file FILE, -f FILE  Raw data input files.
  --directory DIRECTORY, -d DIRECTORY
                        Directory containing raw data input files.
  --regex [REGEX], -r [REGEX]
                        Regex to match raw files in directory.
  --library [LIBRARY], -l [LIBRARY]
                        Spectral library.
  --fasta FASTA         Fasta file(s) used to generate or annotate the spectral library.
  --config [CONFIG], -c [CONFIG]
                        Config yaml which will be used to update the default config.
  --wsl, -w             Set if running on Windows Subsystem for Linux.
  --config-dict [CONFIG_DICT]
                        Python Dict which will be used to update the default config.
```
