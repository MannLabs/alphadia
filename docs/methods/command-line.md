# Command line
AlphaDIA offers a command line interface to perform searches. In fact, even the GUI uses the command line interface internally. For reprodicable searches, method optimization or cluster and cloud use, the command line can be quite usefull.

## Usage
Before starting, please make sure that alphaDIA is correctly installed and up to date.
```bash
alphadia --check
```

To get an overview of all possible arguments, you can use

```bash
alphadia -h
```

Which should return

```
usage: alphadia [-h] [--version] [--output [OUTPUT]] [--file FILE]
                [--directory DIRECTORY] [--regex [REGEX]]
                [--library [LIBRARY]] [--fasta FASTA] [--config [CONFIG]]
                [--config-dict [CONFIG_DICT]]

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
  --fasta FASTA         Fasta file(s) used to generate or annotate the
                        spectral library.
  --config [CONFIG], -c [CONFIG]
                        Config yaml which will be used to update the default
                        config.
  --config-dict [CONFIG_DICT]
                        Python Dict which will be used to update the default
                        config.
```

## Setting up a search
The follwoing section will focus on how to set up a search using bash on mac or linux. For windows, please use a `.ps1` powershell script.
Create a new bash script named `search.sh` and enter it.

### 1. Basics
```bash
vim search.sh
```
We will write the command across multiple lines to make it more legible.
First, choose an output folder with the `-o` option, where all of the search results will be stored.

```
alphadia \
    -o /my/folder/output \
```

### 2. Input Files
Next add some raw files to the search with the `-f` option

```
    -f /my/raw/files1/exploris007_experiment1_rep1.raw \
    -f /my/raw/files1/exploris007_experiment1_rep2.raw \
```
AlphaDIA also supports to add whole directories to the search using the `d` option
```
    -d /my/raw/files2 \
```

Furthermore, files can be filtered by providing a regex filter using `--regex`. This allows for example to select a mass spec output folder and select by the experiment.
```
    --regex ".*_experiment1_.*\.raw$" \
```

### 3. Libraries
You can provide alphaDIA with an existing alphaBase `.hdf` or generic `.tsv` library.
Use the `--library` option to provide a library location
```
    --library speclib.single_species.hdf \
```

You can also provide a fasta file for reannotation of the library using the `--fasta` option.
Please note, if library prediction is enabled, the fasta file will not only be used for reannotation but will be digested and in-silico prediction of spectral library will be performed with peptDeep.

```
    --fasta human_reviewed_2024_03_11.fasta \
```

### 4. Configuration
To control the behaviour of alphaDIA's search, you will want to provide a `config.yaml` file.
All settings defined in this file will update the default parameters of alphaDIA.

You can either write it from scratch or use a `config.yaml` generated in a GUI search.

```
    --config "config_astral_first_pass.yaml" \
```

Sometimes, for example when optimizing a single parameter or building multi step workflows, you want to change only a single parameters on top of an existing configuration. Therefore, alphaDIA offers the `--config-dict` option, which leds you access all parameters from the command line.
```
    --config-dict "{\"library_prediction\":{\"nce\":26}}"
```

### 5. Summary
The full script looks like:
```bash
alphadia \
    -o /my/folder/output \
    -f /my/raw/files1/exploris007_experiment1_rep1.raw \
    -f /my/raw/files1/exploris007_experiment1_rep2.raw \
    -d /my/raw/files2 \
    --regex ".*_experiment1_.*\.raw$" \
    --library speclib.single_species.hdf \
    --fasta human_reviewed_2024_03_11.fasta \
    --config config_astral_first_pass.yaml \
    --config-dict "{\"library_prediction\":{\"nce\":26}}"
```
