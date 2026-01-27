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
Note: this can be passed multiple times, with the later ones taking precedence in case of overlapping keys.

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

## Advanced

### Error handling
By default, AlphaDIA will continue processing the next file when an error occurs for a raw file.
In case of errors, at the end of a (one- or multistep) search, all errors will be logged and AlphaDIA will exit with a nonzero exit code.
You can then check the log file to identify the issues, and, after fixing them, use the
restarting functionality described [below](#restarting) to continue processing only the missing file(s).
Beware that for multistep searches, fixed errors in earlier steps require re-running all subsequent steps.

In case you want AlphaDIA to stop processing immediately when an error occurs, you can set the config option `general.fail_fast` to `True`.

### Restarting
During the main search, alphaDIA processes each raw file independently.
After each file, quantification results are saved to `<output_folder>/quant/<raw_file_name>`,
which can be used as a checkpoint in case the processing is interrupted.

The config switch `general.reuse_quant` enables skipping raw file processing
when quantification results already exist, which is useful for
distributed searches or for re-running the consensus step with protein inference, FDR and LFQ quantification with different parameters.

When enabled: Before processing each raw file, checks if quantification results already exist for that
file (by default looking into `<output_folder>/quant`).
If so, skips processing entirely and reuses existing quantification.
If not, the file is being searched.
After all quantifications are available, the workflow continues normally, combining results from all files.
This way, an alphaDIA run that failed at file 9/10 (e.g. due to a cluster timeout) can simply be restarted,
as only the missing files (9 and 10) will be processed.

The `--quant-dir` CLI parameter (Config: `quant_directory`, default: `null`)
can be used to specify a directory containing quantification results different from `<output_folder>/quant`.
Note: this parameter is not supported with multistep search as each search step has its own quant directory.
If you want more detailed control over this, build a custom multistep workflow using the concepts of [distributed search](./dist_search_setup.md).

On startup, the current configuration is dumped as `frozen_config.yaml`, which contains all information to reproduce this run.

Combining these three concepts, here's an example how to reuse an existing quantification (from the `previous_run` directory), but create additional
output (`peptide_level_lfq`) through a custom `--config-dict`:
```
alphadia -o ./output_dir --quant-dir ./previous_run/quant --config ./previous_run/frozen_config.yaml --config-dict '{"general": {"reuse_quant": "True"}, "search_output": {"peptide_level_lfq": "True"}}'
```

Cf. also the documentation on [distributed search](./dist_search_setup.md).
