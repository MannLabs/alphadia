# AlphaDIA — Initial Setup Friction Notes

Date: 2026-06-24
Goal: Clone github.com/MannLabs/alphadia, install into a new conda environment, run the basic e2e test.
Perspective: A new user following the documentation for the first time.

These notes capture **unclarities**, **errors**, and **documentation gaps** encountered during setup, to help reduce friction for new users.

---

## Environment (starting point)
- OS: Ubuntu (Linux 6.8.0)
- Python (system): 3.10.12 at /usr/bin/python3
- git: 2.34.1
- conda/mamba: **NOT installed** (see Friction #1)

---

## Friction log

### #1 — `conda` is not installed at all
- **Type:** Error / prerequisite gap
- The task and the docs assume conda is available, but `conda`, `mamba`, miniconda, anaconda, miniforge are all absent from this machine.
- A brand-new user on a clean machine hits this wall immediately. The alphadia docs say "create a conda environment" but do not link to / explain how to install conda (miniconda/miniforge) in the first place.
- **Suggestion:** The installation docs should start with a one-line pointer to install Miniforge/Miniconda (with a link), for users who don't already have it.
- **Resolution:** I installed Miniforge into `~/miniforge3` to proceed.
USER_COMMENT: add the one-liner

### #2 — The "basic" e2e test requires Mono (not flagged at the e2e step)
- **Type:** Documentation gap / hidden prerequisite
- The `basic` e2e test case (`tests/e2e_tests/e2e_test_cases.yaml`) downloads a Thermo **`.raw`** file. Reading `.raw` requires `alpharaw` + **Mono**.
- The Mono requirement is mentioned in the pip/developer install sections ("if you want to use `.raw` files"), but the AGENTS.md "Running Tests → End-to-end" instructions (`./run_e2e_tests.sh basic <env>`) do **not** mention that the basic case needs Mono. A new user running the documented e2e command in a fresh env will hit a `.raw` read failure with no upfront warning.
- **Suggestion:** Note in the e2e test docs that `basic` needs Mono, or provide a `.raw`-free smallest test case as the true "hello world".
USER_COMMENT: no action

### #3 — Two different recommended env names across docs
- **Type:** Unclarity / minor inconsistency
- `docs/installation.md` uses `conda create -n alphadia ...`
- `AGENTS.md` uses `conda create -n alphadia_env ...`
- The GUI install note also hard-codes that the env must be named `alphadia` (unless `profile.js` is edited).
- **Suggestion:** Standardize the env name across docs, or call out explicitly that the name is arbitrary for CLI use.
USER_COMMENT: no action

### #4 — `alphadia --check` reports success even when Thermo `.raw` reading is broken
- **Type:** Unclarity / misleading success signal
- After install, `alphadia --check` prints `Importing AlphaDIA works!` and exits 0, **but** earlier in the same output it emits:
  ```
  UserWarning: Dotnet-based dependencies could not be loaded. Thermo support is disabled.
  UserWarning: Dotnet-based dependencies could not be loaded. Sciex support is disabled.
  ```
  These are easy to miss (buried above the success line). A new user would reasonably conclude they're ready to run, then fail when the basic e2e test tries to read a `.raw` file.
- **Suggestion:** `--check` could explicitly report vendor-format support status (Thermo/Sciex enabled/disabled) as part of its summary, rather than only as buried warnings.
USER_COMMENT: report vendor-format support status

### #5 — Mono must be installed separately; not part of the pip/editable install
- **Type:** Prerequisite gap
- Mono is required for `.raw` reading but is not pulled in by `pip install alphadia[stable]`. It must be installed via `conda install mono=6.12.0.182 -c anaconda -y` (per the Slurm section) or via the AlphaRaw guide.
- This is documented but spread across sections; a linear reader doing "pip install → run" misses it.
- **Resolution:** Installed mono into the env.
USER_COMMENT: no action

### #6 — `docs/quickstart.md` "Quickstart using CLI" has a wrong path: `cd test`
- **Type:** Error (copy/paste bug in docs)
- Step 3 says: `cd into the root folder of the repository, then `cd test``. The directory is **`tests`** (plural). `cd test` fails with "No such file or directory".
- The same section also does not mention installing Mono, even though `./run_e2e_tests.sh basic alphadia` downloads a `.raw` file that needs it.
- **Suggestion:** Fix `cd test` → `cd tests`, and add a one-line Mono prerequisite to the CLI quickstart.
USER_COMMENT: no action

### #7 — Minor typos in docs
- **Type:** Documentation polish
- `docs/quickstart.md` line ~20: "select the **thre** `.raw` files" → "three".
- **Suggestion:** Spell-check pass on docs.
USER_COMMENT: no action

### #8 — `run_e2e_tests.sh` requires `conda` on PATH, but is undocumented as such
- **Type:** Unclarity
- The script calls `conda run -n <env> ...` directly. If the user's shell hasn't had `conda init` run (so `conda` isn't a shell function/binary on PATH), the script fails immediately. The docs don't state this dependency; they assume an initialized conda shell.
- **Suggestion:** Mention that the e2e script needs `conda` available on PATH (i.e. run `conda init` / use a conda-initialized shell first).
USER_COMMENT: add a comment

### #9 — The "basic" test downloads ~3 GB (no size warning, no progress in log)
- **Type:** Unclarity / UX
- The `basic` case downloads a 2.9 GB `.raw` file + a 94 MB library before anything runs. There is no upfront note of the download size or disk requirement, and `conda run` buffers stdout so the run log appears empty during the multi-minute download — looks like a hang.
- **Suggestion:** State the expected download size / disk + time in the e2e docs, and/or surface download progress.
USER_COMMENT: add a comment

### #10 — Noisy warnings during the run (cosmetic, but alarming to newcomers)
- **Type:** UX / polish
- The run prints many repeated warnings that a first-time user may mistake for errors:
  - numpy: `UserWarning: Signature ... for <class 'numpy.longdouble'> does not match any known type ... This warnings indicates broken support for the dtype!` (printed ~once per worker; harmless, known numpy probe issue).
  - alphabase: `DeprecationWarning: Support for whitespaces in modifications will be dropped ...`
- **Suggestion:** Filter/suppress the known-benign numpy longdouble warning, or note in docs that these are expected.
USER_COMMENT: add a comment

### #11 — ❗ The e2e `calc_metrics.py` step is broken / out of sync with current output schema (HIGH)
- **Type:** Error (the most impactful one — the documented "basic e2e test" does not fully pass)
- The **search itself succeeds** and writes all outputs (precursors, matrices, MBR library; 101,529 precursors / 8,368 proteins). But the final metrics step fails:
  ```
  loading basic/output/stat.parquet
  Exception calculating metrics:  [Errno 2] No such file or directory: 'basic/output/stat.parquet'
  ```
- **Two distinct mismatches** between `tests/e2e_tests/calc_metrics.py` and the current alphaDIA output:
  1. **Filename:** `OutputFiles.STAT = "stat.parquet"` and it is read via `_load_parquet`, but alphaDIA writes **`stat.tsv`** (TSV). (Note the file's own docstring on line ~67 still says `DataStore(...)["stat.tsv"]` — internal inconsistency.)
  2. **Column names:** even with the file found, `BasicStats._calc()` requests columns that no longer exist:
     - `proteins`   → now `search.proteins`
     - `precursors` → now `search.precursors`
     - `calibration.ms2_median_accuracy` / `..._precision` → now `calibration.ms2_bias` / `calibration.ms2_variance` (same for ms1)
     - Only the 4 `optimization.{ms1,ms2,rt,mobility}_error` columns still match.
- **Severity:** This is what a new user runs to confirm their setup works. It silently "passes" (exit code 0) because `calc_metrics.py` wraps everything in `try/except` and only prints the exception — so neither a human nor CI would notice the metrics never computed.
- **Suggestions:**
  - Update `calc_metrics.py` to read `stat.tsv` (or make alphaDIA emit `stat.parquet` when `search_output.file_format: parquet`) and to use the current `search.*` / `calibration.*_bias|variance` column names.
  - Make the metrics step fail loudly (non-zero exit) instead of swallowing exceptions, so schema drift is caught.
  - Add a CI test that asserts `calc_metrics.py` produces the expected keys, to prevent future drift.
USER_COMMENT: fix this
