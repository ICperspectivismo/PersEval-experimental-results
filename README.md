# PersEval paper reproduction and results

This repository stores PersEval paper results and new experiments, and hosts the
leaderboard on GitHub Pages. Experiments run in the separate
[PersEval](https://github.com/lebraga1/PersEval) benchmark.

## Installation

Use Python 3.11 or newer in a separate environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, replace the activation command with:
* Command Prompt (CMD): .venv\Scripts\activate.bat
* PowerShell: .\.venv\Scripts\Activate.ps1

Configure your Git author with `git config --local user.name` and
`git config --local user.email`. Before the first publication, maintainers must
review and commit the repository setup and configure its GitHub remote.
Install Git LFS and run `git lfs install --local` if importing predictions or
artifacts of at least 10 MiB.

## 1. Export from PersEval

In your experiment notebook after training:

```python
model.predict(trainer)
bundle = model.last_run.export_result(output_dir="exports/")
print(bundle)
```

Copy the resulting `.perseval-result.tar.gz` file to your computer. In Colab:

```python
from google.colab import files
files.download(str(bundle))
```

Export is explicit and optional. It does not publish or upload anything.

## 2. Publish results

In an up-to-date, clean checkout of this repository, with its environment active:

```sh
perseval-results publish-bundle /path/to/result.perseval-result.tar.gz \
  --name "EPIC RoBERTa reproduction" --submitter "UFMG" --commit
```

Review and confirm the printed plan. The command validates the archive, stores
the run, creates a submission, calculates rankings, builds a local preview, and
commits the result records. Keep downloaded bundles outside the checkout.
Add `--dry-run` to preview without writing, or `--yes` for noninteractive use.

## 3. Push

```sh
git push
```

GitHub Actions validates the records, builds the leaderboard, and deploys
GitHub Pages. Enable Pages once in the repository settings, selecting
**GitHub Actions** as the deployment source.

## Multiple seeds

Pass compatible runs together to publish their mean, sample standard deviation,
and run count. A single run has no standard deviation and no best seed is selected.

```sh
perseval-results publish-bundle /downloads/seed-42.perseval-result.tar.gz \
  /downloads/seed-43.perseval-result.tar.gz \
  --name "EPIC RoBERTa reproduction" --submitter "UFMG" --commit
```

To add another run, use the submission ID printed by the first command:

```sh
perseval-results publish-bundle /downloads/seed-44.perseval-result.tar.gz \
  --submission sub_<uuid> --commit
```

Updates preserve previous revisions. An exact publication retry creates no
duplicate entry or empty commit.

## Local preview

```sh
perseval-results site build --output site-build/
perseval-results site serve --output site-build/ --port 8000
```

Open `http://127.0.0.1:8000/`. The generated `site-build/` directory is ignored by Git.

## Paper results and historical evidence

`catalog/` describes the paper experiments; `papers/` records their reported
values and citations. The checksummed snapshot at `historical/perseval-97cf3ae/`
preserves the original PersEval commit. These are separate from new measured runs.

```sh
perseval-results catalog list --scope main
perseval-results catalog show <experiment-id>
perseval-results historical audit
```

Measured entries use macro-F1 and rank within compatible test/scoring groups.
Reported paper values retain their original scoring rules and separate rankings.
