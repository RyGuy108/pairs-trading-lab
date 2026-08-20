# Publishing the repository to GitHub

The local repository is configured on the `main` branch. Generated outputs,
downloaded data, virtual environments, caches, `.env`, and API keys are excluded
by `.gitignore`.

## Option A: GitHub website and Git

1. On GitHub, create a new repository named `pairs-trading-lab`.
2. Do **not** initialize it with a README, `.gitignore`, or license; those files
   already exist locally.
3. Copy the repository's HTTPS or SSH URL.
4. From this project directory, run:

```bash
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

Example URL formats:

```text
https://github.com/YOUR_USERNAME/pairs-trading-lab.git
git@github.com:YOUR_USERNAME/pairs-trading-lab.git
```

## Option B: GitHub CLI

If the GitHub CLI is installed and authenticated:

```bash
gh repo create pairs-trading-lab --public --source=. --remote=origin --push
```

Use `--private` instead of `--public` if you do not want the project visible.

## Verify before the first push

Inspect exactly what Git tracks:

```bash
git status
git ls-files
```

Confirm that secrets are absent:

```bash
git grep -n "ALPACA_API_KEY="
git grep -n "ALPACA_SECRET_KEY="
```

Only placeholder values in `.env.example` should appear. The real `.env` file
must not be listed by `git ls-files`.

Run the same checks as GitHub Actions:

```bash
make check
```

## What happens after pushing

`.github/workflows/ci.yml` runs on each push and pull request. It tests Python
3.11, 3.13, and 3.14, with read-only repository permissions. Each job:

1. Installs the project and development dependencies.
2. Runs Ruff.
3. Runs all unit tests.
4. Builds the source distribution and wheel.

Open the repository's **Actions** tab after the first push. A green CI run proves
that another clean machine can install, test, and package the project.

## Suggested repository settings

After the first successful push:

- Add the description: `Leakage-aware educational pairs-trading research lab`
- Add topics: `python`, `quantitative-finance`, `pairs-trading`, `cointegration`,
  `algorithmic-trading`, `alpaca`
- Enable private vulnerability reporting if the repository is public
- Protect `main` and require the CI check if other people will contribute

Do not add Alpaca credentials as GitHub repository secrets unless a future
workflow genuinely requires them. The current CI is fully offline and needs no
brokerage access.
