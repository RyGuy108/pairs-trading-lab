# Security policy

## Credentials

The project reads Alpaca credentials from environment variables or an untracked
`.env` file. Never commit real values. `.env` is ignored; `.env.example` contains
placeholders only.

If a key is exposed:

1. Revoke it immediately in the Alpaca dashboard.
2. Generate a replacement.
3. Remove the secret from Git history before publishing the repository.
4. Review paper-account orders and activity.

## Execution boundary

The included trading client is hardcoded with `paper=True`. There is no supported
live-trading mode. Pair legs are submitted sequentially, not atomically, so even
paper execution can demonstrate partial-fill and leg-imbalance failure modes.

Do not add live execution without fill reconciliation, short-availability checks,
position limits, idempotent order handling, monitoring, and an emergency flattening
procedure.

## Reporting a vulnerability

For a public repository, report security issues privately through GitHub's
security-advisory feature. Do not publish credentials or exploitable details in a
public issue.
