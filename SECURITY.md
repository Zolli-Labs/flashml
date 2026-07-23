# Security Policy

FlashRuntime runs untrusted workloads on heterogeneous, sometimes untrusted,
machines. We take security seriously and appreciate responsible disclosure.

## Supported versions

FlashRuntime is pre-1.0. Security fixes land on `main` and in the latest
`0.x` release. Pin a version you have reviewed; upgrade promptly when a
security release is published.

| Version | Supported            |
| ------- | -------------------- |
| 0.1.x   | ✅ (current)         |
| < 0.1   | ❌                   |

## Reporting a vulnerability — privately

**Do not open a public issue for a security vulnerability.** Public issues are
visible to everyone before a fix exists.

Instead, report privately through **GitHub's private vulnerability reporting**:

> Repository → **Security** tab → **Report a vulnerability**
> (https://github.com/Zolli-Labs/flashruntime/security/advisories/new)

This opens a private advisory only the maintainers can see. If you cannot use
GitHub, email **security@zolli-labs.com** (an org alias that reaches the
maintainers) — encrypt if you can, and never include a working exploit against
third-party infrastructure.

Please include:

- the affected component and version / commit,
- a description of the impact (what an attacker gains),
- reproduction steps or a minimal proof of concept,
- any suggested remediation.

### What to expect

- **Acknowledgement** within 3 business days.
- An initial assessment and severity within 7 business days.
- Coordinated disclosure: we agree on a timeline with you, fix privately, then
  publish an advisory crediting you (unless you prefer to remain anonymous).

Please give us a reasonable window to remediate before any public disclosure.

## Our security philosophy: fail closed

Security-relevant decisions in FlashRuntime **deny by default**. This is a
design rule, not a case-by-case choice:

- A missing, malformed, or unverifiable security-relevant field **denies** the
  operation — it never falls through to a permissive default.
- Node registration is gated by join codes where configured; an absent or
  invalid code is refused, not waved through.
- Artifact and checkpoint commits are **verified before they are trusted**:
  parts are written first, hashes are checked against the task's committed
  key, and the manifest is written *last*. No manifest → no checkpoint. A hash
  mismatch rejects the commit rather than accepting unverified bytes.
- Recovery actions are typed, deterministic, and logged — never driven by an
  LLM or by unbounded automation. Correlated incidents freeze automated
  recovery instead of amplifying a failure.
- Untrusted nodes are isolated by tier (sandbox/VM), gated by the isolation
  tier rather than a code allowlist.

## Secrets

Secrets never live in the repository. They belong only in a local, gitignored
`.env`. `scripts/audit_secrets.sh` scans the full git history and the tracked
worktree for credential-shaped strings (RunPod, AWS, GitHub tokens, private
keys, OpenAI-style keys) and asserts `.env` is untracked; it runs clean on
`main` and should stay that way. If you believe a secret was ever committed,
report it privately using the process above.
