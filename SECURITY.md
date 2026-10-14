# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

0.1.0 is the first published version. The earlier numbers in `CHANGELOG.md` (up to an
internal 1.0.0) were development milestones that were never released.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

To report a vulnerability, email **wujian.thu@gmail.com** with the subject line
`[QuantX Security] <brief description>`.

Include as much of the following as possible:

- Type of issue (e.g. arbitrary code execution, dependency confusion, path traversal)
- Full paths of any source files involved
- Steps to reproduce
- Proof-of-concept or exploit code (if available)
- Impact assessment — what an attacker could achieve

You will receive an acknowledgement within **48 hours** and a resolution timeline
within **7 days**.

## Scope

In scope:
- `quantx-engine` Python package
- Native C++ engine (`libquantx`)
- REST API server (`server/`)
- GitHub Actions workflows

Out of scope:
- Third-party dependencies (report upstream)
- Issues requiring physical access to hardware backends
- Denial-of-service via intentionally large quantum circuits

## Disclosure Policy

We follow coordinated disclosure. Once a fix is ready we will:

1. Release a patched version to PyPI
2. Credit the reporter in the release notes (unless anonymity is requested)
3. Publish a GitHub Security Advisory

## Preferred Languages

Reports in English or Chinese are both welcome.
