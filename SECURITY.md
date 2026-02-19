# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in OMEGA, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, email us at: **omega-memory@proton.me**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will acknowledge receipt within 48 hours and provide a timeline for a fix. We aim to release a patch within 90 days of a confirmed vulnerability. We follow coordinated disclosure — we will work with you on timing before any public announcement. We are happy to credit reporters in the release notes unless you prefer to remain anonymous.

## Scope

This policy covers the `omega-memory` Python package and its MCP server. Vulnerabilities in third-party dependencies should be reported to the respective maintainers, though we welcome a heads-up if you believe a dependency vulnerability affects OMEGA users.

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.8.x   | Yes                |
| 0.7.x   | Yes                |
| 0.6.x   | Yes                |
| 0.5.x   | Yes                |
| < 0.5   | No                 |

## Security Design

OMEGA is designed with security in mind:

- **Local-first**: All data stays on your machine by default. No external network calls for core functionality.
- **Encryption at rest**: Optional AES-256-GCM encryption for stored memories (`OMEGA_ENCRYPT=1`).
- **File permissions**: Database and key files are created with `0o600` permissions.
- **No credential storage**: OMEGA never stores API keys or credentials in its database.
- **Rate limiting**: The MCP server includes built-in rate limiting to prevent abuse.
