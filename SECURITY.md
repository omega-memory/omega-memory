# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.0.x   | Yes                |
| < 1.0   | No                 |

## Reporting a Vulnerability

If you discover a security vulnerability in OMEGA, please report it responsibly.

**Do NOT open a public issue.**

Instead, use one of these methods:

1. **GitHub Security Advisories** (preferred): Go to the [Security Advisories page](https://github.com/omega-memory/omega-memory/security/advisories/new) and create a private advisory.
2. **Email**: Contact the maintainers at hello@omega-memory.dev.

### What to include

- Description of the vulnerability
- Steps to reproduce
- Affected versions
- Potential impact

### Response timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 1 week
- **Fix or mitigation**: Depends on severity, targeting within 2 weeks for critical issues

### Scope

This policy covers the `omega-memory` Python package only. Issues with Claude Code itself should be reported to [Anthropic](https://github.com/anthropics/claude-code/issues).

### Security design

- All SQL queries use parameterized statements (no string interpolation)
- File paths are validated against traversal attacks on export/import
- The optional encryption layer uses `cryptography` with `secrets.token_bytes()` for key generation
- The encryption key is stored at `~/.omega/.key` with restricted permissions
- The UDS hook socket is created with mode `0o600`
