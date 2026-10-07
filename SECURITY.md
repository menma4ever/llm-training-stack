# Security Policy

## 🛡️ Supported Versions

| Version | Supported |
|---|---|
| 0.1.x | Yes |
| < 0.1.0 | No |

---

## 🔒 Reporting a Vulnerability

We take the security of LLM training workloads, weights, and orchestration infrastructure seriously. If you discover a security vulnerability (such as pickle deserialization risks, unauthorized code execution paths, or MCP permission bypasses):

1. **Do Not Open a Public Issue**: Please do not file public GitHub issues for security vulnerabilities.
2. **Responsible Disclosure**: Contact the project maintainers directly through secure channels or private repository vulnerability reporting on GitHub.
3. **Assessment & Patching**: The team will acknowledge receipt of the report, assess the impact, and release a patched version promptly.

---

## 🔐 Model Context Protocol (MCP) Boundaries

The `llm-training-stack` MCP server is designed with the principle of least privilege:
- Read-only tools cannot modify system state or allocate persistent compute resources.
- Training jobs require explicit `authorized=True` confirmation.
- Arbitrary shell execution and unvalidated Python `eval` tools are strictly prohibited.
