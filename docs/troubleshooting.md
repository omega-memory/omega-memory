# Troubleshooting

Common issues and solutions when using OMEGA.

---

## Model Download Fails

**Symptom**: First query hangs or errors with "Failed to download model" or connection timeout.

**Cause**: OMEGA downloads the bge-small-en-v1.5 ONNX model (~90 MB) on first use to `~/.cache/omega/models/`.

**Solutions**:

1. **Proxy or firewall**: If behind a corporate proxy, set `HTTPS_PROXY` before starting the MCP server:
   ```bash
   export HTTPS_PROXY=http://proxy.example.com:8080
   ```

2. **Disk space**: Ensure at least 200 MB free in `~/.cache/omega/`.

3. **Manual download**: Download the model files manually from [Hugging Face](https://huggingface.co/BAAI/bge-small-en-v1.5) and place them in `~/.cache/omega/models/bge-small-en-v1.5-onnx/`.

4. **Verify with doctor**: Run `omega doctor` to check model status.

---

## ONNX Runtime Not Found

**Symptom**: `ImportError: onnxruntime not found` or embedding generation fails.

**Cause**: The `onnxruntime` package is missing from your Python environment.

**Solution**:

```bash
pip install onnxruntime
```

Or install OMEGA with all dependencies:

```bash
pip install "omega-memory[all]"
```

Note: CoreML acceleration is intentionally disabled due to a memory leak in Apple's ANE runtime. CPU-only inference is used.

---

## sqlite-vec Not Available

**Symptom**: Warning about "sqlite-vec not available, using hash-based fallback" or degraded search quality.

**Cause**: The `sqlite-vec` extension (used for vector similarity search) isn't installed or can't be loaded.

**Impact**: OMEGA falls back to hash-based approximate nearest neighbors. Search still works but with lower accuracy for semantic queries.

**Solution**:

```bash
pip install sqlite-vec
```

If `pip install` fails (e.g., no wheel for your platform), OMEGA will continue to function with the fallback. Run `omega doctor` to verify the status.

---

## High RSS Memory (300-400 MB)

**Symptom**: `omega_health` reports "critical" RSS memory at 300-400 MB, or system monitor shows high memory usage.

**Cause**: This is **expected behavior**, not a memory leak. The ONNX embedding model loads ~300 MB into RAM on first semantic query.

**Lifecycle**:

```
~31 MB idle → ~337 MB after first query → ~31 MB after 10 min idle
```

The model auto-unloads after 10 minutes without queries. The health check critical threshold is set to 800 MB to avoid false alarms during normal peak usage.

**When to worry**: If RSS stays above 500 MB with no active queries for more than 15 minutes, or if it grows unboundedly over time, that may indicate an actual issue. File a bug report with the output of `omega_health`.

---

## Database Locked

**Symptom**: `sqlite3.OperationalError: database is locked`

**Cause**: Multiple processes are trying to write to `~/.omega/omega.db` simultaneously. SQLite handles concurrent reads but serializes writes.

**Solutions**:

1. **Check for stuck processes**:
   ```bash
   ps aux | grep omega
   ```
   Kill any orphaned OMEGA server processes.

2. **WAL mode**: OMEGA uses WAL mode by default, which allows concurrent reads during writes. If your database isn't in WAL mode:
   ```bash
   omega doctor
   ```
   This will report the journal mode and fix it if needed.

---

## Hook Daemon Not Running

**Symptom**: `~/.omega/hooks.log` shows core hooks as `OK (0ms, skipped)`,
memories are not captured or surfaced during a session, or `omega doctor`
warns that an MCP server is running but no hook socket exists.

**Cause**: The hook daemon runs inside the MCP server process; there is no
separate service to start. `fast_hook.py` reaches it over
`~/.omega/hook.sock` (TCP loopback on Windows). When the socket is missing,
hooks fall back to a cold path that runs only the safety guards and
best-effort captures, so `session_start`, `session_stop`, `auto_capture`,
and `surface_memories` are skipped.

**Solutions**:

1. **Check status**:
   ```bash
   omega doctor
   ```
   The "Hook Daemon" section reports whether the socket is listening, absent,
   or stale, and how many core hook runs were skipped recently.

2. **Restart the session**: the daemon starts with the MCP server, so restart
   Claude Code (or whichever client launched the server).

3. **Stale socket**: a socket file nobody answers on is normal after a server
   was killed (`claude mcp list` does this while probing). The next server
   replaces it. Only if `omega doctor` warns that a server is running *and*
   nothing answers, remove the file and restart the session:
   ```bash
   rm ~/.omega/hook.sock
   ```

4. **Still skipped**: if `omega doctor` reports the daemon module as not
   importable, reinstall the package. Releases before 1.5.16 shipped without
   the daemon (issue #76):
   ```bash
   pip install -U "omega-memory[server]"
   ```

Note: Hooks fail open. If the daemon is unreachable, Claude Code keeps working
normally; you only lose automatic capture and surfacing until it is back.

---

## Setup Issues

For initial setup problems, run the diagnostic tool:

```bash
omega doctor
```

This checks:
- Python version compatibility
- Required and optional dependencies
- Database status and schema version
- Model availability
- Hook server status
- Disk space and permissions
