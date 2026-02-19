# Technical Debt

## 2026-02-19: MCP Transport object cleanup (stateless mode)

**What:** In stateless mode, each `/mcp` request creates a fresh `Server` + `StreamableHTTPServerTransport`
pair. Currently these objects are released to GC after the request completes. Explicit `close()` calls
were tested but caused test failures (transport closes SSE stream before data reaches client).

**Why deferred:** Low value for single-user setup. Stateless transport holds no system resources
(no timers, no file descriptors). GC reclaims the objects after each request.

**Impact if not done:** Negligible for low-volume usage. Could become relevant at high request rates
if Node.js GC pressure increases. No memory leaks observed in testing.

**Effort when ready:** Known pattern — needs investigation of correct timing for `transport.close()`
relative to SSE response completion. Possible approach: `res.on('finish', () => transport.close())`.

**Context:** `mcp-sse-server/server.js` lines 605-615. Introduced in stateless mode refactor.

---

## 2026-02-19: SSE session sweep (ungraceful disconnects)

**What:** The old stateful code had a TTL sweep for Streamable HTTP sessions. This sweep was removed
(no longer needed in stateless mode). The legacy SSE sessions (`/mcp/sse`) use `res.on('close')`
for cleanup — works for normal disconnects but not for ungraceful network failures.

**Why deferred:** SSE transport (`/mcp/sse`) is deprecated. Low value to fix cleanup for a
deprecated endpoint. Single-user setup doesn't accumulate sessions.

**Impact if not done:** Ungraceful network disconnects may leave orphaned session entries in the
`sessions` Map. In practice negligible — sessions are small objects and `/mcp/sse` is rarely used.

**Effort when ready:** Known pattern — add a TTL sweep for `type === 'sse'` sessions only.
No session.lastAccess tracking currently added for SSE sessions (would need to be added).

**Context:** `mcp-sse-server/server.js` lines 425-430 (sessions Map declaration).

---

## 2026-02-19: Per-endpoint rate limiting (M-4 follow-up)

**What:** Rate limiting currently applies a single global default (1000/hour per IP).
The remediation plan called for differentiated limits: write endpoints (100/h),
read endpoints (500/h), admin endpoints (20/h). These could not be applied because
all routes are defined inside blueprint factory closures (dependency injection pattern),
and flask-limiter's `@limiter.limit()` decorator cannot be applied post-registration.

**Why deferred:** Global 1000/hour limit still provides meaningful protection against
abuse. Per-endpoint limits require either: (a) injecting `limiter` into all 8 blueprint
factories, or (b) using flask-limiter's `request_filter` approach with URL inspection.
Risk of introducing bugs across 8 factories is non-trivial.

**Impact if not done:** Admin endpoints (/admin/reembed, /admin/sync) and write endpoints
(/memory POST, /associate POST) are somewhat easier to abuse than with per-endpoint limits.
Still protected by auth token requirement.

**Effort when ready:** Medium value, medium risk. Requires adding `limiter` parameter to
all blueprint factory functions or switching to URL-based filter approach.

**Context:** `app.py` (Limiter setup, lines ~115-140), `automem/api/` (8 blueprint factories).
Env vars RATE_LIMIT_WRITE_ENDPOINTS, RATE_LIMIT_READ_ENDPOINTS, RATE_LIMIT_ADMIN_ENDPOINTS
are available but unused — ready for when the refactoring happens.
