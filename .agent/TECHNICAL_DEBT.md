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
