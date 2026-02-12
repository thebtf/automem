# AutoMem Code Review Remediation Plan

**Date:** 2026-02-12
**Based on:** .agent/code-review-report.md
**Total Issues:** 25 (5 CRITICAL, 6 HIGH, 10 MEDIUM, 4 LOW)

---

## Overview

This plan addresses all findings from the comprehensive code review in prioritized phases:
- **Phase 1:** Security Hardening (6 tasks) - CRITICAL/HIGH security issues
- **Phase 2:** Code Quality (5 tasks) - HIGH maintainability issues
- **Phase 3:** Operational Improvements (6 tasks) - MEDIUM reliability issues
- **Phase 4:** Polish (4 tasks) - LOW quality-of-life improvements

---

## Phase 1: Security Hardening (IMMEDIATE)

### Task 1.1: Fix .gitignore to prevent secret leakage (C-1)

**Issue:** `.env.local` not in `.gitignore`, contains hardcoded tokens at risk of accidental commit.

**Steps:**
1. Add `.env.*` pattern to `.gitignore`
2. Verify `.env.local` remains untracked: `git status`
3. Add warning comment in `.gitignore` about secret files
4. Commit change

**Files:**
- `.gitignore`

**Tests:** Verify `git status` shows no `.env.local`

---

### Task 1.2: Replace timing-vulnerable token comparisons (C-2, C-3)

**Issue:** API and admin token comparisons use `!=` instead of constant-time comparison, enabling timing attacks.

**Steps:**
1. Import `hmac` module in `app.py`
2. Replace `token != API_TOKEN` at line 1175 with `hmac.compare_digest(token or "", API_TOKEN or "")`
3. Replace `provided != ADMIN_TOKEN` at line 1160 with `hmac.compare_digest(provided or "", ADMIN_TOKEN or "")`
4. Add test case for empty token handling
5. Commit change

**Files:**
- `app.py:1160` (admin token check)
- `app.py:1175` (API token check)

**Tests:**
- Verify authentication still works with valid tokens
- Verify 401 returned for invalid tokens
- Verify empty tokens handled safely

---

### Task 1.3: Remove client-supplied memory IDs (C-4)

**Issue:** Legacy `/memory` POST endpoint accepts client-supplied IDs, enabling memory tampering.

**Steps:**
1. Remove `payload.get("id") or` from `app.py:2588`
2. Force server-side UUID generation: `memory_id = str(uuid.uuid4())`
3. Add comment documenting security rationale
4. Test memory creation still works
5. Commit change

**Files:**
- `app.py:2588`

**Tests:**
- POST /memory with `id` field → should ignore and generate new UUID
- Verify existing memories cannot be overwritten

---

### Task 1.4: Remove exception details from API responses (C-5, M-6)

**Issue:** 9 locations leak internal exception details via `str(e)` in JSON/abort responses.

**Steps:**
1. For each location, log exception server-side: `logger.exception("Operation failed")`
2. Return generic error message: `{"error": "Operation failed"}` (no details)
3. Locations to fix:
   - `app.py:3211` → consolidation
   - `app.py:3243` → analyze
   - `app.py:3310` → /associate
   - `automem/api/recall.py:1611` → startup-recall
   - `automem/api/recall.py:1762` → related-memories
   - `automem/api/consolidation.py:41` → POST /consolidate
   - `automem/api/consolidation.py:69` → GET /consolidate/status
   - `automem/api/graph.py:204` → graph/snapshot
   - `automem/api/graph.py:452` → graph/neighborhood
4. Run integration tests to verify error responses
5. Commit changes

**Files:**
- `app.py` (3 locations)
- `automem/api/recall.py` (2 locations)
- `automem/api/consolidation.py` (2 locations)
- `automem/api/graph.py` (2 locations)

**Tests:**
- Trigger error conditions
- Verify response contains generic message, not exception details
- Verify logs contain full exception stack trace

---

### Task 1.5: Deprecate query parameter token authentication (M-3)

**Issue:** Accepting tokens via query params (`?api_key=`) leaks credentials in logs/history.

**Steps:**
1. Add deprecation warning when token provided via query param
2. Document in CHANGELOG.md that query param auth will be removed in next major version
3. Update API documentation to show only header-based auth
4. Add log warning: `logger.warning("Token via query param is deprecated")`
5. Commit changes

**Files:**
- `app.py` (_extract_api_token function)
- `CHANGELOG.md`
- `docs/API.md`

**Tests:**
- Verify header auth still works
- Verify query param auth still works (with warning logged)

---

### Task 1.6: Update requests dependency (M-9)

**Issue:** `requests==2.31.0` vulnerable to CVE-2024-35195 (session cookie leakage).

**Steps:**
1. Update `requirements.txt`: `requests>=2.32.3`
2. Run `pip install --upgrade requests` in venv
3. Run full test suite to verify no regressions
4. Commit change

**Files:**
- `requirements.txt`

**Tests:** Full test suite must pass

---

## Phase 2: Code Quality (NEAR-TERM)

### Task 2.1: Remove legacy route handlers (H-4)

**Issue:** Massive code duplication between `app.py` legacy routes and `automem/api/` blueprints.

**Steps:**
1. Create integration test suite comparing legacy vs blueprint responses
2. Verify 100% behavioral parity for all endpoints:
   - POST /memory
   - GET /recall
   - POST /associate
   - PATCH /memory/<id>
   - DELETE /memory/<id>
   - GET /memory/by-tag
   - POST /consolidate
   - GET /analyze
3. Remove legacy route handlers from `app.py` (12+ functions)
4. Run full test suite
5. Commit removal

**Files:**
- `app.py` (remove ~2000 lines of legacy routes)
- `tests/` (add parity tests)

**Tests:**
- Integration tests verify blueprint equivalence
- All existing tests pass after removal

---

### Task 2.2: Extract app.py concerns to modules (H-1)

**Issue:** `app.py` is 3853 lines (5x threshold) with mixed concerns.

**Steps:**
1. Create `automem/workers/` directory
2. Move embedding worker → `automem/workers/embedding.py`
3. Move enrichment worker → `automem/workers/enrichment.py`
4. Move consolidation worker → `automem/workers/consolidation.py`
5. Move sync worker → `automem/workers/sync.py`
6. Create `automem/db.py` and move database initialization
7. Create `automem/auth.py` and move authentication functions
8. Update imports in `app.py`
9. Target: `app.py` under 500 lines (factory function + config + worker startup)
10. Run full test suite
11. Commit refactoring

**Files:**
- `app.py` (reduce from 3853 → ~400 lines)
- `automem/workers/` (new directory)
- `automem/db.py` (new file)
- `automem/auth.py` (new file)

**Tests:** Full test suite must pass, no behavioral changes

---

### Task 2.3: Split recall.py into focused modules (H-2)

**Issue:** `automem/api/recall.py` is 1851 lines (2.3x threshold).

**Steps:**
1. Create `automem/api/recall/` directory
2. Extract modules:
   - `recall/core.py` → core recall endpoint (~400 lines)
   - `recall/startup.py` → startup-recall logic
   - `recall/analyze.py` → analytics endpoint
   - `recall/expansion.py` → entity/relation expansion
   - `recall/related.py` → related-memories endpoint
3. Update blueprint registration in `app.py`
4. Run recall-specific tests
5. Commit refactoring

**Files:**
- `automem/api/recall.py` → `automem/api/recall/` (directory)
- `app.py` (update imports)

**Tests:** All recall tests pass

---

### Task 2.4: Add input validation for graph API query params (H-5)

**Issue:** Uncaught `ValueError` on non-numeric query params in 3 locations.

**Steps:**
1. Create validation helper: `validate_int_param(value, default, min_val, max_val)`
2. Wrap `int()` calls at:
   - `graph.py:73` (limit parameter)
   - `graph.py:219` (depth parameter)
   - `graph.py:221` (semantic_limit parameter)
3. Return 400 Bad Request with clear message on invalid input
4. Add test cases for invalid inputs
5. Commit changes

**Files:**
- `automem/api/graph.py` (3 locations)
- `tests/test_api_endpoints.py` (add validation tests)

**Tests:**
- `GET /graph/snapshot?limit=abc` → 400 with error message
- Valid numeric params still work

---

### Task 2.5: Refactor handle_recall to use dependencies dataclass (H-6)

**Issue:** `handle_recall` accepts ~17 parameters, extreme coupling.

**Steps:**
1. Create `automem/api/recall/dependencies.py`
2. Define `RecallDependencies` dataclass with all deps
3. Refactor `handle_recall` signature: `def handle_recall(deps: RecallDependencies, query_params: dict)`
4. Update all call sites to construct `RecallDependencies`
5. Run recall tests
6. Commit refactoring

**Files:**
- `automem/api/recall.py` or `automem/api/recall/dependencies.py`

**Tests:** All recall tests pass, behavior unchanged

---

## Phase 3: Operational Improvements (MEDIUM-TERM)

### Task 3.1: Convert f-string logging to lazy %s (M-1)

**Issue:** 70 occurrences of f-string logging cause unconditional string interpolation.

**Steps:**
1. Batch replace across files:
   - `scripts/health_monitor.py` (22 occurrences)
   - `scripts/restore_from_backup.py` (16 occurrences)
   - `scripts/backup_automem.py` (15 occurrences)
   - `automem/api/graph.py` (4 occurrences)
   - `automem/api/admin.py` (3 occurrences)
   - `app.py` (3 occurrences)
   - `consolidation.py` (2 occurrences)
   - `automem/api/recall.py` (2 occurrences)
   - `automem/api/consolidation.py` (2 occurrences)
   - `automem/api/memory.py` (1 occurrence)
2. Pattern: `logger.error(f"msg {var}")` → `logger.error("msg %s", var)`
3. Run tests
4. Commit changes

**Files:** 10 files with 70 total replacements

**Tests:** Verify logging still works, no exceptions

---

### Task 3.2: Add authentication to /stream/status (M-2)

**Issue:** `/stream/status` endpoint lacks authentication, leaks operational info.

**Steps:**
1. Add `@require_api_token` decorator to `/stream/status` route
2. Verify authentication is checked before returning subscriber count
3. Add test case for unauthenticated request → 401
4. Commit change

**Files:**
- `automem/api/stream.py:104-113`

**Tests:**
- Unauthenticated GET /stream/status → 401
- Authenticated request works

---

### Task 3.3: Implement rate limiting (M-4)

**Issue:** No rate limiting on any endpoint, risk of resource exhaustion.

**Steps:**
1. Add `flask-limiter` to `requirements.txt`
2. Configure rate limiter in `app.py`
3. Apply rate limits:
   - Write endpoints (POST /memory, POST /associate): 100/hour per IP
   - Read endpoints (GET /recall, GET /analyze): 1000/hour per IP
   - Admin endpoints (POST /admin/reembed): 10/hour per IP
4. Add rate limit exceeded test cases
5. Commit changes

**Files:**
- `requirements.txt`
- `app.py` (limiter setup)
- `automem/api/` blueprints (apply decorators)

**Tests:**
- Exceed rate limit → 429 Too Many Requests
- Within limit works normally

---

### Task 3.4: Add safety assertions for Cypher f-string interpolations (M-5)

**Issue:** 3 locations use f-strings for Cypher structural elements without explicit documentation.

**Steps:**
1. Add safety assertion comments at:
   - `automem/api/graph.py:238`
   - `automem/api/memory.py:584-593`
   - `app.py:3147-3155`
2. Comment format: `# SAFETY: rel_type validated against RELATIONSHIP_TYPES allowlist (line X)`
3. Verify validation exists for each interpolated value
4. Commit documentation

**Files:**
- `automem/api/graph.py`
- `automem/api/memory.py`
- `app.py`

**Tests:** No behavioral changes, documentation only

---

### Task 3.5: Migrate to Flask factory pattern (M-8)

**Issue:** Global mutable `ServiceState` dataclass complicates testing and deployment.

**Steps:**
1. Create `create_app(config=None)` factory function
2. Move state to `app.extensions['automem']`
3. Use `g` object for request-scoped state
4. Update all state access: `state.graph` → `current_app.extensions['automem'].graph`
5. Update tests to use factory pattern
6. Commit refactoring

**Files:**
- `app.py` (create_app factory)
- All files accessing `state` global
- `tests/conftest.py` (update fixtures)

**Tests:** Full test suite passes with factory pattern

---

### Task 3.6: Audit exception handlers for silent swallowing (M-10)

**Issue:** Some except blocks may silently swallow errors without logging.

**Steps:**
1. Search for all `except` blocks: `rg "except " --type py`
2. For each block, verify one of:
   - `logger.exception()` called
   - `logger.error()` with exception info
   - `logger.debug()` with rationale for ignoring
3. Add logging where missing
4. Document intentionally ignored exceptions with comments
5. Commit improvements

**Files:** Various (audit all Python files)

**Tests:** Trigger error paths, verify logged

---

## Phase 4: Polish (WHEN CONVENIENT)

### Task 4.1: Standardize error response format (L-1)

**Issue:** Inconsistent error formats: `{"error": ...}`, `{"description": ...}`, `{"status": "error", ...}`.

**Steps:**
1. Define standard error envelope:
   ```json
   {
     "status": "error",
     "error": "Human-readable message",
     "code": "ERROR_CODE"
   }
   ```
2. Create error response helper: `error_response(message, code, status_code)`
3. Update all endpoints to use helper
4. Update API documentation
5. Commit standardization

**Files:**
- All `automem/api/` blueprints
- `app.py`

**Tests:** Verify all error responses follow format

---

### Task 4.2: Add missing type annotations (L-2)

**Issue:** Some utility functions lack type hints.

**Steps:**
1. Run `mypy automem/` to find missing annotations
2. Add type hints to all functions in:
   - `automem/utils/`
   - `automem/embedding/`
   - Helper functions throughout codebase
3. Enable `mypy --strict` in CI
4. Commit type improvements

**Files:** Various utility modules

**Tests:** `mypy automem/` passes with --strict

---

### Task 4.3: Add __all__ exports to packages (L-3)

**Issue:** No `__all__` definitions in package `__init__.py` files.

**Steps:**
1. Add `__all__` to:
   - `automem/__init__.py`
   - `automem/api/__init__.py`
   - `automem/embedding/__init__.py`
   - `automem/utils/__init__.py`
2. List only intended public API in each `__all__`
3. Update documentation with public API surface
4. Commit exports

**Files:** All `__init__.py` files

**Tests:** Verify `from automem import *` works as expected

---

### Task 4.4: Replace console.log with structured logger in MCP server (L-4)

**Issue:** MCP SSE server uses `console.log` throughout (789 lines).

**Steps:**
1. Add `pino` logging library to `mcp-sse-server/package.json`
2. Create logger instance with levels: debug, info, warn, error
3. Replace all `console.log` → `logger.info()`
4. Replace all `console.error` → `logger.error()`
5. Configure log levels via environment variable
6. Commit improvements

**Files:**
- `mcp-sse-server/server.js`
- `mcp-sse-server/package.json`

**Tests:** Verify MCP server logs structured JSON

---

## Execution Strategy

**Priority Order:**
1. **Phase 1 (Security):** Must complete before any deployment
2. **Phase 2 (Quality):** Should complete before adding major features
3. **Phase 3 (Operations):** Can be done incrementally alongside features
4. **Phase 4 (Polish):** Nice-to-have improvements

**Testing Requirements:**
- Each phase must pass full test suite before merging
- Phase 1 requires security review after completion
- Create PR after each phase for review

**Commit Strategy:**
- One commit per task (atomic changes)
- Descriptive commit messages referencing issue IDs
- Run pre-commit checks before each commit

---

*Plan generated from .agent/code-review-report.md*
*Ready for autonomous execution*
