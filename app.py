"""AutoMem Memory Service API.

Provides a small Flask API that stores memories in FalkorDB and Qdrant.
This module focuses on being resilient: it validates requests, handles
transient outages, and degrades gracefully when one of the backing services
is unavailable.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import random
import re
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from falkordb import FalkorDB
from flask import Blueprint, Flask, abort, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models

try:
    from qdrant_client.http.exceptions import UnexpectedResponse
except ImportError:  # Allow tests to import without full qdrant client installed
    UnexpectedResponse = Exception  # type: ignore[misc,assignment]

try:  # Allow tests to import without full qdrant client installed
    from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams
except Exception:  # pragma: no cover - degraded import path
    try:
        from qdrant_client.http import models as _qmodels

        Distance = getattr(_qmodels, "Distance", None)
        PointStruct = getattr(_qmodels, "PointStruct", None)
        VectorParams = getattr(_qmodels, "VectorParams", None)
        PayloadSchemaType = getattr(_qmodels, "PayloadSchemaType", None)
    except Exception:
        Distance = PointStruct = VectorParams = None
        PayloadSchemaType = None

# Provide a simple PointStruct shim for tests/environments lacking qdrant models
if PointStruct is None:  # pragma: no cover - test shim

    class PointStruct:  # type: ignore[no-redef]
        def __init__(self, id: str, vector: List[float], payload: Dict[str, Any]):
            self.id = id
            self.vector = vector
            self.payload = payload


from werkzeug.exceptions import HTTPException

from consolidation import ConsolidationScheduler, MemoryConsolidator

# Make OpenAI import optional to allow running without it
try:
    from openai import OpenAI  # type: ignore
except ImportError:
    OpenAI = None  # type: ignore

# SSE streaming for real-time observability
from automem.api.stream import create_stream_blueprint, emit_event

# Import only the interface; import backends lazily in init_embedding_provider()
from automem.embedding.provider import EmbeddingProvider

try:
    import spacy  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    spacy = None

# Environment is loaded by automem.config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,  # Write to stdout so Railway correctly parses log levels
)
logger = logging.getLogger("automem.api")

# Configure Flask and Werkzeug loggers to use stdout instead of stderr
# This ensures Railway correctly parses log levels instead of treating everything as "error"
for logger_name in ["werkzeug", "flask.app"]:
    framework_logger = logging.getLogger(logger_name)
    framework_logger.handlers.clear()
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    framework_logger.addHandler(stdout_handler)
    framework_logger.setLevel(logging.INFO)

# Ensure local package imports work when only app.py is copied
try:
    import automem  # type: ignore
except Exception:
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Rate limiting (M-4)
# ---------------------------------------------------------------------------
# flask-limiter protects the API from abuse.  Limits are configurable via
# environment variables so they can be tuned per deployment without code
# changes.  Rate limiting is disabled entirely in test environments via the
# RATELIMIT_ENABLED Flask config key (set to False in conftest.py).
# ---------------------------------------------------------------------------
_RATE_LIMIT_ENABLED = os.environ.get("RATELIMIT_ENABLED", "true").lower() not in ("false", "0", "no")
_RATE_LIMIT_DEFAULT = os.environ.get("RATE_LIMIT_DEFAULT", "1000 per hour")

app.config.setdefault("RATELIMIT_ENABLED", _RATE_LIMIT_ENABLED)
app.config.setdefault("RATELIMIT_STORAGE_URI", "memory://")

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[_RATE_LIMIT_DEFAULT],
    storage_uri=app.config["RATELIMIT_STORAGE_URI"],
)

# Import canonical configuration constants
from automem.config import (
    ADMIN_TOKEN,
    ALLOWED_RELATIONS,
    API_TOKEN,
    CLASSIFICATION_MODEL,
    COLLECTION_NAME,
    CONSOLIDATION_ARCHIVE_THRESHOLD,
    CONSOLIDATION_CLUSTER_INTERVAL_SECONDS,
    CONSOLIDATION_CONTROL_LABEL,
    CONSOLIDATION_CONTROL_NODE_ID,
    CONSOLIDATION_CREATIVE_INTERVAL_SECONDS,
    CONSOLIDATION_DECAY_IMPORTANCE_THRESHOLD,
    CONSOLIDATION_DECAY_INTERVAL_SECONDS,
    CONSOLIDATION_DELETE_THRESHOLD,
    CONSOLIDATION_FORGET_INTERVAL_SECONDS,
    CONSOLIDATION_GRACE_PERIOD_DAYS,
    CONSOLIDATION_HISTORY_LIMIT,
    CONSOLIDATION_IMPORTANCE_PROTECTION_THRESHOLD,
    CONSOLIDATION_PROTECTED_TYPES,
    CONSOLIDATION_RUN_LABEL,
    CONSOLIDATION_TASK_FIELDS,
    CONSOLIDATION_TICK_SECONDS,
    EMBEDDING_MODEL,
    ENRICHMENT_ENABLE_SUMMARIES,
    ENRICHMENT_FAILURE_BACKOFF_SECONDS,
    ENRICHMENT_IDLE_SLEEP_SECONDS,
    ENRICHMENT_MAX_ATTEMPTS,
    ENRICHMENT_SIMILARITY_LIMIT,
    ENRICHMENT_SIMILARITY_THRESHOLD,
    ENRICHMENT_SPACY_MODEL,
    FALKORDB_PORT,
    GRAPH_NAME,
    MEMORY_TYPES,
    RECALL_EXPANSION_LIMIT,
    RECALL_RELATION_LIMIT,
    RELATIONSHIP_TYPES,
    SEARCH_WEIGHT_CONFIDENCE,
    SEARCH_WEIGHT_EXACT,
    SEARCH_WEIGHT_IMPORTANCE,
    SEARCH_WEIGHT_KEYWORD,
    SEARCH_WEIGHT_RECENCY,
    SEARCH_WEIGHT_TAG,
    SEARCH_WEIGHT_VECTOR,
    SYNC_AUTO_REPAIR,
    SYNC_CHECK_INTERVAL_SECONDS,
    TYPE_ALIASES,
    VECTOR_SIZE,
    normalize_memory_type,
)
from automem.stores.graph_store import _build_graph_tag_predicate
from automem.stores.vector_store import _build_qdrant_tag_filter
from automem.utils.graph import _serialize_node, _summarize_relation_node
from automem.utils.scoring import _compute_metadata_score, _parse_metadata_field
from automem.utils.tags import (
    _compute_tag_prefixes,
    _expand_tag_prefixes,
    _normalize_tag_list,
    _prepare_tag_filters,
)

# Shared utils and helpers
from automem.utils.time import (
    _normalize_timestamp,
    _parse_iso_datetime,
    _parse_time_expression,
    utc_now,
)
from automem.utils.validation import get_effective_vector_size, validate_vector_dimensions

# Embedding batching configuration
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "20"))
EMBEDDING_BATCH_TIMEOUT_SECONDS = float(os.getenv("EMBEDDING_BATCH_TIMEOUT_SECONDS", "2.0"))

"""Note: default types/relations/weights are imported from automem.config"""

# Keyword/NER constants come from automem.utils.text if available
SEARCH_STOPWORDS: Set[str] = set()
ENTITY_STOPWORDS: Set[str] = set()
ENTITY_BLOCKLIST: Set[str] = set()

# Search weights are imported from automem.config

# Maximum number of results returned by /recall
RECALL_MAX_LIMIT = int(os.getenv("RECALL_MAX_LIMIT", "100"))

# API tokens are imported from automem.config


try:
    from automem.utils.text import ENTITY_BLOCKLIST as _AM_ENTITY_BLOCKLIST
    from automem.utils.text import ENTITY_STOPWORDS as _AM_ENTITY_STOPWORDS
    from automem.utils.text import SEARCH_STOPWORDS as _AM_SEARCH_STOPWORDS
    from automem.utils.text import _extract_keywords as _AM_extract_keywords

    # Override local constants if package is available
    SEARCH_STOPWORDS = _AM_SEARCH_STOPWORDS
    ENTITY_STOPWORDS = _AM_ENTITY_STOPWORDS
    ENTITY_BLOCKLIST = _AM_ENTITY_BLOCKLIST
    _extract_keywords = _AM_extract_keywords
except Exception:
    # Define local fallback for keyword extraction
    def _extract_keywords(text: str) -> List[str]:
        if not text:
            return []
        words = re.findall(r"[A-Za-z0-9_\-]+", text.lower())
        keywords: List[str] = []
        seen: set[str] = set()
        for word in words:
            cleaned = word.strip("-_")
            if len(cleaned) < 3:
                continue
            if cleaned in SEARCH_STOPWORDS:
                continue
            if cleaned in seen:
                continue
            seen.add(cleaned)
            keywords.append(cleaned)
        return keywords


# Local scoring/metadata helpers (fallback if package not available)


def _result_passes_filters(
    result: Dict[str, Any],
    start_time: Optional[str],
    end_time: Optional[str],
    tag_filters: Optional[List[str]] = None,
    tag_mode: str = "any",
    tag_match: str = "prefix",
    exclude_tags: Optional[List[str]] = None,
) -> bool:
    memory = result.get("memory", {}) or {}
    timestamp = memory.get("timestamp")
    if start_time or end_time:
        parsed = _parse_iso_datetime(timestamp) if timestamp else None
        parsed_start = _parse_iso_datetime(start_time) if start_time else None
        parsed_end = _parse_iso_datetime(end_time) if end_time else None
        if parsed is None:
            return False
        if parsed_start and parsed < parsed_start:
            return False
        if parsed_end and parsed > parsed_end:
            return False

    if tag_filters:
        normalized_filters = _prepare_tag_filters(tag_filters)
        if normalized_filters:
            normalized_mode = "all" if tag_mode == "all" else "any"
            normalized_match = "prefix" if tag_match == "prefix" else "exact"

            tags = memory.get("tags") or []
            lowered_tags = [
                str(tag).strip().lower()
                for tag in tags
                if isinstance(tag, str) and str(tag).strip()
            ]

            if normalized_match == "exact":
                tag_set = set(lowered_tags)
                if not tag_set:
                    return False
                if normalized_mode == "all":
                    if not all(filter_tag in tag_set for filter_tag in normalized_filters):
                        return False
                else:
                    if not any(filter_tag in tag_set for filter_tag in normalized_filters):
                        return False
            else:
                prefixes = memory.get("tag_prefixes") or []
                prefix_set = {
                    str(prefix).strip().lower()
                    for prefix in prefixes
                    if isinstance(prefix, str) and str(prefix).strip()
                }

                def _tags_start_with() -> bool:
                    if not lowered_tags:
                        return False
                    if normalized_mode == "all":
                        return all(
                            any(tag.startswith(filter_tag) for tag in lowered_tags)
                            for filter_tag in normalized_filters
                        )
                    return any(
                        tag.startswith(filter_tag)
                        for filter_tag in normalized_filters
                        for tag in lowered_tags
                    )

                if prefix_set:
                    if normalized_mode == "all":
                        if not all(filter_tag in prefix_set for filter_tag in normalized_filters):
                            return False
                    else:
                        if not any(filter_tag in prefix_set for filter_tag in normalized_filters):
                            return False
                else:
                    if not _tags_start_with():
                        return False

    # Apply exclude_tags filter - exclude if ANY excluded tag matches
    if exclude_tags:
        normalized_exclude = _prepare_tag_filters(exclude_tags)
        if normalized_exclude:
            tags = memory.get("tags") or []
            lowered_tags = [
                str(tag).strip().lower()
                for tag in tags
                if isinstance(tag, str) and str(tag).strip()
            ]

            # Check exact matches first
            tag_set = set(lowered_tags)
            if any(exclude_tag in tag_set for exclude_tag in normalized_exclude):
                return False

            # Check prefix matches
            if any(
                tag.startswith(exclude_tag)
                for exclude_tag in normalized_exclude
                for tag in lowered_tags
            ):
                return False

    return True


def _format_graph_result(
    graph: Any,
    node: Any,
    score: Optional[float],
    match_type: str,
    seen_ids: set[str],
) -> Optional[Dict[str, Any]]:
    data = _serialize_node(node)
    memory_id = str(data.get("id")) if data.get("id") is not None else None
    if not memory_id or memory_id in seen_ids:
        return None

    seen_ids.add(memory_id)
    relations: List[Dict[str, Any]] = _fetch_relations(graph, memory_id)

    numeric_score = float(score) if score is not None else 0.0
    return {
        "id": memory_id,
        "score": numeric_score,
        "match_score": numeric_score,
        "match_type": match_type,
        "source": "graph",
        "memory": data,
        "relations": relations,
    }


def _graph_trending_results(
    graph: Any,
    limit: int,
    seen_ids: set[str],
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    tag_filters: Optional[List[str]] = None,
    tag_mode: str = "any",
    tag_match: str = "prefix",
) -> List[Dict[str, Any]]:
    """Return high-importance memories when no specific query is supplied."""
    try:
        sort_param = (
            ((request.args.get("sort") or request.args.get("order_by") or "score") or "")
            .strip()
            .lower()
        )
        order_by = {
            "time_asc": "m.timestamp ASC, m.importance DESC",
            "time_desc": "m.timestamp DESC, m.importance DESC",
            "updated_asc": "coalesce(m.updated_at, m.timestamp) ASC, m.importance DESC",
            "updated_desc": "coalesce(m.updated_at, m.timestamp) DESC, m.importance DESC",
        }.get(sort_param, "m.importance DESC, m.timestamp DESC")

        where_clauses = ["coalesce(m.archived, false) = false"]
        params: Dict[str, Any] = {"limit": limit}
        if start_time:
            where_clauses.append("m.timestamp >= $start_time")
            params["start_time"] = start_time
        if end_time:
            where_clauses.append("m.timestamp <= $end_time")
            params["end_time"] = end_time
        if tag_filters:
            normalized_filters = _prepare_tag_filters(tag_filters)
            if normalized_filters:
                where_clauses.append(_build_graph_tag_predicate(tag_mode, tag_match))
                params["tag_filters"] = normalized_filters

        query = f"""
            MATCH (m:Memory)
            WHERE {' AND '.join(where_clauses)}
            RETURN m
            ORDER BY {order_by}
            LIMIT $limit
        """
        result = graph.query(query, params)
    except Exception:
        logger.exception("Failed to load trending memories")
        return []

    trending: List[Dict[str, Any]] = []
    for row in getattr(result, "result_set", []) or []:
        record = _format_graph_result(graph, row[0], None, "trending", seen_ids)
        if record is None:
            continue
        # Use importance as a pseudo-score for ordering consistency
        importance = record["memory"].get("importance")
        record["score"] = float(importance) if isinstance(importance, (int, float)) else 0.0
        record["match_score"] = record["score"]
        trending.append(record)

    return trending


def _graph_keyword_search(
    graph: Any,
    query_text: str,
    limit: int,
    seen_ids: set[str],
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    tag_filters: Optional[List[str]] = None,
    tag_mode: str = "any",
    tag_match: str = "prefix",
) -> List[Dict[str, Any]]:
    """Perform a keyword-oriented search in the graph store."""
    normalized = query_text.strip().lower()
    if not normalized or normalized == "*":
        return _graph_trending_results(
            graph,
            limit,
            seen_ids,
            start_time,
            end_time,
            tag_filters,
            tag_mode,
            tag_match,
        )

    keywords = _extract_keywords(normalized)
    phrase = normalized if len(normalized) >= 3 else ""

    try:
        base_where = ["m.content IS NOT NULL"]
        params: Dict[str, Any] = {"limit": limit}
        if start_time:
            base_where.append("m.timestamp >= $start_time")
            params["start_time"] = start_time
        if end_time:
            base_where.append("m.timestamp <= $end_time")
            params["end_time"] = end_time
        if tag_filters:
            normalized_filters = _prepare_tag_filters(tag_filters)
            if normalized_filters:
                base_where.append(_build_graph_tag_predicate(tag_mode, tag_match))
                params["tag_filters"] = normalized_filters

        where_clause = " AND ".join(base_where)

        if keywords:
            params.update({"keywords": keywords, "phrase": phrase})
            query = f"""
                MATCH (m:Memory)
                WHERE {where_clause}
                WITH m,
                     toLower(m.content) AS content,
                     [tag IN coalesce(m.tags, []) | toLower(tag)] AS tags
                UNWIND $keywords AS kw
                WITH m, content, tags, kw,
                     CASE WHEN content CONTAINS kw THEN 2 ELSE 0 END +
                     CASE WHEN any(tag IN tags WHERE tag CONTAINS kw) THEN 1 ELSE 0 END AS kw_score
                WITH m, content, tags, SUM(kw_score) AS keyword_score
                WITH m, keyword_score +
                     CASE WHEN $phrase <> '' AND content CONTAINS $phrase THEN 2 ELSE 0 END +
                     CASE WHEN $phrase <> '' AND any(tag IN tags WHERE tag CONTAINS $phrase) THEN 1 ELSE 0 END AS score
                WHERE score > 0
                RETURN m, score
                ORDER BY score DESC, m.importance DESC, m.timestamp DESC
                LIMIT $limit
            """
            result = graph.query(query, params)
        elif phrase:
            params.update({"phrase": phrase})
            query = f"""
                MATCH (m:Memory)
                WHERE {where_clause}
                WITH m,
                     toLower(m.content) AS content,
                     [tag IN coalesce(m.tags, []) | toLower(tag)] AS tags
                WITH m,
                     CASE WHEN content CONTAINS $phrase THEN 2 ELSE 0 END +
                     CASE WHEN any(tag IN tags WHERE tag CONTAINS $phrase) THEN 1 ELSE 0 END AS score
                WHERE score > 0
                RETURN m, score
                ORDER BY score DESC, m.importance DESC, m.timestamp DESC
                LIMIT $limit
            """
            result = graph.query(query, params)
        else:
            return _graph_trending_results(
                graph,
                limit,
                seen_ids,
                start_time,
                end_time,
                tag_filters,
                tag_mode,
                tag_match,
            )
    except Exception:
        logger.exception("Graph keyword search failed")
        return []

    matches: List[Dict[str, Any]] = []
    for row in getattr(result, "result_set", []) or []:
        node = row[0]
        score = row[1] if len(row) > 1 else None
        record = _format_graph_result(graph, node, score, "keyword", seen_ids)
        if record is None:
            continue
        matches.append(record)

    return matches


def _vector_filter_only_tag_search(
    qdrant_client: Optional[QdrantClient],
    tag_filters: Optional[List[str]],
    tag_mode: str,
    tag_match: str,
    limit: int,
    seen_ids: set[str],
) -> List[Dict[str, Any]]:
    """Fallback scroll search when only tags are provided."""
    if qdrant_client is None or not tag_filters or limit <= 0:
        return []

    query_filter = _build_qdrant_tag_filter(tag_filters, tag_mode, tag_match)
    if query_filter is None:
        return []

    try:
        points, _ = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
    except Exception:
        logger.exception("Qdrant tag-only scroll failed")
        return []

    results: List[Dict[str, Any]] = []
    for point in points or []:
        memory_id = str(point.id)
        if memory_id in seen_ids:
            continue
        seen_ids.add(memory_id)

        payload = point.payload or {}
        importance = payload.get("importance")
        if isinstance(importance, (int, float)):
            score = float(importance)
        else:
            try:
                score = float(importance) if importance is not None else 0.0
            except (TypeError, ValueError):
                score = 0.0

        results.append(
            {
                "id": memory_id,
                "score": score,
                "match_score": score,
                "match_type": "tag",
                "source": "qdrant",
                "memory": payload,
                "relations": [],
            }
        )

    return results


def _vector_search(
    qdrant_client: Optional[QdrantClient],
    graph: Any,
    query_text: str,
    embedding_param: Optional[str],
    limit: int,
    seen_ids: set[str],
    tag_filters: Optional[List[str]] = None,
    tag_mode: str = "any",
    tag_match: str = "prefix",
) -> List[Dict[str, Any]]:
    """Perform vector search against Qdrant when configured."""
    if qdrant_client is None:
        return []

    normalized = (query_text or "").strip()
    if not embedding_param and normalized in {"", "*"}:
        return []

    embedding: Optional[List[float]] = None

    if embedding_param:
        try:
            embedding = _coerce_embedding(embedding_param)
        except ValueError as exc:
            abort(400, description=str(exc))
    elif normalized:
        logger.debug("Generating embedding for query: %s", normalized)
        embedding = _generate_real_embedding(normalized)

    if not embedding:
        return []

    query_filter = _build_qdrant_tag_filter(tag_filters, tag_mode, tag_match)

    try:
        vector_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedding,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        )
    except Exception:
        logger.exception("Qdrant search failed")
        return []

    matches: List[Dict[str, Any]] = []
    for hit in vector_results:
        memory_id = str(hit.id)
        if memory_id in seen_ids:
            continue

        seen_ids.add(memory_id)
        payload = hit.payload or {}
        relations = _fetch_relations(graph, memory_id) if graph is not None else []
        score = float(hit.score) if hit.score is not None else 0.0

        matches.append(
            {
                "id": memory_id,
                "score": score,
                "match_score": score,
                "match_type": "vector",
                "source": "qdrant",
                "memory": payload,
                "relations": relations,
            }
        )

    return matches


class MemoryClassifier:
    """Classifies memories into specific types based on content patterns."""

    PATTERNS = {
        "Decision": [
            r"decided to",
            r"chose (\w+) over",
            r"going with",
            r"picked",
            r"selected",
            r"will use",
            r"choosing",
            r"opted for",
        ],
        "Pattern": [
            r"usually",
            r"typically",
            r"tend to",
            r"pattern i noticed",
            r"often",
            r"frequently",
            r"regularly",
            r"consistently",
        ],
        "Preference": [
            r"prefer",
            r"like.*better",
            r"favorite",
            r"always use",
            r"rather than",
            r"instead of",
            r"favor",
        ],
        "Style": [
            r"wrote.*in.*style",
            r"communicated",
            r"responded to",
            r"formatted as",
            r"using.*tone",
            r"expressed as",
        ],
        "Habit": [
            r"always",
            r"every time",
            r"habitually",
            r"routine",
            r"daily",
            r"weekly",
            r"monthly",
        ],
        "Insight": [
            r"realized",
            r"discovered",
            r"learned that",
            r"understood",
            r"figured out",
            r"insight",
            r"revelation",
        ],
        "Context": [
            r"during",
            r"while working on",
            r"in the context of",
            r"when",
            r"at the time",
            r"situation was",
        ],
    }

    SYSTEM_PROMPT = """You are a memory classification system. Classify each memory into exactly ONE of these types:

- **Decision**: Choices made, selected options, what was decided
- **Pattern**: Recurring behaviors, typical approaches, consistent tendencies
- **Preference**: Likes/dislikes, favorites, personal tastes
- **Style**: Communication approach, formatting, tone used
- **Habit**: Regular routines, repeated actions, schedules
- **Insight**: Discoveries, learnings, realizations, key findings
- **Context**: Situational background, what was happening, circumstances

Return JSON with: {"type": "<type>", "confidence": <0.0-1.0>}"""

    def classify(self, content: str, *, use_llm: bool = True) -> tuple[str, float]:
        """
        Classify memory type and return confidence score.
        Returns: (type, confidence)

        Args:
            content: Memory content to classify
            use_llm: If True, falls back to LLM when regex patterns don't match
        """
        content_lower = content.lower()

        # Try regex patterns first (fast, free)
        for memory_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    # Start with base confidence based on pattern match
                    confidence = 0.6

                    # Boost confidence for multiple pattern matches
                    matches = sum(1 for p in patterns if re.search(p, content_lower))
                    if matches > 1:
                        confidence = min(0.95, confidence + (matches * 0.1))

                    return memory_type, confidence

        # If no regex match and LLM enabled, use LLM classification
        if use_llm:
            try:
                result = self._classify_with_llm(content)
                if result:
                    return result
            except Exception:
                logger.exception("LLM classification failed, using fallback")

        # Default to base Memory type with lower confidence
        return "Memory", 0.3

    def _classify_with_llm(self, content: str) -> Optional[tuple[str, float]]:
        """Use OpenAI to classify memory type (fallback for complex content)."""
        # Reuse existing client if available
        if state.openai_client is None:
            init_openai()

        if state.openai_client is None:
            return None

        try:
            # Build model-specific params (o-series and gpt-5 don't support temperature)
            extra_params: dict = {}
            if CLASSIFICATION_MODEL.startswith(("o", "gpt-5")):
                extra_params["max_completion_tokens"] = 50
            else:
                extra_params["max_tokens"] = 50
                extra_params["temperature"] = 0.3
            response = state.openai_client.chat.completions.create(
                model=CLASSIFICATION_MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": content[:1000]},  # Limit to 1000 chars
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "classification",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "confidence": {"type": "number"},
                            },
                            "required": ["type", "confidence"],
                            "additionalProperties": False,
                        },
                    },
                },
                **extra_params,
            )

            result = json.loads(response.choices[0].message.content)
            raw_type = result.get("type", "Memory")
            confidence = float(result.get("confidence", 0.7))

            # Normalize type (handles aliases and case variations)
            memory_type, was_normalized = normalize_memory_type(raw_type)
            if not memory_type:
                logger.warning("LLM returned unmappable type '%s', using Context", raw_type)
                return "Context", 0.5

            if was_normalized and memory_type != raw_type:
                logger.debug("LLM type normalized '%s' -> '%s'", raw_type, memory_type)

            logger.info("LLM classified as %s (confidence: %.2f)", memory_type, confidence)
            return memory_type, confidence

        except Exception as exc:
            logger.warning("LLM classification failed: %s", exc)
            return None


memory_classifier = MemoryClassifier()


_SPACY_NLP = None
_SPACY_INIT_LOCK = Lock()


def _get_spacy_nlp():  # type: ignore[return-type]
    global _SPACY_NLP
    if spacy is None:
        return None

    with _SPACY_INIT_LOCK:
        if _SPACY_NLP is not None:
            return _SPACY_NLP

        try:
            _SPACY_NLP = spacy.load(ENRICHMENT_SPACY_MODEL)
            logger.info("Loaded spaCy model '%s' for enrichment", ENRICHMENT_SPACY_MODEL)
        except Exception:  # pragma: no cover - optional dependency
            logger.warning("Failed to load spaCy model '%s'", ENRICHMENT_SPACY_MODEL)
            _SPACY_NLP = None

        return _SPACY_NLP


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower())
    return cleaned.strip("-")


def _is_valid_entity(
    value: str, *, allow_lower: bool = False, max_words: Optional[int] = None
) -> bool:
    if not value:
        return False

    cleaned = value.strip()
    if len(cleaned) < 3:
        return False

    words = cleaned.split()
    if max_words is not None and len(words) > max_words:
        return False

    lowered = cleaned.lower()
    if lowered in SEARCH_STOPWORDS or lowered in ENTITY_STOPWORDS:
        return False

    # Reject error codes and technical noise
    if lowered in ENTITY_BLOCKLIST:
        return False

    if not any(ch.isalpha() for ch in cleaned):
        return False

    if not allow_lower and cleaned[0].islower() and not cleaned.isupper():
        return False

    # Reject strings starting with markdown/formatting or code characters
    if cleaned[0] in {"-", "*", "#", ">", "|", "[", "]", "{", "}", "(", ")", "_", "'", '"'}:
        return False

    # Reject common code artifacts (suffixes that indicate class names)
    code_suffixes = (
        "Adapter",
        "Handler",
        "Manager",
        "Service",
        "Controller",
        "Provider",
        "Factory",
        "Builder",
        "Helper",
        "Util",
    )
    if any(cleaned.endswith(suffix) for suffix in code_suffixes):
        return False

    # Reject boolean/null literals and common JSON noise
    if lowered in {"true", "false", "null", "none", "undefined"}:
        return False

    # Reject environment variables (all caps with underscores) and text fragments ending with colons
    if ("_" in cleaned and cleaned.isupper()) or cleaned.endswith(":"):
        return False

    return True


def generate_summary(
    content: str, fallback: Optional[str] = None, *, max_length: int = 240
) -> Optional[str]:
    text = (content or "").strip()
    if not text:
        return fallback

    sentences = re.split(r"(?<=[.!?])\s+", text)
    summary = sentences[0] if sentences else text
    summary = summary.strip()

    if not summary:
        return fallback

    if len(summary) > max_length:
        truncated = summary[:max_length].rsplit(" ", 1)[0]
        summary = truncated.strip() if truncated else summary[:max_length].strip()

    if fallback and fallback.strip() == summary:
        return fallback

    return summary


def extract_entities(content: str) -> Dict[str, List[str]]:
    """Extract entities from memory content using spaCy when available."""
    result: Dict[str, Set[str]] = {
        "tools": set(),
        "projects": set(),
        "people": set(),
        "concepts": set(),
        "organizations": set(),
    }

    text = (content or "").strip()
    if not text:
        return {key: [] for key in result}

    nlp = _get_spacy_nlp()
    if nlp is not None:
        try:
            doc = nlp(text)
            for ent in doc.ents:
                value = ent.text.strip()
                if not _is_valid_entity(value, allow_lower=False, max_words=6):
                    continue
                if ent.label_ in {"PERSON"}:
                    result["people"].add(value)
                elif ent.label_ in {"ORG"}:
                    result["organizations"].add(value)
                elif ent.label_ in {"PRODUCT", "WORK_OF_ART", "LAW"}:
                    result["tools"].add(value)
                elif ent.label_ in {"EVENT", "GPE", "LOC", "NORP"}:
                    result["concepts"].add(value)
        except Exception:  # pragma: no cover - defensive
            logger.exception("spaCy entity extraction failed")

    # Regex-based fallbacks to capture simple patterns
    for match in re.findall(
        r"(?:with|met with|meeting with|talked to|spoke with)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        text,
    ):
        result["people"].add(match.strip())

    tool_patterns = [
        r"(?:use|using|deploy|deployed|with|via)\s+([A-Z][\w\-]+)",
        r"([A-Z][\w\-]+)\s+(?:vs|versus|over|instead of)",
    ]
    for pattern in tool_patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            cleaned = match.strip()
            if _is_valid_entity(cleaned):
                result["tools"].add(cleaned)

    for match in re.findall(r"`([^`]+)`", text):
        cleaned = match.strip()
        if _is_valid_entity(cleaned, allow_lower=False, max_words=4):
            result["projects"].add(cleaned)

    # Extract project names from "project called/named 'X'" pattern
    for match in re.findall(
        r'(?:project|repo|repository)\s+(?:called|named)\s+"([^"]+)"', text, re.IGNORECASE
    ):
        cleaned = match.strip()
        if _is_valid_entity(cleaned, allow_lower=False, max_words=4):
            result["projects"].add(cleaned)

    # Extract project names from 'project "X"' pattern
    for match in re.findall(r'(?:project|repo|repository)\s+"([^"]+)"', text, re.IGNORECASE):
        cleaned = match.strip()
        if _is_valid_entity(cleaned, allow_lower=False, max_words=4):
            result["projects"].add(cleaned)

    for match in re.findall(r"Project\s+([A-Z][\w\-]+)", text):
        cleaned = match.strip()
        if _is_valid_entity(cleaned):
            result["projects"].add(cleaned)

    # Extract project names from "project: project-name" pattern (common in session starts)
    for match in re.findall(r"(?:in |on )?project:\s+([a-z][a-z0-9\-]+)", text, re.IGNORECASE):
        cleaned = match.strip()
        if _is_valid_entity(cleaned, allow_lower=True):
            result["projects"].add(cleaned)

    result["tools"].difference_update(result["people"])

    cleaned = {key: sorted({value for value in values if value}) for key, values in result.items()}
    return cleaned


@dataclass
class EnrichmentStats:
    processed_total: int = 0
    successes: int = 0
    failures: int = 0
    last_success_id: Optional[str] = None
    last_success_at: Optional[str] = None
    last_error: Optional[str] = None
    last_error_at: Optional[str] = None

    def record_success(self, memory_id: str) -> None:
        self.processed_total += 1
        self.successes += 1
        self.last_success_id = memory_id
        self.last_success_at = utc_now()

    def record_failure(self, error: str) -> None:
        self.processed_total += 1
        self.failures += 1
        self.last_error = error
        self.last_error_at = utc_now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "processed_total": self.processed_total,
            "successes": self.successes,
            "failures": self.failures,
            "last_success_id": self.last_success_id,
            "last_success_at": self.last_success_at,
            "last_error": self.last_error,
            "last_error_at": self.last_error_at,
        }


@dataclass
class EnrichmentJob:
    memory_id: str
    attempt: int = 0
    forced: bool = False


@dataclass
class ServiceState:
    falkordb: Optional[FalkorDB] = None
    memory_graph: Any = None
    qdrant: Optional[QdrantClient] = None
    openai_client: Optional[OpenAI] = (
        None  # Keep for backward compatibility (e.g., memory type classification)
    )
    embedding_provider: Optional[EmbeddingProvider] = None  # New provider pattern for embeddings
    enrichment_queue: Optional[Queue] = None
    enrichment_thread: Optional[Thread] = None
    enrichment_stats: EnrichmentStats = field(default_factory=EnrichmentStats)
    enrichment_inflight: Set[str] = field(default_factory=set)
    enrichment_pending: Set[str] = field(default_factory=set)
    enrichment_lock: Lock = field(default_factory=Lock)
    consolidation_thread: Optional[Thread] = None
    consolidation_stop_event: Optional[Event] = None
    # Async embedding generation
    embedding_queue: Optional[Queue] = None
    embedding_thread: Optional[Thread] = None
    embedding_inflight: Set[str] = field(default_factory=set)
    embedding_pending: Set[str] = field(default_factory=set)
    embedding_lock: Lock = field(default_factory=Lock)
    # Background sync worker
    sync_thread: Optional[Thread] = None
    sync_stop_event: Optional[Event] = None
    sync_last_run: Optional[str] = None
    sync_last_result: Optional[Dict[str, Any]] = None
    # Effective vector size (auto-detected from existing collection or config default)
    effective_vector_size: int = VECTOR_SIZE


state = ServiceState()


def _extract_api_token() -> Optional[str]:
    if not API_TOKEN:
        return None

    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()

    api_key_header = request.headers.get("X-API-Key")
    if api_key_header:
        return api_key_header.strip()

    api_key_param = request.args.get("api_key")
    if api_key_param:
        # DEPRECATED: Query parameter auth will be removed in next major version
        # Tokens in query params are logged in access logs, browser history, proxy logs
        logger.warning(
            "DEPRECATED: Token via query parameter is deprecated and will be removed. "
            "Use 'Authorization: Bearer' or 'X-API-Key' header instead."
        )
        return api_key_param.strip()

    return None


def get_openai_client() -> Optional[OpenAI]:
    return state.openai_client


def _require_admin_token() -> None:
    if not ADMIN_TOKEN:
        abort(403, description="Admin token not configured")

    provided = (
        request.headers.get("X-Admin-Token")
        or request.headers.get("X-Admin-Api-Key")
        or request.args.get("admin_token")
    )

    # Use constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(provided or "", ADMIN_TOKEN or ""):
        abort(401, description="Admin authorization required")


@app.before_request
def require_api_token() -> None:
    if not API_TOKEN:
        return

    # Allow unauthenticated health checks (supports blueprint endpoint names)
    endpoint = request.endpoint or ""
    if endpoint.endswith("health") or request.path == "/health":
        return

    token = _extract_api_token()
    # Use constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(token or "", API_TOKEN or ""):
        abort(401, description="Unauthorized")


def _get_base_url_for_logging(url: str | None) -> str:
    """Sanitize URL for logging by removing userinfo and query params."""
    if not url:
        return "default"
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        # Return only scheme://host:port
        sanitized = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else "custom"
        return sanitized
    except Exception:
        return "custom"


def init_openai() -> None:
    """Initialize OpenAI client for memory type classification (not embeddings)."""
    if state.openai_client is not None:
        return

    # Check if OpenAI is available at all
    if OpenAI is None:
        logger.info("OpenAI package not installed (used for memory type classification)")
        return

    # Use separate classification credentials if provided, otherwise fall back to OPENAI_*
    api_key = os.getenv("CLASSIFICATION_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("CLASSIFICATION_BASE_URL") or os.getenv("OPENAI_BASE_URL")

    if not api_key:
        logger.info("OpenAI API key not provided (used for memory type classification)")
        return

    try:
        state.openai_client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(
            "OpenAI client initialized for memory type classification (base_url=%s)",
            _get_base_url_for_logging(base_url)
        )
    except Exception:
        logger.exception("Failed to initialize OpenAI client")
        state.openai_client = None


def _create_openai_embedding_provider(api_key: str, vector_size: int) -> "OpenAIEmbeddingProvider":
    """Create OpenAI embedding provider with consistent configuration."""
    from automem.embedding.openai import OpenAIEmbeddingProvider
    base_url = os.getenv("OPENAI_BASE_URL")
    return OpenAIEmbeddingProvider(
        api_key=api_key, model=EMBEDDING_MODEL, dimension=vector_size, base_url=base_url
    )


def init_embedding_provider() -> None:
    """Initialize embedding provider with auto-selection fallback.

    Priority order:
    1. Voyage API (if VOYAGE_API_KEY is set)
    2. OpenAI API (if OPENAI_API_KEY is set)
    3. Ollama local server (if configured)
    4. Local fastembed model (no API key needed)
    5. Placeholder hash-based embeddings (fallback)

    Can be controlled via EMBEDDING_PROVIDER env var:
    - "auto" (default): Try Voyage, then OpenAI, then Ollama, then fastembed, then placeholder
    - "voyage": Use Voyage only, fail if unavailable
    - "openai": Use OpenAI only, fail if unavailable
    - "local": Use fastembed only, fail if unavailable
    - "ollama": Use Ollama only, fail if unavailable
    - "placeholder": Use placeholder embeddings
    """
    if state.embedding_provider is not None:
        return

    provider_config = (os.getenv("EMBEDDING_PROVIDER", "auto") or "auto").strip().lower()
    # Use effective dimension (auto-detected from existing collection or config default).
    # If Qdrant hasn't set it (or config was changed in-process), align to VECTOR_SIZE.
    if state.qdrant is None and state.effective_vector_size != VECTOR_SIZE:
        state.effective_vector_size = VECTOR_SIZE
    vector_size = state.effective_vector_size

    # Explicit provider selection
    if provider_config == "voyage":
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=voyage but VOYAGE_API_KEY not set")
        try:
            from automem.embedding.voyage import VoyageEmbeddingProvider

            voyage_model = os.getenv("VOYAGE_MODEL", "voyage-4")
            state.embedding_provider = VoyageEmbeddingProvider(
                api_key=api_key, model=voyage_model, dimension=vector_size
            )
            logger.info("Embedding provider: %s", state.embedding_provider.provider_name())
            return
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Voyage provider: {e}") from e

    elif provider_config == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=openai but OPENAI_API_KEY not set")
        try:
            state.embedding_provider = _create_openai_embedding_provider(api_key, vector_size)
            logger.info("Embedding provider: %s", state.embedding_provider.provider_name())
            return
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI provider: {e}") from e

    elif provider_config == "local":
        try:
            from automem.embedding.fastembed import FastEmbedProvider

            state.embedding_provider = FastEmbedProvider(dimension=vector_size)
            logger.info("Embedding provider: %s", state.embedding_provider.provider_name())
            return
        except Exception as e:
            raise RuntimeError(f"Failed to initialize local fastembed provider: {e}") from e

    elif provider_config == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "nomic-embed-text")
        try:
            timeout = float(os.getenv("OLLAMA_TIMEOUT", "30"))
            max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))
        except ValueError as ve:
            raise RuntimeError(f"Invalid OLLAMA_TIMEOUT or OLLAMA_MAX_RETRIES value: {ve}") from ve
        try:
            from automem.embedding.ollama import OllamaEmbeddingProvider

            state.embedding_provider = OllamaEmbeddingProvider(
                base_url=base_url,
                model=model,
                dimension=vector_size,
                timeout=timeout,
                max_retries=max_retries,
            )
            logger.info("Embedding provider: %s", state.embedding_provider.provider_name())
            return
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Ollama provider: {e}") from e

    elif provider_config == "placeholder":
        from automem.embedding.placeholder import PlaceholderEmbeddingProvider

        state.embedding_provider = PlaceholderEmbeddingProvider(dimension=vector_size)
        logger.info("Embedding provider: %s", state.embedding_provider.provider_name())
        return

    # Auto-selection: Try Voyage → OpenAI → Ollama → fastembed → placeholder
    if provider_config == "auto":
        # Try Voyage first (preferred)
        voyage_key = os.getenv("VOYAGE_API_KEY")
        if voyage_key:
            try:
                from automem.embedding.voyage import VoyageEmbeddingProvider

                voyage_model = os.getenv("VOYAGE_MODEL", "voyage-4")
                state.embedding_provider = VoyageEmbeddingProvider(
                    api_key=voyage_key, model=voyage_model, dimension=vector_size
                )
                logger.info(
                    "Embedding provider (auto-selected): %s",
                    state.embedding_provider.provider_name(),
                )
                return
            except Exception as e:
                logger.warning("Failed to initialize Voyage provider, trying OpenAI: %s", str(e))

        # Try OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                state.embedding_provider = _create_openai_embedding_provider(api_key, vector_size)
                logger.info(
                    "Embedding provider (auto-selected): %s",
                    state.embedding_provider.provider_name(),
                )
                return
            except Exception as e:
                logger.warning(
                    "Failed to initialize OpenAI provider, trying local model: %s", str(e)
                )

        ollama_base_url = os.getenv("OLLAMA_BASE_URL")
        ollama_model = os.getenv("OLLAMA_MODEL")
        if ollama_base_url or ollama_model:
            try:
                from automem.embedding.ollama import OllamaEmbeddingProvider

                base_url = ollama_base_url or "http://localhost:11434"
                model = ollama_model or "nomic-embed-text"
                try:
                    timeout = float(os.getenv("OLLAMA_TIMEOUT", "30"))
                    max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))
                except ValueError:
                    logger.warning("Invalid OLLAMA_TIMEOUT or OLLAMA_MAX_RETRIES, using defaults")
                    timeout = 30.0
                    max_retries = 2
                state.embedding_provider = OllamaEmbeddingProvider(
                    base_url=base_url,
                    model=model,
                    dimension=vector_size,
                    timeout=timeout,
                    max_retries=max_retries,
                )
                logger.info(
                    "Embedding provider (auto-selected): %s",
                    state.embedding_provider.provider_name(),
                )
                return
            except Exception as e:
                logger.warning(
                    "Failed to initialize Ollama provider, trying local model: %s", str(e)
                )

        # Try local fastembed
        try:
            from automem.embedding.fastembed import FastEmbedProvider

            state.embedding_provider = FastEmbedProvider(dimension=vector_size)
            logger.info(
                "Embedding provider (auto-selected): %s", state.embedding_provider.provider_name()
            )
            return
        except Exception as e:
            logger.warning("Failed to initialize fastembed provider, using placeholder: %s", str(e))

        # Fallback to placeholder
        from automem.embedding.placeholder import PlaceholderEmbeddingProvider

        state.embedding_provider = PlaceholderEmbeddingProvider(dimension=vector_size)
        logger.warning(
            "Using placeholder embeddings (no semantic search). "
            "Install fastembed or set VOYAGE_API_KEY/OPENAI_API_KEY for semantic embeddings."
        )
        logger.info(
            "Embedding provider (auto-selected): %s", state.embedding_provider.provider_name()
        )
        return

    # Invalid config
    raise ValueError(
        f"Invalid EMBEDDING_PROVIDER={provider_config}. "
        f"Valid options: auto, voyage, openai, local, ollama, placeholder"
    )


def init_falkordb() -> None:
    """Initialize FalkorDB connection if not already connected."""
    if state.memory_graph is not None:
        return

    host = (
        os.getenv("FALKORDB_HOST")
        or os.getenv("RAILWAY_PRIVATE_DOMAIN")
        or os.getenv("RAILWAY_PUBLIC_DOMAIN")
        or "localhost"
    )
    password = os.getenv("FALKORDB_PASSWORD")

    try:
        logger.info("Connecting to FalkorDB at %s:%s", host, FALKORDB_PORT)

        # Only pass authentication if password is actually configured
        connection_params = {
            "host": host,
            "port": FALKORDB_PORT,
        }
        if password:
            connection_params["password"] = password
            connection_params["username"] = "default"

        state.falkordb = FalkorDB(**connection_params)
        state.memory_graph = state.falkordb.select_graph(GRAPH_NAME)
        logger.info(
            "FalkorDB connection established (auth: %s)", "enabled" if password else "disabled"
        )
    except Exception:  # pragma: no cover - log full stack trace in production
        logger.exception("Failed to initialize FalkorDB connection")
        state.falkordb = None
        state.memory_graph = None


def init_qdrant() -> None:
    """Initialize Qdrant connection and ensure the collection exists."""
    if state.qdrant is not None:
        return

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url:
        logger.info("Qdrant URL not provided; skipping client initialization")
        return

    try:
        logger.info("Connecting to Qdrant at %s", url)
        state.qdrant = QdrantClient(url=url, api_key=api_key)
        _ensure_qdrant_collection()
        logger.info("Qdrant connection established")
    except ValueError:
        # Surface migration guidance (e.g., vector dimension mismatch) and halt startup
        state.qdrant = None
        raise
    except Exception:  # pragma: no cover - log full stack trace in production
        logger.exception("Failed to initialize Qdrant client")
        state.qdrant = None


def _ensure_qdrant_collection() -> None:
    """Create the Qdrant collection if it does not already exist."""
    if state.qdrant is None:
        return

    try:
        # Auto-detect vector dimension from existing collection (backwards compatibility)
        # This ensures users with 768d embeddings aren't broken by default change to 3072d
        effective_dim, source = get_effective_vector_size(state.qdrant)
        state.effective_vector_size = effective_dim

        if source == "collection":
            logger.info(
                "Using existing collection dimension: %dd (config default: %dd)",
                effective_dim,
                VECTOR_SIZE,
            )
        else:
            logger.info("Using configured vector dimension: %dd", effective_dim)

        collections = state.qdrant.get_collections()
        existing = {collection.name for collection in collections.collections}
        if COLLECTION_NAME not in existing:
            logger.info(
                "Creating Qdrant collection '%s' with %dd vectors", COLLECTION_NAME, effective_dim
            )
            state.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=effective_dim, distance=Distance.COSINE),
            )

        # Ensure payload indexes exist for tag filtering
        logger.info("Ensuring Qdrant payload indexes for collection '%s'", COLLECTION_NAME)
        if PayloadSchemaType:
            # Use enum if available
            state.qdrant.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="tags",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            state.qdrant.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="tag_prefixes",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        else:
            # Fallback to string values when enum not available
            state.qdrant.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="tags",
                field_schema="keyword",
            )
            state.qdrant.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="tag_prefixes",
                field_schema="keyword",
            )
    except ValueError:
        # Bubble up migration guidance so the service fails fast instead of silently degrading
        raise
    except Exception:  # pragma: no cover - log full stack trace in production
        logger.exception("Failed to ensure Qdrant collection; disabling client")
        state.qdrant = None


def get_memory_graph() -> Any:
    init_falkordb()
    return state.memory_graph


def get_qdrant_client() -> Optional[QdrantClient]:
    init_qdrant()
    return state.qdrant


def init_enrichment_pipeline() -> None:
    """Initialize the background enrichment pipeline."""
    if state.enrichment_queue is not None:
        return

    state.enrichment_queue = Queue()
    state.enrichment_thread = Thread(target=enrichment_worker, daemon=True)
    state.enrichment_thread.start()
    logger.info("Enrichment pipeline initialized")


def enqueue_enrichment(memory_id: str, *, forced: bool = False, attempt: int = 0) -> None:
    if not memory_id or state.enrichment_queue is None:
        return

    job = EnrichmentJob(memory_id=memory_id, attempt=attempt, forced=forced)

    with state.enrichment_lock:
        if not forced and (
            memory_id in state.enrichment_pending or memory_id in state.enrichment_inflight
        ):
            return

        state.enrichment_pending.add(memory_id)
        state.enrichment_queue.put(job)


# ---------------------------------------------------------------------------
# Access Tracking (updates last_accessed on recall)
# ---------------------------------------------------------------------------


def update_last_accessed(memory_ids: List[str]) -> None:
    """Update last_accessed timestamp for retrieved memories (direct, synchronous)."""
    if not memory_ids:
        return

    graph = get_memory_graph()
    if graph is None:
        return

    timestamp = utc_now()
    try:
        graph.query(
            """
            UNWIND $ids AS mid
            MATCH (m:Memory {id: mid})
            SET m.last_accessed = $ts
            """,
            {"ids": memory_ids, "ts": timestamp},
        )
        logger.debug("Updated last_accessed for %d memories", len(memory_ids))
    except Exception:
        logger.exception("Failed to update last_accessed for memories")


def _load_control_record(graph: Any) -> Dict[str, Any]:
    """Fetch or create the consolidation control node."""
    bootstrap_timestamp = utc_now()
    bootstrap_fields = sorted(set(CONSOLIDATION_TASK_FIELDS.values()))
    bootstrap_set_clause = ",\n                ".join(
        f"c.{field} = coalesce(c.{field}, $now)" for field in bootstrap_fields
    )
    try:
        result = graph.query(
            f"""
            MERGE (c:{CONSOLIDATION_CONTROL_LABEL} {{id: $id}})
            SET {bootstrap_set_clause}
            RETURN c
            """,
            {"id": CONSOLIDATION_CONTROL_NODE_ID, "now": bootstrap_timestamp},
        )
    except Exception:
        logger.exception("Failed to load consolidation control record")
        return {}

    if not getattr(result, "result_set", None):
        return {}

    node = result.result_set[0][0]
    properties = getattr(node, "properties", None)
    if isinstance(properties, dict):
        return dict(properties)
    if isinstance(node, dict):
        return dict(node)
    return {}


def _load_recent_runs(graph: Any, limit: int) -> List[Dict[str, Any]]:
    """Return recent consolidation run records."""
    try:
        result = graph.query(
            f"""
            MATCH (r:{CONSOLIDATION_RUN_LABEL})
            RETURN r
            ORDER BY r.started_at DESC
            LIMIT $limit
            """,
            {"limit": limit},
        )
    except Exception:
        logger.exception("Failed to load consolidation history")
        return []

    runs: List[Dict[str, Any]] = []
    for row in getattr(result, "result_set", []) or []:
        node = row[0]
        properties = getattr(node, "properties", None)
        if isinstance(properties, dict):
            runs.append(dict(properties))
        elif isinstance(node, dict):
            runs.append(dict(node))
    return runs


def _apply_scheduler_overrides(scheduler: ConsolidationScheduler) -> None:
    """Override default scheduler intervals using configuration."""
    overrides = {
        "decay": timedelta(seconds=CONSOLIDATION_DECAY_INTERVAL_SECONDS),
        "creative": timedelta(seconds=CONSOLIDATION_CREATIVE_INTERVAL_SECONDS),
        "cluster": timedelta(seconds=CONSOLIDATION_CLUSTER_INTERVAL_SECONDS),
        "forget": timedelta(seconds=CONSOLIDATION_FORGET_INTERVAL_SECONDS),
    }

    for task, interval in overrides.items():
        if task in scheduler.schedules:
            scheduler.schedules[task]["interval"] = interval


def _tasks_for_mode(mode: str) -> List[str]:
    """Map a consolidation mode to its task identifiers."""
    if mode == "full":
        return ["decay", "creative", "cluster", "forget", "full"]
    if mode in CONSOLIDATION_TASK_FIELDS:
        return [mode]
    return [mode]


def _persist_consolidation_run(graph: Any, result: Dict[str, Any]) -> None:
    """Record consolidation outcomes and update scheduler metadata."""
    mode = result.get("mode", "unknown")
    completed_at = result.get("completed_at") or utc_now()
    started_at = result.get("started_at") or completed_at
    success = bool(result.get("success"))
    dry_run = bool(result.get("dry_run"))

    try:
        graph.query(
            f"""
            CREATE (r:{CONSOLIDATION_RUN_LABEL} {{
                id: $id,
                mode: $mode,
                task: $task,
                success: $success,
                dry_run: $dry_run,
                started_at: $started_at,
                completed_at: $completed_at,
                result: $result
            }})
            """,
            {
                "id": uuid.uuid4().hex,
                "mode": mode,
                "task": mode,
                "success": success,
                "dry_run": dry_run,
                "started_at": started_at,
                "completed_at": completed_at,
                "result": json.dumps(result, default=str),
            },
        )
    except Exception:
        logger.exception("Failed to record consolidation run history")

    for task in _tasks_for_mode(mode):
        field = CONSOLIDATION_TASK_FIELDS.get(task)
        if not field:
            continue
        try:
            graph.query(
                f"""
                MERGE (c:{CONSOLIDATION_CONTROL_LABEL} {{id: $id}})
                SET c.{field} = $timestamp
                """,
                {
                    "id": CONSOLIDATION_CONTROL_NODE_ID,
                    "timestamp": completed_at,
                },
            )
        except Exception:
            logger.exception("Failed to update consolidation control for task %s", task)

    try:
        graph.query(
            f"""
            MATCH (r:{CONSOLIDATION_RUN_LABEL})
            WITH r ORDER BY r.started_at DESC
            SKIP $keep
            DELETE r
            """,
            {"keep": CONSOLIDATION_HISTORY_LIMIT},
        )
    except Exception:
        logger.exception("Failed to prune consolidation history")


def _build_scheduler_from_graph(graph: Any) -> Optional[ConsolidationScheduler]:
    vector_store = get_qdrant_client()
    consolidator = _build_consolidator_from_config(graph, vector_store)
    scheduler = ConsolidationScheduler(consolidator)
    _apply_scheduler_overrides(scheduler)

    control = _load_control_record(graph)
    for task, field in CONSOLIDATION_TASK_FIELDS.items():
        iso_value = control.get(field)
        last_run = _parse_iso_datetime(iso_value)
        if last_run and task in scheduler.schedules:
            scheduler.schedules[task]["last_run"] = last_run

    return scheduler


def _build_consolidator_from_config(graph: Any, vector_store: Any) -> MemoryConsolidator:
    return MemoryConsolidator(
        graph,
        vector_store,
        delete_threshold=CONSOLIDATION_DELETE_THRESHOLD,
        archive_threshold=CONSOLIDATION_ARCHIVE_THRESHOLD,
        grace_period_days=CONSOLIDATION_GRACE_PERIOD_DAYS,
        importance_protection_threshold=CONSOLIDATION_IMPORTANCE_PROTECTION_THRESHOLD,
        protected_types=set(CONSOLIDATION_PROTECTED_TYPES),
    )


def _run_consolidation_tick() -> None:
    graph = get_memory_graph()
    if graph is None:
        return

    scheduler = _build_scheduler_from_graph(graph)
    if scheduler is None:
        return

    try:
        tick_start = time.perf_counter()
        results = scheduler.run_scheduled_tasks(
            decay_threshold=CONSOLIDATION_DECAY_IMPORTANCE_THRESHOLD
        )
        for result in results:
            _persist_consolidation_run(graph, result)

            # Emit SSE event for real-time monitoring
            task_type = result.get("mode", "unknown")
            steps = result.get("steps", {})
            affected_count = 0

            # Count affected memories from each step
            if "decay" in steps:
                affected_count += steps["decay"].get("updated", 0)
            if "creative" in steps:
                affected_count += steps["creative"].get("created", 0)
            if "cluster" in steps:
                affected_count += steps["cluster"].get("meta_memories_created", 0)
            if "forget" in steps:
                affected_count += steps["forget"].get("archived", 0)
                affected_count += steps["forget"].get("deleted", 0)

            elapsed_ms = int((time.perf_counter() - tick_start) * 1000)
            next_runs = scheduler.get_next_runs()

            emit_event(
                "consolidation.run",
                {
                    "task_type": task_type,
                    "affected_count": affected_count,
                    "elapsed_ms": elapsed_ms,
                    "success": result.get("success", False),
                    "next_scheduled": next_runs.get(task_type, "unknown"),
                    "steps": list(steps.keys()),
                },
                utc_now,
            )
    except Exception:
        logger.exception("Consolidation scheduler tick failed")


def consolidation_worker() -> None:
    """Background loop that triggers consolidation tasks."""
    logger.info("Consolidation scheduler thread started")
    while state.consolidation_stop_event and not state.consolidation_stop_event.wait(
        CONSOLIDATION_TICK_SECONDS
    ):
        _run_consolidation_tick()


def init_consolidation_scheduler() -> None:
    """Ensure the background consolidation scheduler is running."""
    if state.consolidation_thread and state.consolidation_thread.is_alive():
        return

    stop_event = Event()
    state.consolidation_stop_event = stop_event
    state.consolidation_thread = Thread(
        target=consolidation_worker,
        daemon=True,
        name="consolidation-scheduler",
    )
    state.consolidation_thread.start()
    # Kick off an initial tick so schedules are populated quickly.
    _run_consolidation_tick()
    logger.info("Consolidation scheduler initialized")


def enrichment_worker() -> None:
    """Background worker that processes memories for enrichment."""
    while True:
        try:
            if state.enrichment_queue is None:
                time.sleep(ENRICHMENT_IDLE_SLEEP_SECONDS)
                continue

            try:
                job: EnrichmentJob = state.enrichment_queue.get(
                    timeout=ENRICHMENT_IDLE_SLEEP_SECONDS
                )
            except Empty:
                continue

            with state.enrichment_lock:
                state.enrichment_pending.discard(job.memory_id)
                state.enrichment_inflight.add(job.memory_id)

            enrich_start = time.perf_counter()
            emit_event(
                "enrichment.start",
                {
                    "memory_id": job.memory_id,
                    "attempt": job.attempt + 1,
                },
                utc_now,
            )

            try:
                processed = enrich_memory(job.memory_id, forced=job.forced)
                state.enrichment_stats.record_success(job.memory_id)
                elapsed_ms = int((time.perf_counter() - enrich_start) * 1000)
                emit_event(
                    "enrichment.complete",
                    {
                        "memory_id": job.memory_id,
                        "success": True,
                        "elapsed_ms": elapsed_ms,
                        "skipped": not processed,
                    },
                    utc_now,
                )
                if not processed:
                    logger.debug("Enrichment skipped for %s (already processed)", job.memory_id)
            except Exception as exc:  # pragma: no cover - background thread
                state.enrichment_stats.record_failure(str(exc))
                elapsed_ms = int((time.perf_counter() - enrich_start) * 1000)
                emit_event(
                    "enrichment.failed",
                    {
                        "memory_id": job.memory_id,
                        "error": str(exc)[:100],
                        "attempt": job.attempt + 1,
                        "elapsed_ms": elapsed_ms,
                        "will_retry": job.attempt + 1 < ENRICHMENT_MAX_ATTEMPTS,
                    },
                    utc_now,
                )
                logger.exception("Failed to enrich memory %s", job.memory_id)
                if job.attempt + 1 < ENRICHMENT_MAX_ATTEMPTS:
                    time.sleep(ENRICHMENT_FAILURE_BACKOFF_SECONDS)
                    enqueue_enrichment(job.memory_id, forced=job.forced, attempt=job.attempt + 1)
                else:
                    logger.error(
                        "Giving up on enrichment for %s after %s attempts",
                        job.memory_id,
                        job.attempt + 1,
                    )
            finally:
                with state.enrichment_lock:
                    state.enrichment_inflight.discard(job.memory_id)
                state.enrichment_queue.task_done()
        except Exception:  # pragma: no cover - defensive catch-all
            logger.exception("Error in enrichment worker loop")
            time.sleep(ENRICHMENT_FAILURE_BACKOFF_SECONDS)


def init_embedding_pipeline() -> None:
    """Initialize the background embedding generation pipeline."""
    if state.embedding_queue is not None:
        return

    state.embedding_queue = Queue()
    state.embedding_thread = Thread(target=embedding_worker, daemon=True)
    state.embedding_thread.start()
    logger.info("Embedding pipeline initialized")


def enqueue_embedding(memory_id: str, content: str) -> None:
    """Queue a memory for async embedding generation."""
    if not memory_id or not content or state.embedding_queue is None:
        return

    with state.embedding_lock:
        if memory_id in state.embedding_pending or memory_id in state.embedding_inflight:
            return

        state.embedding_pending.add(memory_id)
        state.embedding_queue.put((memory_id, content))


def embedding_worker() -> None:
    """Background worker that generates embeddings and stores them in Qdrant with batching."""
    batch: List[Tuple[str, str]] = []  # List of (memory_id, content) tuples
    batch_deadline = time.time() + EMBEDDING_BATCH_TIMEOUT_SECONDS

    while True:
        try:
            if state.embedding_queue is None:
                time.sleep(1)
                continue

            # Calculate remaining time until batch deadline
            timeout = max(0.1, batch_deadline - time.time())

            try:
                memory_id, content = state.embedding_queue.get(timeout=timeout)
                batch.append((memory_id, content))

                # Process batch if full
                if len(batch) >= EMBEDDING_BATCH_SIZE:
                    _process_embedding_batch(batch)
                    batch = []
                    batch_deadline = time.time() + EMBEDDING_BATCH_TIMEOUT_SECONDS

            except Empty:
                # Timeout reached - process whatever we have
                if batch:
                    _process_embedding_batch(batch)
                    batch = []
                batch_deadline = time.time() + EMBEDDING_BATCH_TIMEOUT_SECONDS
                continue

        except Exception:  # pragma: no cover - defensive catch-all
            logger.exception("Error in embedding worker loop")
            # Process any pending batch before sleeping
            if batch:
                try:
                    _process_embedding_batch(batch)
                except Exception:
                    logger.exception("Failed to process batch during error recovery")
                batch = []
            time.sleep(1)
            batch_deadline = time.time() + EMBEDDING_BATCH_TIMEOUT_SECONDS


def _process_embedding_batch(batch: List[Tuple[str, str]]) -> None:
    """Process a batch of embeddings efficiently."""
    if not batch:
        return

    memory_ids = [item[0] for item in batch]
    contents = [item[1] for item in batch]

    # Mark all as inflight
    with state.embedding_lock:
        for memory_id in memory_ids:
            state.embedding_pending.discard(memory_id)
            state.embedding_inflight.add(memory_id)

    try:
        # Generate embeddings in batch
        embeddings = _generate_real_embeddings_batch(contents)

        # Store each embedding individually (Qdrant operations are fast)
        for memory_id, content, embedding in zip(memory_ids, contents, embeddings):
            try:
                _store_embedding_in_qdrant(memory_id, content, embedding)
                logger.debug("Generated and stored embedding for %s", memory_id)
            except Exception:  # pragma: no cover
                logger.exception("Failed to store embedding for %s", memory_id)
    except Exception:  # pragma: no cover
        logger.exception("Failed to generate batch embeddings")
    finally:
        # Mark all as complete
        with state.embedding_lock:
            for memory_id in memory_ids:
                state.embedding_inflight.discard(memory_id)

        # Mark all queue items as done
        for _ in batch:
            state.embedding_queue.task_done()


def _store_embedding_in_qdrant(memory_id: str, content: str, embedding: List[float]) -> None:
    """Store a pre-generated embedding in Qdrant with memory metadata."""
    qdrant_client = get_qdrant_client()
    if qdrant_client is None:
        return

    graph = get_memory_graph()
    if graph is None:
        return

    # Fetch latest memory data from FalkorDB for payload
    result = graph.query("MATCH (m:Memory {id: $id}) RETURN m", {"id": memory_id})
    if not getattr(result, "result_set", None):
        logger.warning("Memory %s not found in FalkorDB, skipping Qdrant update", memory_id)
        return

    node = result.result_set[0][0]
    properties = getattr(node, "properties", {})

    # Store in Qdrant
    try:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=memory_id,
                    vector=embedding,
                    payload={
                        "content": properties.get("content", content),
                        "tags": properties.get("tags", []),
                        "tag_prefixes": properties.get("tag_prefixes", []),
                        "importance": properties.get("importance", 0.5),
                        "timestamp": properties.get("timestamp", utc_now()),
                        "type": properties.get("type", "Context"),
                        "confidence": properties.get("confidence", 0.5),
                        "updated_at": properties.get("updated_at", utc_now()),
                        "last_accessed": properties.get("last_accessed", utc_now()),
                        "metadata": json.loads(properties.get("metadata", "{}")),
                        "relevance_score": properties.get("relevance_score"),
                    },
                )
            ],
        )
        logger.info("Stored embedding for %s in Qdrant", memory_id)
    except Exception:  # pragma: no cover - log full stack trace
        logger.exception("Qdrant upsert failed for %s", memory_id)


def generate_and_store_embedding(memory_id: str, content: str) -> None:
    """Generate embedding for content and store in Qdrant (legacy single-item API)."""
    embedding = _generate_real_embedding(content)
    _store_embedding_in_qdrant(memory_id, content, embedding)


# ---------------------------------------------------------------------------
# Background Sync Worker
# ---------------------------------------------------------------------------


def init_sync_worker() -> None:
    """Initialize the background sync worker if auto-repair is enabled."""
    if not SYNC_AUTO_REPAIR:
        logger.info("Sync auto-repair disabled (SYNC_AUTO_REPAIR=false)")
        return

    if state.sync_thread is not None:
        return

    state.sync_stop_event = Event()
    state.sync_thread = Thread(target=sync_worker, daemon=True)
    state.sync_thread.start()
    logger.info("Sync worker initialized (interval: %ds)", SYNC_CHECK_INTERVAL_SECONDS)


def sync_worker() -> None:
    """Background worker that detects and repairs FalkorDB/Qdrant sync drift.

    This is non-destructive: only adds missing embeddings, never removes existing ones.
    """
    while not state.sync_stop_event.is_set():
        try:
            # Wait for the interval (or until stop event)
            if state.sync_stop_event.wait(timeout=SYNC_CHECK_INTERVAL_SECONDS):
                break  # Stop event set

            _run_sync_check()

        except Exception:
            logger.exception("Error in sync worker")
            # Sleep briefly on error before retrying
            time.sleep(60)


def _run_sync_check() -> None:
    """Check for sync drift and repair if needed."""
    graph = get_memory_graph()
    qdrant = get_qdrant_client()

    if graph is None or qdrant is None:
        logger.debug("Sync check skipped: services unavailable")
        return

    try:
        # Get memory IDs from FalkorDB
        falkor_result = graph.query("MATCH (m:Memory) RETURN m.id AS id")
        falkor_ids: Set[str] = set()
        for row in getattr(falkor_result, "result_set", []) or []:
            if row[0]:
                falkor_ids.add(str(row[0]))

        # Get point IDs from Qdrant
        qdrant_ids: Set[str] = set()
        offset = None
        while True:
            result = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            points, next_offset = result
            for point in points:
                qdrant_ids.add(str(point.id))
            if next_offset is None:
                break
            offset = next_offset

        # Check for drift
        missing_ids = falkor_ids - qdrant_ids

        state.sync_last_run = utc_now()
        state.sync_last_result = {
            "falkordb_count": len(falkor_ids),
            "qdrant_count": len(qdrant_ids),
            "missing_count": len(missing_ids),
        }

        if not missing_ids:
            logger.debug("Sync check: no drift detected (%d memories)", len(falkor_ids))
            return

        logger.warning(
            "Sync drift detected: %d memories missing from Qdrant (will auto-repair)",
            len(missing_ids),
        )

        # Queue missing memories for embedding
        for memory_id in missing_ids:
            # Fetch content to queue
            mem_result = graph.query(
                "MATCH (m:Memory {id: $id}) RETURN m.content", {"id": memory_id}
            )
            if getattr(mem_result, "result_set", None):
                content = mem_result.result_set[0][0]
                if content:
                    enqueue_embedding(memory_id, content)

        logger.info("Queued %d memories for sync repair", len(missing_ids))

    except Exception:
        logger.exception("Sync check failed")


def enrich_memory(memory_id: str, *, forced: bool = False) -> bool:
    """Enrich a memory with relationships, patterns, and entity extraction."""
    graph = get_memory_graph()
    if graph is None:
        raise RuntimeError("FalkorDB unavailable for enrichment")

    result = graph.query("MATCH (m:Memory {id: $id}) RETURN m", {"id": memory_id})

    if not result.result_set:
        logger.debug("Skipping enrichment for %s; memory not found", memory_id)
        return False

    node = result.result_set[0][0]
    properties = getattr(node, "properties", None)
    if not isinstance(properties, dict):
        properties = dict(getattr(node, "__dict__", {}))

    metadata_raw = properties.get("metadata")
    metadata = _parse_metadata_field(metadata_raw) or {}
    if not isinstance(metadata, dict):
        metadata = {"_raw_metadata": metadata}

    already_processed = bool(properties.get("processed"))
    if already_processed and not forced:
        return False

    content = properties.get("content", "") or ""
    entities = extract_entities(content)

    tags = list(dict.fromkeys(_normalize_tag_list(properties.get("tags"))))
    entity_tags: Set[str] = set()

    if entities:
        entities_section = metadata.setdefault("entities", {})
        if not isinstance(entities_section, dict):
            entities_section = {}
        for category, values in entities.items():
            if not values:
                continue
            entities_section[category] = sorted(values)
            for value in values:
                slug = _slugify(value)
                if slug:
                    entity_tags.add(f"entity:{category}:{slug}")
        metadata["entities"] = entities_section

    if entity_tags:
        tags = list(dict.fromkeys(tags + sorted(entity_tags)))

    tag_prefixes = _compute_tag_prefixes(tags)

    temporal_links = find_temporal_relationships(graph, memory_id)
    pattern_info = detect_patterns(graph, memory_id, content)
    semantic_neighbors = link_semantic_neighbors(graph, memory_id)

    if ENRICHMENT_ENABLE_SUMMARIES:
        existing_summary = properties.get("summary")
        summary = generate_summary(content, existing_summary if forced else None)
    else:
        summary = properties.get("summary")

    enrichment_meta = metadata.setdefault("enrichment", {})
    if not isinstance(enrichment_meta, dict):
        enrichment_meta = {}
    enrichment_meta.update(
        {
            "last_run": utc_now(),
            "forced": forced,
            "temporal_links": temporal_links,
            "patterns_detected": pattern_info,
            "semantic_neighbors": [
                {"id": neighbour_id, "score": score} for neighbour_id, score in semantic_neighbors
            ],
        }
    )
    metadata["enrichment"] = enrichment_meta

    update_payload = {
        "id": memory_id,
        "metadata": json.dumps(metadata, default=str),
        "tags": tags,
        "tag_prefixes": tag_prefixes,
        "summary": summary,
        "enriched_at": utc_now(),
    }

    graph.query(
        """
        MATCH (m:Memory {id: $id})
        SET m.metadata = $metadata,
            m.tags = $tags,
            m.tag_prefixes = $tag_prefixes,
            m.summary = $summary,
            m.enriched = true,
            m.enriched_at = $enriched_at,
            m.processed = true
        """,
        update_payload,
    )

    qdrant_client = get_qdrant_client()
    if qdrant_client is not None:
        try:
            qdrant_client.set_payload(
                collection_name=COLLECTION_NAME,
                points=[memory_id],
                payload={
                    "tags": tags,
                    "tag_prefixes": tag_prefixes,
                    "metadata": metadata,
                },
            )
        except UnexpectedResponse as exc:
            # 404 means embedding upload hasn't completed yet (race condition)
            if exc.status_code == 404:
                logger.debug(
                    "Qdrant payload sync skipped - point not yet uploaded: %s", memory_id[:8]
                )
            else:
                logger.warning("Qdrant payload sync failed (%d): %s", exc.status_code, memory_id)
        except Exception:
            logger.exception("Failed to sync Qdrant payload for enriched memory %s", memory_id)

    logger.debug(
        "Enriched memory %s (temporal=%s, patterns=%s, semantic=%s)",
        memory_id,
        temporal_links,
        pattern_info,
        len(semantic_neighbors),
    )

    return True


def _temporal_cutoff() -> str:
    """Return an ISO timestamp 7 days ago to bound temporal queries."""
    return (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()


def find_temporal_relationships(graph: Any, memory_id: str, limit: int = 5) -> int:
    """Find and create temporal relationships with recent memories."""
    created = 0
    try:
        result = graph.query(
            """
            MATCH (m1:Memory {id: $id})
            WITH m1, m1.timestamp AS ts
            WHERE ts IS NOT NULL
            MATCH (m2:Memory)
            WHERE m2.id <> $id
                AND m2.timestamp IS NOT NULL
                AND m2.timestamp < ts
                AND m2.timestamp > $cutoff
            RETURN m2.id
            ORDER BY m2.timestamp DESC
            LIMIT $limit
            """,
            {"id": memory_id, "limit": limit, "cutoff": _temporal_cutoff()},
            timeout=5000,
        )

        timestamp = utc_now()
        for (related_id,) in result.result_set:
            if not related_id:
                continue
            graph.query(
                """
                MATCH (m1:Memory {id: $id1})
                MATCH (m2:Memory {id: $id2})
                MERGE (m1)-[r:PRECEDED_BY]->(m2)
                SET r.updated_at = $timestamp,
                    r.count = COALESCE(r.count, 0) + 1
                """,
                {"id1": memory_id, "id2": related_id, "timestamp": timestamp},
            )
            created += 1
    except Exception:
        logger.exception("Failed to find temporal relationships")

    return created


def detect_patterns(graph: Any, memory_id: str, content: str) -> List[Dict[str, Any]]:
    """Detect if this memory exemplifies or creates patterns."""
    detected: List[Dict[str, Any]] = []

    try:
        memory_type, confidence = memory_classifier.classify(content)
        result = graph.query(
            """
            MATCH (m:Memory)
            WHERE m.type = $type
                AND m.id <> $id
                AND m.confidence > 0.5
            RETURN m.id, m.content
            LIMIT 10
            """,
            {"type": memory_type, "id": memory_id},
        )

        similar_texts = [content]
        similar_texts.extend(row[1] for row in result.result_set if len(row) > 1)
        similar_count = len(result.result_set)

        if similar_count >= 3:
            tokens = Counter()
            for text in similar_texts:
                for token in re.findall(r"[a-zA-Z]{4,}", (text or "").lower()):
                    if token in SEARCH_STOPWORDS:
                        continue
                    tokens[token] += 1

            top_terms = [term for term, _ in tokens.most_common(5)]
            pattern_id = f"pattern-{memory_type}-{uuid.uuid4().hex[:8]}"
            description = f"Pattern across {similar_count + 1} {memory_type} memories" + (
                f" highlighting {', '.join(top_terms)}" if top_terms else ""
            )

            graph.query(
                """
                MERGE (p:Pattern {type: $type})
                ON CREATE SET
                    p.id = $pattern_id,
                    p.content = $description,
                    p.confidence = $initial_confidence,
                    p.observations = 1,
                    p.key_terms = $key_terms,
                    p.created_at = $timestamp
                ON MATCH SET
                    p.confidence = CASE
                        WHEN p.confidence < 0.95 THEN p.confidence + 0.05
                        ELSE 0.95
                    END,
                    p.observations = p.observations + 1,
                    p.key_terms = $key_terms,
                    p.updated_at = $timestamp
                """,
                {
                    "type": memory_type,
                    "pattern_id": pattern_id,
                    "description": description,
                    "initial_confidence": 0.35,
                    "key_terms": top_terms,
                    "timestamp": utc_now(),
                },
            )

            graph.query(
                """
                MATCH (m:Memory {id: $memory_id})
                MATCH (p:Pattern {type: $type})
                MERGE (m)-[r:EXEMPLIFIES]->(p)
                SET r.confidence = $confidence,
                    r.updated_at = $timestamp
                """,
                {
                    "type": memory_type,
                    "memory_id": memory_id,
                    "confidence": confidence,
                    "timestamp": utc_now(),
                },
            )

            detected.append(
                {
                    "type": memory_type,
                    "similar_memories": similar_count,
                    "key_terms": top_terms,
                }
            )
    except Exception:
        logger.exception("Failed to detect patterns")

    return detected


def link_semantic_neighbors(graph: Any, memory_id: str) -> List[Tuple[str, float]]:
    client = get_qdrant_client()
    if client is None:
        return []

    try:
        points = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[memory_id],
            with_vectors=True,
            with_payload=False,
        )
    except Exception:
        logger.exception("Failed to fetch vector for memory %s", memory_id)
        return []

    if not points or getattr(points[0], "vector", None) is None:
        return []

    query_vector = points[0].vector

    try:
        neighbors = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=ENRICHMENT_SIMILARITY_LIMIT + 1,
            with_payload=False,
        )
    except Exception:
        logger.exception("Semantic neighbor search failed for %s", memory_id)
        return []

    created: List[Tuple[str, float]] = []
    timestamp = utc_now()

    for neighbour in neighbors:
        neighbour_id = str(neighbour.id)
        if neighbour_id == memory_id:
            continue

        score = float(neighbour.score or 0.0)
        if score < ENRICHMENT_SIMILARITY_THRESHOLD:
            continue

        params = {
            "id1": memory_id,
            "id2": neighbour_id,
            "score": score,
            "timestamp": timestamp,
        }

        graph.query(
            """
            MATCH (a:Memory {id: $id1})
            MATCH (b:Memory {id: $id2})
            MERGE (a)-[r:SIMILAR_TO]->(b)
            SET r.score = $score,
                r.updated_at = $timestamp
            """,
            params,
        )

        graph.query(
            """
            MATCH (a:Memory {id: $id1})
            MATCH (b:Memory {id: $id2})
            MERGE (b)-[r:SIMILAR_TO]->(a)
            SET r.score = $score,
                r.updated_at = $timestamp
            """,
            params,
        )

        created.append((neighbour_id, score))

    return created


@app.errorhandler(Exception)
def handle_exceptions(exc: Exception):
    """Return JSON responses for both HTTP and unexpected errors."""
    if isinstance(exc, HTTPException):
        response = {
            "status": "error",
            "code": exc.code,
            "message": exc.description or exc.name,
        }
        return jsonify(response), exc.code

    logger.exception("Unhandled error")
    response = {
        "status": "error",
        "code": 500,
        "message": "Internal server error",
    }
    return jsonify(response), 500



def _normalize_tags(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(tag, str) for tag in value):
        return value
    abort(400, description="'tags' must be a list of strings or a single string")


def _coerce_importance(value: Any) -> float:
    if value is None:
        return 0.5
    try:
        score = float(value)
    except (TypeError, ValueError):
        abort(400, description="'importance' must be a number")
    if score < 0 or score > 1:
        abort(400, description="'importance' must be between 0 and 1")
    return score


def _coerce_embedding(value: Any) -> Optional[List[float]]:
    if value is None or value == "":
        return None
    vector: List[Any]
    if isinstance(value, list):
        vector = value
    elif isinstance(value, str):
        vector = [part.strip() for part in value.split(",") if part.strip()]
    else:
        raise ValueError("Embedding must be a list of floats or a comma-separated string")

    # Use effective dimension (auto-detected or config)
    expected_dim = state.effective_vector_size
    if len(vector) != expected_dim:
        raise ValueError(f"Embedding must contain exactly {expected_dim} values")

    try:
        return [float(component) for component in vector]
    except ValueError as exc:
        raise ValueError("Embedding must contain numeric values") from exc


def _generate_placeholder_embedding(content: str) -> List[float]:
    """Generate a deterministic embedding vector from the content."""
    digest = hashlib.sha256(content.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = random.Random(seed)
    # Use effective dimension (auto-detected or config)
    return [rng.random() for _ in range(state.effective_vector_size)]


def _generate_real_embedding(content: str) -> List[float]:
    """Generate an embedding using the configured provider."""
    init_embedding_provider()

    if state.embedding_provider is None:
        logger.warning("No embedding provider available, using placeholder")
        return _generate_placeholder_embedding(content)

    expected_dim = state.effective_vector_size
    try:
        embedding = state.embedding_provider.generate_embedding(content)
        if not isinstance(embedding, list) or len(embedding) != expected_dim:
            logger.warning(
                "Provider %s returned %s dims (expected %d); falling back to placeholder",
                state.embedding_provider.provider_name(),
                len(embedding) if isinstance(embedding, list) else "invalid",
                expected_dim,
            )
            return _generate_placeholder_embedding(content)
        return embedding
    except Exception as e:
        logger.warning("Failed to generate embedding: %s", str(e))
        return _generate_placeholder_embedding(content)


def _generate_real_embeddings_batch(contents: List[str]) -> List[List[float]]:
    """Generate multiple embeddings in a single batch for efficiency."""
    init_embedding_provider()

    if not contents:
        return []

    if state.embedding_provider is None:
        logger.debug("No embedding provider available, falling back to placeholder embeddings")
        return [_generate_placeholder_embedding(c) for c in contents]

    expected_dim = state.effective_vector_size
    try:
        embeddings = state.embedding_provider.generate_embeddings_batch(contents)
        if not embeddings or any(len(e) != expected_dim for e in embeddings):
            logger.warning(
                "Provider %s returned invalid dims in batch; using placeholders",
                state.embedding_provider.provider_name() if state.embedding_provider else "unknown",
            )
            return [_generate_placeholder_embedding(c) for c in contents]
        return embeddings
    except Exception as e:
        logger.warning("Failed to generate batch embeddings: %s", str(e))
        return [_generate_placeholder_embedding(c) for c in contents]


def _fetch_relations(graph: Any, memory_id: str) -> List[Dict[str, Any]]:
    try:
        records = graph.query(
            """
            MATCH (m:Memory {id: $id})-[r]->(related:Memory)
            RETURN type(r) as relation_type, r.strength as strength, related
            ORDER BY coalesce(r.updated_at, related.timestamp) DESC
            LIMIT $limit
            """,
            {"id": memory_id, "limit": RECALL_RELATION_LIMIT},
        )
    except Exception:  # pragma: no cover - log full stack trace in production
        logger.exception("Failed to fetch relations for memory %s", memory_id)
        return []

    connections: List[Dict[str, Any]] = []
    for relation_type, strength, related in records.result_set:
        connections.append(
            {
                "type": relation_type,
                "strength": strength,
                "memory": _summarize_relation_node(_serialize_node(related)),
            }
        )
    return connections




from automem.api.admin import create_admin_blueprint_full
from automem.api.consolidation import create_consolidation_blueprint_full
from automem.api.enrichment import create_enrichment_blueprint
from automem.api.graph import create_graph_blueprint

# Register blueprints after all routes are defined
from automem.api.health import create_health_blueprint
from automem.api.memory import create_memory_blueprint_full
from automem.api.recall import create_recall_blueprint

health_bp = create_health_blueprint(
    get_memory_graph,
    get_qdrant_client,
    state,
    GRAPH_NAME,
    COLLECTION_NAME,
    utc_now,
)

enrichment_bp = create_enrichment_blueprint(
    _require_admin_token,
    state,
    enqueue_enrichment,
    ENRICHMENT_MAX_ATTEMPTS,
)

recall_bp = create_recall_blueprint(
    get_memory_graph,
    get_qdrant_client,
    _normalize_tag_list,
    _normalize_timestamp,
    _parse_time_expression,
    _extract_keywords,
    _compute_metadata_score,
    _result_passes_filters,
    _graph_keyword_search,
    _vector_search,
    _vector_filter_only_tag_search,
    RECALL_MAX_LIMIT,
    logger,
    ALLOWED_RELATIONS,
    RECALL_RELATION_LIMIT,
    _serialize_node,
    _summarize_relation_node,
    update_last_accessed,
)

memory_bp = create_memory_blueprint_full(
    get_memory_graph,
    get_qdrant_client,
    _normalize_tags,
    _normalize_tag_list,
    _compute_tag_prefixes,
    _coerce_importance,
    _coerce_embedding,
    _normalize_timestamp,
    utc_now,
    _serialize_node,
    _parse_metadata_field,
    _generate_real_embedding,
    enqueue_enrichment,
    enqueue_embedding,
    lambda content: memory_classifier.classify(content),
    PointStruct,
    COLLECTION_NAME,
    ALLOWED_RELATIONS,
    RELATIONSHIP_TYPES,
    state,
    logger,
    update_last_accessed,
    get_openai_client,
)

admin_bp = create_admin_blueprint_full(
    _require_admin_token,
    init_openai,
    get_openai_client,
    get_qdrant_client,
    get_memory_graph,
    PointStruct,
    COLLECTION_NAME,
    lambda: state.effective_vector_size,  # Use runtime-detected dimension
    EMBEDDING_MODEL,
    utc_now,
    logger,
)

consolidation_bp = create_consolidation_blueprint_full(
    get_memory_graph,
    get_qdrant_client,
    _build_consolidator_from_config,
    _persist_consolidation_run,
    _build_scheduler_from_graph,
    _load_recent_runs,
    state,
    CONSOLIDATION_TICK_SECONDS,
    CONSOLIDATION_HISTORY_LIMIT,
    logger,
)

graph_bp = create_graph_blueprint(
    get_memory_graph,
    get_qdrant_client,
    _serialize_node,
    COLLECTION_NAME,
    logger,
)

stream_bp = create_stream_blueprint(
    require_api_token=require_api_token,
)

app.register_blueprint(health_bp)
app.register_blueprint(enrichment_bp)
app.register_blueprint(memory_bp)
app.register_blueprint(admin_bp)
app.register_blueprint(recall_bp)
app.register_blueprint(consolidation_bp)
app.register_blueprint(graph_bp)
app.register_blueprint(stream_bp)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8001"))
    logger.info("Starting Flask API on port %s", port)
    init_falkordb()
    init_qdrant()
    init_openai()  # Still needed for memory type classification
    init_embedding_provider()  # New provider pattern for embeddings
    init_enrichment_pipeline()
    init_embedding_pipeline()
    init_consolidation_scheduler()
    init_sync_worker()
    # Use :: for IPv6 dual-stack (Railway), 0.0.0.0 for IPv4-only (Windows/local)
    host = os.environ.get("FLASK_HOST", "::")
    app.run(host=host, port=port, debug=False)
