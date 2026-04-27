#!/usr/bin/env python3
"""E2E test harness against the LIVE LocalRAG production stack.

Sends 40 queries (10 per intent) through the full bridge pipeline via
``GET /api/rag/stream/{chat_id}`` (which exercises intent classification,
shadow logging, and ``_log_rag_query``). The SSE ``hits`` event carries
the ranked top-K with doc_ids + filenames + the bridge's final intent.

Captures per-query telemetry by correlating:
  * SSE stream stages: embed, retrieve, rerank, mmr, expand, budget, hits, done
  * SSE ``hits`` event: ``intent`` + ``intent_reason`` + ranked doc list
  * Stdout container logs: "rag: request started req=..." (req_id) +
    ``_log_rag_query`` JSON line (final_intent, kbs, hits, total_ms)
  * Shadow log (/var/log/orgchat/qu_shadow.jsonl): regex/llm labels,
    confidence, agree, escalation_reason
  * vllm-qu logs: count of POST /v1/chat/completions in the time window
  * Prometheus counters (silent_failure_total, qu_invocations,
    qu_escalations) — diff before/after the 40-query run

Usage::

    python3 tests/e2e/run_e2e.py
    # writes results to /tmp/e2e_results.json
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx


WEBUI = "http://localhost:6100"
QDRANT = "http://localhost:6333"
PROM = "http://localhost:9091"
TOKEN = Path("/tmp/.rag_admin_token").read_text().strip()
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


# --- 40 queries (10 per intent) -------------------------------------------

QUERIES: List[Tuple[str, str]] = [
    # metadata (10)
    ("metadata", "What are total files available with you?"),
    ("metadata", "How many reports do you have?"),
    ("metadata", "From when to when do you have data?"),
    ("metadata", "Show me everything you know"),
    ("metadata", "List all documents in the KB"),
    ("metadata", "What's in your knowledge base?"),
    ("metadata", "Total documents in the corpus?"),
    ("metadata", "What date range do the reports cover?"),
    ("metadata", "Catalog of all files"),
    ("metadata", "Do you have documents about OFC?"),
    # global (10)
    ("global", "List all dates of reports"),
    ("global", "Summarize the entire knowledge base"),
    ("global", "Every report's high-level theme"),
    ("global", "Across all months, what trends are visible?"),
    ("global", "Overview of all communication outages"),
    ("global", "What are all the BSF posts mentioned anywhere?"),
    ("global", "Recap of all OFC issues"),
    ("global", "Highlights from all months"),
    ("global", "State of communications across the entire corpus"),
    ("global", "Full list of alternative comm methods used"),
    # specific (10)
    ("specific", "What is JFC?"),
    ("specific", "Which posts had OFC breaks pending restoration?"),
    ("specific", "What's the alternative for OFC failure?"),
    ("specific", "Tell me about Hulu post"),
    ("specific", "What is the role of MCCS in alternative communications?"),
    ("specific", "Which post is Bombay OP?"),
    ("specific", "What does DMR mean in this context?"),
    ("specific", "What was the issue with Khapuri post?"),
    ("specific", "Which formation does 75 BSF belong to?"),
    ("specific", "Resolution status of pending foot-link items?"),
    # specific_date (10)
    ("specific_date", "What happened on 5 Jan 2026?"),
    ("specific_date", "Show me the 17 Feb 2026 report"),
    ("specific_date", "What was the OFC status on 26 Mar 2026?"),
    ("specific_date", "Communication state on 1 April 2026"),
    ("specific_date", "Report from 10 February 2026"),
    ("specific_date", "What issues were noted on 9 March 2026?"),
    ("specific_date", "Summary of 4 February 2026 events"),
    ("specific_date", "Activities on 12 March 2026"),
    ("specific_date", "What did the 16 March 2026 report say?"),
    ("specific_date", "April 6 2026 OFC update"),
]


# --- Postgres helpers (via docker exec) -----------------------------------

PG_PASSWORD = "5330b21d1cfd22a3bd545c45ba74c397"


def docker_logs_since(container: str, since_unix: int, contains: Optional[str] = None) -> List[str]:
    """Read docker logs for ``container`` since unix-epoch ``since_unix``.

    Optionally filters lines containing the given substring.
    """
    proc = subprocess.run(
        ["docker", "logs", container, "--since", str(since_unix)],
        capture_output=True, text=True, timeout=15,
    )
    lines: List[str] = []
    for line in (proc.stdout + proc.stderr).splitlines():
        if contains is None or contains in line:
            lines.append(line)
    return lines


def docker_exec_cat(container: str, path: str) -> str:
    proc = subprocess.run(
        ["docker", "exec", container, "cat", path],
        capture_output=True, text=True, timeout=15,
    )
    return proc.stdout


# --- Setup: create one chat per intent batch ------------------------------

def create_chat(title: str) -> str:
    r = httpx.post(
        f"{WEBUI}/api/v1/chats/new",
        headers={**HEADERS, "Content-Type": "application/json"},
        json={"chat": {"title": title}},
        timeout=10.0,
    )
    r.raise_for_status()
    return r.json()["id"]


def attach_kb(chat_id: str, kb_id: int = 1) -> None:
    r = httpx.put(
        f"{WEBUI}/api/chats/{chat_id}/kb_config",
        headers={**HEADERS, "Content-Type": "application/json"},
        json={"config": [{"kb_id": kb_id, "subtag_ids": []}]},
        timeout=10.0,
    )
    r.raise_for_status()


# --- Hit a single query (full pipeline + telemetry) ----------------------

def hit_query(chat_id: str, query: str) -> Dict[str, Any]:
    """Run one query through /api/rag/stream and capture telemetry."""
    started_unix = int(time.time())
    started_t = time.time()

    sse_events: List[Dict[str, Any]] = []
    timed_out = False
    try:
        with httpx.stream(
            "GET",
            f"{WEBUI}/api/rag/stream/{chat_id}",
            params={"q": query},
            headers=HEADERS,
            timeout=httpx.Timeout(connect=10.0, read=90.0, write=10.0, pool=10.0),
        ) as r:
            current_event = None
            for line in r.iter_lines():
                if not line:
                    current_event = None
                    continue
                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    payload = line.split(":", 1)[1].strip()
                    try:
                        data = json.loads(payload)
                        if isinstance(data, dict):
                            data["_event"] = current_event
                            sse_events.append(data)
                    except json.JSONDecodeError:
                        pass
                if current_event == "done" and any(
                    e.get("_event") == "done" for e in sse_events
                ):
                    # we got the terminal "done" event with empty payload
                    pass
    except httpx.ReadTimeout:
        timed_out = True
    except Exception as e:
        sse_events.append({"_event": "harness_error", "err": str(e)})
    stream_ms = int((time.time() - started_t) * 1000)

    # Pull the SSE "hits" event for top-K + intent
    hits_event = next(
        (e for e in sse_events if e.get("stage") == "hits" or e.get("_event") == "hits"),
        {},
    )
    hits_list = hits_event.get("hits", []) if isinstance(hits_event, dict) else []
    intent_from_hits = hits_event.get("intent") if isinstance(hits_event, dict) else None
    intent_reason = hits_event.get("intent_reason") if isinstance(hits_event, dict) else None
    # hits_list often starts with "current-datetime" + "kb-catalog" preamble
    real_hits = [
        h for h in hits_list
        if h.get("doc_id") not in (None, "null") and h.get("kb_id")
    ]
    top3_doc_ids = [h.get("doc_id") for h in real_hits[:3]]
    top3_filenames = [h.get("filename") for h in real_hits[:3]]
    top3_scores = [round(h.get("score", 0.0), 4) for h in real_hits[:3] if "score" in h]

    # Find the SSE done payload (stage==done with total_ms + sources)
    sse_done = next(
        (e for e in sse_events if e.get("stage") == "done" and "total_ms" in e),
        {},
    )
    sse_total_ms = sse_done.get("total_ms")
    sse_sources = sse_done.get("sources")

    # Wait for log flush.
    time.sleep(0.6)

    # Pull req_id + intent + kbs + hits + total_ms from container logs
    log_lines = docker_logs_since("orgchat-open-webui", started_unix - 1)
    req_id = None
    final_intent = None
    final_kbs: List[int] = []
    final_hits = None
    final_total_ms = None
    rag_started_lines = [ln for ln in log_lines if "rag: request started" in ln]
    rag_query_lines = [ln for ln in log_lines if '"event":"rag_query"' in ln]
    # Take the most recent matching pair (last in window).
    if rag_started_lines:
        m = re.search(r"req=([0-9a-f-]+)", rag_started_lines[-1])
        if m:
            req_id = m.group(1)
    if rag_query_lines:
        m = re.search(r"\{\"event\":\"rag_query\".*?\}", rag_query_lines[-1])
        if m:
            try:
                payload = json.loads(m.group(0))
                final_intent = payload.get("intent")
                final_kbs = payload.get("kbs", [])
                final_hits = payload.get("hits")
                final_total_ms = payload.get("total_ms")
            except json.JSONDecodeError:
                pass

    # Pull most recent shadow log entry matching this query.
    shadow_raw = docker_exec_cat("orgchat-open-webui", "/var/log/orgchat/qu_shadow.jsonl")
    regex_label = regex_reason = llm_label = None
    llm_confidence = None
    agree = None
    escalation_reason = None
    for ln in reversed(shadow_raw.splitlines()):
        try:
            entry = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if entry.get("query") == query:
            regex_label = entry.get("regex_label")
            regex_reason = entry.get("regex_reason")
            llm_label = entry.get("llm_label")
            llm_confidence = entry.get("llm_confidence")
            agree = entry.get("agree")
            escalation_reason = entry.get("escalation_reason")
            break

    # Look at vllm-qu logs for chat/completions calls in our window
    qu_logs = docker_logs_since(
        "orgchat-vllm-qu", started_unix - 1,
        contains="POST /v1/chat/completions",
    )
    vllm_qu_called_count = len(qu_logs)

    # SSE stage timings
    stage_ms = {}
    for ev in sse_events:
        stg = ev.get("stage")
        if not stg or "ms" not in ev:
            continue
        if ev.get("status") in (None, "done", "running") and stg not in stage_ms:
            stage_ms[stg] = ev.get("ms")

    return {
        "query": query,
        "req_id": req_id,
        "regex_label": regex_label,
        "regex_reason": regex_reason,
        "llm_label": llm_label,
        "llm_confidence": llm_confidence,
        "agree": agree,
        "escalation_reason": escalation_reason,
        "final_intent": final_intent,
        "final_kbs": final_kbs,
        "final_hits": final_hits,
        "final_total_ms": final_total_ms,
        "intent_from_hits": intent_from_hits,
        "intent_reason": intent_reason,
        "sse_total_ms": sse_total_ms,
        "sse_sources": sse_sources,
        "stage_ms": stage_ms,
        "stream_ms": stream_ms,
        "stream_timed_out": timed_out,
        "top3_doc_ids": top3_doc_ids,
        "top3_filenames": top3_filenames,
        "top3_scores": top3_scores,
        "all_hits_count": len(hits_list),
        "real_hits_count": len(real_hits),
        "vllm_qu_called_count": vllm_qu_called_count,
        "sse_event_count": len(sse_events),
    }


# --- Prometheus snapshot ---------------------------------------------------

def prom_query(q: str) -> List[Dict[str, Any]]:
    try:
        r = httpx.get(f"{PROM}/api/v1/query", params={"query": q}, timeout=5.0)
        r.raise_for_status()
        return r.json().get("data", {}).get("result", [])
    except Exception:
        return []


def prom_snapshot() -> Dict[str, Any]:
    return {
        "qu_invocations": prom_query("sum by (source) (rag_qu_invocations_total)"),
        "qu_escalations": prom_query("sum by (reason) (rag_qu_escalations_total)"),
        "silent_failures": prom_query("sum by (where) (rag_silent_failure_total)"),
        "rag_retrieval_p50": prom_query(
            "histogram_quantile(0.5, sum by (le) (rate(rag_retrieval_latency_seconds_bucket[5m])))"
        ),
        "rag_retrieval_p95": prom_query(
            "histogram_quantile(0.95, sum by (le) (rate(rag_retrieval_latency_seconds_bucket[5m])))"
        ),
    }


# --- Main loop -------------------------------------------------------------

def main() -> None:
    print(f"E2E run started at {datetime.utcnow().isoformat()}Z")
    print(f"Token: {TOKEN[:30]}…")
    print(f"WebUI: {WEBUI}")
    print()

    prom_before = prom_snapshot()
    print("Prometheus snapshot (before):")
    for k, v in prom_before.items():
        print(f"  {k}: {len(v)} series")
    print()

    intents = ["metadata", "global", "specific", "specific_date"]
    chats: Dict[str, str] = {}
    for intent in intents:
        chat_id = create_chat(f"e2e-{intent}-{int(time.time())}")
        attach_kb(chat_id, kb_id=1)
        chats[intent] = chat_id
        print(f"  chat for {intent}: {chat_id}")

    results: List[Dict[str, Any]] = []
    for i, (expected, query) in enumerate(QUERIES, start=1):
        chat_id = chats[expected]
        print(f"  [{i:02d}/40] {expected:14} | {query[:62]}", flush=True)
        try:
            r = hit_query(chat_id, query)
            r["expected_intent"] = expected
            # The bridge's "final_intent" from _log_rag_query is the canonical
            # routing decision. Fall back to intent_from_hits SSE event if the
            # _log_rag_query line didn't make it (early return path may differ).
            actual = r.get("final_intent") or r.get("intent_from_hits")
            r["actual_intent"] = actual
            r["correct_routing"] = (actual == expected)
            results.append(r)
            print(
                f"           regex={r.get('regex_label')!s:14} "
                f"llm={r.get('llm_label')!s:14} "
                f"actual={actual!s:14} "
                f"hits={r.get('real_hits_count'):2d} "
                f"ms={r.get('final_total_ms') or r.get('sse_total_ms')!s:>5} "
                f"corr={'Y' if r['correct_routing'] else 'N'}"
            )
        except Exception as e:
            print(f"           ERR: {e}")
            results.append({
                "query": query, "expected_intent": expected,
                "harness_error": str(e),
            })
        time.sleep(0.5)

    prom_after = prom_snapshot()

    out = {
        "started_at": datetime.utcnow().isoformat() + "Z",
        "total_queries": len(QUERIES),
        "chats": chats,
        "prom_before": prom_before,
        "prom_after": prom_after,
        "results": results,
    }
    Path("/tmp/e2e_results.json").write_text(json.dumps(out, indent=2, default=str))
    print()
    print(f"Wrote /tmp/e2e_results.json — {len(results)} results")


if __name__ == "__main__":
    main()
