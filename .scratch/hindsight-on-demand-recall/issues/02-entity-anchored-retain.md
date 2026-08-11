# 02 — Entity-anchored retain enrichment

**What to build:** Retained Hindsight memories carry entity-name tags (e.g. 陳志明) alongside the existing `["corpchat","search"]`, so Hindsight's own compaction and any future recall have entity anchors — protecting against "content preserved, entity dropped". The tags are derived from the query and/or the retained hit messages, using existing entity sources (no new extraction machinery).

**Blocked by:** None — independent of ticket 01 (write side).

**Status:** resolved

- [x] Retain tags include person/entity names found in the hit messages — `extract_entity_tags` reads `customer_name` (fallback `external_userid`) + `company` from each hit's metadata
- [x] Existing `["corpchat","search"]` tags and `context` payload are preserved — entity tags are appended after the base tags; `context` unchanged
- [x] No new entity-extraction dependencies — reuses the metadata already carried on `raw_hits`; query-side names left to Hindsight's own entity extraction from `content`/`context`
- [x] Tests green: 6 new unit tests for `extract_entity_tags` (extraction, dedup, fallback, cap-5, malformed inputs); full suite 215 passed (209 + 6), 0 failures — run 2026-08-11
