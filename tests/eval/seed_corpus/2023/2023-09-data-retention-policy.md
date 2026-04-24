# Data Retention Policy v2

**Date:** 2023-09-04
**Author:** Legal & Compliance
**Doc ID:** 3002

## Scope

Applies to all customer-facing data stores as of 2023-09-01. Supersedes the
2021 retention policy.

## Retention Windows

- Transactional logs: 90 days hot, 2 years cold storage.
- Customer chat transcripts: 18 months, then hashed.
- Auth events: 7 years (regulatory requirement).

## Deletion Workflow

Right-to-erasure requests are processed within 30 days per the compliance
SLA. Escalate ambiguous cases to the legal review queue.
