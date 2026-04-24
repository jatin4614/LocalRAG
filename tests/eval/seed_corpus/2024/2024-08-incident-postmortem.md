# Incident Postmortem — 2024-08-17 Auth Outage

**Date:** 2024-08-19
**Author:** SRE On-call
**Doc ID:** 4002

## Summary

A 47-minute auth outage started 2024-08-17 14:22 UTC after a Redis cluster
failover promoted a lagging replica as primary. Login requests returned 503
until the session store was manually re-seeded.

## Root Cause

Replica lag alerts were silenced during a prior maintenance window and never
re-enabled. The monitor config drift went undetected for six weeks.

## Action Items

- Restore the replica-lag alert; owner: SRE team, due 2024-08-24.
- Add a weekly cron that reconciles silenced alerts against ticket IDs.
- Review the runbook entry for Redis primary loss (currently 2022 version).
