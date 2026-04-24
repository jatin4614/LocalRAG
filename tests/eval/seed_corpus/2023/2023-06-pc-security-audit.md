# PC Security Audit — June 2023

**Date:** 2023-06-12
**Author:** Platform Security
**Doc ID:** 3001

## Findings

- Audit of endpoint baseline completed 2023-06-08 covering 142 production hosts.
- Two hosts flagged with stale kernel versions; patch window scheduled 2023-06-19.
- Disk encryption verified on all laptops issued after 2022-10-01.

## Remediation

- Rotate shared service credentials by 2023-07-15.
- Enable MFA enforcement for the ops-admin group (tracked as SEC-1142).
- Close out three long-standing exceptions granted during the 2022 migration.
