# Shelved — H1 DuckDB-handoff spike (core repo)

**Archived 2026-06-16 (Session 83).** Pairs with the app-repo shelf
`PhotoLibrarian/Docs/BackgroundProcesses/SMAppServiceHelper-Shelved-2026-06-16/`
(which holds the Swift helper source, entitlements, and plist).

## What's here

- `duckdb_handoff_spike.rs` — the **Spike H1** binary. Proves the DuckDB
  single-writer lock / `CHECKPOINT` / reopen handoff that the after-quit
  background-enrichment helper would have relied on (only one read-write
  process per catalogue, so the daemon runs only while the app is closed).
  Result and scope are written up in the app repo's
  `Docs/SPIKE-H1-DuckDB-Handoff.md`.

## Why it was moved out of `src/bin/`

The background-enrichment helper was **shelved at S83** (the real
`BackgroundEnrichmentHelper` was AMFI-killed before `main()`; leading cause a
stale BTM/Login-Items orphan; shelved on cost/benefit — see
`SESSION_83_HANDOFF.md`). This file lived at `src/bin/duckdb_handoff_spike.rs`,
where Cargo **auto-discovers** it as a binary target and compiles it on every
build of the product crate. With the helper shelved it is dead weight in the
build, so it was archived here (out of the compile path) rather than deleted —
the spike is still worth re-running if the helper is ever resumed.

## How to re-run (if the helper is resumed)

Copy it back into `photolibrariancore/src/bin/`, then:

```
cargo run --bin duckdb_handoff_spike -- run <catalogue-path>
```

It writes only to a namespaced `_photolib_h1_handoff_spike` table and never
touches real catalogue tables; it works on a WAL-preserving copy of the real
catalogue. Modes: `run` (orchestrator) / `agent` / `open-once`.
