# `src/schema_fixtures/` — the historical schema batches, FROZEN

Each `.sql` file here is the **verbatim** contents of one historical commit's
schema batch — the string literal `let schema = r#" … "#;` inside
`open_and_migrate_catalogue` — extracted mechanically and checked in as inert
data. `src/lib.rs`'s `#[cfg(test)] mod schema_upgrade_fixture_tests` pulls each
one in with `include_str!`, replays it onto a temp **file** with a raw
`Connection::open`, and then opens that file through the **production**
`open_and_migrate_catalogue`.

They exist because nothing else in this crate measures the migration: every
other fixture starts from a **current-schema** catalogue that the current code
just created. One failing statement in the 765-line batch aborts it,
`open_and_migrate_catalogue` returns `None`, and the product shows an **empty
library** on every existing catalogue (S93) — or an additive Restore that
silently does nothing (`merge_catalogue_from_backup`). Slice K, register R-41.

## ⛔ NEVER EDIT A FIXTURE

**These are frozen artifacts.** If a future schema change makes a test here
fail, the finding is in the change, not in the fixture. Never "update" a file
here to make the suite green — **that deletes the pin**, and the class it
catches (`R-60`, `R-82`, `R-96`: a CREATE-body edit that never reached an
existing catalogue) has already shipped undetected three times.

A **new** schema state gets a **new** file plus a new row in `VINTAGES` and a
new `#[test]`. `EXPECTED_VINTAGE_COUNT` in that module is the vacuity guard that
makes forgetting impossible.

## Provenance — reproduce any file byte-for-byte

```sh
cd photolibrariancore
git show "<sha>:src/lib.rs" > /tmp/old-lib.rs
s=$(grep -n 'let schema = r#"' /tmp/old-lib.rs | head -1 | cut -d: -f1)
e=$(awk -v st="$s" 'NR>st && /^[[:space:]]*"#;[[:space:]]*$/ {print NR; exit}' /tmp/old-lib.rs)
sed -n "$((s+1)),$((e-1))p" /tmp/old-lib.rs | diff - src/schema_fixtures/<sha>.sql
```

⚠️ Two traps, both hit during this work:

1. **Do not cut at `conn.execute_batch(schema)`** — there is Rust between the
   closing `"#;` and that call, and a `//` comment in the output becomes
   `Parser Error: syntax error at or near "//"`. The `awk` boundary above is the
   fix.
2. **Quote the git revision** (`git show "$c:src/lib.rs"`). Unquoted in zsh,
   `$c:src/…` is eaten as a history modifier and `git show` prints the whole
   commit as a patch instead of the blob.

## The files

The **reachable window opens on 2026-07-03**: a backup is the only inbound path
for an old catalogue, backups did not exist until S111 (`8193d5d`, 2026-07-03),
and restore gates `manifest.formatVersion <= 1`. Q-27 bounds everything earlier
to zero (S114 renamed the bundle ID; S127 was a fresh-catalogue production
baseline). Fingerprinting **every** commit that touched the batch collapses the
window into **six** distinct states.

| tag | file | commit | date | lines | bytes | tables | live retired indexes | note |
|---|---|---|---|---|---|---|---|---|
| **V1** | `8193d5d.sql` | `8193d5d` | 2026-07-03 | 614 | 36332 | 15 | 14 | the first reachable vintage — S111, backups begin. ⚠️ The DDL state itself originates at `a9300ac` (2026-06-25) and is byte-identical there; the file is named for the S111 commit because that is the earliest vintage a **backup** can hold. |
| **V2** | `3c5895b.sql` | `3c5895b` | 2026-07-04 | 649 | 38259 | 17 | 14 | operation log |
| **V3** | `68f88f9.sql` | `68f88f9` | 2026-07-20 | 666 | 39157 | 18 | 14 | |
| **V4** | `ee9640c.sql` | `ee9640c` | 2026-07-21 | 679 | 40001 | 18 | 14 | `19fedf7` is the same state |
| **V5** | `b8ea0a6.sql` | `b8ea0a6` | 2026-08-06 | 688 | 40500 | 19 | 14 | `09834a6` (S173, 2026-09-15) is the same state — it changed **Rust**, and only comments and blank lines in the batch |
| **V6** | `2bc221e.sql` | `2bc221e` | 2026-09-18 | 765 | 45598 | 19 | 0 | S179. **The identity control** — byte-identical to the working tree's batch, so its upgrade must produce ZERO divergence |
| **G** | `b3f9998.sql` | `b3f9998` | 2026-05-04 | 52 | 1827 | 1 | 0 | **genesis**, archival. `87f2e93` is the same state. |

`sha256` of each file, for the record:

```
9efbd986d28b823d3d49a36c0b3c2a8c747ba39b9b8468348b516eacb6779f01  b3f9998.sql
2fbc2656cdd314c2215a7bb06d66f71272737faffb1d84475e6f7c9ead9d24f4  8193d5d.sql
ee146d834f1689f605ddd84906c656627c4ba135fdc0abfd2f142c53e3839b24  3c5895b.sql
955dcd794bf8c5c3ca7c26f12d1730bd00ee94257da92e17589949e47e80f64d  68f88f9.sql
e79617a9a3bd17a51f009ecf2eec46356818f069d017266dbe156045a3a1102d  ee9640c.sql
e0710199517ba3871e479743ce905db7ebf23a16960ee13a016fab3b7eb7c525  b8ea0a6.sql
b49df4acf4cc9bef312f81fc05491407cf4e13d73992e9198ab570567d3edeb6  2bc221e.sql
```

## Why genesis is kept

All seven current `images` indexes are over **genesis-era** columns, and every
reachable vintage already carries every column the batch touches — so **no
reachable fixture can be made to fail** the "the open returns `Some`" assertion
by the S93 mutation (a `CREATE INDEX` hoisted above the `ALTER` that adds its
column). Genesis has **1 table instead of 19** and **29 `images` columns instead
of 69**, so 40 columns and 18 tables arrive through the migration. It is the
only fixture that gives that mutation something to break.

It is also the only fixture with an **allow-list** — four entries, each carrying
its register number and its reason, and **asserted in both directions** (an
unlisted divergence fails; a listed divergence that is not observed fails). The
list lives in `src/lib.rs`, not here.

## How the vintage set was derived

```sh
# distinct states across every commit that touched src/lib.rs
git log --format='%H %cd' --date=short --reverse -- src/lib.rs
# per commit: extract (above), then fingerprint
sed -e 's/--.*$//' "$f" | sed -e 's/[[:space:]]\{1,\}/ /g' -e 's/^ //' -e 's/ $//' \
  | grep -v '^$' | md5
```

⚠️ **The fingerprint must drop blank lines as well as comments.** A
comment-only edit that also adds a blank line otherwise reads as a new schema
state — which is exactly what `b8ea0a6` → `09834a6` (S173) does.
