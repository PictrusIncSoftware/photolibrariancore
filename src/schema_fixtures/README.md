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
new `#[test]`.

⚠️ **`EXPECTED_VINTAGE_COUNT` does NOT make forgetting impossible** — that claim
was wrong and was corrected in fix round 1 (2026-09-25, reviewer finding K-F3).
It fires only when a `VINTAGES` row is added or removed without updating it, so
it is structurally blind to the case that actually matters: **a schema edit that
ships with no new fixture.** A *correct* future edit — a new column in the
`images` CREATE body plus its matching `ALTER … ADD COLUMN IF NOT EXISTS` — was
executed by the reviewer and left every test in the module GREEN while the
identity control silently became a stale vintage still wearing the flag.

The guard that actually catches it is the assertion
**`the_identity_control_fixture_is_byte_identical_to_the_in_tree_schema_batch`**,
added in fix round 1: it re-extracts the batch from `lib.rs` at the same two
boundaries this README's recipe cuts at and compares it to the
`is_identity_control` vintage's DDL. Move the batch without adding a fixture and
that test goes red naming the four things to do.

## Provenance — reproduce any file byte-for-byte

```sh
cd photolibrariancore
c=<sha>
git show "${c}:src/lib.rs" > /tmp/old-lib.rs         # ⚠️ ${c}: — see trap 2
s=$(grep -n 'let schema = r#"' /tmp/old-lib.rs | head -1 | cut -d: -f1)
e=$(awk -v st="$s" 'NR>st && /^[[:space:]]*"#;[[:space:]]*$/ {print NR; exit}' /tmp/old-lib.rs)
sed -n "$((s+1)),$((e-1))p" /tmp/old-lib.rs | diff - "src/schema_fixtures/${c}.sql"
```

⚠️ Three traps, all hit during this work:

1. **Do not cut at `conn.execute_batch(schema)`** — there is Rust between the
   closing `"#;` and that call, and a `//` comment in the output becomes
   `Parser Error: syntax error at or near "//"`. The `awk` boundary above is the
   fix.
2. ⭐ **BRACE the git revision: `git show "${c}:src/lib.rs"`.** Quoting is NOT
   the fix — this project's shell is **zsh**, where `$c:s…` is consumed as a
   *parameter-expansion modifier*, so even the quoted `git show "$c:src/lib.rs"`
   prints **the commit as a patch** instead of the blob. Measured on 2026-09-25:
   the bare form returned 3,374 lines (a patch) where `${c}:` returned 27,918
   (the blob), and the patch's own `+ let schema = r#"` line then broke the
   `awk` boundary. `git cat-file -p "${c}:src/lib.rs"` is an equally safe form.
   *(This README previously presented quoting as the fix — corrected in fix
   round 1, reviewer finding K-F5.)*
3. The fixture files end with a trailing newline, as `sed -n` emits; `diff` above
   will tell you if yours does not.

## The files

A backup is the only inbound path for an old catalogue, backups did not exist
until S111 (`8193d5d`, 2026-07-03), and restore gates
`manifest.formatVersion <= 1`; the launch path cannot see anything older (S114
renamed the bundle ID, S127 was a fresh-catalogue production baseline).
Fingerprinting **every** commit that touched the batch collapses the
post-2026-07-03 window into **six** distinct states, V1…V6.

⚠️ **NARROWED in fix round 1 (2026-09-25, reviewer finding K-F1).** This section
used to say "the reachable window opens on 2026-07-03". ⛔ **That does not
follow.** 2026-07-03 is when a backup could first be **taken**, not the earliest
schema a backup can **hold**: a ZIP taken in the 2026-07-03 → 2026-07-22 window
(after S111 began backups, before S127's fresh-catalogue baseline) contains a
catalogue whose tables were **CREATED** in May or June 2026. V1…V6 each model a
catalogue *born* at their own state, so the June-born-and-migrated-forward shape
a real archive of that window has was modelled by nothing — which is why
`4599235` (tier 3) and `b3f9998` (genesis) are in the set and why neither is
"archival". Whether such an archive exists on Richard's disk is Q-29, still
unanswered.

⭐ What bounds the risk is **measurement**, not unreachability. All **32**
distinct historical batch states have been driven through the production
`open_and_migrate_catalogue` on the bundled DuckDB 1.5.5 — by the reviewer in
round 1 and again independently in fix round 1 (2026-09-25): **every one opens
`Some`, keeps its seeded rows with values intact, gets the S173 marker and
converges `directory_path`.** **21** of them diverge — **18** in R-42's columns
and nothing else (2026-05-10 … 2026-06-20), and **3** (2026-05-04, 05-06, 05-08)
that additionally show `images.rotation` (R-60), `images.id` (R-82) and
`images.file_size` (R-96), which is what genesis models. Divergence ends at
2026-06-22 (`326314c`). ⚠️ The round-1 review states **13** where the
re-measurement says 21/18; the measured numbers are the ones recorded here.
R-42 is
therefore **Restore-reachable and harmless** — harmless because the batch pairs
each bare ALTER with a **value backfill** (`UPDATE images SET is_video = FALSE
WHERE is_video IS NULL` and the three `UPDATE keyword SET …` siblings), so only
the column *attributes* ever diverge and never the data. Not harmless because it
cannot happen.

| tag | file | commit | date | lines | bytes | tables | live retired indexes | note |
|---|---|---|---|---|---|---|---|---|
| **V1** | `8193d5d.sql` | `8193d5d` | 2026-07-03 | 614 | 36332 | 15 | 14 | the first reachable vintage — S111, backups begin. ⚠️ The DDL state itself originates at `a9300ac` (2026-06-25) and is byte-identical there; the file is named for the S111 commit because that is the earliest vintage a **backup** can hold. |
| **V2** | `3c5895b.sql` | `3c5895b` | 2026-07-04 | 649 | 38259 | 17 | 14 | operation log |
| **V3** | `68f88f9.sql` | `68f88f9` | 2026-07-20 | 666 | 39157 | 18 | 14 | |
| **V4** | `ee9640c.sql` | `ee9640c` | 2026-07-21 | 679 | 40001 | 18 | 14 | `19fedf7` is the same state |
| **V5** | `b8ea0a6.sql` | `b8ea0a6` | 2026-08-06 | 688 | 40500 | 19 | 14 | `09834a6` (S173, 2026-09-15) is the same state — it changed **Rust**, and only comments and blank lines in the batch |
| **V6** | `2bc221e.sql` | `2bc221e` | 2026-09-18 | 765 | 45598 | 19 | 0 | S179. **The identity control** — byte-identical to the working tree's batch, so its upgrade must produce ZERO divergence |
| **T3** | `4599235.sql` | `4599235` | 2026-06-01 | 128 | 7370 | 2 | 0 | ⭐ **tier 3**, added in fix round 1 (K-F1). A catalogue **BORN 2026-06-01** — the shape a real 2026-07-03…07-22 archive holds. The commit that introduced the `keyword` table, so it is the only fixture with a **June-born `keyword`**: nullable `origin` with no default, and no `collection`/`color`/`is_video`. Shows **all five** of R-42's columns at once (13 of the 32 historical states diverge; this is the strongest single choice, and by 2026-06-22 the divergence is gone). |
| **G** | `b3f9998.sql` | `b3f9998` | 2026-05-04 | 52 | 1827 | 1 | 0 | **genesis** — ⛔ **not "archival"**: the EXTREME instance of the Restore-reachable class, and the only fixture that can fail the "open returns `Some`" assertion under the S93 mutation. `87f2e93` is the same state. **Do not delete it.** |

`sha256` of each file, for the record:

```
9efbd986d28b823d3d49a36c0b3c2a8c747ba39b9b8468348b516eacb6779f01  b3f9998.sql
fe0f2d43a31f465e26eeb945ae63e7d1286e808730546c07edbb844102f5dc75  4599235.sql
2fbc2656cdd314c2215a7bb06d66f71272737faffb1d84475e6f7c9ead9d24f4  8193d5d.sql
ee146d834f1689f605ddd84906c656627c4ba135fdc0abfd2f142c53e3839b24  3c5895b.sql
955dcd794bf8c5c3ca7c26f12d1730bd00ee94257da92e17589949e47e80f64d  68f88f9.sql
e79617a9a3bd17a51f009ecf2eec46356818f069d017266dbe156045a3a1102d  ee9640c.sql
e0710199517ba3871e479743ce905db7ebf23a16960ee13a016fab3b7eb7c525  b8ea0a6.sql
b49df4acf4cc9bef312f81fc05491407cf4e13d73992e9198ab570567d3edeb6  2bc221e.sql
```

## Why genesis is kept — ⛔ and why it must not be deleted

All seven current `images` indexes are over **genesis-era** columns, and every
vintage from V1 on already carries every column the batch touches — so the S93
mutation (a `CREATE INDEX` hoisted above the `ALTER` that adds its column) has
nothing to break in the reachable window. Genesis has **1 table instead of 19**
and **29 `images` columns instead of 69**, so 40 columns and 18 tables arrive
through the migration: it is the **broadest** pin on that assertion.
**Mutation-proved three times on the same mutation** — the S93 shape over
`images.is_video` is RED on GENESIS and GREEN on V1…V6 (implementer M1, reviewer
M-B, fix-round-1 T1).

⚠️ Fix round 1 **corrects** the earlier "the only fixture that can fail it"
wording: T1 is red on GENESIS **and** on the new tier-3 fixture, which lacks
`images.is_video` too. Genesis remains the only fixture that reaches the columns
and tables which arrived **before 2026-06-01**.

⚠️ The earlier framing — "genesis is archival … documentation of a closed class,
not live risk" — was an **invitation to trim the flagship pin**, and is
withdrawn (fix round 1, reviewer finding K-F1). Genesis is the extreme instance
of the Restore-reachable class.

Two fixtures carry an **allow-list**: genesis (four entries — `images.file_size`
R-96, `images.id` R-82, `images.is_video` R-42, `images.rotation` R-60) and tier
3 (five entries — R-42's `images.is_video` plus `keyword.collection`, `.color`,
`.is_video` and `.origin`). Every entry carries its register number, its
**measured shape on both sides**, and its reason, and each list is **asserted in
three directions**: an unlisted divergence fails, a listed divergence that is
not observed fails, and a listed column that starts diverging in a *different*
way fails. The lists live in `src/lib.rs`, not here.

## ⚠️ Known limit of assertion (e): it sees COLUMNS, and the blind spot is empty today

The fresh-vs-upgraded comparison is a set equality over
`(table_name, column_name, data_type, is_nullable, column_default)` from
`duckdb_columns()`. **Constraints and indexes are outside it**: indexes are
covered only by the marker assertion (one name, `idx_images_directory_path`) and
the retired-index assertion (the 14 `FOCUS_WRITEBACK_RETIRED_INDEXES` names);
table constraints are not compared at all. So a future `CHECK` or `UNIQUE` added
to a CREATE body, or an index removed from the batch without a matching `DROP`,
would be an R-60-class divergence this pin cannot see.

⭐ **Measured 2026-09-25**, all eight fixtures, bundled DuckDB 1.5.5,
`duckdb_constraints()` and `duckdb_indexes()` as multisets fresh vs upgraded:

| fixture | constraints (fresh / upgraded) | indexes | differing keys |
|---|---|---|---|
| V1 … V6 | 134 / 134 | 38 / 38 | **0** |
| T3 (`4599235`) | 134 / **129** | 38 / 38 | 2 — `images NOT NULL` 7 vs 6, `keyword NOT NULL` 9 vs 5 |
| G (`b3f9998`) | 134 / **133** | 38 / 38 | 1 — `images NOT NULL` 7 vs 6 |

Every one of those constraint differences is the **nullability of a column the
four-tuple already reports as an allow-list entry** (genesis: `images.is_video`;
T3: that plus `keyword.origin`/`.collection`/`.color`/`.is_video`), and there is
**no index difference at all** on any fixture. `keyword_visible` is created with
`CREATE OR REPLACE VIEW`, in the current batch and in every fixture that has it,
so a stale view definition is not a gap either. ⇒ **The blind spot is EMPTY as
of this date and the four-tuple is currently sufficient.** Closing the class
before it opens is cheap — the same set equality over `duckdb_constraints()` and
`duckdb_indexes()` passes as-is for V1…V6 — and is a brief-scope decision for
the coordinator, not a deviation: the brief specified the four-tuple.

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

⭐ Run over the **whole** history rather than the post-2026-07-03 window, the
same fingerprint yields **32 distinct batch states**. Every one has been driven
through the production open path (reviewer round 1; re-measured independently
2026-09-25). Measured divergence sets, verbatim from that run:

| born | states | diverged columns |
|---|---|---|
| 2026-05-04 `b3f9998` | 1 | `images.file_size`, `images.id`, `images.is_video`, `images.rotation` — **genesis, the extreme instance** |
| 2026-05-06 `1e1bd01` | 1 | `images.file_size`, `images.is_video`, `images.rotation` |
| 2026-05-08 `14e9810` | 1 | `images.is_video`, `images.rotation` |
| 2026-05-10 … 2026-05-18 | 3 | `images.is_video` |
| 2026-06-01 … 2026-06-02 | 2 | all **five** of R-42's — ⭐ `4599235` is the tier-3 fixture |
| 2026-06-10 … 2026-06-12 | 5 | 4 then 3 of R-42's, as `keyword.collection` (06-10) and then `keyword.color` (06-11) join the CREATE body with their defaults |
| 2026-06-13 … 2026-06-20 | 8 | `keyword.origin` alone — `images.is_video` and `keyword.is_video` converge at `9cd0bf8` |
| 2026-06-22 `326314c` onward | 11 | **none** — divergence ends here |

21 of 32 diverge; 18 of those in R-42's columns only. `4599235` is the tier-3
fixture because it is the single state that shows all five at once, and extending
the set past it buys duplicates of the same class rather than new information.
