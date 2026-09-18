//! S179 — live reproduction harness for the focus-analysis writeback's
//! one-target/one-row invariant ("UPDATE images ... WHERE id = ?1 updated 2 rows").
//!
//! WHY THIS IS AN IN-CRATE `#[cfg(test)]` MODULE AND NOT A CLI SCRIPT
//! ------------------------------------------------------------------
//! The trip was produced by the BUNDLED DuckDB engine (`v1.5.5`, pinned by
//! `duckdb = "=1.10505.0"`), not by the Homebrew `duckdb` CLI (v1.5.2). A CLI
//! reproduction can hide a bundled-engine bug entirely. This module therefore
//! drives the REAL production functions
//! (`plan_focus_analysis_writebacks` -> `write_focus_analysis_plans_in_transaction`
//! -> `update_focus_analysis_target`, all reached through
//! `update_focus_analysis_results_impl`) against a writable COPY of the real
//! catalogue, through the statically linked engine, exactly as the app does.
//!
//! It mirrors the S124/S126 `live_repro_tests` pattern: a private
//! `Connection` that never touches the global `CATALOGUE`, and an env-var gate
//! so `cargo test --lib` stays green (and silent) on any machine where the
//! variable is unset.
//!
//! HOW TO RUN
//! ----------
//!   # ALWAYS against a throwaway copy-on-write copy; NEVER the live catalogue.
//!   cp -c /path/to/snapshot/catalogue.db     /tmp/repro-A/catalogue.db
//!   cp -c /path/to/snapshot/catalogue.db.wal /tmp/repro-A/catalogue.db.wal   # if present
//!
//!   PLDIAG_REPRO_DB=/tmp/repro-A/catalogue.db \
//!   PLDIAG_REPRO_ARM=A \
//!   PLDIAG_REPRO_MAX_MINUTES=25 \
//!     cargo test --lib live_repro_focus_writeback -- --nocapture --test-threads=1
//!
//!   RUN IT UNDER `--release`. The dev profile compiles the bundled DuckDB at
//!   -O0, which makes a whole-catalogue arm hours slower than the app:
//!
//!   PLDIAG_REPRO_DB=... PLDIAG_REPRO_ARM=A PLDIAG_REPRO_MAX_MINUTES=25 \
//!     cargo test --release --lib live_repro_focus_writeback \
//!       -- --nocapture --test-threads=1
//!
//! ⚠️ WHAT THE SIGNAL IS, POST-FIX
//! ------------------------------
//! This harness was written against the PRE-fix code, where a bad change count
//! surfaced as an `Err` — a failed chunk with stage `apply`. After S179 that is
//! no longer the primary signal:
//!
//!   * an anomaly the catalogue CONFIRMS is now ACCEPTED, and appears only as a
//!     `focus_writeback_count_anomaly` warning on the SUCCESS receipt;
//!   * an anomaly the catalogue CANNOT confirm rolls back and is REPLAYED once,
//!     appearing as a `focus_writeback_chunk_retried` warning — on the success
//!     receipt if the replay carried it, on the failure receipt if it did not.
//!
//! So the driver counts and prints every warning on EVERY receipt, success or
//! failure, and the per-arm summary reports those counts beside the trips. A
//! run whose summary shows anomalies > 0 and trips = 0 is the fix WORKING, not
//! a no-repro.
//!
//! ⚠️ ARM A IS NO LONGER "AS SHIPPED" UNDER THE DEFAULT `PLDIAG_REPRO_MIGRATE=1`.
//! The production open now DROPS the fourteen secondary indexes on the columns
//! the writeback UPDATE writes, so a migrating open turns arm A into arm B. To
//! drive the genuine pre-fix index shape, run with `PLDIAG_REPRO_MIGRATE=0`
//! (which opens the copy raw) or recreate the fourteen by hand first.
//!
//! ENVIRONMENT
//! -----------
//!   PLDIAG_REPRO_DB            (REQUIRED) path to a WRITABLE catalogue copy.
//!                              Unset => the test returns immediately and passes.
//!   PLDIAG_REPRO_ARM           A | B | C   (default A)
//!       A  baseline: indexes exactly as shipped.
//!       B  hypothesis: DROP every secondary ART index that covers a column the
//!          writeback UPDATE writes — there are exactly FOURTEEN of them on
//!          `images` (`idx_focus_score`, `idx_focus_human_score`,
//!          `idx_focus_animal_score`, `idx_focus_foreground_score`,
//!          `idx_focus_saliency_score`, `idx_focus_animal_pose_score`,
//!          `idx_focus_whole_image_score`, `idx_focus_analysis_status`,
//!          `idx_face_count`, `idx_face_quality_best`,
//!          `idx_face_quality_average`, `idx_face_quality_min`,
//!          `idx_face_eyes_open_count`, `idx_face_blink_risk_count`). The PK,
//!          the `file_path` UNIQUE and the other eight secondary indexes
//!          (`camera_model`, `capture_datetime`, `color_label`,
//!          `created_timestamp`, `file_extension`, `flag`, `rating`,
//!          `directory_path`) stay.
//!          WHY THIS IS THE DISCRIMINATING ARM: with ANY index on an updated
//!          column DuckDB executes the UPDATE as DELETE + INSERT, and the
//!          change count it returns is the number of rows the SCAN FED to the
//!          operator, NOT the number of rows written (the operator dedupes row
//!          ids first). So a trip under arm A means a double-emitting scan and
//!          the committed data should still be correct. Dropping all fourteen
//!          turns the statement back into an in-place update, which takes the
//!          delete+insert branch out of the picture entirely.
//!       C  stress: arm A plus a CHECKPOINT every
//!          PLDIAG_REPRO_CHECKPOINT_EVERY chunks, and faces/keywords on EVERY
//!          result so the post-UPDATE statements run on every target.
//!   PLDIAG_REPRO_MAX_MINUTES   wall-clock budget for the whole arm (default 25).
//!   PLDIAG_REPRO_MAX_PASSES    passes over the candidate list (default 3).
//!   PLDIAG_REPRO_CHUNK         writeback chunk size (default 14 = the Mac
//!                              Studio's lane count, which is the production
//!                              chunk size on that machine).
//!   PLDIAG_REPRO_MAX_CANDIDATES  cap the candidate list (0 = no cap, default).
//!   PLDIAG_REPRO_SKIP_CANDIDATES skip the first N queue entries (default 0).
//!                              Useful because the low-id head of this
//!                              catalogue's queue is almost all lone files;
//!                              the RAW-twin sources (2 targets per source,
//!                              the shape job 6 tripped on) start further in.
//!   PLDIAG_REPRO_FACE_EVERY    1 result in N carries a face observation
//!                              (default 4; 0 = never; arm C forces 1).
//!   PLDIAG_REPRO_KEYWORD_EVERY 1 result in N carries auto keywords
//!                              (default 8; 0 = never; arm C forces 1).
//!   PLDIAG_REPRO_INCOMPLETE_EVERY 1 result in N is a non-`complete` status
//!                              (default 20; 0 = never).
//!   PLDIAG_REPRO_CHECKPOINT_EVERY  CHECKPOINT cadence in chunks for arm C
//!                              (default 2000; 0 = never).
//!   PLDIAG_REPRO_PROGRESS_EVERY    progress line cadence in chunks (default 200).
//!   PLDIAG_REPRO_MIGRATE       1 (default) = open through the production
//!                              `open_and_migrate_catalogue`; 0 = bare
//!                              `Connection::open` (the production initializer
//!                              sets NO threads/memory_limit/checkpoint_threshold
//!                              pragmas, so a bare open already matches the
//!                              engine settings the app runs with).
//!   PLDIAG_REPRO_STOP_ON_TRIP  1 (default) = stop the arm at the first
//!                              invariant trip after capturing diagnostics;
//!                              0 = keep going and count them.
//!
//! WHAT IT WRITES
//! --------------
//! Real COMMITTED writes to the copy, one transaction per chunk, exactly like
//! production. The copy is disposable. Nothing here rolls back, because the
//! trip is a property of a long committed write stream, not of one statement.

use super::*;
use std::time::{Duration, Instant};

const REPRO_ALGORITHM: &str = "pldiag-s179-focus-writeback-v1";

// ---------------------------------------------------------------------------
// env helpers
// ---------------------------------------------------------------------------

fn env_string(key: &str, fallback: &str) -> String
{
    std::env::var(key).unwrap_or_else(|_| fallback.to_string())
}

fn env_usize(key: &str, fallback: usize) -> usize
{
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(fallback)
}

fn env_f64(key: &str, fallback: f64) -> f64
{
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<f64>().ok())
        .unwrap_or(fallback)
}

// ---------------------------------------------------------------------------
// deterministic value generation (no `rand`/`uuid` dependency in this crate)
// ---------------------------------------------------------------------------

fn mix64(value: u64) -> u64
{
    let mut x = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Uppercase UUID-shaped attempt id, the same shape Swift's `UUID().uuidString`
/// produces for `analysis_run_id` in production.
fn synthetic_attempt_id(pass: usize) -> String
{
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mut state = mix64(nanos ^ ((std::process::id() as u64) << 32) ^ (pass as u64));
    let digits = b"0123456789ABCDEF";
    let mut out = String::with_capacity(36);
    for position in 0..32
    {
        if position == 8 || position == 12 || position == 16 || position == 20
        {
            out.push('-');
        }
        state = mix64(state);
        out.push(digits[((state >> 59) & 0xF) as usize] as char);
    }
    out
}

fn repro_observation(image_id: i64) -> FaceObservationResult
{
    let seed = mix64(image_id as u64);
    let unit = |shift: u32| ((mix64(seed ^ (shift as u64)) >> 11) as f64) / ((1u64 << 53) as f64);
    FaceObservationResult {
        face_index: 0,
        bounding_box_x: 0.1 + 0.2 * unit(1),
        bounding_box_y: 0.1 + 0.2 * unit(2),
        bounding_box_width: 0.2 + 0.1 * unit(3),
        bounding_box_height: 0.2 + 0.1 * unit(4),
        detection_confidence: Some(0.80 + 0.19 * unit(5)),
        face_capture_quality: Some(0.40 + 0.55 * unit(6)),
        face_focus_score: Some(60.0 + 400.0 * unit(7)),
        left_eye_open_score: Some(unit(8)),
        right_eye_open_score: Some(unit(9)),
        eyes_open_score: Some(unit(10)),
        blink_risk_score: Some(unit(11)),
        left_eye_x: Some(0.2),
        left_eye_y: Some(0.6),
        right_eye_x: Some(0.4),
        right_eye_y: Some(0.6),
        nose_x: Some(0.3),
        nose_y: Some(0.45),
        mouth_left_x: Some(0.23),
        mouth_left_y: Some(0.35),
        mouth_right_x: Some(0.37),
        mouth_right_y: Some(0.35),
    }
}

struct ResultShape
{
    face_every: usize,
    keyword_every: usize,
    incomplete_every: usize,
}

/// One synthesized writeback row whose values match what a real
/// `multi-subject-laplacian-v7` run writes for a scored image
/// (cf. the real trip row: focus_score 253.0454781582044, basis `human_face`,
/// status `complete`).
fn repro_result(
    image_id: i64,
    ordinal: usize,
    attempt_id: &str,
    shape: &ResultShape,
) -> FocusAnalysisResult
{
    let seed = mix64(image_id as u64 ^ mix64(ordinal as u64));
    let unit = |shift: u32| ((mix64(seed ^ (shift as u64)) >> 11) as f64) / ((1u64 << 53) as f64);

    let incomplete =
        shape.incomplete_every > 0 && ordinal % shape.incomplete_every == shape.incomplete_every - 1;
    let with_face = !incomplete && shape.face_every > 0 && ordinal % shape.face_every == 0;
    let with_keywords =
        !incomplete && shape.keyword_every > 0 && ordinal % shape.keyword_every == 0;

    if incomplete
    {
        // The analyzer's non-scored outcomes. `plan_focus_analysis_writebacks`
        // does NOT expand these to the RAW twin, and
        // `replace_face_observations_for_targets` only DELETEs for them.
        let status = if ordinal % (2 * shape.incomplete_every) == shape.incomplete_every - 1
        {
            "online_only"
        }
        else
        {
            "unreadable"
        };
        return FocusAnalysisResult {
            id: image_id,
            focus_score: None,
            focus_basis: None,
            algorithm_version: REPRO_ALGORITHM.to_string(),
            analysis_run_id: attempt_id.to_string(),
            status: status.to_string(),
            focus_human_score: None,
            focus_animal_score: None,
            focus_foreground_score: None,
            focus_saliency_score: None,
            focus_animal_pose_score: None,
            focus_whole_image_score: None,
            face_count: None,
            face_quality_best: None,
            face_quality_average: None,
            face_quality_min: None,
            face_eyes_open_count: None,
            face_blink_risk_count: None,
            auto_keywords: Vec::new(),
            face_observations: Vec::new(),
        };
    }

    let basis = if with_face { "human_face" } else { ["whole_image", "foreground", "saliency", "animal"][(seed % 4) as usize] };
    let score = 40.0 + 900.0 * unit(20);

    FocusAnalysisResult {
        id: image_id,
        focus_score: Some(score),
        focus_basis: Some(basis.to_string()),
        algorithm_version: REPRO_ALGORITHM.to_string(),
        analysis_run_id: attempt_id.to_string(),
        status: "complete".to_string(),
        focus_human_score: if with_face { Some(score) } else { None },
        focus_animal_score: Some(10.0 + 200.0 * unit(21)),
        focus_foreground_score: Some(10.0 + 400.0 * unit(22)),
        focus_saliency_score: Some(10.0 + 400.0 * unit(23)),
        focus_animal_pose_score: None,
        focus_whole_image_score: Some(10.0 + 800.0 * unit(24)),
        face_count: Some(if with_face { 1 } else { 0 }),
        face_quality_best: if with_face { Some(unit(25)) } else { None },
        face_quality_average: if with_face { Some(unit(26)) } else { None },
        face_quality_min: if with_face { Some(unit(27)) } else { None },
        face_eyes_open_count: Some(if with_face { 1 } else { 0 }),
        face_blink_risk_count: Some(0),
        auto_keywords: if with_keywords
        {
            vec!["PLDiag/S179".to_string(), "PLDiag/Repro".to_string()]
        }
        else
        {
            Vec::new()
        },
        face_observations: if with_face
        {
            vec![repro_observation(image_id)]
        }
        else
        {
            Vec::new()
        },
    }
}

// ---------------------------------------------------------------------------
// catalogue helpers
// ---------------------------------------------------------------------------

fn wal_bytes(db_path: &str) -> u64
{
    std::fs::metadata(format!("{}.wal", db_path))
        .map(|meta| meta.len())
        .unwrap_or(0)
}

fn db_bytes(db_path: &str) -> u64
{
    std::fs::metadata(db_path).map(|meta| meta.len()).unwrap_or(0)
}

/// Every `images` column the writeback UPDATE assigns (18 bound parameters plus
/// the `CURRENT_TIMESTAMP` literal). Arm B drops the secondary indexes over
/// these.
const UPDATED_COLUMNS: &[&str] = &[
    "focus_score",
    "focus_basis",
    "focus_human_score",
    "focus_animal_score",
    "focus_foreground_score",
    "focus_saliency_score",
    "focus_animal_pose_score",
    "focus_whole_image_score",
    "focus_algorithm_version",
    "focus_analysis_status",
    "focus_analysis_attempt_id",
    "focus_scored_at",
    "face_count",
    "face_quality_best",
    "face_quality_average",
    "face_quality_min",
    "face_eyes_open_count",
    "face_blink_risk_count",
];

struct IndexRow
{
    name: String,
    is_unique: bool,
    is_primary: bool,
    expressions: String,
}

fn images_indexes(conn: &Connection) -> Vec<IndexRow>
{
    let mut stmt = match conn.prepare(
        "SELECT index_name, is_unique, is_primary, CAST(expressions AS VARCHAR)
         FROM duckdb_indexes()
         WHERE table_name = 'images'
         ORDER BY index_name",
    )
    {
        Ok(stmt) => stmt,
        Err(e) =>
        {
            eprintln!("[repro] duckdb_indexes() prepare failed: {}", e);
            return Vec::new();
        }
    };

    let mapped = stmt.query_map([], |row| {
        Ok(IndexRow {
            name: row.get::<_, String>(0)?,
            is_unique: row.get::<_, bool>(1).unwrap_or(false),
            is_primary: row.get::<_, bool>(2).unwrap_or(false),
            expressions: row.get::<_, String>(3).unwrap_or_default(),
        })
    });

    match mapped
    {
        Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
        Err(e) =>
        {
            eprintln!("[repro] duckdb_indexes() query failed: {}", e);
            Vec::new()
        }
    }
}

/// Arm B: drop every NON-unique, NON-primary index on `images` whose expression
/// names a column this UPDATE writes. Returns the dropped names.
fn drop_updated_column_indexes(conn: &Connection) -> Vec<String>
{
    let mut dropped = Vec::new();
    for index in images_indexes(conn)
    {
        if index.is_unique || index.is_primary
        {
            continue;
        }
        let covers = UPDATED_COLUMNS
            .iter()
            .any(|column| index.expressions.contains(column));
        if !covers
        {
            continue;
        }
        let sql = format!("DROP INDEX IF EXISTS \"{}\";", index.name);
        match conn.execute_batch(&sql)
        {
            Ok(()) => dropped.push(index.name),
            Err(e) => eprintln!("[repro] DROP INDEX {} failed: {}", index.name, e),
        }
    }
    dropped
}

/// The production focus-analysis queue, in production order (`ORDER BY id`),
/// with no `LIMIT`. Built from `focus_analysis_queue_where_clause` so the
/// candidate set is byte-for-byte the queue the app feeds the analyzer.
fn repro_candidate_ids(conn: &Connection, attempt_id: &str) -> Vec<i64>
{
    let where_clause = focus_analysis_queue_where_clause(None);
    let sql = format!("SELECT id FROM images {} ORDER BY id", where_clause);
    let mut stmt = match conn.prepare(&sql)
    {
        Ok(stmt) => stmt,
        Err(e) =>
        {
            eprintln!("[repro] candidate prepare failed: {}", e);
            return Vec::new();
        }
    };
    let mapped = stmt.query_map(params![REPRO_ALGORITHM, attempt_id], |row| {
        row.get::<_, i64>(0)
    });
    match mapped
    {
        Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
        Err(e) =>
        {
            eprintln!("[repro] candidate query failed: {}", e);
            Vec::new()
        }
    }
}

/// The post-trip forensic snapshot the S179 brief asks for.
fn dump_row_state(conn: &Connection, label: &str, image_id: i64)
{
    let sql = "SELECT rowid, id, file_path, image_kind, file_stem, directory_path,
                      focus_analysis_attempt_id, focus_algorithm_version,
                      focus_analysis_status, focus_score,
                      CAST(focus_scored_at AS VARCHAR)
               FROM images WHERE id = ?1";
    let mut stmt = match conn.prepare(sql)
    {
        Ok(stmt) => stmt,
        Err(e) =>
        {
            eprintln!("[repro] {} row-state prepare failed: {}", label, e);
            return;
        }
    };
    let mapped = stmt.query_map(params![image_id], |row| {
        Ok(format!(
            "rowid={:?} id={:?} path={:?} kind={:?} stem={:?} dir={:?} attempt={:?} version={:?} status={:?} score={:?} scored_at={:?}",
            row.get::<_, i64>(0).ok(),
            row.get::<_, i64>(1).ok(),
            row.get::<_, String>(2).ok(),
            row.get::<_, String>(3).ok(),
            row.get::<_, String>(4).ok(),
            row.get::<_, String>(5).ok(),
            row.get::<_, String>(6).ok(),
            row.get::<_, String>(7).ok(),
            row.get::<_, String>(8).ok(),
            row.get::<_, f64>(9).ok(),
            row.get::<_, String>(10).ok(),
        ))
    });
    match mapped
    {
        Ok(iter) =>
        {
            let mut seen = 0usize;
            for row in iter.flatten()
            {
                seen += 1;
                eprintln!("[repro] {} row#{} {}", label, seen, row);
            }
            eprintln!("[repro] {} SELECT for id={} returned {} row(s)", label, image_id, seen);
        }
        Err(e) => eprintln!("[repro] {} row-state query failed: {}", label, e),
    }

    // Does the engine agree the id is unique right now?
    let counted: Result<(i64, i64), duckdb::Error> = conn.query_row(
        "SELECT COUNT(*), COUNT(DISTINCT rowid) FROM images WHERE id = ?1",
        params![image_id],
        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
    );
    match counted
    {
        Ok((count, distinct_rowids)) => eprintln!(
            "[repro] {} COUNT(*)={} COUNT(DISTINCT rowid)={} for id={}",
            label, count, distinct_rowids, image_id
        ),
        Err(e) => eprintln!("[repro] {} count query failed: {}", label, e),
    }
}

// ---------------------------------------------------------------------------
// the arm
// ---------------------------------------------------------------------------

struct Totals
{
    chunks: usize,
    sources: usize,
    target_updates: u64,
    trips: usize,
    other_failures: usize,
    /// S179 — accepted row-count anomalies (the catalogue confirmed the write).
    count_anomalies: usize,
    /// S179 — chunks whose first transaction could not be verified and were
    /// replayed once.
    chunk_retries: usize,
    checkpoints: usize,
}

#[test]
fn live_repro_focus_writeback_invariant_on_real_catalogue()
{
    // Unset => silently pass, exactly like the S124/S126 live_repro_tests.
    let Ok(db_path) = std::env::var("PLDIAG_REPRO_DB")
    else
    {
        return;
    };

    let arm = env_string("PLDIAG_REPRO_ARM", "A").to_uppercase();
    let max_minutes = env_f64("PLDIAG_REPRO_MAX_MINUTES", 25.0);
    let max_passes = env_usize("PLDIAG_REPRO_MAX_PASSES", 3).max(1);
    let chunk_size = env_usize("PLDIAG_REPRO_CHUNK", 14).max(1);
    let max_candidates = env_usize("PLDIAG_REPRO_MAX_CANDIDATES", 0);
    let skip_candidates = env_usize("PLDIAG_REPRO_SKIP_CANDIDATES", 0);
    let progress_every = env_usize("PLDIAG_REPRO_PROGRESS_EVERY", 200).max(1);
    let stop_on_trip = env_usize("PLDIAG_REPRO_STOP_ON_TRIP", 1) != 0;
    let migrate = env_usize("PLDIAG_REPRO_MIGRATE", 1) != 0;

    let is_arm_c = arm == "C";
    let shape = ResultShape {
        face_every: if is_arm_c { 1 } else { env_usize("PLDIAG_REPRO_FACE_EVERY", 4) },
        keyword_every: if is_arm_c { 1 } else { env_usize("PLDIAG_REPRO_KEYWORD_EVERY", 8) },
        incomplete_every: env_usize("PLDIAG_REPRO_INCOMPLETE_EVERY", 20),
    };
    let checkpoint_every = if is_arm_c
    {
        env_usize("PLDIAG_REPRO_CHECKPOINT_EVERY", 2000)
    }
    else
    {
        env_usize("PLDIAG_REPRO_CHECKPOINT_EVERY", 0)
    };

    let budget = Duration::from_secs_f64((max_minutes * 60.0).max(1.0));
    let started = Instant::now();

    eprintln!("================================================================");
    eprintln!("[repro] S179 focus-writeback invariant reproduction — ARM {}", arm);
    eprintln!("[repro] db={}", db_path);
    eprintln!(
        "[repro] budget={:.1} min  passes<={}  chunk={}  face_every={}  keyword_every={}  incomplete_every={}  checkpoint_every={}  migrate={}",
        max_minutes,
        max_passes,
        chunk_size,
        shape.face_every,
        shape.keyword_every,
        shape.incomplete_every,
        checkpoint_every,
        migrate
    );
    eprintln!(
        "[repro] db_bytes={} wal_bytes={} (before open)",
        db_bytes(&db_path),
        wal_bytes(&db_path)
    );

    // --- open the catalogue the way the app opens it -----------------------
    let open_started = Instant::now();
    let conn = if migrate
    {
        match open_and_migrate_catalogue(std::path::Path::new(&db_path))
        {
            Some(conn) => conn,
            None => panic!("[repro] open_and_migrate_catalogue failed for {}", db_path),
        }
    }
    else
    {
        Connection::open(&db_path).expect("[repro] Connection::open failed")
    };
    eprintln!(
        "[repro] catalogue open+migrate took {:.1}s",
        open_started.elapsed().as_secs_f64()
    );

    // --- doctrine: this MUST be the bundled engine, not the Homebrew CLI ----
    let engine_version: String = conn
        .query_row("SELECT version()", [], |row| row.get(0))
        .expect("[repro] SELECT version() failed");
    eprintln!("[repro] bundled engine version = {}", engine_version);
    assert_eq!(
        engine_version, EXPECTED_BUNDLED_DUCKDB_VERSION,
        "this reproduction must run through the app's BUNDLED DuckDB, not a CLI"
    );

    // --- table facts -------------------------------------------------------
    let (row_count, distinct_ids, distinct_rowids): (i64, i64, i64) = conn
        .query_row(
            "SELECT COUNT(*), COUNT(DISTINCT id), COUNT(DISTINCT rowid) FROM images",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .expect("[repro] images fact query failed");
    eprintln!(
        "[repro] images: rows={} distinct_id={} distinct_rowid={}",
        row_count, distinct_ids, distinct_rowids
    );

    let before = images_indexes(&conn);
    eprintln!("[repro] images carries {} index entries before the arm", before.len());
    for index in &before
    {
        eprintln!(
            "[repro]   index {} unique={} primary={} expressions={}",
            index.name, index.is_unique, index.is_primary, index.expressions
        );
    }

    if arm == "B"
    {
        let dropped = drop_updated_column_indexes(&conn);
        eprintln!(
            "[repro] ARM B dropped {} secondary index(es) on updated columns: {:?}",
            dropped.len(),
            dropped
        );
        let after = images_indexes(&conn);
        eprintln!("[repro] images carries {} index entries after the drops", after.len());
    }

    // --- the candidate queue ----------------------------------------------
    let probe_attempt = synthetic_attempt_id(0);
    let queue_started = Instant::now();
    let mut candidates = repro_candidate_ids(&conn, &probe_attempt);
    eprintln!(
        "[repro] production candidate queue: {} ids in {:.1}s",
        candidates.len(),
        queue_started.elapsed().as_secs_f64()
    );
    if skip_candidates > 0 && skip_candidates < candidates.len()
    {
        candidates.drain(0..skip_candidates);
        eprintln!(
            "[repro] skipped the first {} candidates; {} remain (first id {:?})",
            skip_candidates,
            candidates.len(),
            candidates.first()
        );
    }
    if max_candidates > 0 && candidates.len() > max_candidates
    {
        candidates.truncate(max_candidates);
        eprintln!("[repro] candidate list truncated to {}", candidates.len());
    }
    // How much twin expansion does this slice carry? (2 targets per source is
    // the shape job 6 tripped on.)
    {
        let probe: Vec<i64> = candidates.iter().copied().take(400).collect();
        let mut expanded = 0usize;
        for id in &probe
        {
            if let Ok(targets) = focus_analysis_writeback_target_ids(&conn, *id)
            {
                if targets.len() > 1
                {
                    expanded += 1;
                }
            }
        }
        eprintln!(
            "[repro] twin probe: {} of the first {} candidates expand to >1 target",
            expanded,
            probe.len()
        );
    }
    assert!(
        !candidates.is_empty(),
        "[repro] the production focus-analysis queue is empty on this copy — nothing to drive"
    );

    // --- the driver --------------------------------------------------------
    let mut totals = Totals {
        chunks: 0,
        sources: 0,
        target_updates: 0,
        trips: 0,
        other_failures: 0,
        count_anomalies: 0,
        chunk_retries: 0,
        checkpoints: 0,
    };
    let mut first_trip: Option<String> = None;
    let mut stop = false;
    let mut first_warning: Option<String> = None;

    'passes: for pass in 1..=max_passes
    {
        let attempt_id = synthetic_attempt_id(pass);
        let pass_started = Instant::now();
        eprintln!(
            "[repro] ---- PASS {} of {} — attempt_id={} — {} candidates ----",
            pass,
            max_passes,
            attempt_id,
            candidates.len()
        );

        let mut chunk_index = 0usize;
        let mut ordinal = 0usize;
        let mut chunk_start = 0usize;

        while chunk_start < candidates.len()
        {
            if started.elapsed() >= budget
            {
                eprintln!(
                    "[repro] TIME BUDGET reached ({:.1} min) during pass {} chunk {} — stopping",
                    max_minutes, pass, chunk_index
                );
                stop = true;
                break 'passes;
            }

            let chunk_end = (chunk_start + chunk_size).min(candidates.len());
            let chunk_ids = &candidates[chunk_start..chunk_end];
            let results: Vec<FocusAnalysisResult> = chunk_ids
                .iter()
                .map(|id| {
                    let result = repro_result(*id, ordinal, &attempt_id, &shape);
                    ordinal += 1;
                    result
                })
                .collect();
            let source_count = results.len();

            let chunk_started = Instant::now();
            // THE PRODUCTION PATH: plan -> BEGIN -> apply -> COMMIT.
            let receipt = update_focus_analysis_results_impl(&conn, results, None);
            let chunk_elapsed = chunk_started.elapsed();

            totals.chunks += 1;
            totals.sources += source_count;
            totals.target_updates += receipt.updated;

            // S179 — the primary signal. Warnings ride BOTH the success and the
            // failure receipt, so this runs before the failure branch below.
            for warning in &receipt.warnings
            {
                if warning.reason_code == FOCUS_WRITEBACK_WARNING_CHUNK_RETRIED
                {
                    totals.chunk_retries += 1;
                }
                else
                {
                    totals.count_anomalies += 1;
                }
                eprintln!("================================================================");
                eprintln!(
                    "[repro] *** WRITEBACK WARNING *** {} pass={} chunk={} elapsed={:.3}s",
                    warning.reason_code,
                    pass,
                    chunk_index,
                    chunk_elapsed.as_secs_f64()
                );
                eprintln!(
                    "[repro] source_image_id={} target_image_id={} row_count={} matching_rows={} distinct_rowids={}",
                    warning.source_image_id,
                    warning.target_image_id,
                    warning.row_count,
                    warning.matching_rows,
                    warning.distinct_rowids
                );
                eprintln!("[repro] detail={}", warning.detail);
                eprintln!(
                    "[repro] wal_bytes={} db_bytes={} total_target_updates_so_far={}",
                    wal_bytes(&db_path),
                    db_bytes(&db_path),
                    totals.target_updates
                );
                dump_row_state(&conn, "post-COMMIT ANOMALY TARGET", warning.target_image_id);
                eprintln!("================================================================");

                if first_warning.is_none()
                {
                    first_warning = Some(format!(
                        "arm={} pass={} chunk={} {} source={} target={} row_count={}",
                        arm,
                        pass,
                        chunk_index,
                        warning.reason_code,
                        warning.source_image_id,
                        warning.target_image_id,
                        warning.row_count
                    ));
                }
            }

            if receipt.failure_stage.is_some() || receipt.failed_reason.is_some()
            {
                let stage = receipt.failure_stage.clone().unwrap_or_else(|| "unknown".to_string());
                let reason = receipt
                    .failed_reason
                    .clone()
                    .unwrap_or_else(|| "(no reason)".to_string());
                let is_invariant_trip = reason.contains("one-target/one-row invariant");
                if is_invariant_trip
                {
                    totals.trips += 1;
                }
                else
                {
                    totals.other_failures += 1;
                }

                eprintln!("================================================================");
                eprintln!(
                    "[repro] *** {} *** pass={} chunk={} elapsed={:.3}s",
                    if is_invariant_trip { "INVARIANT TRIP" } else { "WRITEBACK FAILURE" },
                    pass,
                    chunk_index,
                    chunk_elapsed.as_secs_f64()
                );
                eprintln!("[repro] stage={}", stage);
                eprintln!("[repro] reason={}", reason);
                eprintln!(
                    "[repro] source_image_id={:?} target_image_id={:?} updated={}",
                    receipt.source_image_id, receipt.target_image_id, receipt.updated
                );
                eprintln!("[repro] chunk ids = {:?}", chunk_ids);
                eprintln!(
                    "[repro] wal_bytes={} db_bytes={} total_target_updates_so_far={}",
                    wal_bytes(&db_path),
                    db_bytes(&db_path),
                    totals.target_updates
                );

                // The impl already ROLLBACKed. Snapshot the row(s) now.
                if let Some(source_id) = receipt.source_image_id
                {
                    dump_row_state(&conn, "post-rollback SOURCE", source_id);
                    match focus_analysis_writeback_target_ids(&conn, source_id)
                    {
                        Ok(targets) =>
                        {
                            eprintln!("[repro] target_ids for source {} = {:?}", source_id, targets);
                            for target in targets
                            {
                                if Some(target) != receipt.source_image_id
                                {
                                    dump_row_state(&conn, "post-rollback TWIN", target);
                                }
                            }
                        }
                        Err(failure) =>
                        {
                            eprintln!("[repro] target_ids re-query failed: {:?}", failure)
                        }
                    }
                }
                if let Some(target_id) = receipt.target_image_id
                {
                    if Some(target_id) != receipt.source_image_id
                    {
                        dump_row_state(&conn, "post-rollback TARGET", target_id);
                    }
                }

                // Does the SAME chunk succeed when re-run right now?
                let mut retry_ordinal = chunk_index * chunk_size;
                let retry: Vec<FocusAnalysisResult> = chunk_ids
                    .iter()
                    .map(|id| {
                        let result = repro_result(*id, retry_ordinal, &attempt_id, &shape);
                        retry_ordinal += 1;
                        result
                    })
                    .collect();
                let retry_receipt = update_focus_analysis_results_impl(&conn, retry, None);
                eprintln!(
                    "[repro] SAME-CHUNK RETRY: updated={} stage={:?} reason={:?}",
                    retry_receipt.updated, retry_receipt.failure_stage, retry_receipt.failed_reason
                );
                for warning in &retry_receipt.warnings
                {
                    eprintln!(
                        "[repro] SAME-CHUNK RETRY warning: {} source={} target={} row_count={} matching_rows={} distinct_rowids={} detail={}",
                        warning.reason_code,
                        warning.source_image_id,
                        warning.target_image_id,
                        warning.row_count,
                        warning.matching_rows,
                        warning.distinct_rowids,
                        warning.detail
                    );
                }

                // Is the DATA correct once a chunk does COMMIT?
                //
                // With any index on an updated column DuckDB runs the UPDATE as
                // DELETE + INSERT, and the returned change count is the number
                // of rows the SCAN FED to the operator, not the number of rows
                // written (the operator deduplicates row ids before writing).
                // A trip in arm A therefore means a DOUBLE-EMITTING SCAN, and
                // the committed data is expected to be correct. This records
                // whether it actually is. (The driver itself never rolls back —
                // `update_focus_analysis_results_impl` rolls the failed chunk
                // back on its own, so the retry is what produces a COMMIT.)
                if retry_receipt.failure_stage.is_none() && retry_receipt.failed_reason.is_none()
                {
                    eprintln!(
                        "[repro] post-COMMIT check — every row below should carry attempt_id={} version={} ",
                        attempt_id, REPRO_ALGORITHM
                    );
                    if let Some(source_id) = receipt.source_image_id
                    {
                        dump_row_state(&conn, "post-COMMIT SOURCE", source_id);
                        if let Ok(targets) = focus_analysis_writeback_target_ids(&conn, source_id)
                        {
                            for target in targets
                            {
                                if Some(target) != receipt.source_image_id
                                {
                                    dump_row_state(&conn, "post-COMMIT TWIN", target);
                                }
                            }
                        }
                    }
                    if let Some(target_id) = receipt.target_image_id
                    {
                        if Some(target_id) != receipt.source_image_id
                        {
                            dump_row_state(&conn, "post-COMMIT TARGET", target_id);
                        }
                    }
                }
                eprintln!("================================================================");

                if first_trip.is_none()
                {
                    first_trip = Some(format!(
                        "arm={} pass={} chunk={} stage={} reason={}",
                        arm, pass, chunk_index, stage, reason
                    ));
                }

                if stop_on_trip
                {
                    stop = true;
                    break 'passes;
                }
            }
            else if receipt.updated == 0 && source_count > 0
            {
                // The Swift-side S123/F4 zero-row guard, mirrored here.
                eprintln!(
                    "[repro] ZERO-ROW chunk (Swift would raise stage=invariant): pass={} chunk={} ids={:?}",
                    pass, chunk_index, chunk_ids
                );
                totals.other_failures += 1;
            }

            if checkpoint_every > 0 && totals.chunks % checkpoint_every == 0
            {
                let checkpoint_started = Instant::now();
                match conn.execute_batch("CHECKPOINT;")
                {
                    Ok(()) =>
                    {
                        totals.checkpoints += 1;
                        eprintln!(
                            "[repro] CHECKPOINT #{} after chunk {} took {:.2}s (wal {} -> {})",
                            totals.checkpoints,
                            totals.chunks,
                            checkpoint_started.elapsed().as_secs_f64(),
                            "pre",
                            wal_bytes(&db_path)
                        );
                    }
                    Err(e) => eprintln!("[repro] CHECKPOINT failed after chunk {}: {}", totals.chunks, e),
                }
            }

            if chunk_index % progress_every == 0
            {
                eprintln!(
                    "[repro] pass {} chunk {:>6} src {:>7}/{} targets={} last_chunk={:.3}s pass_elapsed={:.1}s total_elapsed={:.1}s wal={} db={}",
                    pass,
                    chunk_index,
                    chunk_end,
                    candidates.len(),
                    totals.target_updates,
                    chunk_elapsed.as_secs_f64(),
                    pass_started.elapsed().as_secs_f64(),
                    started.elapsed().as_secs_f64(),
                    wal_bytes(&db_path),
                    db_bytes(&db_path)
                );
            }

            chunk_index += 1;
            chunk_start = chunk_end;
        }

        eprintln!(
            "[repro] ---- PASS {} DONE in {:.1}s — chunks={} target_updates={} trips={} ----",
            pass,
            pass_started.elapsed().as_secs_f64(),
            totals.chunks,
            totals.target_updates,
            totals.trips
        );
    }

    eprintln!("================================================================");
    eprintln!("[repro] ARM {} SUMMARY", arm);
    eprintln!("[repro]   stopped_early      = {}", stop);
    eprintln!("[repro]   elapsed            = {:.1}s", started.elapsed().as_secs_f64());
    eprintln!("[repro]   chunks committed   = {}", totals.chunks);
    eprintln!("[repro]   source results     = {}", totals.sources);
    eprintln!("[repro]   TARGET UPDATES     = {}", totals.target_updates);
    eprintln!("[repro]   invariant trips    = {}", totals.trips);
    eprintln!("[repro]   other failures     = {}", totals.other_failures);
    eprintln!("[repro]   ACCEPTED ANOMALIES = {}", totals.count_anomalies);
    eprintln!("[repro]   CHUNK RETRIES      = {}", totals.chunk_retries);
    eprintln!("[repro]   checkpoints        = {}", totals.checkpoints);
    eprintln!("[repro]   wal_bytes (end)    = {}", wal_bytes(&db_path));
    eprintln!("[repro]   db_bytes  (end)    = {}", db_bytes(&db_path));
    match &first_trip
    {
        Some(trip) => eprintln!("[repro]   FIRST TRIP: {}", trip),
        None => eprintln!("[repro]   NO TRIP REPRODUCED"),
    }
    match &first_warning
    {
        Some(warning) => eprintln!("[repro]   FIRST WARNING: {}", warning),
        None => eprintln!("[repro]   NO WARNING REPRODUCED"),
    }
    eprintln!("================================================================");

    // The harness reports; it never fails the suite on a no-repro. A trip is a
    // finding, not a test failure — the run log is the deliverable.
}
