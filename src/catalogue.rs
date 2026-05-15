//! CatalogueService and the shared `with_connection` helper.
//!
//! This module implements Phase 1a of
//! `Docs/DESIGN-async-catalogue-serialization.md`. Read that doc before
//! changing anything in this file.
//!
//! ## Phase 1a constraints
//!
//! - The legacy `static CATALOGUE` in `lib.rs` is still alive and is
//!   still locked directly by the ~15 unmigrated catalogue functions.
//!   `CatalogueService::with_arc` exists specifically so the service
//!   can share its inner `Arc<Mutex<Option<Connection>>>` with that
//!   static. Migrated methods routed through `with_connection` and
//!   unmigrated functions calling `CATALOGUE.lock()` therefore contend
//!   on the same mutex, giving single-connection serialization across
//!   the entire migration window.
//! - The `Option<Connection>` is retained because pre-init "may not yet
//!   be ready" semantics match the legacy static's expectations. The
//!   helper returns `CatalogueError::NotInitialized` when the slot is
//!   `None`.
//! - `CatalogueError` is module-private semantically: it is `pub` so
//!   inherent-impl methods on `CatalogueService` defined in `lib.rs`
//!   can name it in return types, but no variant is exposed across the
//!   UniFFI boundary. The shim layer in `lib.rs` catches each variant,
//!   logs via `eprintln!`, and returns the pre-refactor sentinel value
//!   so Swift callers see identical behavior.
//! - The `new(db_path: &Path)` constructor specified in the design doc
//!   is intentionally absent from Phase 1a. It will land in Phase 1b
//!   when the legacy static is deleted and the service owns the
//!   connection exclusively.

use std::fmt;
use std::sync::{Arc, Mutex};

use duckdb::Connection;
use once_cell::sync::OnceCell;
use tokio::task::JoinError;

/// Errors produced by `CatalogueService::with_connection` and the
/// `initialize_catalogue` shim.
///
/// Hand-rolled (no `thiserror`) to match the existing crate style.
/// Not exposed across the UniFFI boundary — the shim layer in `lib.rs`
/// converts every variant to an `eprintln!` log plus the pre-refactor
/// sentinel return value (`false`, `0`, `vec![]`, `None`, etc.).
#[derive(Debug)]
pub enum CatalogueError
{
    /// `with_connection` was invoked but the connection slot in the
    /// shared `Arc<Mutex<Option<Connection>>>` is `None`. In Phase 1a
    /// this is reachable if a migrated method runs before
    /// `initialize_catalogue` completes its open + schema work; in
    /// Phase 1b it will be reachable only via deliberate teardown.
    NotInitialized,

    /// `tokio::task::spawn_blocking` returned an error — the blocking
    /// task panicked or was cancelled. Treated as fatal-ish: the
    /// closure never produced its `R`.
    TaskJoin(JoinError),

    /// `DEFAULT_SERVICE.set(...)` failed because the process-wide
    /// service was already initialized. Reachable only if Swift calls
    /// `initialize_catalogue` more than once in a single process
    /// lifetime — not expected, but cheap to surface explicitly.
    AlreadyInitialized,

    /// A DuckDB call inside a `with_connection` closure failed. Wraps
    /// the underlying `duckdb::Error` for context. Conversion is
    /// automatic via the `From<duckdb::Error>` impl, so closures can
    /// use `?` on any DuckDB call.
    Duck(duckdb::Error),
}

impl fmt::Display for CatalogueError
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
    {
        match self
        {
            Self::NotInitialized => write!(f, "catalogue not initialized"),
            Self::TaskJoin(e) => write!(f, "catalogue task join error: {}", e),
            Self::AlreadyInitialized => write!(f, "catalogue already initialized"),
            Self::Duck(e) => write!(f, "duckdb error: {}", e),
        }
    }
}

impl From<duckdb::Error> for CatalogueError
{
    fn from(e: duckdb::Error) -> Self
    {
        Self::Duck(e)
    }
}

/// Owns serialized access to the DuckDB connection.
///
/// In Phase 1a, the inner `Arc<Mutex<Option<Connection>>>` is shared
/// with the legacy `static CATALOGUE` in `lib.rs` (see module-level
/// doc above). This is intentional and is the mechanism that keeps
/// the app working continuously across the migration.
///
/// Future inherent-impl blocks (in `lib.rs`) attach catalogue methods
/// like `get_image_count`, `update_image_rating`, etc. Each method
/// routes through `with_connection` and never touches `connection`
/// directly.
pub struct CatalogueService
{
    connection: Arc<Mutex<Option<Connection>>>,
}

impl CatalogueService
{
    /// Construct a service that shares an externally-supplied
    /// connection arc.
    ///
    /// Phase 1a constructor. `initialize_catalogue` calls this with
    /// `Arc::clone(&CATALOGUE)` after opening the connection and
    /// running schema migrations the legacy way; the service and the
    /// legacy static then point at the same `Mutex<Option<Connection>>`
    /// and serialize against each other.
    ///
    /// Will be replaced by `new(db_path: &Path)` in Phase 1b once the
    /// legacy static is deleted and the service owns the connection.
    pub fn with_arc(connection: Arc<Mutex<Option<Connection>>>) -> Self
    {
        Self { connection }
    }

    /// Run a synchronous closure with exclusive access to the DuckDB
    /// connection, off the async runtime's worker threads.
    ///
    /// Properties (see the design doc for the full rationale):
    ///
    /// - `F: FnOnce(&mut Connection) -> Result<R, CatalogueError> +
    ///    Send + 'static` — closure is consumed, owns its captures.
    /// - `R: Send + 'static` — return value crosses thread boundaries.
    /// - The closure runs on `tokio::task::spawn_blocking`'s blocking
    ///   thread pool, never on the async runtime's worker threads.
    /// - The mutex is locked **inside** the blocking task, so the
    ///   lock is never held across an `.await`. Deadlock-by-await is
    ///   structurally impossible.
    /// - Mutex poisoning is fatal (`.expect(...)`): a poisoned
    ///   catalogue mutex means database state is suspect anyway.
    /// - If the connection slot is `None`, returns
    ///   `CatalogueError::NotInitialized` (the legacy "catalogue not
    ///   initialized" branch).
    /// - `spawn_blocking` join failure is mapped to
    ///   `CatalogueError::TaskJoin`.
    pub async fn with_connection<F, R>(&self, f: F) -> Result<R, CatalogueError>
    where
        F: FnOnce(&mut Connection) -> Result<R, CatalogueError> + Send + 'static,
        R: Send + 'static,
    {
        let conn = Arc::clone(&self.connection);
        tokio::task::spawn_blocking(move ||
        {
            let mut guard = conn.lock().expect("catalogue mutex poisoned");
            match guard.as_mut()
            {
                Some(connection) => f(connection),
                None => Err(CatalogueError::NotInitialized),
            }
        })
        .await
        .map_err(CatalogueError::TaskJoin)?
    }
}

/// Process-wide default service instance.
///
/// Populated exactly once by `initialize_catalogue`'s shim body. All
/// migrated public catalogue functions (in `lib.rs`) delegate to this
/// instance via `default_service()`.
pub static DEFAULT_SERVICE: OnceCell<CatalogueService> = OnceCell::new();

/// Accessor for the process-wide default service.
///
/// Panics if `initialize_catalogue` has not yet completed. The legacy
/// static `CATALOGUE` survives a missing-init by returning sentinels;
/// the default service goes a step further and panics, because being
/// in this code path at all means a migrated method was called before
/// init — a programming error rather than expected runtime state.
///
/// (Once-`initialized` but `NotInitialized` from the connection-slot
/// being `None` is a distinct, non-panic path through `with_connection`
/// → `CatalogueError::NotInitialized`.)
pub fn default_service() -> &'static CatalogueService
{
    DEFAULT_SERVICE
        .get()
        .expect("catalogue not initialized: initialize_catalogue must run before any catalogue call")
}

#[cfg(test)]
mod tests
{
    use super::*;
    use duckdb::params;
    use std::sync::Arc;
    use tempfile::TempDir;

    /// Concurrency test: prove that `with_connection` actually
    /// serializes concurrent writes.
    ///
    /// Each of N tasks reads the current counter value and writes
    /// `value + 1`. If the mutex serializes correctly, the final
    /// value equals N. If serialization were broken, lost updates
    /// would produce a value < N.
    ///
    /// Multi-thread runtime with 4 worker threads — a single-threaded
    /// runtime would mask the bug we're testing for by accidentally
    /// serializing all tasks at the runtime level rather than at the
    /// mutex level.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn with_connection_serializes_concurrent_writes()
    {
        let tmp = TempDir::new().expect("create temp dir");
        let db_path = tmp.path().join("test.duckdb");

        // Open connection and seed the counter table.
        let conn = Connection::open(&db_path).expect("open duckdb");
        conn.execute_batch(
            "CREATE TABLE counter (value BIGINT NOT NULL); \
             INSERT INTO counter (value) VALUES (0);",
        )
        .expect("create + seed counter table");

        // Wrap in the production shape and construct a service that
        // shares that arc.
        let arc = Arc::new(Mutex::new(Some(conn)));
        let service = Arc::new(CatalogueService::with_arc(arc));

        // Spawn N concurrent tasks. Each performs a single
        // with_connection call that reads the current value and
        // writes value + 1.
        const N: usize = 50;
        let mut handles = Vec::with_capacity(N);
        for _ in 0..N
        {
            let svc = Arc::clone(&service);
            handles.push(tokio::spawn(async move
            {
                svc.with_connection(|conn|
                {
                    let current: i64 = conn.query_row(
                        "SELECT value FROM counter",
                        [],
                        |row| row.get(0),
                    )?;
                    conn.execute(
                        "UPDATE counter SET value = ?",
                        params![current + 1],
                    )?;
                    Ok(())
                })
                .await
                .expect("with_connection should succeed");
            }));
        }

        for h in handles
        {
            h.await.expect("task should complete");
        }

        // Verify final value. Anything less than N means lost updates.
        let final_value = service
            .with_connection(|conn|
            {
                let v: i64 = conn.query_row(
                    "SELECT value FROM counter",
                    [],
                    |row| row.get(0),
                )?;
                Ok(v)
            })
            .await
            .expect("read final value");

        assert_eq!(
            final_value, N as i64,
            "concurrent writes through with_connection must serialize"
        );
    }
}
