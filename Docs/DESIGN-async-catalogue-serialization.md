**Status:** Approved for implementation
**Session introduced:** Session 15
**Scope:** `photolibrariancore` — catalogue access layer
**Related:** `DESIGN-image-classification.md` (sibling Rust-side design doc)

---

## Problem

The Rust core exposes catalogue operations to Swift via UniFFI async bindings. DuckDB, the underlying catalogue store, is an embedded analytical database with single-writer semantics — it is not designed for concurrent write access, and concurrent reads against an actively-written database can return inconsistent results.

Without explicit serialization on the Rust side, two simultaneous calls from Swift (for example: a thumbnail-cache scan writing classification results while the UI reads photo metadata) can:

- Corrupt the catalogue file
- Return partial or inconsistent reads
- Panic the Rust core
- Deadlock at the DuckDB driver level

This is currently latent. PhotoLibrarian's UI is largely sequential today, but as features accumulate (background classification, async thumbnail builds, semantic search indexing) the assumption that "only one catalogue call is in flight at a time" stops holding. STATUS.md flags this as a Pre-v1 Must-Do and explicitly warns against a quick patch — whichever approach is chosen shapes every future catalogue operation, so it deserves a deliberate design.

---

## Decision summary

1. **Serialize all catalogue access through a single connection guarded by a mutex.** No connection pool in v1.
2. **Use `std::sync::Mutex`, not `tokio::sync::Mutex`.** DuckDB is a blocking C library underneath; an async-aware mutex offers no benefit and risks holding the lock across `.await` points.
3. **Run all DuckDB work inside `tokio::task::spawn_blocking`.** This moves the blocking call off the async runtime's worker threads and onto the blocking thread pool, where it belongs.
4. **Expose access only through a `with_connection<F, R>` helper.** Call sites never touch the mutex or the connection directly. This is the abstraction layer that makes a future swap to a connection pool a localized change rather than a project-wide refactor.

---

## Rejected alternatives

### Alternative A — Connection pool from day one

A pool of N connections, each mutex-wrapped, with callers grabbing whichever connection is free.

**Rejected because:**
- DuckDB's single-writer model means readers and writers still serialize at the database level. The pool would not actually parallelize the workload it claims to.
- PhotoLibrarian is single-user desktop. Contention is structurally low. There is no measured evidence that a pool would deliver any user-visible benefit.
- Pool infrastructure (size tuning, exhaustion handling, connection lifecycle) is real complexity for speculative gain.

### Alternative B — Short-lived borrow / open-and-close per call

Each catalogue operation opens a fresh connection, runs its query, closes the connection. No long-lived shared state.

**Rejected because:**
- DuckDB connection open is non-trivial. Doing it on every call would tax the hot path.
- Eliminates a single coordination point, replacing it with implicit reliance on DuckDB's own concurrency behavior — which, again, is single-writer. We would end up reimplementing serialization at the call sites, less cleanly.

### Alternative C — `Arc<Mutex<Connection>>` exposed directly, no helper

Call sites grab the lock themselves: `self.connection.lock().unwrap().execute(...)`.

**Rejected because:**
- Couples every call site to the concrete implementation. A future migration to a pool, to a read/write split, or to per-table connections would require touching every call site.
- Makes lock-held duration harder to audit — easy to accidentally hold the lock across slow work.

---

## Design

### The `with_connection` helper

A single method on the catalogue service that owns the connection:

```rust
impl CatalogueService {
    pub async fn with_connection<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&mut Connection) -> Result<R> + Send + 'static,
        R: Send + 'static,
    {
        let conn = Arc::clone(&self.connection);
        tokio::task::spawn_blocking(move || {
            let mut guard = conn.lock().expect("catalogue mutex poisoned");
            f(&mut *guard)
        })
        .await
        .map_err(|join_err| CatalogueError::TaskJoin(join_err))?
    }
}
```

Properties:

- **`F: FnOnce`** — closure is consumed, runs once. Forces call sites to be explicit about what they capture.
- **`Send + 'static`** — required by `spawn_blocking`. Closures cannot borrow from local scope, which is intentional: it forces inputs to be owned data, which keeps the locked section self-contained.
- **`spawn_blocking`** — the closure runs on Tokio's blocking thread pool, not on the async runtime's worker threads. This is critical: DuckDB calls are CPU-bound or I/O-blocking, and running them on async workers would stall the entire runtime.
- **Lock is acquired *inside* `spawn_blocking`, not outside.** This means the lock is held only on a blocking thread, never across an `.await`. Deadlock-by-await is structurally impossible.
- **Mutex poisoning** — if a previous holder panicked while holding the lock, `lock()` returns `Err`. We treat this as fatal (`.expect(...)`) because a poisoned catalogue mutex means the database state is suspect anyway.

### Call site shape

Every catalogue operation looks like:

```rust
pub async fn get_photo_by_id(&self, id: PhotoId) -> Result<Option<Photo>> {
    self.with_connection(move |conn| {
        let mut stmt = conn.prepare("SELECT ... FROM photos WHERE id = ?")?;
        let row = stmt.query_row([id.0], |r| Photo::try_from_row(r)).optional()?;
        Ok(row)
    })
    .await
}
```

The closure body is plain synchronous DuckDB code. No `.await` inside it (the closure isn't async). No lock management at the call site. The async boundary is at `with_connection`'s edge.

### What the service struct looks like

```rust
pub struct CatalogueService {
    connection: Arc<Mutex<Connection>>,
    // ... other fields: paths, config, etc.
}

impl CatalogueService {
    pub fn new(db_path: &Path) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        // schema migrations, pragmas, etc. run here, single-threaded
        Ok(Self {
            connection: Arc::new(Mutex::new(conn)),
        })
    }

    pub async fn with_connection<F, R>(&self, f: F) -> Result<R> { /* as above */ }

    // All public catalogue methods are thin wrappers around with_connection.
}
```

### Rules for call sites

1. **All catalogue access goes through `with_connection`.** No exceptions, no direct `self.connection.lock()` anywhere outside the helper itself.
2. **Closures must be self-contained.** Capture owned data (clone if necessary). The `Send + 'static` bound enforces this at compile time.
3. **Keep closures tight.** Do query work inside; do post-processing (formatting, transformation that doesn't touch the DB) outside, after the closure returns. The lock-held duration should be the query duration, nothing more.
4. **No nested `with_connection` calls.** A closure must not itself call another catalogue method that takes the lock — that's an immediate deadlock. If two queries need to run in one logical operation, write a single closure that does both.

---

## Migration path to a pool, if ever needed

The design preserves the option to upgrade without disruption. If future profiling shows real contention (e.g., long classification jobs blocking interactive reads), the upgrade is:

1. Replace the `Arc<Mutex<Connection>>` field with a pool type (e.g., `r2d2` or a custom one).
2. Rewrite `with_connection` to check out a connection from the pool, run the closure, return it.
3. Call sites are unchanged.

The closure signature would not need to change — it still takes `&mut Connection`. The only externally visible difference would be that two `with_connection` calls could run truly concurrently, where today they serialize.

A read/write split is similarly localized: split into `with_read_connection` and `with_write_connection`, both routing through their own pool or lock. Call sites pick the right one; the abstraction holds.

---

## Open questions deferred from this design

- **Transaction support.** v1 doesn't expose explicit transactions across multiple `with_connection` calls. If a future feature needs cross-call transactional semantics, we'll need either a `with_transaction` variant (closure runs inside a single transaction) or a way to hand out a borrowed `Transaction<'_>` from inside the closure. Punt until needed.
- **Cancellation.** `spawn_blocking` tasks cannot be cancelled. If a long-running catalogue query needs to be interruptible (e.g., user cancels a slow filter), we'll need to either chunk the query at the SQL level or accept that cancellation lands after the current query completes. Not relevant for v1.
- **Metrics / observability.** No instrumentation on `with_connection` for v1. If contention becomes suspected, adding a histogram of lock-wait duration is a one-line change in the helper.

---

## Implementation checklist (Session 15)

- [ ] Add `Arc<Mutex<Connection>>` field to `CatalogueService` (or whichever struct currently owns the connection).
- [ ] Implement `with_connection<F, R>` as specified.
- [ ] Migrate existing catalogue methods to route through `with_connection`. Each one becomes a thin async wrapper around a sync closure.
- [ ] Add a `CatalogueError::TaskJoin` variant for `spawn_blocking` join failures.
- [ ] Verify no `.await` calls remain inside locked sections (should be impossible by construction, but worth a manual pass).
- [ ] Run the existing test suite. Add at least one test that fires multiple concurrent `with_connection` calls and verifies serialized behavior.
- [ ] Update `CLAUDE.md` (Rust side) with the "all catalogue access through `with_connection`" rule.
