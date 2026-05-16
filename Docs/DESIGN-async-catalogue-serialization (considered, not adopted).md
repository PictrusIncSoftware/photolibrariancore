# DESIGN: Async Catalogue Serialization

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
5. **Migrate from the current module-level static to a `CatalogueService` struct.** The service struct owns the connection, holds the mutex, and exposes catalogue operations as instance methods. This replaces the current `static CATALOGUE: Lazy<Arc<Mutex<Option<Connection>>>>` pattern.
6. **Use a shim layer during migration to keep the app working continuously.** The existing module-level public functions are preserved as thin shims that forward to a process-wide default `CatalogueService` instance. This lets the Rust-side refactor and the Swift-side migration land in separate sessions without ever leaving the app in a broken state.

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

### Alternative D — Keep the global static, add `with_connection` as a module-level function

Preserve the existing `static CATALOGUE: Lazy<Arc<Mutex<Option<Connection>>>>` pattern. Add a `with_connection` free function that closures over it. Migrate the ~20 existing call sites to use the helper but keep the public API as bare module-level functions.

**Rejected because:**
- Global statics are a known anti-pattern in Rust. They persist across test cases, preventing isolated unit tests of catalogue logic.
- The pattern forecloses on having more than one catalogue open simultaneously — a constraint that may matter for future features (multiple libraries, library import preview, A/B catalogue migration).
- Statics cannot be passed as dependencies to subsystems that need catalogue access. Subsystems are forced to reach into the global, which couples them to the catalogue's storage mechanism.
- The work to migrate from the global to a service struct only gets harder as the codebase grows. Doing it now, while we are already touching every catalogue call site, is strictly cheaper than doing it later as a dedicated refactor.
- The shim layer (see migration plan below) means we get the benefits of the service struct without paying the Swift-side migration cost in the same session.

---

## Design

### The `CatalogueService` struct

```rust
pub struct CatalogueService {
    connection: Arc<Mutex<Option<Connection>>>,
}

impl CatalogueService {
    /// Create a new service backed by the database at `db_path`.
    /// Runs schema migrations and pragmas during initialization.
    pub fn new(db_path: &Path) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        // schema migrations, pragmas, etc. run here, single-threaded
        Ok(Self {
            connection: Arc::new(Mutex::new(Some(conn))),
        })
    }

    pub async fn with_connection<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&mut Connection) -> Result<R> + Send + 'static,
        R: Send + 'static,
    {
        let conn = Arc::clone(&self.connection);
        tokio::task::spawn_blocking(move || {
            let mut guard = conn.lock().expect("catalogue mutex poisoned");
            match guard.as_mut() {
                Some(connection) => f(connection),
                None => Err(CatalogueError::NotInitialized),
            }
        })
        .await
        .map_err(CatalogueError::TaskJoin)?
    }

    // All catalogue methods are thin wrappers around with_connection.
}
```

**The `Option<Connection>` is retained** because the current code expects "may not yet be initialized" semantics. `with_connection` returns `CatalogueError::NotInitialized` if the connection slot is `None`. This matches existing error-handling expectations.

### Properties of `with_connection`

- **`F: FnOnce`** — closure is consumed, runs once. Forces call sites to be explicit about what they capture.
- **`Send + 'static`** — required by `spawn_blocking`. Closures cannot borrow from local scope, which is intentional: it forces inputs to be owned data, which keeps the locked section self-contained.
- **`spawn_blocking`** — the closure runs on Tokio's blocking thread pool, not on the async runtime's worker threads. This is critical: DuckDB calls are CPU-bound or I/O-blocking, and running them on async workers would stall the entire runtime.
- **Lock is acquired *inside* `spawn_blocking`, not outside.** This means the lock is held only on a blocking thread, never across an `.await`. Deadlock-by-await is structurally impossible.
- **Mutex poisoning** — if a previous holder panicked while holding the lock, `lock()` returns `Err`. We treat this as fatal (`.expect(...)`) because a poisoned catalogue mutex means the database state is suspect anyway.

### Call site shape

Every catalogue operation looks like:

```rust
impl CatalogueService {
    pub async fn get_photo_by_id(&self, id: PhotoId) -> Result<Option<Photo>> {
        self.with_connection(move |conn| {
            let mut stmt = conn.prepare("SELECT ... FROM photos WHERE id = ?")?;
            let row = stmt.query_row([id.0], |r| Photo::try_from_row(r)).optional()?;
            Ok(row)
        })
        .await
    }
}
```

The closure body is plain synchronous DuckDB code. No `.await` inside it (the closure isn't async). No lock management at the call site. The async boundary is at `with_connection`'s edge.

### Rules for call sites

1. **All catalogue access goes through `with_connection`.** No exceptions, no direct `self.connection.lock()` anywhere outside the helper itself.
2. **Closures must be self-contained.** Capture owned data (clone if necessary). The `Send + 'static` bound enforces this at compile time.
3. **Keep closures tight.** Do query work inside; do post-processing (formatting, transformation that doesn't touch the DB) outside, after the closure returns. The lock-held duration should be the query duration, nothing more.
4. **No nested `with_connection` calls.** A closure must not itself call another catalogue method that takes the lock — that's an immediate deadlock. If two queries need to run in one logical operation, write a single closure that does both.

---

## Migration plan: the shim layer

The migration from the current global-static design to the service-struct design must keep the app continuously runnable. A flag-day rewrite is rejected on risk grounds — there is no point at which the team can be confident the app works as expected if every Rust and Swift call site changes simultaneously.

The strategy: introduce the `CatalogueService` struct, expose its functionality through a process-wide default instance, and preserve the existing module-level public functions as thin shims that forward to that default instance. This decouples the Rust-side refactor from the Swift-side migration in time, so each phase ends with a fully working, testable application.

### Phase 1 — Rust-side refactor (Session 15b)

At the end of Phase 1, the Rust code looks like:

```rust
// New: the proper service struct
pub struct CatalogueService { /* as designed above */ }

impl CatalogueService {
    pub fn new(db_path: &Path) -> Result<Self> { ... }
    pub async fn with_connection<F, R>(&self, f: F) -> Result<R> { ... }

    // All ~20 catalogue methods, as instance methods.
    pub async fn get_photos(&self, ...) -> Result<...> { ... }
    pub async fn update_rating(&self, ...) -> Result<...> { ... }
    // ... etc.
}

// Process-wide default instance, initialized on first use.
static DEFAULT_SERVICE: OnceCell<CatalogueService> = OnceCell::new();

fn default_service() -> &'static CatalogueService {
    DEFAULT_SERVICE.get().expect("catalogue not initialized")
}

// Old public API preserved as shims.
pub async fn init_catalogue(path: &str) -> Result<()> {
    let service = CatalogueService::new(Path::new(path))?;
    DEFAULT_SERVICE.set(service).map_err(|_| CatalogueError::AlreadyInitialized)?;
    Ok(())
}

pub async fn get_photos(...) -> Result<...> {
    default_service().get_photos(...).await
}
// ... shim for every existing public function.
```

**Swift does not change in Phase 1.** Every existing Swift call site continues to call the same module-level function. Under the hood, those calls now flow through the new service struct. UniFFI bindings are unchanged because the exported function signatures are unchanged.

### Phase 1 verification gate

Before Phase 1 is considered complete:

- App launches without error
- Catalogue loads existing records (currently 32,855 in the development library)
- Thumbnails display in BrowseView
- Rating a photo persists across relaunch (write path)
- Rotating a photo persists across relaunch (write path)
- Right-click context menu works (read path — reads JPEG/RAW counterparts)
- Scanning a small directory adds records to the catalogue (full ingest path)
- Quit and relaunch — state preserved (catalogue persistence intact)

If all of the above pass, Phase 1 is complete and the Rust core is verified working through the existing API surface.

### Phase 2 — Swift-side migration (Session 16)

Swift call sites move from the module-level shim functions to direct instance-method calls on a held `CatalogueService`. The migration happens in logical groups (e.g., all rating calls together, all scan calls together), with verification after each group.

During Phase 2, the shim functions remain in place — migrated and unmigrated Swift call sites coexist, both routing to the same underlying `CatalogueService` instance through different paths. The app stays runnable throughout.

When every Swift call site is migrated, the shim functions on the Rust side are deleted in a final cleanup commit.

### Why this sequencing works

At every commit between now and the end of Session 16, the app is in a working state. There is no "broken for a week while we refactor" window. The Rust-side refactor in Phase 1 is verifiable on its own merits before any Swift work begins, which means a Phase 2 regression can be localized to Phase 2 with confidence.

---

## Migration path to a pool, if ever needed

The design preserves the option to upgrade without disruption. If future profiling shows real contention (e.g., long classification jobs blocking interactive reads), the upgrade is:

1. Replace the `Arc<Mutex<Option<Connection>>>` field with a pool type (e.g., `r2d2` or a custom one).
2. Rewrite `with_connection` to check out a connection from the pool, run the closure, return it.
3. Call sites are unchanged.

The closure signature would not need to change — it still takes `&mut Connection`. The only externally visible difference would be that two `with_connection` calls could run truly concurrently, where today they serialize.

A read/write split is similarly localized: split into `with_read_connection` and `with_write_connection`, both routing through their own pool or lock. Call sites pick the right one; the abstraction holds.

---

## Open questions deferred from this design

- **Transaction support.** v1 doesn't expose explicit transactions across multiple `with_connection` calls. If a future feature needs cross-call transactional semantics, we'll need either a `with_transaction` variant (closure runs inside a single transaction) or a way to hand out a borrowed `Transaction<'_>` from inside the closure. Punt until needed.
- **Cancellation.** `spawn_blocking` tasks cannot be cancelled. If a long-running catalogue query needs to be interruptible (e.g., user cancels a slow filter), we'll need to either chunk the query at the SQL level or accept that cancellation lands after the current query completes. Not relevant for v1.
- **Metrics / observability.** No instrumentation on `with_connection` for v1. If contention becomes suspected, adding a histogram of lock-wait duration is a one-line change in the helper.
- **Multiple catalogue instances.** The service-struct design makes this possible architecturally, but Phase 1 still uses a process-wide singleton via `DEFAULT_SERVICE`. True multi-catalogue support would require Swift to hold and route between multiple instances, which is out of scope for the Phase 1/2 migration.

---

## Implementation checklist

### Phase 1 (Session 15b)

- [ ] Define `CatalogueService` struct with `Arc<Mutex<Option<Connection>>>` field.
- [ ] Implement `CatalogueService::new(db_path)` performing init, migrations, pragmas.
- [ ] Implement `with_connection<F, R>` as specified.
- [ ] Add `CatalogueError::NotInitialized` and `CatalogueError::TaskJoin` variants.
- [ ] Add `CatalogueError::AlreadyInitialized` variant for re-init attempts.
- [ ] Migrate all ~20 catalogue functions from module-level to `impl CatalogueService` methods.
- [ ] Each migrated method routes through `self.with_connection(...)`.
- [ ] Add `DEFAULT_SERVICE: OnceCell<CatalogueService>` and `default_service()` accessor.
- [ ] Rewrite the existing module-level public functions as shims that delegate to `default_service()`.
- [ ] Verify no `.await` calls remain inside locked sections (should be impossible by construction, but worth a manual pass).
- [ ] Verify no nested `with_connection` calls exist among the migrated methods.
- [ ] Run the existing test suite.
- [ ] Add a concurrency test: fire multiple concurrent `with_connection` calls and verify serialized behavior (e.g., interleaved writes to a counter table produce the expected final value).
- [ ] Execute the Phase 1 verification gate checklist (eight items above).
- [ ] Update Rust-side `CLAUDE.md` with the "all catalogue access through `with_connection`" rule and the "no nested `with_connection`" rule.

### Phase 2 (Session 16)

- [ ] Introduce a Swift-side holder for the `CatalogueService` instance (decision: app-level singleton vs `@StateObject` — to be designed in Session 16 kickoff).
- [ ] Migrate Swift call sites in logical groups, with verification between groups.
- [ ] Delete the Rust-side shim functions once all Swift call sites are migrated.
- [ ] Final verification pass on the eight-item checklist.
