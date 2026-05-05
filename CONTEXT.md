# PhotoLibrarian — Project Context

**Last Updated:** May 2026  
**Purpose:** Architectural decisions, dev environment state, and session continuity for Claude Code and other AI-assisted development sessions. Read this before writing any code.

---

## 1. What This Project Is

A native macOS photo library manager and organiser — a subscription-free, one-time-purchase Lightroom alternative focused exclusively on organisation and cataloguing, not editing. Full specification is in `SPEC.md`.

---

## 2. Repository & Project Structure

- **GitHub org:** https://github.com/PictrusIncSoftware
- **Repo:** https://github.com/PictrusIncSoftware/photolibrariancore
- **Rust crate root:** `photolibrariancore/`
- **Crate name:** `photolibrariancore` (lowercase, no hyphens, no caps — Cargo convention)
- **Two development machines:**
  - MacBook Pro M1 — `/Users/richardwagner/PhotoLibrarian-core/photolibrariancore`
  - Mac Studio M4 — `/Users/richardwagner/DeveloperProjects/PhotoLibrarian/photolibrariancore`

---

## 3. Technology Stack

| Layer | Technology | Notes |
|---|---|---|
| UI | Swift / SwiftUI | Native macOS, Apple Silicon only |
| Backend / Core Logic | Rust | High-throughput file I/O, metadata processing, database operations |
| Rust–Swift Binding | UniFFI | Mozilla's binding generator — not yet scaffolded, first priority |
| Catalogue Database | DuckDB | Embedded, no server process, bundled via `duckdb` crate with `bundled` feature |
| Vector / Semantic Index | LanceDB | Embedded vector database, `lancedb` crate |
| ML / AI | Apple Core ML | On-device inference, auto-tagging, face recognition |
| EXIF Extraction | Apple ImageIO (Swift) | Chosen over Rust EXIF crates for native RAW format coverage |

---

## 4. Cargo.toml Dependencies (current)

```toml
[package]
name = "photolibrariancore"
version = "0.1.0"
edition = "2021"
publish = false

[dependencies]
duckdb = { version = "1.1", features = ["bundled"] }
lancedb = "0.14"
arrow-array = "53"
arrow-schema = "53"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

---

## 5. Architectural Decisions (Locked)

### 5.1 EXIF Extraction — Swift/ImageIO, not Rust
EXIF and metadata extraction is handled on the Swift side using Apple's `ImageIO` framework. Rationale: native RAW format coverage (CR2, CR3, NEF, ARW, RAF, and all Apple-supported formats) is free and automatic. Rust EXIF crates do not cover the full RAW format matrix required by the spec. If Apple adds new RAW format support, the app inherits it automatically.

### 5.2 Data Flow for Image Ingestion
```
SwiftUI (directory picker)
    → Swift image scanner (recursive file traversal)
        → ImageIO (EXIF extraction per file)
            → UniFFI bridge
                → Rust/DuckDB (store structured metadata records)
```

### 5.3 UniFFI for Rust–Swift Bridge
All Rust–Swift interop goes through UniFFI. The API surface is declared in a `.udl` interface definition file. UniFFI generates both the Rust scaffolding and the Swift bindings. All long-running backend operations are exposed as async Rust functions and consumed as Swift async/await calls. **This bridge has not yet been scaffolded — it is the first development priority.**

### 5.4 App Store Sandboxing
The application targets Mac App Store distribution. All libraries — DuckDB, LanceDB, Rust backend — must be bundled within the `.app` container as a static library. No external processes, no shared library installations, no subprocess spawning. The Rust crate compiles to a static lib embedded in the Xcode project. DuckDB's `bundled` feature ensures no system library dependency. Sandbox compliance must be validated early via a signed `.app` build — this is a known early validation requirement.

### 5.5 File Path Discipline
The Rust backend never constructs file paths internally. All paths are passed in from the Swift layer, which obtains them via sandbox-legal means (`NSOpenPanel`, security-scoped bookmarks). This is required for App Store compliance and must be maintained throughout development.

### 5.6 No Subprocess Architecture
The Rust code runs in-process as a static library. No separate daemon, helper binary, or inter-process communication. This is a hard constraint imposed by App Store sandboxing.

### 5.7 Apple Silicon Only
`aarch64-apple-darwin` exclusively. No Intel support now or in future versions. No fat binaries required.

---

## 6. Dev Environment — Both Machines

**Prerequisites (confirmed installed on both machines):**
- Xcode Command Line Tools
- Homebrew
- Rust stable (`rustup`) — 1.95.0 as of May 2026
- `aarch64-apple-darwin` target — confirmed
- `protobuf` (`brew install protobuf`) — required by LanceDB's lance-encoding build
- VS Code with `rust-analyzer` extension

**Build status:** `cargo build` compiles clean on both machines as of May 2026.

---

## 7. Current Development State

- [x] Rust crate scaffolded (`photolibrariancore`)
- [x] Core dependencies compiling clean (DuckDB, LanceDB, Arrow, Tokio)
- [x] GitHub repo created and pushed
- [x] Both dev machines in sync
- [x] UniFFI bridge — Rust side complete; Swift bindings generated and wired into Xcode
- [x] DuckDB schema for image catalogue
- [ ] SwiftUI directory picker
- [ ] Recursive image scanner (Swift)
- [ ] EXIF extraction via ImageIO
- [ ] Rust ingest function (receives EXIF records, writes to DuckDB)

---

## 8. First Milestone Scope

**Milestone 1: Image Scan → EXIF → DuckDB**

A SwiftUI interface that allows the user to select a directory (or the entire machine), recursively scans it for image files, extracts EXIF metadata via ImageIO, and stores the results in DuckDB via the Rust backend. No editing, no tagging, no AI, no vector search. Just images in the catalogue with their metadata. This pipeline is the foundation everything else builds on.

---

## 9. Distribution & Licensing

- Mac App Store exclusively
- One-time purchase, $10.00 USD
- Apple Small Business Program (15% commission) — to apply at first submission
- Apple Developer Program enrollment deferred until approaching submission
- No subscription, no telemetry, no cloud dependency

---

## 10. Developer Notes

- The lead developer has strong database design background (identity resolution systems), molecular biology/bioinformatics training, and recent Rust and SwiftUI experience
- Dyslexic/dysgraphic — uses Wispr Flow for speech-to-text input
- AI-assisted development is the primary coding methodology (Claude / Claude Code)
- Preference for direct, honest communication — no sycophancy, corrections welcome
- One thing at a time — scope each Claude Code session to a single well-defined deliverable

---

## 11. Known Limitations (To Address Later)

- **DuckDB connection handling:** Global `CATALOGUE` using 
  `Arc<Mutex<Option<Connection>>>` is adequate for Milestone 1 
  sequential batch inserts but will bottleneck under concurrent 
  writes. Replace with a connection pool in a later milestone.

*End of CONTEXT.md*
