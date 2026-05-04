# PhotoLibrarian Core

**Rust backend for PhotoLibrarian — a native macOS photo library manager**

This is the core Rust library that powers PhotoLibrarian, a subscription-free, one-time-purchase alternative to Adobe Lightroom's cataloguing functionality. This library handles all metadata storage, database operations, and will eventually provide semantic search capabilities through LanceDB.

## What This Library Provides

A UniFFI-exposed API for Swift/SwiftUI frontends to:

- Initialize and manage a DuckDB-backed photo catalogue
- Ingest image metadata extracted by Swift's ImageIO framework
- Query the catalogue for image counts and statistics
- (Future) Perform semantic and visual similarity search via LanceDB

## Current Implementation Status

✅ **Milestone 1 Complete**: Image Scan → EXIF → DuckDB

- [x] UniFFI bridge configured and tested
- [x] DuckDB catalogue schema implemented
- [x] Async API for metadata ingestion
- [x] Thread-safe global catalogue state
- [x] EXIF metadata structure matching ImageIO output
- [x] Batch insert with duplicate detection
- [x] Indexed queries on common metadata fields

## Architecture

- **Language**: Rust 2021 edition
- **Database**: DuckDB (embedded, bundled via `bundled` feature)
- **Vector Search**: LanceDB (embedded, not yet integrated)
- **FFI Bridge**: UniFFI 0.28 for Rust ↔ Swift interop
- **Async Runtime**: Tokio (full feature set)
- **Platform**: Apple Silicon (aarch64-apple-darwin) exclusively

## Project Structure

```
photolibrariancore/
├── src/
│   ├── lib.rs                     # Main library, UniFFI scaffolding, DuckDB ops
│   └── photolibrariancore.udl     # UniFFI interface definition
├── build.rs                       # UniFFI scaffolding generator
├── Cargo.toml                     # Dependencies and build config
├── SPEC.md                        # Full product specification
├── CONTEXT.md                     # Architectural decisions and dev state
├── UNIFFI_INTEGRATION.md          # Swift/Xcode integration guide
└── build_for_xcode.sh             # Helper script to build for Xcode

```

## API Overview

### Functions

```rust
// Initialize the catalogue at the given path
async fn initialize_catalogue(catalogue_path: String) -> bool

// Ingest a batch of image metadata records
async fn ingest_metadata(metadata: Vec<ImageMetadata>) -> u32

// Get total count of images in catalogue
async fn get_image_count() -> u64
```

### Data Types

```rust
struct ImageMetadata {
    file_path: String,
    file_size: u64,
    file_name: String,
    file_extension: Option<String>,
    created_timestamp: i64,
    modified_timestamp: i64,

    // Camera metadata
    camera_make: Option<String>,
    camera_model: Option<String>,
    lens_model: Option<String>,
    focal_length: Option<f64>,
    aperture: Option<f64>,
    shutter_speed: Option<f64>,
    iso: Option<u32>,
    capture_datetime: Option<String>,

    // Image properties
    pixel_width: Option<u32>,
    pixel_height: Option<u32>,
    color_space: Option<String>,
    bit_depth: Option<u32>,

    // GPS
    gps_latitude: Option<f64>,
    gps_longitude: Option<f64>,
    gps_altitude: Option<f64>,

    // IPTC/copyright
    copyright: Option<String>,
    creator: Option<String>,
    description: Option<String>,
}
```

## Building

### Prerequisites

- Rust stable (1.95.0+)
- Xcode Command Line Tools
- Homebrew
- `protobuf` (`brew install protobuf`) — required for LanceDB

### Build Commands

```bash
# Standard debug build
cargo build

# Release build for Xcode integration
./build_for_xcode.sh

# Run tests
cargo test

# Check without building
cargo check
```

The `build_for_xcode.sh` script:
1. Builds the library for `aarch64-apple-darwin` (release mode)
2. Copies `libphotolibrariancore.a` to `xcode-libs/`
3. Reports the library size and location

Output: `xcode-libs/libphotolibrariancore.a` (static library for Xcode)

## Swift Integration

See **[UNIFFI_INTEGRATION.md](./UNIFFI_INTEGRATION.md)** for complete instructions on integrating this library into an Xcode SwiftUI project.

Quick summary:
1. Run `./build_for_xcode.sh`
2. Add `libphotolibrariancore.a` to your Xcode project
3. Set up a build phase to generate Swift bindings from `photolibrariancore.udl`
4. Import and use the async API in Swift

## Database Schema

The DuckDB catalogue stores images in a single table:

```sql
CREATE TABLE images (
    id INTEGER PRIMARY KEY,
    file_path TEXT NOT NULL UNIQUE,
    file_size INTEGER NOT NULL,
    file_name TEXT NOT NULL,
    file_extension TEXT,
    created_timestamp INTEGER NOT NULL,
    modified_timestamp INTEGER NOT NULL,

    -- Camera metadata
    camera_make TEXT,
    camera_model TEXT,
    lens_model TEXT,
    focal_length REAL,
    aperture REAL,
    shutter_speed REAL,
    iso INTEGER,
    capture_datetime TEXT,

    -- Image properties
    pixel_width INTEGER,
    pixel_height INTEGER,
    color_space TEXT,
    bit_depth INTEGER,

    -- GPS
    gps_latitude REAL,
    gps_longitude REAL,
    gps_altitude REAL,

    -- IPTC
    copyright TEXT,
    creator TEXT,
    description TEXT,

    indexed_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_camera_model ON images(camera_model);
CREATE INDEX idx_capture_datetime ON images(capture_datetime);
CREATE INDEX idx_created_timestamp ON images(created_timestamp);
CREATE INDEX idx_file_extension ON images(file_extension);
```

## Architectural Decisions

Key constraints from **CONTEXT.md**:

1. **EXIF extraction happens in Swift** using Apple's ImageIO framework (not in Rust). This ensures native RAW format support for all Apple-recognized formats.

2. **File paths are never constructed in Rust**. All paths come from Swift via sandbox-legal mechanisms (`NSOpenPanel`, security-scoped bookmarks).

3. **App Store sandboxing**: All libraries (DuckDB, LanceDB) are bundled and run in-process. No subprocesses, no external binaries.

4. **Apple Silicon only**: `aarch64-apple-darwin` exclusively. No Intel support.

5. **In-process architecture**: The Rust library is a static lib embedded in the macOS app. No daemon, no IPC.

## Data Flow (Milestone 1)

```
User selects directory (NSOpenPanel)
    ↓
Swift scans recursively for image files
    ↓
ImageIO extracts EXIF per file
    ↓
Swift populates ImageMetadata structs
    ↓
Swift calls ingest_metadata() via UniFFI
    ↓
Rust writes to DuckDB (batch insert, duplicate detection)
    ↓
Swift calls get_image_count() to display results
```

## Testing

Currently includes:
- UniFFI scaffolding generation
- DuckDB schema creation
- Catalogue initialization
- Metadata ingestion with duplicate handling
- Count queries

Run tests:

```bash
cargo test
```

## Future Milestones

- [ ] Milestone 2: LanceDB integration for semantic search
- [ ] Milestone 3: Vector embedding generation via Core ML
- [ ] Milestone 4: Smart collections and filtering
- [ ] Milestone 5: External editor integration API
- [ ] Milestone 6: Export and archiving functions

## Dependencies

| Crate | Version | Purpose |
|---|---|---|
| `duckdb` | 1.1 | Embedded analytical database (bundled feature) |
| `lancedb` | 0.14 | Embedded vector database for semantic search |
| `arrow-array` | 53 | Apache Arrow data structures |
| `arrow-schema` | 53 | Apache Arrow schemas |
| `tokio` | 1 | Async runtime (full feature set) |
| `serde` | 1 | Serialization (derive feature) |
| `serde_json` | 1 | JSON serialization |
| `uniffi` | 0.28 | Rust-Swift FFI bridge |
| `once_cell` | 1.20 | Global state initialization |

## Sandboxing & App Store Compliance

This library is designed for Mac App Store distribution with full sandboxing:

- DuckDB uses `bundled` feature (no system library dependencies)
- LanceDB runs entirely in-process
- No file I/O initiated by Rust (all paths from Swift)
- No subprocess spawning
- No network access (future cloud sync handled by Swift layer)

Tested configurations:
- [x] Compiles with `cargo build`
- [x] UniFFI scaffolding generates correctly
- [ ] Sandbox validation (requires signed `.app` build)
- [ ] DuckDB file creation in sandboxed environment
- [ ] LanceDB operation in sandboxed environment

## Contributing

This is a commercial project for Mac App Store distribution. Development is AI-assisted using Claude Code and Anthropic's Claude.

For bug reports and feedback after public release, see the main PhotoLibrarian repository.

## License

Proprietary. © 2026 Pictrus Inc Software. All rights reserved.

This code is part of PhotoLibrarian, a commercial macOS application. Not licensed for redistribution or reuse.

---

**Developer**: Richard Wagner
**Organization**: Pictrus Inc Software
**Repository**: https://github.com/PictrusIncSoftware/photolibrariancore
**Status**: Active development — Milestone 1 complete
**Last Updated**: May 2026
