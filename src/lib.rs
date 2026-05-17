// Import DuckDB for embedded database operations
// DuckDB is used instead of SQLite for better analytical query performance on large image catalogues
use duckdb::{Connection, params};
use std::sync::{Arc, Mutex};
use std::path::PathBuf;

// UniFFI scaffolding — generates the Rust-to-Swift FFI layer
// This macro reads the .udl file and creates the necessary C-compatible interface code
uniffi::include_scaffolding!("photolibrariancore");

// Global catalogue connection singleton
//
// Design decision: Using a global connection wrapped in Arc<Mutex<>> rather than passing
// connections through function parameters. This is adequate for Milestone 1's sequential
// batch inserts but will become a bottleneck under concurrent writes.
//
// Why Arc<Mutex<Option<>>>:
// - Arc: Enables shared ownership across threads (required for UniFFI async calls)
// - Mutex: Provides interior mutability and thread-safe access
// - Option: Allows initialization to None before the database is opened
//
// Data flow: Swift calls initialize_catalogue() → stores connection here →
//           subsequent calls (ingest_metadata, get_image_count) use the stored connection
//
// Known limitation: Multiple threads will serialize on this lock. Replace with a
// connection pool (e.g., r2d2) in a later milestone when concurrent operations are needed.
static CATALOGUE: once_cell::sync::Lazy<Arc<Mutex<Option<Connection>>>> =
    once_cell::sync::Lazy::new(|| Arc::new(Mutex::new(None)));

/// Represents the metadata for a single image file
///
/// This struct mirrors the database schema and serves as the data transfer object
/// between Swift and Rust layers via UniFFI. It must also be defined in the .udl file.
///
/// Design decision: Most fields are Option<T> because EXIF metadata is inherently
/// incomplete and varies by camera/file type. Only file system properties (path, size,
/// timestamps) are guaranteed to exist.
///
/// Data flow:
/// 1. Swift extracts EXIF via ImageIO (native RAW format support)
/// 2. Swift constructs ImageMetadata instances
/// 3. Swift passes Vec<ImageMetadata> to Rust via ingest_metadata()
/// 4. Rust inserts records into DuckDB
///
/// Why not extract EXIF in Rust? Rust EXIF crates don't support the full matrix of
/// RAW formats (CR2, CR3, NEF, ARW, RAF, etc.) required by the spec. Apple's ImageIO
/// framework provides this coverage automatically and inherits new format support as
/// Apple adds it.
#[derive(Debug, Clone)]
pub struct ImageMetadata {
    // File system properties (always present)
    pub file_path: String,        // Absolute path; passed from Swift's sandbox-legal picker
    pub file_size: u64,
    pub file_name: String,         // Basename extracted from path
    pub file_extension: Option<String>,  // Lowercase extension (e.g., "jpg", "cr3")
    pub created_timestamp: i64,    // Unix epoch seconds
    pub modified_timestamp: i64,   // Unix epoch seconds

    // Camera/capture metadata (from EXIF, may be absent)
    pub camera_make: Option<String>,       // e.g., "Canon", "Nikon"
    pub camera_model: Option<String>,      // e.g., "EOS R5", "D850"
    pub lens_model: Option<String>,
    pub focal_length: Option<f64>,         // Millimeters
    pub aperture: Option<f64>,             // F-stop value
    pub shutter_speed: Option<f64>,        // Seconds (e.g., 0.0005 for 1/2000)
    pub iso: Option<u32>,                  // ISO speed rating
    pub capture_datetime: Option<String>,  // ISO 8601 string from EXIF

    // Image properties (resolution and color)
    pub pixel_width: Option<u32>,
    pub pixel_height: Option<u32>,
    pub color_space: Option<String>,       // e.g., "sRGB", "Adobe RGB"
    pub bit_depth: Option<u32>,            // Bits per channel

    // GPS coordinates (from geotagged images)
    pub gps_latitude: Option<f64>,         // Decimal degrees
    pub gps_longitude: Option<f64>,        // Decimal degrees
    pub gps_altitude: Option<f64>,         // Meters above sea level

    // IPTC/copyright metadata (user-editable in many cameras/software)
    pub copyright: Option<String>,
    pub creator: Option<String>,           // Photographer name
    pub description: Option<String>,       // Caption/alt text

    // Organization metadata (user-assigned within PhotoLibrarian)
    // These fields are initially None and populated by user interactions in the app
    pub rating: Option<u8>,                // 0-5 stars
    pub flag: Option<String>,              // "pick", "reject", or None
    pub color_label: Option<String>,       // "red", "green", "blue", etc.
}

/// Represents a complete image record from the database
///
/// This struct contains all columns from the images table, including the auto-generated
/// id and indexed_timestamp. Used for querying and displaying catalogue contents.
///
/// Unlike ImageMetadata (which is used for ingestion), this struct is read-only and
/// includes the database-generated fields.
#[derive(Debug, Clone)]
pub struct ImageRecord {
    // Database-generated fields
    pub id: i64,                           // Auto-generated primary key
    pub indexed_timestamp: String,         // When record was added to catalogue (ISO 8601)

    // File system properties
    pub file_path: String,
    pub file_size: u64,
    pub file_name: String,
    pub file_extension: Option<String>,
    pub created_timestamp: i64,
    pub modified_timestamp: i64,

    // Camera/capture metadata
    pub camera_make: Option<String>,
    pub camera_model: Option<String>,
    pub lens_model: Option<String>,
    pub focal_length: Option<f64>,
    pub aperture: Option<f64>,
    pub shutter_speed: Option<f64>,
    pub iso: Option<u32>,
    pub capture_datetime: Option<String>,

    // Image properties
    pub pixel_width: Option<u32>,
    pub pixel_height: Option<u32>,
    pub color_space: Option<String>,
    pub bit_depth: Option<u32>,

    // GPS coordinates
    pub gps_latitude: Option<f64>,
    pub gps_longitude: Option<f64>,
    pub gps_altitude: Option<f64>,

    // IPTC/copyright metadata
    pub copyright: Option<String>,
    pub creator: Option<String>,
    pub description: Option<String>,

    // Organization metadata
    pub rating: Option<u8>,
    pub flag: Option<String>,
    pub color_label: Option<String>,
    pub rotation: i32,

    // Cross-directory duplicate consolidation (DESIGN-Duplicate-Consolidation.md)
    //
    // Computed at query time via window function. For each row in the result set,
    // duplicate_group_id is:
    //   - NULL if capture_datetime IS NULL (singletons by exemption rule)
    //   - NULL if the row is the only row in its (capture_datetime, camera_model,
    //     pixel_width, pixel_height) group (singletons)
    //   - Otherwise, the id of the row in the group with the lexicographically
    //     smallest file_path (the canonical/representative row)
    //
    // Consumed by Swift `PhotosView.displayedImages` filter: when "Show duplicates"
    // is OFF, rows where `Int64(id) != duplicate_group_id` are hidden.
    pub duplicate_group_id: Option<i64>,
}

/// Recognized RAW image file extensions
///
/// Compile-time list of file extensions (lowercase, no leading dot) that
/// PhotoLibrarian classifies as RAW images. RAW images are the camera-original
/// negative; in JPEG+RAW pair workflows the RAW is the authoritative source
/// and the JPEG is a preview.
///
/// Design decision: `const &[&str]` rather than `static` or runtime list:
/// - Compile-time known, zero runtime overhead, thread-safe by construction.
/// - These are facts about the world (formats we recognize), not user state.
/// - Single source of truth at the Rust layer; SQL queries fetch candidates
///   and classify in Rust rather than encoding the list in SQL.
///
/// Adding a new RAW format = one-line edit here. User-configurable lists are
/// an explicit non-goal of the current design.
const RAW_EXTENSIONS: &[&str] = &[
    "nef",  // Nikon
    "cr2",  // Canon (older)
    "cr3",  // Canon (newer)
    "arw",  // Sony
    "dng",  // Adobe / Pentax / Leica / etc.
    "raf",  // Fujifilm
    "rw2",  // Panasonic
    "orf",  // Olympus / OM System
];

/// Recognized JPEG image file extensions
///
/// Companion to RAW_EXTENSIONS. Two-element list because the JPEG ecosystem
/// has settled on these two spellings in the workflows this app targets.
const JPEG_EXTENSIONS: &[&str] = &["jpg", "jpeg"];

/// Image classification categories for JPEG+RAW pair handling
///
/// Used by classify_extension to answer "what kind of image is this?" and by
/// find_counterpart_image (added separately) to decide whether a counterpart
/// query makes sense — Other-kind images have no JPEG+RAW counterpart by
/// definition.
///
/// Three variants chosen deliberately over a boolean is_raw:
/// - `Other` is a real category (PNG, TIFF, HEIC, etc.), not a "not RAW"
///   bucket. These formats don't participate in pair workflows and should
///   be classified explicitly.
/// - Pattern-matching on three variants in Swift is clearer than chained
///   booleans, and future format categories extend the enum without
///   changing call-site shape.
///
/// This is the first enum exposed across the UniFFI boundary in this
/// project. UniFFI generates a Swift enum with cases `.jpeg`, `.raw`,
/// `.other` (lowercased first character). Verify casing after binding
/// generation; surface any deviation before changing call sites.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageKind
{
    Jpeg,
    Raw,
    Other,
}

/// Classify a file extension into JPEG, RAW, or Other
///
/// Pure string-to-enum lookup. Case-insensitive: lowercases the input before
/// checking the constant tables.
///
/// **Synchronous** — this is the project's first synchronous UniFFI export.
/// All prior exports are `[Async]`. The semantic match is correct: no I/O,
/// no shared state, no DuckDB. Forcing it async would push the Swift-side
/// consumer (a SwiftUI computed property that cannot `await`) into awkward
/// shapes.
///
/// The function takes the extension as a separate parameter (not parsed from
/// a path) because the catalogue already stores `file_extension` as a
/// separate, lowercase-normalized column. Callers pass it directly without
/// further preprocessing.
///
/// Parameters:
/// - ext: File extension without leading dot (e.g., "jpg", "NEF", "Cr3");
///        empty string returns ImageKind::Other.
///
/// Returns:
/// - ImageKind::Jpeg if ext (lowercased) matches JPEG_EXTENSIONS
/// - ImageKind::Raw if ext (lowercased) matches RAW_EXTENSIONS
/// - ImageKind::Other otherwise
pub fn classify_extension(ext: String) -> ImageKind
{
    let lower = ext.to_lowercase();
    if JPEG_EXTENSIONS.contains(&lower.as_str())
    {
        ImageKind::Jpeg
    }
    else if RAW_EXTENSIONS.contains(&lower.as_str())
    {
        ImageKind::Raw
    }
    else
    {
        ImageKind::Other
    }
}

/// Parsed components of a filename
///
/// Returned by `parse_filename` — owns the canonical filename-parsing contract
/// for the catalogue (DESIGN-Duplicate-Consolidation.md §7).
///
/// Fields:
/// - `stem`: everything before the last `.` in the filename, in its **original
///   case**. The duplicate-detection partition compares stems case-insensitively
///   via `LOWER(file_stem)` at the SQL boundary (Section 12), so the stored
///   form preserves display case while comparison applies case-folding.
/// - `extension_lower`: everything after the last `.`, **always lowercased**.
///   Set to the synthetic placeholder `"metype"` (a DOS-8.3-era mnemonic for
///   "missing extension type") when the filename is malformed — no dot, leading
///   dot, or trailing dot — per Section 7's parsing rule.
/// - `kind`: result of `classify_extension(extension_lower)`. Always
///   `ImageKind::Other` for the malformed cases (since `"metype"` is in neither
///   JPEG_EXTENSIONS nor RAW_EXTENSIONS).
///
/// Why the Rust field is `extension_lower` rather than `extension`: `extension`
/// is a reserved keyword in Swift, and UniFFI's generated Swift binding would
/// either require backtick-escaping at every call site or surface a binding
/// error. The renamed field is unambiguous about its case-folded contract and
/// keeps both call sides clean. The lowercased contract is documented by the
/// field name itself.
#[derive(Debug, Clone)]
pub struct ParsedFilename
{
    pub stem: String,
    pub extension_lower: String,
    pub kind: ImageKind,
}

/// Parse a filename into its stem, lowercased extension, and ImageKind.
///
/// **Synchronous** — same rationale as `classify_extension`: no I/O, no DuckDB
/// access, no shared state. Forcing async would force Swift consumers into
/// awkward shapes.
///
/// This is the canonical filename-parsing entry point for the catalogue.
/// Composes on top of `classify_extension`:
/// - `classify_extension` is the canonical extension-string → kind mapping.
/// - `parse_filename` is the higher-level entry point that accepts a full
///   filename, parses it, and delegates extension classification to
///   `classify_extension`.
///
/// Parsing rule (DESIGN-Duplicate-Consolidation.md §7):
/// 1. Find the LAST occurrence of `.` in `file_name`.
/// 2. If no dot exists, OR the dot is at position 0, OR everything after the
///    dot is empty:
///       stem = file_name (or "metype" if file_name is empty)
///       extension_lower = "metype"
///       kind = ImageKind::Other
/// 3. Otherwise:
///       stem = everything before the last dot (original case)
///       extension_lower = everything after the last dot, lowercased
///       kind = classify_extension(extension_lower)
///
/// Examples:
/// - `"IMG_1234.NEF"` → stem `"IMG_1234"`, ext `"nef"`, kind Raw
/// - `"photo.jpg"`    → stem `"photo"`,    ext `"jpg"`, kind Jpeg
/// - `"foo.bar.baz"`  → stem `"foo.bar"`,  ext `"baz"`, kind Other
/// - `".DS_Store"`    → stem `".DS_Store"`, ext `"metype"`, kind Other
/// - `"README"`       → stem `"README"`,    ext `"metype"`, kind Other
/// - `"foo."`         → stem `"foo."`,      ext `"metype"`, kind Other
/// - `""`             → stem `"metype"`,    ext `"metype"`, kind Other
///
/// Parameters:
/// - file_name: The filename to parse (no path components). Caller is expected
///   to pass just the file name, not a full path; the function does not strip
///   directory components.
///
/// Returns:
/// - ParsedFilename with the three derived fields populated per the rule.
pub fn parse_filename(file_name: String) -> ParsedFilename
{
    // Empty filename: synthetic stem so the stored column is never empty.
    if file_name.is_empty()
    {
        return ParsedFilename
        {
            stem: "metype".to_string(),
            extension_lower: "metype".to_string(),
            kind: ImageKind::Other,
        };
    }

    match file_name.rfind('.')
    {
        // No dot at all — case 2.
        None => ParsedFilename
        {
            stem: file_name,
            extension_lower: "metype".to_string(),
            kind: ImageKind::Other,
        },
        // Leading dot, e.g. ".DS_Store" — case 2.
        // The whole name is treated as the stem; there is no extension to parse.
        Some(0) => ParsedFilename
        {
            stem: file_name,
            extension_lower: "metype".to_string(),
            kind: ImageKind::Other,
        },
        // Trailing dot, e.g. "foo." — case 2.
        // Note: idx is a byte offset; for the trailing-dot test we compare against
        // file_name.len() - 1. Byte offsets are safe here because '.' is single-byte
        // ASCII; rfind on a multi-byte string still returns a byte offset on a code
        // unit boundary.
        Some(idx) if idx == file_name.len() - 1 => ParsedFilename
        {
            stem: file_name,
            extension_lower: "metype".to_string(),
            kind: ImageKind::Other,
        },
        // Normal case: split at the last dot.
        Some(idx) =>
        {
            let stem = file_name[..idx].to_string();
            let ext = file_name[idx + 1..].to_lowercase();
            let kind = classify_extension(ext.clone());
            ParsedFilename
            {
                stem,
                extension_lower: ext,
                kind,
            }
        }
    }
}

/// Initialize the catalogue database at the given path
///
/// This function must be called once before any other catalogue operations. It creates
/// the DuckDB database file, establishes the schema, and stores the connection in the
/// global CATALOGUE singleton.
///
/// Design decision: Returns bool instead of Result<> because UniFFI's async support
/// for Result is more complex. Errors are logged to stderr and false is returned.
///
/// Data flow:
/// 1. Swift obtains a sandbox-legal path (e.g., via NSOpenPanel or app container)
/// 2. Swift calls this function with the path
/// 3. Rust creates the DB file and schema
/// 4. Connection is stored in CATALOGUE for subsequent operations
///
/// Why async? Although this operation is I/O-bound, it's declared async to maintain
/// consistency with the UniFFI async pattern used throughout the API surface.
/// UniFFI generates Swift async/await bindings for Rust async functions.
///
/// Parameters:
/// - catalogue_path: Absolute path to the .duckdb file (e.g., ~/Library/Application Support/PhotoLibrarian/catalogue.duckdb)
///
/// Returns:
/// - true if initialization succeeded
/// - false if directory creation, file open, or schema creation failed
pub async fn initialize_catalogue(catalogue_path: String) -> bool {
    let path = PathBuf::from(&catalogue_path);

    // Create parent directory if it doesn't exist
    // This handles the case where the app container directory structure isn't yet created
    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            eprintln!("Failed to create catalogue directory: {}", e);
            return false;
        }
    }

    // Open or create the database
    // DuckDB's bundled feature ensures the database engine is statically linked
    // (no system library dependency — required for App Store sandboxing)
    let conn = match Connection::open(&path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to open catalogue database: {}", e);
            return false;
        }
    };

    // EXPERIMENT 3: Query DuckDB version to confirm bundled library is being used
    match conn.query_row("SELECT version()", [], |row| row.get::<_, String>(0)) {
        Ok(version) => eprintln!("DuckDB version: {}", version),
        Err(e) => eprintln!("Failed to query DuckDB version: {}", e),
    }

    // Create the images table schema
    // Design decision: Single denormalized table for Milestone 1. This schema prioritizes
    // query simplicity over normalization. Future milestones may introduce related tables
    // for tags, collections, face recognition data, and vector embeddings.
    let schema = r#"
        CREATE SEQUENCE IF NOT EXISTS images_id_seq START 1;

        CREATE TABLE IF NOT EXISTS images (
            -- Primary key: ID auto-generated via images_id_seq sequence
            -- (DuckDB 1.2.2 silently drops GENERATED BY DEFAULT AS IDENTITY,
            -- so we use the underlying mechanism directly)
            id INTEGER PRIMARY KEY DEFAULT nextval('images_id_seq'),

            -- File system properties
            -- UNIQUE constraint on file_path prevents duplicate imports
            -- This allows INSERT OR IGNORE to silently skip duplicates during batch ingestion
            file_path TEXT NOT NULL UNIQUE,
            file_size BIGINT NOT NULL,  -- BIGINT (64-bit) required: large TIFFs and PSDs can exceed 2.1GB INT32 limit
            file_name TEXT NOT NULL,
            file_extension TEXT,
            -- Filename-derived columns for cross-directory duplicate consolidation
            -- (DESIGN-Duplicate-Consolidation.md §5). Populated at ingest by
            -- parse_filename(). Nullable in DDL — the non-null invariant is enforced
            -- at the populate-time contract (new-record ingest + one-time backfill),
            -- the same pattern as `rotation INTEGER DEFAULT 0`. file_stem preserves
            -- original case (display); image_kind is always lowercase ("jpeg" /
            -- "raw" / "other"). Comparisons that need case-folding apply LOWER() at
            -- the boundary (see Section 12, Canonical Case for String Comparisons).
            file_stem VARCHAR,
            image_kind VARCHAR,
            created_timestamp INTEGER NOT NULL,  -- Unix epoch seconds
            modified_timestamp INTEGER NOT NULL, -- Unix epoch seconds

            -- Camera/capture metadata (nullable — not all images have complete EXIF)
            camera_make TEXT,
            camera_model TEXT,
            lens_model TEXT,
            focal_length REAL,      -- Millimeters
            aperture REAL,          -- F-stop value
            shutter_speed REAL,     -- Seconds
            iso INTEGER,            -- ISO speed rating
            capture_datetime TEXT,  -- ISO 8601 string

            -- Image properties
            pixel_width INTEGER,
            pixel_height INTEGER,
            color_space TEXT,
            bit_depth INTEGER,

            -- GPS coordinates (from geotagged images)
            gps_latitude REAL,
            gps_longitude REAL,
            gps_altitude REAL,

            -- IPTC/copyright metadata
            copyright TEXT,
            creator TEXT,
            description TEXT,

            -- Organization metadata (user-assigned)
            rating INTEGER,      -- 0-5 stars
            flag TEXT,           -- "pick", "reject", or NULL
            color_label TEXT,    -- "red", "green", "blue", etc.
            rotation INTEGER DEFAULT 0,  -- Image rotation in degrees: 0, 90, 180, or 270

            -- Audit timestamp: when this record was added to the catalogue
            indexed_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Schema upgrade for pre-existing catalogues
        -- (DESIGN-Duplicate-Consolidation.md §5, decision C1 from the Step (a')
        -- read-and-confirm pass). CREATE TABLE IF NOT EXISTS is a no-op when the
        -- images table already exists, so it cannot add columns to pre-existing
        -- catalogues. The ALTER TABLE ADD COLUMN IF NOT EXISTS pair below is the
        -- migration path: on a fresh database these are no-ops (the columns were
        -- just defined above); on an existing 32,855-record catalogue these add
        -- the columns the backfill migration then populates.
        ALTER TABLE images ADD COLUMN IF NOT EXISTS file_stem VARCHAR;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS image_kind VARCHAR;

        -- Indexes for efficient filtering and querying
        -- These columns are commonly used in WHERE clauses and ORDER BY operations
        -- Rationale: Photographers frequently filter by camera, date, rating, and organization markers
        CREATE INDEX IF NOT EXISTS idx_camera_model ON images(camera_model);
        CREATE INDEX IF NOT EXISTS idx_capture_datetime ON images(capture_datetime);
        CREATE INDEX IF NOT EXISTS idx_created_timestamp ON images(created_timestamp);
        CREATE INDEX IF NOT EXISTS idx_file_extension ON images(file_extension);
        CREATE INDEX IF NOT EXISTS idx_rating ON images(rating);
        CREATE INDEX IF NOT EXISTS idx_flag ON images(flag);
        CREATE INDEX IF NOT EXISTS idx_color_label ON images(color_label);
    "#;

    // Execute schema creation as a batch
    // execute_batch is used instead of execute because we're running multiple statements
    if let Err(e) = conn.execute_batch(schema) {
        eprintln!("Failed to create catalogue schema: {}", e);
        return false;
    }

    // EXPERIMENT 4: Verify the actual schema that was created
    match conn.query_row(
        "SELECT sql FROM duckdb_tables() WHERE table_name = 'images'",
        [],
        |row| row.get::<_, String>(0)
    ) {
        Ok(actual_schema) => eprintln!("Actual schema created:\n{}", actual_schema),
        Err(e) => eprintln!("Failed to query schema: {}", e),
    }

    // --- Backfill migration (DESIGN-Duplicate-Consolidation.md §8) ----------
    //
    // Runs once at app launch, after schema creation (CREATE TABLE + ALTER
    // TABLE IF NOT EXISTS), before the catalogue connection is stored in
    // the CATALOGUE singleton. Atomic-with-init from the app's perspective —
    // no UI query can race the migration because the singleton is the only
    // public access path and is set only after this block returns.
    //
    // Two responsibilities, both wrapped in a single transaction:
    //   1. Populate file_stem and image_kind for any row where either column
    //      is NULL. On a fresh database this loop is a no-op (no rows). On
    //      an existing 32,855-record catalogue this is a one-time cost.
    //      The estimate per Section 14 is well under a second.
    //   2. Idempotent lowercase normalization of file_extension. The current
    //      catalogue spot-check shows file_extension is already lowercase,
    //      but the design wants a belt-and-braces UPDATE to enforce the
    //      lowercase-canonical convention even on data ingested before
    //      it was a stated invariant.
    //
    // Idempotency: the WHERE clauses on both steps mean that running this
    // block a second time has no work to do — rows with non-NULL file_stem
    // and image_kind are skipped, and rows whose file_extension already
    // matches LOWER(file_extension) are skipped.
    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;")
    {
        eprintln!("[migration] Failed to begin transaction: {}", e);
        return false;
    }

    // Step 1: backfill file_stem and image_kind.
    let rows_needing_backfill: Vec<(i64, String)> = match conn.prepare(
        "SELECT id, file_name FROM images WHERE file_stem IS NULL OR image_kind IS NULL"
    )
    {
        Ok(mut stmt) =>
        {
            match stmt.query_map([], |row|
            {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })
            {
                Ok(mapped) => mapped.filter_map(Result::ok).collect(),
                Err(e) =>
                {
                    eprintln!("[migration] Failed to query rows needing backfill: {}", e);
                    Vec::new()
                }
            }
        }
        Err(e) =>
        {
            eprintln!("[migration] Failed to prepare backfill query: {}", e);
            Vec::new()
        }
    };

    let mut backfilled_count = 0u64;
    for (row_id, file_name) in &rows_needing_backfill
    {
        let parsed = parse_filename(file_name.clone());
        let kind_str = match parsed.kind
        {
            ImageKind::Jpeg => "jpeg",
            ImageKind::Raw  => "raw",
            ImageKind::Other => "other",
        };
        match conn.execute(
            "UPDATE images SET file_stem = ?1, image_kind = ?2 WHERE id = ?3",
            params![parsed.stem, kind_str, row_id],
        )
        {
            Ok(_) => backfilled_count += 1,
            Err(e) => eprintln!("[migration] Failed to update row id={}: {}", row_id, e),
        }
    }
    eprintln!("[migration] backfilled file_stem/image_kind for {} rows", backfilled_count);

    // Step 2: idempotent lowercase normalization of file_extension.
    // The WHERE clause restricts the UPDATE to rows that actually differ;
    // DuckDB returns the count of changed rows in `changed`.
    match conn.execute(
        "UPDATE images SET file_extension = LOWER(file_extension) \
         WHERE file_extension IS NOT NULL AND file_extension != LOWER(file_extension)",
        [],
    )
    {
        Ok(changed) => eprintln!("[migration] file_extension lowercased: {} rows", changed),
        Err(e) => eprintln!("[migration] Failed to normalize file_extension: {}", e),
    }

    if let Err(e) = conn.execute_batch("COMMIT;")
    {
        eprintln!("[migration] Failed to commit migration transaction: {}", e);
        return false;
    }
    // --- End backfill migration --------------------------------------------

    // Store the connection in the global state
    // This connection will be reused by all subsequent catalogue operations
    // Mutex ensures thread-safe access when called from multiple Swift async tasks
    let mut catalogue = CATALOGUE.lock().unwrap();
    *catalogue = Some(conn);

    true
}

/// Ingest a batch of image metadata records into the catalogue
///
/// This is the primary data ingestion function. It receives a vector of ImageMetadata
/// structs from Swift and inserts them into the DuckDB catalogue.
///
/// Design decision: Uses INSERT OR IGNORE to silently skip duplicates. This is
/// appropriate because re-scanning a directory is a common operation and shouldn't
/// fail or require explicit duplicate checking on the Swift side.
///
/// Data flow:
/// 1. Swift scans directory and extracts EXIF for each image
/// 2. Swift builds Vec<ImageMetadata> and passes it via UniFFI
/// 3. Rust iterates and inserts each record
/// 4. Returns count of actually inserted records (excludes skipped duplicates)
///
/// Why not batch insert? DuckDB supports batch inserts via prepared statements, but
/// this per-record approach provides better error isolation — one malformed record
/// doesn't abort the entire batch. For Milestone 1's sequential batch workflow, the
/// performance difference is negligible. Consider batch inserts in a future milestone
/// if throughput becomes a bottleneck.
///
/// Parameters:
/// - metadata: Vector of ImageMetadata structs
///
/// Returns:
/// - Number of records successfully inserted (excludes duplicates)
pub async fn ingest_metadata(metadata: Vec<ImageMetadata>) -> u32 {
    // Acquire lock on global catalogue connection
    // Known limitation: All concurrent calls serialize on this lock
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    // Prepared statement with positional parameters (?1, ?2, ...)
    // INSERT OR IGNORE: Skip records where file_path already exists (UNIQUE constraint)
    // The id column is omitted and auto-filled by DuckDB using the DEFAULT nextval() clause
    //
    // file_stem and image_kind are positioned immediately after file_extension to
    // keep the filename-parsing family (file_name → file_extension → file_stem →
    // image_kind) visually grouped (decision C6 from the Step (a')
    // read-and-confirm pass). They are populated per-record by parse_filename()
    // below.
    let insert_sql = r#"
        INSERT OR IGNORE INTO images (
            file_path, file_size, file_name, file_extension,
            file_stem, image_kind,
            created_timestamp, modified_timestamp,
            camera_make, camera_model, lens_model,
            focal_length, aperture, shutter_speed, iso,
            capture_datetime,
            pixel_width, pixel_height, color_space, bit_depth,
            gps_latitude, gps_longitude, gps_altitude,
            copyright, creator, description,
            rating, flag, color_label
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10,
            ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20,
            ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29
        )
    "#;

    let mut inserted_count = 0u32;

    // Iterate through each metadata record and insert individually
    // Per-record approach: Better error isolation at the cost of slightly lower throughput
    for record in metadata {
        // Derive filename-parsing columns via the canonical parser. The Rust core
        // owns this logic so Swift can be ignorant of the "metype" placeholder
        // and the case-folding convention; both are enforced here at write time.
        // (DESIGN-Duplicate-Consolidation.md §5, §7.)
        let parsed = parse_filename(record.file_name.clone());
        let image_kind_str = match parsed.kind
        {
            ImageKind::Jpeg => "jpeg",
            ImageKind::Raw  => "raw",
            ImageKind::Other => "other",
        };

        // Execute the prepared statement with positional parameters
        // Type conversions: u32/u64 → i64 for DuckDB INTEGER columns
        // Option<T> fields are passed directly — DuckDB handles NULL for None
        let result = conn.execute(
            insert_sql,
            params![
                record.file_path,                           // ?1
                record.file_size as i64,                    // ?2  (u64 → i64 cast safe for file sizes < 9 exabytes)
                record.file_name,                           // ?3
                record.file_extension,                      // ?4
                parsed.stem,                                // ?5  file_stem (original case preserved)
                image_kind_str,                             // ?6  image_kind (always lowercase: "jpeg"/"raw"/"other")
                record.created_timestamp,                   // ?7
                record.modified_timestamp,                  // ?8
                record.camera_make,                         // ?9
                record.camera_model,                        // ?10
                record.lens_model,                          // ?11
                record.focal_length,                        // ?12
                record.aperture,                            // ?13
                record.shutter_speed,                       // ?14
                record.iso.map(|v| v as i64),              // ?15 (u32 → i64)
                record.capture_datetime,                    // ?16
                record.pixel_width.map(|v| v as i64),      // ?17 (u32 → i64)
                record.pixel_height.map(|v| v as i64),     // ?18 (u32 → i64)
                record.color_space,                         // ?19
                record.bit_depth.map(|v| v as i64),        // ?20 (u32 → i64)
                record.gps_latitude,                        // ?21
                record.gps_longitude,                       // ?22
                record.gps_altitude,                        // ?23
                record.copyright,                           // ?24
                record.creator,                             // ?25
                record.description,                         // ?26
                record.rating.map(|v| v as i64),           // ?27 (u8 → i64)
                record.flag,                                // ?28
                record.color_label,                         // ?29
            ],
        );

        match result {
            // Ok(changed) returns the number of rows affected (1 for insert, 0 for duplicate skip)
            Ok(changed) => inserted_count += changed as u32,
            // Log error but continue processing remaining records
            // This ensures one bad record doesn't abort the entire batch
            Err(e) => eprintln!("Failed to insert record {}: {}", record.file_path, e),
        }
    }

    // Return count of successfully inserted records
    // Note: This excludes skipped duplicates (which return Ok(0))
    inserted_count
}

/// Get the total count of images in the catalogue
///
/// Simple utility function for UI display and validation. Used by Swift to show
/// "X images in catalogue" status or verify ingestion results.
///
/// Data flow:
/// Swift calls this after ingestion or on app launch to populate UI statistics
///
/// Returns:
/// - Total number of image records in the catalogue
/// - 0 if catalogue not initialized or query fails
pub async fn get_image_count() -> u64 {
    // Acquire lock and validate connection
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    // Execute COUNT(*) query
    // query_row is used for single-row results (more ergonomic than query + fetch)
    // The closure |row| row.get(0) extracts the first column (the count)
    let count_result: Result<i64, _> = conn.query_row(
        "SELECT COUNT(*) FROM images",
        [],           // No query parameters
        |row| row.get(0),
    );

    match count_result {
        // DuckDB returns i64, cast to u64 for the unsigned count semantic
        Ok(count) => count as u64,
        Err(e) => {
            eprintln!("Failed to query image count: {}", e);
            0  // Return 0 on error (catalogue is effectively empty to the caller)
        }
    }
}

/// Format Unix epoch seconds to ISO 8601 string format
///
/// Converts DuckDB epoch timestamp (i64 seconds since Unix epoch) to
/// human-readable format: "YYYY-MM-DD HH:MM:SS"
///
/// This is a helper for the TIMESTAMP→String conversion pattern used throughout
/// this project, since UniFFI has no native TIMESTAMP type support.
fn format_epoch_to_iso8601(epoch_secs: i64) -> String {
    // Convert to local time components
    // Note: This uses a simple calculation assuming UTC
    let total_secs = epoch_secs;
    let days_since_epoch = total_secs / 86400;
    let seconds_today = total_secs % 86400;

    let hours = (seconds_today / 3600) % 24;
    let minutes = (seconds_today / 60) % 60;
    let seconds = seconds_today % 60;

    // Calculate year, month, day from days since epoch (1970-01-01)
    // Simplified calculation for demonstration
    let mut year = 1970;
    let mut remaining_days = days_since_epoch;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    let month_days = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1;
    for days in month_days.iter() {
        if remaining_days < *days {
            break;
        }
        remaining_days -= days;
        month += 1;
    }

    let day = remaining_days + 1;

    format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
            year, month, day, hours, minutes, seconds)
}

fn is_leap_year(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Get images from the catalogue with pagination support
///
/// Returns a paginated list of image records in the catalogue, ordered by ID.
/// This enables the Swift UI layer to load large catalogues incrementally without
/// freezing the interface.
///
/// Pagination pattern:
/// - First page: get_all_images(limit: 500, offset: 0)
/// - Second page: get_all_images(limit: 500, offset: 500)
/// - Third page: get_all_images(limit: 500, offset: 1000)
/// - Continue until returned Vec is empty or smaller than limit
///
/// Data flow:
/// - Swift calls this function to populate the browse view, loading pages on demand
/// - Rust queries a window of records using LIMIT and OFFSET
/// - Returns full ImageRecord structs including database-generated fields
///
/// Parameters:
/// - limit: Maximum number of records to return (page size)
/// - offset: Number of records to skip (page_number * page_size)
///
/// Returns:
/// - Vec of ImageRecord structs, empty vec if catalogue is empty or not initialized
pub async fn get_all_images(limit: u32, offset: u32) -> Vec<ImageRecord> {
    // Acquire lock and validate connection
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    // Query images with pagination, ordered by ID
    // Note: DuckDB TIMESTAMP columns must be cast to epoch seconds in SQL,
    // then formatted to String in Rust, because the UniFFI bridge has no
    // TIMESTAMP type. This is the standard pattern for all TIMESTAMP columns.
    // LIMIT ?1 OFFSET ?2 enables paginated loading for large catalogues
    let query_sql = r#"
        SELECT
            id, epoch(indexed_timestamp) as indexed_ts_epoch,
            file_path, file_size, file_name, file_extension,
            created_timestamp, modified_timestamp,
            camera_make, camera_model, lens_model,
            focal_length, aperture, shutter_speed, iso,
            capture_datetime,
            pixel_width, pixel_height, color_space, bit_depth,
            gps_latitude, gps_longitude, gps_altitude,
            copyright, creator, description,
            rating, flag, color_label, rotation,
            CASE
                WHEN capture_datetime IS NULL THEN NULL
                WHEN COUNT(*) OVER (
                    PARTITION BY
                        capture_datetime,
                        camera_model,
                        pixel_width,
                        pixel_height,
                        image_kind,
                        LOWER(file_stem)
                ) = 1 THEN NULL
                ELSE FIRST_VALUE(id) OVER (
                    PARTITION BY
                        capture_datetime,
                        camera_model,
                        pixel_width,
                        pixel_height,
                        image_kind,
                        LOWER(file_stem)
                    ORDER BY file_path ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                )
            END AS duplicate_group_id
        FROM images
        ORDER BY id
        LIMIT ?1 OFFSET ?2
    "#;

    // Execute query with limit and offset parameters
    let mut stmt = match conn.prepare(query_sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to prepare query: {}", e);
            return Vec::new();
        }
    };

    let rows = match stmt.query_map(params![limit as i64, offset as i64], |row| {
        // Extract all columns from the row
        // Type conversions: i64 from DuckDB → u64/u32/u8 for Rust

        // Format indexed_timestamp from Unix epoch seconds to ISO 8601 string
        let epoch_secs: i64 = row.get(1)?;
        let indexed_ts = format_epoch_to_iso8601(epoch_secs);

        Ok(ImageRecord {
            id: row.get(0)?,
            indexed_timestamp: indexed_ts,
            file_path: row.get(2)?,
            file_size: row.get::<_, i64>(3)? as u64,
            file_name: row.get(4)?,
            file_extension: row.get(5)?,
            created_timestamp: row.get(6)?,
            modified_timestamp: row.get(7)?,
            camera_make: row.get(8)?,
            camera_model: row.get(9)?,
            lens_model: row.get(10)?,
            focal_length: row.get(11)?,
            aperture: row.get(12)?,
            shutter_speed: row.get(13)?,
            iso: row.get::<_, Option<i64>>(14)?.map(|v| v as u32),
            capture_datetime: row.get(15)?,
            pixel_width: row.get::<_, Option<i64>>(16)?.map(|v| v as u32),
            pixel_height: row.get::<_, Option<i64>>(17)?.map(|v| v as u32),
            color_space: row.get(18)?,
            bit_depth: row.get::<_, Option<i64>>(19)?.map(|v| v as u32),
            gps_latitude: row.get(20)?,
            gps_longitude: row.get(21)?,
            gps_altitude: row.get(22)?,
            copyright: row.get(23)?,
            creator: row.get(24)?,
            description: row.get(25)?,
            rating: row.get::<_, Option<i64>>(26)?.map(|v| v as u8),
            flag: row.get(27)?,
            color_label: row.get(28)?,
            rotation: row.get::<_, i64>(29)? as i32,
            duplicate_group_id: row.get::<_, Option<i64>>(30)?,
        })
    }) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to execute query: {}", e);
            return Vec::new();
        }
    };

    // Collect results, logging errors but continuing for other rows
    let mut records = Vec::new();
    for row_result in rows {
        match row_result {
            Ok(record) => records.push(record),
            Err(e) => eprintln!("Failed to parse row: {}", e),
        }
    }

    records
}

/// Get images from the catalogue with pagination and proper global sort order
///
/// Returns a paginated list of image records in the catalogue, ordered by capture_datetime
/// descending (newest first) with NULL dates at the end, then by created_timestamp descending.
/// This provides a consistent global sort order that matches user expectations for browsing
/// their photo library chronologically.
///
/// Sort order:
/// - Primary: capture_datetime DESC NULLS LAST (images with dates first, newest to oldest)
/// - Secondary: created_timestamp DESC (for images without capture dates)
///
/// Pagination pattern:
/// - First page: get_images_sorted(limit: 50, offset: 0)
/// - Second page: get_images_sorted(limit: 50, offset: 50)
/// - Third page: get_images_sorted(limit: 50, offset: 100)
/// - Continue until returned Vec is empty or smaller than limit
///
/// Data flow:
/// - Swift calls this function to populate the Photos view with thumbnails
/// - Rust queries a window of records using LIMIT and OFFSET with global sort
/// - Returns full ImageRecord structs including database-generated fields
///
/// Parameters:
/// - limit: Maximum number of records to return (page size)
/// - offset: Number of records to skip (page_number * page_size)
///
/// Returns:
/// - Vec of ImageRecord structs sorted by date, empty vec if catalogue is empty or not initialized
pub async fn get_images_sorted(limit: u32, offset: u32) -> Vec<ImageRecord> {
    // Acquire lock and validate connection
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    // Query images with pagination and global sort order
    // ORDER BY capture_datetime DESC NULLS LAST ensures images with dates appear first,
    // sorted newest to oldest, with undated images at the end.
    // Secondary sort by created_timestamp DESC for images without capture dates.
    let query_sql = r#"
        SELECT
            id, epoch(indexed_timestamp) as indexed_ts_epoch,
            file_path, file_size, file_name, file_extension,
            created_timestamp, modified_timestamp,
            camera_make, camera_model, lens_model,
            focal_length, aperture, shutter_speed, iso,
            capture_datetime,
            pixel_width, pixel_height, color_space, bit_depth,
            gps_latitude, gps_longitude, gps_altitude,
            copyright, creator, description,
            rating, flag, color_label, rotation,
            CASE
                WHEN capture_datetime IS NULL THEN NULL
                WHEN COUNT(*) OVER (
                    PARTITION BY
                        capture_datetime,
                        camera_model,
                        pixel_width,
                        pixel_height,
                        image_kind,
                        LOWER(file_stem)
                ) = 1 THEN NULL
                ELSE FIRST_VALUE(id) OVER (
                    PARTITION BY
                        capture_datetime,
                        camera_model,
                        pixel_width,
                        pixel_height,
                        image_kind,
                        LOWER(file_stem)
                    ORDER BY file_path ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                )
            END AS duplicate_group_id
        FROM images
        ORDER BY capture_datetime DESC NULLS LAST, created_timestamp DESC
        LIMIT ?1 OFFSET ?2
    "#;

    // Execute query with limit and offset parameters
    let mut stmt = match conn.prepare(query_sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to prepare query: {}", e);
            return Vec::new();
        }
    };

    let rows = match stmt.query_map(params![limit as i64, offset as i64], |row| {
        // Extract all columns from the row
        // Type conversions: i64 from DuckDB → u64/u32/u8 for Rust

        // Format indexed_timestamp from Unix epoch seconds to ISO 8601 string
        let epoch_secs: i64 = row.get(1)?;
        let indexed_ts = format_epoch_to_iso8601(epoch_secs);

        Ok(ImageRecord {
            id: row.get(0)?,
            indexed_timestamp: indexed_ts,
            file_path: row.get(2)?,
            file_size: row.get::<_, i64>(3)? as u64,
            file_name: row.get(4)?,
            file_extension: row.get(5)?,
            created_timestamp: row.get(6)?,
            modified_timestamp: row.get(7)?,
            camera_make: row.get(8)?,
            camera_model: row.get(9)?,
            lens_model: row.get(10)?,
            focal_length: row.get(11)?,
            aperture: row.get(12)?,
            shutter_speed: row.get(13)?,
            iso: row.get::<_, Option<i64>>(14)?.map(|v| v as u32),
            capture_datetime: row.get(15)?,
            pixel_width: row.get::<_, Option<i64>>(16)?.map(|v| v as u32),
            pixel_height: row.get::<_, Option<i64>>(17)?.map(|v| v as u32),
            color_space: row.get(18)?,
            bit_depth: row.get::<_, Option<i64>>(19)?.map(|v| v as u32),
            gps_latitude: row.get(20)?,
            gps_longitude: row.get(21)?,
            gps_altitude: row.get(22)?,
            copyright: row.get(23)?,
            creator: row.get(24)?,
            description: row.get(25)?,
            rating: row.get::<_, Option<i64>>(26)?.map(|v| v as u8),
            flag: row.get(27)?,
            color_label: row.get(28)?,
            rotation: row.get::<_, i64>(29)? as i32,
            duplicate_group_id: row.get::<_, Option<i64>>(30)?,
        })
    }) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to execute query: {}", e);
            return Vec::new();
        }
    };

    // Collect results, logging errors but continuing for other rows
    let mut records = Vec::new();
    for row_result in rows {
        match row_result {
            Ok(record) => records.push(record),
            Err(e) => eprintln!("Failed to parse row: {}", e),
        }
    }

    records
}

/// Update the rating for an image
///
/// Sets the star rating (0-5) for an image identified by its file path.
/// A rating of 0 clears the rating (sets it to NULL in the database).
///
/// Design decision: Uses file_path as the key because it's the unique identifier
/// available to the Swift layer when a user interacts with a thumbnail. The id field
/// is database-internal and not surfaced in the UI context.
///
/// Data flow:
/// - User taps a star in the Photos view thumbnail
/// - Swift calls this function with the file path and new rating
/// - Rust updates the database
/// - Swift updates its local ImageRecord to reflect the change
///
/// Parameters:
/// - file_path: Absolute path to the image file (must match a record in catalogue)
/// - rating: Star rating from 0-5 (0 clears the rating)
///
/// Returns:
/// - true if update succeeded
/// - false if catalogue not initialized, file not found, or query failed
pub async fn update_image_rating(file_path: String, rating: u32) -> bool {
    // Acquire lock and validate connection
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return false;
        }
    };

    // Convert rating to Option<i64> for database
    // Rating of 0 means clear the rating (set to NULL)
    let rating_value: Option<i64> = if rating == 0 {
        None
    } else {
        Some(rating as i64)
    };

    // Update the rating for the specified file path
    let update_sql = "UPDATE images SET rating = ? WHERE file_path = ?";

    match conn.execute(update_sql, params![rating_value, file_path]) {
        Ok(changed) => {
            if changed == 0 {
                // No rows updated - file path not found
                eprintln!("No image found with file_path: {}", file_path);
                false
            } else {
                // Successfully updated
                true
            }
        }
        Err(e) => {
            // Log error but don't crash
            eprintln!("Failed to update rating for {}: {}", file_path, e);
            false
        }
    }
}

/// Update the rotation angle for an image
///
/// Sets the rotation angle (0, 90, 180, or 270 degrees) for an image identified by its file path.
/// The rotation angle represents how many degrees clockwise the image should be rotated for display.
///
/// Design decision: Uses file_path as the key for the same reasons as update_image_rating -
/// it's the unique identifier available to the Swift layer when a user interacts with a thumbnail.
///
/// Data flow:
/// - User taps a rotation button in the Photos view thumbnail
/// - Swift calculates new rotation value (current + 90 or current - 90, wrapping at 360)
/// - Swift calls this function with the file path and new rotation angle
/// - Rust updates the database
/// - Swift regenerates and caches the rotated thumbnail
///
/// Parameters:
/// - file_path: Absolute path to the image file (must match a record in catalogue)
/// - degrees: Rotation angle in degrees (0, 90, 180, or 270)
///
/// Returns:
/// - true if update succeeded
/// - false if catalogue not initialized, file not found, or query failed
pub async fn update_image_rotation(file_path: String, degrees: i32) -> bool
{
    // Acquire lock and validate connection
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref()
    {
        Some(c) => c,
        None =>
        {
            eprintln!("Catalogue not initialized");
            return false;
        }
    };

    // Update the rotation for the specified file path
    let update_sql = "UPDATE images SET rotation = ? WHERE file_path = ?";

    match conn.execute(update_sql, params![degrees, file_path])
    {
        Ok(changed) =>
        {
            if changed == 0
            {
                // No rows updated - file path not found
                eprintln!("No image found with file_path: {}", file_path);
                false
            }
            else
            {
                // Successfully updated
                true
            }
        }
        Err(e) =>
        {
            // Log error but don't crash
            eprintln!("Failed to update rotation for {}: {}", file_path, e);
            false
        }
    }
}

/// Get all distinct date strings from the catalogue
///
/// Returns the date-only prefixes (first 10 characters: "YYYY:MM:DD") from
/// capture_datetime for all images in the catalogue. Used to populate the date
/// navigation tree in the Photos view.
///
/// Data flow:
/// - Swift calls this function to populate the date navigation sidebar
/// - Rust queries distinct date prefixes from capture_datetime
/// - Returns sorted list (newest first) for Swift to build the tree hierarchy
///
/// Returns:
/// - Vec of date strings in "YYYY:MM:DD" format, sorted descending
/// - Empty vec if catalogue is empty or not initialized
pub async fn get_distinct_date_strings() -> Vec<String> {
    // Acquire lock and validate connection
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    // Query distinct date prefixes from capture_datetime
    // SUBSTRING extracts first 10 characters (YYYY:MM:DD format)
    // Filter out NULL and empty strings
    let query_sql = r#"
        SELECT DISTINCT SUBSTRING(capture_datetime, 1, 10) as date_str
        FROM images
        WHERE capture_datetime IS NOT NULL AND capture_datetime != ''
        ORDER BY date_str ASC
    "#;

    let mut stmt = match conn.prepare(query_sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to prepare query: {}", e);
            return Vec::new();
        }
    };

    let rows = match stmt.query_map([], |row| {
        row.get::<_, String>(0)
    }) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to execute query: {}", e);
            return Vec::new();
        }
    };

    // Collect results, logging errors but continuing for other rows
    let mut dates = Vec::new();
    for row_result in rows {
        match row_result {
            Ok(date_str) => dates.push(date_str),
            Err(e) => eprintln!("Failed to parse row: {}", e),
        }
    }

    dates
}

/// Get all distinct parent directory paths from catalogued images
///
/// Returns a sorted list of all unique parent directory paths that contain catalogued images.
/// This is used to build the source locations tree in the UI, showing the directory structure
/// of scanned locations.
///
/// Implementation: Uses DISTINCT on the directory portion of file_path. DuckDB doesn't have
/// a built-in dirname() function, so we extract the directory path by finding the last '/'
/// and taking everything before it.
///
/// Returns:
/// - Vec of directory path strings, sorted alphabetically
/// - Empty vec if catalogue is empty or not initialized
pub async fn get_distinct_directory_paths() -> Vec<String>
{
    // Acquire lock and validate connection
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref()
    {
        Some(c) => c,
        None =>
        {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    // Query distinct directory paths
    // Extract directory by finding the last '/' and taking everything before it.
    //
    // INSTR(REVERSE(file_path), '/') returns the 1-indexed position of the last '/'
    // counted from the END of the original string. The directory portion is therefore
    // the first (LENGTH(file_path) - INSTR(REVERSE(file_path), '/')) characters, which
    // is everything up to but not including that last '/'.
    let query_sql = r#"
        SELECT DISTINCT
            SUBSTRING(file_path, 1, LENGTH(file_path) - INSTR(REVERSE(file_path), '/')) as dir_path
        FROM images
        WHERE file_path IS NOT NULL AND file_path LIKE '%/%'
        ORDER BY dir_path ASC
    "#;

    let mut stmt = match conn.prepare(query_sql)
    {
        Ok(s) => s,
        Err(e) =>
        {
            eprintln!("Failed to prepare query: {}", e);
            return Vec::new();
        }
    };

    let rows = match stmt.query_map([], |row|
    {
        row.get::<_, String>(0)
    })
    {
        Ok(r) => r,
        Err(e) =>
        {
            eprintln!("Failed to execute query: {}", e);
            return Vec::new();
        }
    };

    // Collect results, logging errors but continuing for other rows
    let mut paths = Vec::new();
    for row_result in rows
    {
        match row_result
        {
            Ok(path) => paths.push(path),
            Err(e) => eprintln!("Failed to parse row: {}", e),
        }
    }

    paths
}

/// Get images from the catalogue with pagination and optional date filtering
///
/// Returns a paginated list of image records, optionally filtered by date prefix.
/// When date_prefix is empty, returns all images. When non-empty, filters to images
/// whose capture_datetime starts with the given prefix.
///
/// Date prefix format:
/// - "" (empty): All images
/// - "2026:": All images from year 2026
/// - "2026:01:": All images from January 2026
/// - "2026:01:09": All images from January 9, 2026
///
/// Sort order: capture_datetime DESC NULLS LAST, created_timestamp DESC
///
/// Parameters:
/// - limit: Maximum number of records to return (page size)
/// - offset: Number of records to skip (page_number * page_size)
/// - date_prefix: Date filter prefix (empty string for no filter)
///
/// Returns:
/// - Vec of ImageRecord structs sorted by date, empty vec if catalogue is empty or not initialized
pub async fn get_images_filtered(limit: i64, offset: i64, date_prefix: String) -> Vec<ImageRecord> {
    // Acquire lock and validate connection
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    // Build query with optional WHERE clause
    let query_sql = if date_prefix.is_empty() {
        // No filter - return all images
        r#"
            SELECT
                id, epoch(indexed_timestamp) as indexed_ts_epoch,
                file_path, file_size, file_name, file_extension,
                created_timestamp, modified_timestamp,
                camera_make, camera_model, lens_model,
                focal_length, aperture, shutter_speed, iso,
                capture_datetime,
                pixel_width, pixel_height, color_space, bit_depth,
                gps_latitude, gps_longitude, gps_altitude,
                copyright, creator, description,
                rating, flag, color_label, rotation,
                CASE
                    WHEN capture_datetime IS NULL THEN NULL
                    WHEN COUNT(*) OVER (
                        PARTITION BY
                            capture_datetime,
                            camera_model,
                            pixel_width,
                            pixel_height,
                            image_kind,
                            LOWER(file_stem)
                    ) = 1 THEN NULL
                    ELSE FIRST_VALUE(id) OVER (
                        PARTITION BY
                            capture_datetime,
                            camera_model,
                            pixel_width,
                            pixel_height,
                            image_kind,
                            LOWER(file_stem)
                        ORDER BY file_path ASC
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                    )
                END AS duplicate_group_id
            FROM images
            ORDER BY capture_datetime DESC NULLS LAST, created_timestamp DESC
            LIMIT ?1 OFFSET ?2
        "#.to_string()
    } else {
        // Filter by date prefix using LIKE
        format!(r#"
            SELECT
                id, epoch(indexed_timestamp) as indexed_ts_epoch,
                file_path, file_size, file_name, file_extension,
                created_timestamp, modified_timestamp,
                camera_make, camera_model, lens_model,
                focal_length, aperture, shutter_speed, iso,
                capture_datetime,
                pixel_width, pixel_height, color_space, bit_depth,
                gps_latitude, gps_longitude, gps_altitude,
                copyright, creator, description,
                rating, flag, color_label, rotation,
                CASE
                    WHEN capture_datetime IS NULL THEN NULL
                    WHEN COUNT(*) OVER (
                        PARTITION BY
                            capture_datetime,
                            camera_model,
                            pixel_width,
                            pixel_height,
                            image_kind,
                            LOWER(file_stem)
                    ) = 1 THEN NULL
                    ELSE FIRST_VALUE(id) OVER (
                        PARTITION BY
                            capture_datetime,
                            camera_model,
                            pixel_width,
                            pixel_height,
                            image_kind,
                            LOWER(file_stem)
                        ORDER BY file_path ASC
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                    )
                END AS duplicate_group_id
            FROM images
            WHERE capture_datetime LIKE '{}%'
            ORDER BY capture_datetime DESC NULLS LAST, created_timestamp DESC
            LIMIT ?1 OFFSET ?2
        "#, date_prefix)
    };

    // Execute query with limit and offset parameters
    let mut stmt = match conn.prepare(&query_sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to prepare query: {}", e);
            return Vec::new();
        }
    };

    let rows = match stmt.query_map(params![limit, offset], |row| {
        // Extract all columns from the row
        // Type conversions: i64 from DuckDB → u64/u32/u8 for Rust

        // Format indexed_timestamp from Unix epoch seconds to ISO 8601 string
        let epoch_secs: i64 = row.get(1)?;
        let indexed_ts = format_epoch_to_iso8601(epoch_secs);

        Ok(ImageRecord {
            id: row.get(0)?,
            indexed_timestamp: indexed_ts,
            file_path: row.get(2)?,
            file_size: row.get::<_, i64>(3)? as u64,
            file_name: row.get(4)?,
            file_extension: row.get(5)?,
            created_timestamp: row.get(6)?,
            modified_timestamp: row.get(7)?,
            camera_make: row.get(8)?,
            camera_model: row.get(9)?,
            lens_model: row.get(10)?,
            focal_length: row.get(11)?,
            aperture: row.get(12)?,
            shutter_speed: row.get(13)?,
            iso: row.get::<_, Option<i64>>(14)?.map(|v| v as u32),
            capture_datetime: row.get(15)?,
            pixel_width: row.get::<_, Option<i64>>(16)?.map(|v| v as u32),
            pixel_height: row.get::<_, Option<i64>>(17)?.map(|v| v as u32),
            color_space: row.get(18)?,
            bit_depth: row.get::<_, Option<i64>>(19)?.map(|v| v as u32),
            gps_latitude: row.get(20)?,
            gps_longitude: row.get(21)?,
            gps_altitude: row.get(22)?,
            copyright: row.get(23)?,
            creator: row.get(24)?,
            description: row.get(25)?,
            rating: row.get::<_, Option<i64>>(26)?.map(|v| v as u8),
            flag: row.get(27)?,
            color_label: row.get(28)?,
            rotation: row.get::<_, i64>(29)? as i32,
            duplicate_group_id: row.get::<_, Option<i64>>(30)?,
        })
    }) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to execute query: {}", e);
            return Vec::new();
        }
    };

    // Collect results, logging errors but continuing for other rows
    let mut records = Vec::new();
    for row_result in rows {
        match row_result {
            Ok(record) => records.push(record),
            Err(e) => eprintln!("Failed to parse row: {}", e),
        }
    }

    records
}

/// Get the count of images matching the date filter
///
/// Returns the total number of images in the catalogue, optionally filtered by
/// date prefix. Used for pagination calculations in the Photos view.
///
/// Date prefix format:
/// - "" (empty): Count all images
/// - "2026:": Count images from year 2026
/// - "2026:01:": Count images from January 2026
/// - "2026:01:09": Count images from January 9, 2026
///
/// Parameters:
/// - date_prefix: Date filter prefix (empty string for no filter)
///
/// Returns:
/// - Total number of image records matching the filter
/// - 0 if catalogue not initialized or query fails
pub async fn get_filtered_image_count(date_prefix: String) -> i64 {
    // Acquire lock and validate connection
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    // Build query with optional WHERE clause
    let query_sql = if date_prefix.is_empty() {
        "SELECT COUNT(*) FROM images".to_string()
    } else {
        format!("SELECT COUNT(*) FROM images WHERE capture_datetime LIKE '{}%'", date_prefix)
    };

    // Execute COUNT query
    let count_result: Result<i64, _> = conn.query_row(
        &query_sql,
        [],
        |row| row.get(0),
    );

    match count_result {
        Ok(count) => count,
        Err(e) => {
            eprintln!("Failed to query filtered image count: {}", e);
            0
        }
    }
}

/// Get the count of images in a directory (matching path prefix)
///
/// Returns the total number of images whose file_path starts with the given
/// path prefix. Used for displaying image counts in the source locations tree.
///
/// Parameters:
/// - path_prefix: Directory path prefix (e.g., "/Users/richard/Photos/")
///
/// Returns:
/// - Total number of image records in the directory tree
/// - 0 if catalogue not initialized or query fails
pub async fn get_image_count_for_path_prefix(path_prefix: String) -> i64
{
    // Acquire lock and validate connection
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref()
    {
        Some(c) => c,
        None =>
        {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    // Build query with path prefix filter
    // Escape single quotes in path to prevent SQL injection
    let escaped_prefix = path_prefix.replace("'", "''");
    let query_sql = format!("SELECT COUNT(*) FROM images WHERE file_path LIKE '{}%'", escaped_prefix);

    // Execute COUNT query
    let count_result: Result<i64, _> = conn.query_row(
        &query_sql,
        [],
        |row| row.get(0),
    );

    match count_result
    {
        Ok(count) => count,
        Err(e) =>
        {
            eprintln!("Failed to query image count for path prefix: {}", e);
            0
        }
    }
}

/// Get images from a directory with pagination (matching path prefix)
///
/// Returns a paginated list of image records whose file_path starts with the
/// given path prefix. Used for filtering the photo grid by selected directory.
///
/// Sort order: capture_datetime DESC NULLS LAST, created_timestamp DESC
///
/// Parameters:
/// - limit: Maximum number of records to return (page size)
/// - offset: Number of records to skip (page_number * page_size)
/// - path_prefix: Directory path prefix (empty string for no filter)
/// - date_prefix: Date filter prefix (empty string for no filter)
///
/// Returns:
/// - Vec of ImageRecord structs sorted by date, empty vec if catalogue is empty or not initialized
pub async fn get_images_for_path_prefix(limit: i64, offset: i64, path_prefix: String, date_prefix: String) -> Vec<ImageRecord>
{
    // Acquire lock and validate connection
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref()
    {
        Some(c) => c,
        None =>
        {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    // Build WHERE clause based on filters
    // Escape single quotes to prevent SQL injection
    let escaped_path = path_prefix.replace("'", "''");
    let escaped_date = date_prefix.replace("'", "''");

    let where_clause = match (path_prefix.is_empty(), date_prefix.is_empty())
    {
        (true, true) => String::new(),
        (false, true) => format!("WHERE file_path LIKE '{}%'", escaped_path),
        (true, false) => format!("WHERE capture_datetime LIKE '{}%'", escaped_date),
        (false, false) => format!("WHERE file_path LIKE '{}%' AND capture_datetime LIKE '{}%'", escaped_path, escaped_date),
    };

    let query_sql = format!(r#"
        SELECT
            id, epoch(indexed_timestamp) as indexed_ts_epoch,
            file_path, file_size, file_name, file_extension,
            created_timestamp, modified_timestamp,
            camera_make, camera_model, lens_model,
            focal_length, aperture, shutter_speed, iso,
            capture_datetime,
            pixel_width, pixel_height, color_space, bit_depth,
            gps_latitude, gps_longitude, gps_altitude,
            copyright, creator, description,
            rating, flag, color_label, rotation,
            CASE
                WHEN capture_datetime IS NULL THEN NULL
                WHEN COUNT(*) OVER (
                    PARTITION BY
                        capture_datetime,
                        camera_model,
                        pixel_width,
                        pixel_height,
                        image_kind,
                        LOWER(file_stem)
                ) = 1 THEN NULL
                ELSE FIRST_VALUE(id) OVER (
                    PARTITION BY
                        capture_datetime,
                        camera_model,
                        pixel_width,
                        pixel_height,
                        image_kind,
                        LOWER(file_stem)
                    ORDER BY file_path ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                )
            END AS duplicate_group_id
        FROM images
        {}
        ORDER BY capture_datetime DESC NULLS LAST, created_timestamp DESC
        LIMIT ?1 OFFSET ?2
    "#, where_clause);

    // Execute query with limit and offset parameters
    let mut stmt = match conn.prepare(&query_sql)
    {
        Ok(s) => s,
        Err(e) =>
        {
            eprintln!("Failed to prepare query: {}", e);
            return Vec::new();
        }
    };

    let rows = match stmt.query_map(params![limit, offset], |row|
    {
        // Extract all columns from the row
        // Type conversions: i64 from DuckDB → u64/u32/u8 for Rust

        // Format indexed_timestamp from Unix epoch seconds to ISO 8601 string
        let epoch_secs: i64 = row.get(1)?;
        let indexed_ts = format_epoch_to_iso8601(epoch_secs);

        Ok(ImageRecord {
            id: row.get(0)?,
            indexed_timestamp: indexed_ts,
            file_path: row.get(2)?,
            file_size: row.get::<_, i64>(3)? as u64,
            file_name: row.get(4)?,
            file_extension: row.get(5)?,
            created_timestamp: row.get(6)?,
            modified_timestamp: row.get(7)?,
            camera_make: row.get(8)?,
            camera_model: row.get(9)?,
            lens_model: row.get(10)?,
            focal_length: row.get(11)?,
            aperture: row.get(12)?,
            shutter_speed: row.get(13)?,
            iso: row.get::<_, Option<i64>>(14)?.map(|v| v as u32),
            capture_datetime: row.get(15)?,
            pixel_width: row.get::<_, Option<i64>>(16)?.map(|v| v as u32),
            pixel_height: row.get::<_, Option<i64>>(17)?.map(|v| v as u32),
            color_space: row.get(18)?,
            bit_depth: row.get::<_, Option<i64>>(19)?.map(|v| v as u32),
            gps_latitude: row.get(20)?,
            gps_longitude: row.get(21)?,
            gps_altitude: row.get(22)?,
            copyright: row.get(23)?,
            creator: row.get(24)?,
            description: row.get(25)?,
            rating: row.get::<_, Option<i64>>(26)?.map(|v| v as u8),
            flag: row.get(27)?,
            color_label: row.get(28)?,
            rotation: row.get::<_, i64>(29)? as i32,
            duplicate_group_id: row.get::<_, Option<i64>>(30)?,
        })
    })
    {
        Ok(r) => r,
        Err(e) =>
        {
            eprintln!("Failed to execute query: {}", e);
            return Vec::new();
        }
    };

    // Collect results, logging errors but continuing for other rows
    let mut records = Vec::new();
    for row_result in rows
    {
        match row_result
        {
            Ok(record) => records.push(record),
            Err(e) => eprintln!("Failed to parse row: {}", e),
        }
    }

    records
}

/// Get count of images matching both path prefix and date filter
///
/// Returns the total number of images whose file_path starts with the given
/// path prefix AND whose capture_datetime starts with the date prefix.
/// Used for pagination calculations when both filters are active.
///
/// Parameters:
/// - path_prefix: Directory path prefix (empty string for no filter)
/// - date_prefix: Date filter prefix (empty string for no filter)
///
/// Returns:
/// - Total number of image records matching both filters
/// - 0 if catalogue not initialized or query fails
pub async fn get_image_count_for_filters(path_prefix: String, date_prefix: String) -> i64
{
    // Acquire lock and validate connection
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref()
    {
        Some(c) => c,
        None =>
        {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    // Build WHERE clause based on filters
    // Escape single quotes to prevent SQL injection
    let escaped_path = path_prefix.replace("'", "''");
    let escaped_date = date_prefix.replace("'", "''");

    let query_sql = match (path_prefix.is_empty(), date_prefix.is_empty())
    {
        (true, true) => "SELECT COUNT(*) FROM images".to_string(),
        (false, true) => format!("SELECT COUNT(*) FROM images WHERE file_path LIKE '{}%'", escaped_path),
        (true, false) => format!("SELECT COUNT(*) FROM images WHERE capture_datetime LIKE '{}%'", escaped_date),
        (false, false) => format!("SELECT COUNT(*) FROM images WHERE file_path LIKE '{}%' AND capture_datetime LIKE '{}%'", escaped_path, escaped_date),
    };

    // Execute COUNT query
    let count_result: Result<i64, _> = conn.query_row(
        &query_sql,
        [],
        |row| row.get(0),
    );

    match count_result
    {
        Ok(count) => count,
        Err(e) =>
        {
            eprintln!("Failed to query image count for filters: {}", e);
            0
        }
    }
}

/// Find the JPEG+RAW counterpart of an image in the catalogue
///
/// Given a JPEG file path, returns the matching RAW image record (same parent
/// directory + same filename stem). Given a RAW, returns the matching JPEG.
/// Returns None if the input is Other-kind (not JPEG, not RAW), if the input
/// is malformed (no parent directory or no extension), or if no counterpart
/// record exists in the catalogue.
///
/// Approach:
/// - Parse parent directory and stem from the input string using pure string
///   operations. No filesystem I/O, no canonicalization. The function trusts
///   the input string and does not look up the input file's own catalogue
///   record. The catalogue stores file_path verbatim from Swift, so byte-
///   exact equality is the matching contract.
/// - Classify the input extension via classify_extension. Other-kind inputs
///   return None immediately — they have no counterpart concept.
/// - SQL fetches all parent+stem candidates EXCLUDING the input's own path,
///   ordered ASC by file_extension. Parent-directory derivation in SQL uses
///   the standard SUBSTRING + LENGTH + INSTR + REVERSE idiom matching
///   get_distinct_directory_paths. Stem derivation in SQL strips
///   file_extension and the dot separator from file_name.
/// - Rust iterates the returned rows in the SQL-imposed order, classifies
///   each candidate's extension, and returns the first row whose kind is
///   the opposite of the input's kind. The LIMIT-1 semantic is enforced
///   at the Rust layer, not the SQL layer: applying SQL LIMIT 1 to the
///   unclassified candidate set would return a same-kind sibling in the
///   multi-RAW edge case (e.g., NEF + ARW with no JPEG) rather than the
///   opposite-kind counterpart we want. Determinism comes from the SQL
///   ORDER BY file_extension ASC.
/// - Parameterized SQL bindings throughout. No format! interpolation, no
///   quote escaping.
///
/// Case sensitivity: SQL `=` comparison is byte-exact. If the same-stem
/// counterpart was indexed under a different filename case than the input,
/// the match will not succeed. Matches existing pair-handling behavior;
/// revisit if real users hit case-mismatch issues.
///
/// Stem derivation in SQL assumes file_extension's stored length matches the
/// literal suffix of file_name. This holds because the scanner normalizes
/// file_extension from file_name's last-dot suffix at ingest; if that
/// invariant is ever weakened, stem derivation will silently produce wrong
/// results.
///
/// This function holds CATALOGUE.lock() across the entire query and row
/// iteration; concurrent async callers serialize on the global mutex.
/// Consistent with all other catalogue query functions in this file.
/// Tracked in STATUS.md for a pre-v1 refactor.
///
/// Parameters:
/// - file_path: Absolute path of the image to find a counterpart for
///
/// Returns:
/// - Some(ImageRecord) for the opposite-kind counterpart, deterministically
///   chosen as the alphabetically-first matching file_extension
/// - None if input is Other-kind, malformed, or has no counterpart in the
///   catalogue
pub async fn find_counterpart_image(file_path: String) -> Option<ImageRecord>
{
    // 1. Parse parent directory and basename from the input path.
    //    Pure string operation — no filesystem I/O, no canonicalization.
    //    Trusts the input. Risk 3 in DESIGN-image-classification.md.
    let last_slash = match file_path.rfind('/')
    {
        Some(pos) => pos,
        None =>
        {
            // No slash → no parent directory. Cannot be a real file path.
            return None;
        }
    };

    let parent_dir = &file_path[..last_slash];
    let basename = &file_path[(last_slash + 1)..];

    // 2. Parse stem and extension from the basename.
    //    rfind('.') finds the LAST dot, which handles multi-dot filenames
    //    such as IMG.2024-05-13.NEF correctly (stem = "IMG.2024-05-13").
    let last_dot = match basename.rfind('.')
    {
        Some(pos) => pos,
        None =>
        {
            // No extension → cannot be classified as JPEG or RAW.
            return None;
        }
    };

    let stem = &basename[..last_dot];
    let ext = &basename[(last_dot + 1)..];

    // 3. Classify the input extension; Other-kind inputs have no counterpart.
    let input_kind = classify_extension(ext.to_string());
    let target_kind = match input_kind
    {
        ImageKind::Jpeg => ImageKind::Raw,
        ImageKind::Raw => ImageKind::Jpeg,
        ImageKind::Other => return None,
    };

    // 4. Acquire lock and validate connection.
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref()
    {
        Some(c) => c,
        None =>
        {
            eprintln!("Catalogue not initialized");
            return None;
        }
    };

    // 5. Parameterized query for parent+stem candidates.
    //    Parent-directory derivation in SQL: SUBSTRING + LENGTH + INSTR +
    //    REVERSE — matches the project idiom in get_distinct_directory_paths.
    //    Stem derivation in SQL: strip file_extension + 1 (the dot) from
    //    file_name; guarded by IS NOT NULL and != '' so the arithmetic is
    //    well-defined.
    //    Exclude-self via file_path != ?3 (exact-path equality).
    //    ORDER BY file_extension ASC gives deterministic resolution when
    //    multiple opposite-kind candidates share the stem.
    let query_sql = r#"
        SELECT
            id, epoch(indexed_timestamp) as indexed_ts_epoch,
            file_path, file_size, file_name, file_extension,
            created_timestamp, modified_timestamp,
            camera_make, camera_model, lens_model,
            focal_length, aperture, shutter_speed, iso,
            capture_datetime,
            pixel_width, pixel_height, color_space, bit_depth,
            gps_latitude, gps_longitude, gps_altitude,
            copyright, creator, description,
            rating, flag, color_label, rotation,
            CASE
                WHEN capture_datetime IS NULL THEN NULL
                WHEN COUNT(*) OVER (
                    PARTITION BY
                        capture_datetime,
                        camera_model,
                        pixel_width,
                        pixel_height,
                        image_kind,
                        LOWER(file_stem)
                ) = 1 THEN NULL
                ELSE FIRST_VALUE(id) OVER (
                    PARTITION BY
                        capture_datetime,
                        camera_model,
                        pixel_width,
                        pixel_height,
                        image_kind,
                        LOWER(file_stem)
                    ORDER BY file_path ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                )
            END AS duplicate_group_id
        FROM images
        WHERE SUBSTRING(file_path, 1, LENGTH(file_path) - INSTR(REVERSE(file_path), '/')) = ?1
          AND file_extension IS NOT NULL
          AND file_extension != ''
          AND SUBSTRING(file_name, 1, LENGTH(file_name) - LENGTH(file_extension) - 1) = ?2
          AND file_path != ?3
        ORDER BY file_extension ASC
    "#;

    let mut stmt = match conn.prepare(query_sql)
    {
        Ok(s) => s,
        Err(e) =>
        {
            eprintln!("Failed to prepare counterpart query: {}", e);
            return None;
        }
    };

    let rows = match stmt.query_map(params![parent_dir, stem, file_path], |row|
    {
        // Standard ImageRecord row decode — mirrors get_images_for_path_prefix.
        // Format indexed_timestamp from Unix epoch seconds to ISO 8601 string.
        let epoch_secs: i64 = row.get(1)?;
        let indexed_ts = format_epoch_to_iso8601(epoch_secs);

        Ok(ImageRecord {
            id: row.get(0)?,
            indexed_timestamp: indexed_ts,
            file_path: row.get(2)?,
            file_size: row.get::<_, i64>(3)? as u64,
            file_name: row.get(4)?,
            file_extension: row.get(5)?,
            created_timestamp: row.get(6)?,
            modified_timestamp: row.get(7)?,
            camera_make: row.get(8)?,
            camera_model: row.get(9)?,
            lens_model: row.get(10)?,
            focal_length: row.get(11)?,
            aperture: row.get(12)?,
            shutter_speed: row.get(13)?,
            iso: row.get::<_, Option<i64>>(14)?.map(|v| v as u32),
            capture_datetime: row.get(15)?,
            pixel_width: row.get::<_, Option<i64>>(16)?.map(|v| v as u32),
            pixel_height: row.get::<_, Option<i64>>(17)?.map(|v| v as u32),
            color_space: row.get(18)?,
            bit_depth: row.get::<_, Option<i64>>(19)?.map(|v| v as u32),
            gps_latitude: row.get(20)?,
            gps_longitude: row.get(21)?,
            gps_altitude: row.get(22)?,
            copyright: row.get(23)?,
            creator: row.get(24)?,
            description: row.get(25)?,
            rating: row.get::<_, Option<i64>>(26)?.map(|v| v as u8),
            flag: row.get(27)?,
            color_label: row.get(28)?,
            rotation: row.get::<_, i64>(29)? as i32,
            duplicate_group_id: row.get::<_, Option<i64>>(30)?,
        })
    })
    {
        Ok(r) => r,
        Err(e) =>
        {
            eprintln!("Failed to execute counterpart query: {}", e);
            return None;
        }
    };

    // 6. Iterate candidates in alphabetical-by-extension order; return the
    //    first record whose extension classifies as the opposite kind of
    //    the input. The SQL ORDER BY guarantees determinism across runs;
    //    the Rust-side filter ensures we don't accidentally return a same-
    //    kind sibling (multi-RAW edge case).
    for row_result in rows
    {
        match row_result
        {
            Ok(record) =>
            {
                // file_extension is filtered NOT NULL/non-empty in SQL,
                // but match defensively in case future schema changes
                // weaken that guard.
                let candidate_ext = match &record.file_extension
                {
                    Some(e) => e.clone(),
                    None => continue,
                };

                if classify_extension(candidate_ext) == target_kind
                {
                    return Some(record);
                }
            }
            Err(e) => eprintln!("Failed to parse counterpart row: {}", e),
        }
    }

    // No opposite-kind candidate found.
    None
}
