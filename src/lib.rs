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
    // NOTE: "dng" intentionally NOT here — DNG is its own ImageKind::Dng
    // (Lightroom import, Docs/DESIGN-Lightroom-Catalog-Import.md §7), so it does
    // NOT pair-collapse as a RAW. Re-adding it here would regress that. See
    // DNG_EXTENSIONS below.
    "raf",  // Fujifilm
    "rw2",  // Panasonic
    "orf",  // Olympus / OM System
];

/// Recognized JPEG image file extensions
///
/// Companion to RAW_EXTENSIONS. Two-element list because the JPEG ecosystem
/// has settled on these two spellings in the workflows this app targets.
const JPEG_EXTENSIONS: &[&str] = &["jpg", "jpeg"];

/// Recognized HEIF image file extensions (Session 41)
///
/// "heic"/"heif" are Apple/phone/generic; "hif" is the extension Nikon,
/// Canon, AND Sony all write HEIF as (e.g. the Nikon Z8's RAW+HEIF mode
/// produces NEF + HIF). HEIF participates in RAW pair-collapse exactly the
/// way JPEG does — it is a lightweight, viewable sibling of the RAW — so it
/// is its own `ImageKind` rather than living in the `Other` bucket.
const HEIF_EXTENSIONS: &[&str] = &["heic", "heif", "hif"];

/// Recognized DNG file extension (Lightroom-import work)
///
/// DNG is its OWN `ImageKind::Dng`, deliberately NOT folded into
/// `RAW_EXTENSIONS`: a single shot can carry RAW + JPEG + DNG, and DNGs are
/// typically edit/conversion artifacts rather than the camera original.
/// Consequence: `.dng` no longer pair-collapses behind a JPEG — it shows as its
/// own kind. See Docs/DESIGN-Lightroom-Catalog-Import.md §7. Trade-off: cameras
/// that shoot NATIVE DNG (Leica/Pentax/some drones) lose RAW-style collapse for
/// those files — accepted.
const DNG_EXTENSIONS: &[&str] = &["dng"];

/// Recognized Photoshop, TIFF, and PNG extensions (Lightroom-import work)
///
/// Promoted out of the `Other` bucket to their own `ImageKind`s so they
/// catalogue, thumbnail, and filter as first-class formats. None participate in
/// RAW+JPEG/HEIF pair-collapse. See Docs/DESIGN-Lightroom-Catalog-Import.md §7.
const PSD_EXTENSIONS: &[&str] = &["psd"];
const TIFF_EXTENSIONS: &[&str] = &["tif", "tiff"];
const PNG_EXTENSIONS: &[&str] = &["png"];

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
    // Session 41: HEIF (heic/heif/hif). Appended LAST to keep the existing
    // UniFFI discriminants stable — must stay in the same position as the
    // UDL `enum ImageKind`.
    Heif,
    // Lightroom import (Docs/DESIGN-Lightroom-Catalog-Import.md §7): DNG
    // promoted out of Raw; PSD/TIFF/PNG out of Other. Each is its own kind so a
    // shot's RAW + JPEG + DNG/PSD/etc. stay distinct, and none participate in
    // RAW+JPEG/HEIF pair-collapse. Appended LAST (after Heif) to keep existing
    // UniFFI discriminants stable — order MUST match the UDL `enum ImageKind`.
    Dng,
    Psd,
    Tiff,
    Png,
}

/// Classify a file extension into its ImageKind (JPEG/RAW/HEIF/DNG/PSD/TIFF/PNG/Other)
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
    else if HEIF_EXTENSIONS.contains(&lower.as_str())
    {
        ImageKind::Heif
    }
    else if DNG_EXTENSIONS.contains(&lower.as_str())
    {
        // Checked before RAW: DNG is its own kind, not a RAW (see DNG_EXTENSIONS).
        ImageKind::Dng
    }
    else if RAW_EXTENSIONS.contains(&lower.as_str())
    {
        ImageKind::Raw
    }
    else if PSD_EXTENSIONS.contains(&lower.as_str())
    {
        ImageKind::Psd
    }
    else if TIFF_EXTENSIONS.contains(&lower.as_str())
    {
        ImageKind::Tiff
    }
    else if PNG_EXTENSIONS.contains(&lower.as_str())
    {
        ImageKind::Png
    }
    else
    {
        ImageKind::Other
    }
}

/// Project the RAW_EXTENSIONS constant table out to the Swift side
///
/// Added for A11 (editor default preferences). The PreferencesView
/// "Default RAW editor" picker needs to enumerate every RAW extension
/// the catalogue recognizes so it can compute the intersection of
/// LaunchServices-capable apps across all of them — an editor only
/// appears as a candidate default if it can open EVERY format in
/// this set. That intersection logic lives Swift-side, but the
/// canonical extension list lives here (this file is the
/// single source of truth — see RAW_EXTENSIONS doc comment, "Single
/// source of truth at the Rust layer").
///
/// **Synchronous** — pure projection of a compile-time constant; no
/// I/O, no DuckDB, no shared state. Matches the synchronous
/// precedent set by `classify_extension`.
///
/// Returns a `Vec<String>` (UniFFI sequence<string>) rather than a
/// borrowed slice so the bridge has a value type to hand over. Each
/// extension is lowercase, no leading dot, as stored in the
/// catalogue's `file_extension` column and as expected by
/// `classify_extension`.
///
/// Order matches the declaration order of RAW_EXTENSIONS. Callers
/// that need a different order should sort Swift-side; this function
/// preserves the canonical order for predictability.
pub fn get_raw_extensions() -> Vec<String>
{
    RAW_EXTENSIONS.iter().map(|s| s.to_string()).collect()
}

/// Project the JPEG_EXTENSIONS constant table out to the Swift side
///
/// Companion to `get_raw_extensions` (A11). The PreferencesView
/// "Default JPEG editor" picker queries by `UTType.jpeg` and does
/// not strictly need this list today, but exposing it here keeps
/// symmetry with the RAW projection and forecloses any future
/// Swift-side need to know which extensions classify as JPEG without
/// asking `classify_extension` one extension at a time.
///
/// **Synchronous** — same rationale as `get_raw_extensions`.
pub fn get_jpeg_extensions() -> Vec<String>
{
    JPEG_EXTENSIONS.iter().map(|s| s.to_string()).collect()
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
            -- Stored parent-directory derivation for filter-aware pagination
            -- (DESIGN-Filter-Aware-Pagination.md §4). Populated at ingest from
            -- file_path via the canonical SUBSTRING/INSTR/REVERSE expression
            -- (see get_distinct_directory_paths for the canonical form, and the
            -- "Directory-Path Extraction SQL — Gotcha" note in CLAUDE.md for the
            -- alternative left-anchored form that must NOT be used). Nullable in
            -- DDL — the non-null invariant is enforced at the populate-time
            -- contract (new-record ingest + one-time backfill), same pattern as
            -- file_stem / image_kind above. Consumed by the RAW+JPEG pair-collapse
            -- subquery introduced in Session 19 Step 4.
            directory_path VARCHAR,
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
        -- read-and-confirm pass; DESIGN-Filter-Aware-Pagination.md §4 added
        -- directory_path in Session 19 Step 1). CREATE TABLE IF NOT EXISTS is a
        -- no-op when the images table already exists, so it cannot add columns
        -- to pre-existing catalogues. The ALTER TABLE ADD COLUMN IF NOT EXISTS
        -- trio below is the migration path: on a fresh database these are no-ops
        -- (the columns were just defined above); on an existing catalogue these
        -- add the columns the backfill migration then populates.
        ALTER TABLE images ADD COLUMN IF NOT EXISTS file_stem VARCHAR;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS image_kind VARCHAR;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS directory_path VARCHAR;

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

        -- === Keyword system (Session 45; Docs/DESIGN-Keyword-System.md) ===
        -- Hierarchical keywords in ONE table. Each applied keyword PATH is
        -- materialized as one row per ancestor level; each row carries `label`
        -- (that node, = last path segment) and `path` (root->that node,
        -- U+001F-joined). Soft-hide via `status` (1 = active/retained, 0 =
        -- hidden); `created_at`/`hidden_at` move together. `label` is the hot
        -- query key; `path` is the structural truth. No stored `level` (it is
        -- derivable from `path`).
        CREATE SEQUENCE IF NOT EXISTS keyword_id_seq START 1;

        CREATE TABLE IF NOT EXISTS keyword (
            id INTEGER PRIMARY KEY DEFAULT nextval('keyword_id_seq'),
            image_id INTEGER NOT NULL,          -- -> images.id (the "image pointer")
            label TEXT NOT NULL,                -- this node's text (= last path segment)
            path TEXT NOT NULL,                 -- root->this node, U+001F-separated
            status INTEGER NOT NULL DEFAULT 1,  -- 1 = active, 0 = hidden (soft-delete)
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            hidden_at TIMESTAMP,                -- set when status->0; NULL while active
            collection BOOLEAN NOT NULL DEFAULT FALSE  -- Collections-panel membership; orthogonal to status (keyword search stays oblivious)
        );

        CREATE INDEX IF NOT EXISTS idx_keyword_image_id ON keyword(image_id);
        CREATE INDEX IF NOT EXISTS idx_keyword_label ON keyword(label);
        CREATE INDEX IF NOT EXISTS idx_keyword_path ON keyword(path);
        CREATE INDEX IF NOT EXISTS idx_keyword_status ON keyword(status);

        -- Migration for pre-existing catalogues (no-op on a fresh DB, where the
        -- CREATE TABLE above already carries the column). Mirrors the images ALTERs:
        -- add the column WITHOUT a DEFAULT, then backfill the literal. An
        -- `ALTER TABLE ... ADD COLUMN ... DEFAULT <expr>` does NOT survive WAL
        -- replay — on reopen DuckDB re-binds the default with no database context
        -- set and throws an internal error (GetDefaultDatabase), wedging the whole
        -- catalogue. `collection = TRUE` excludes the NULLs anyway; the backfill
        -- just makes it tidy.
        ALTER TABLE keyword ADD COLUMN IF NOT EXISTS collection BOOLEAN;
        UPDATE keyword SET collection = FALSE WHERE collection IS NULL;

        -- Active-only view: ALL normal keyword reads/queries hit this so the
        -- status filter can never be forgotten. Recovery reads the raw table.
        CREATE OR REPLACE VIEW keyword_visible AS
            SELECT * FROM keyword WHERE status = 1;

        -- === Videos (Lightroom import; Docs/DESIGN-Lightroom-Catalog-Import.md §8) ===
        -- Video assets live in their OWN table, NOT in `images` (video is not an
        -- ImageKind). Catalog-only for v1: metadata + curation. Poster-frame
        -- thumbnails + playback are Stage 6 (AVFoundation). The column shape
        -- mirrors `images` where they overlap (file props, timestamps, curation)
        -- so the two stay consistent and no migration is needed when videos
        -- become first-class. Video-specific columns: duration_seconds /
        -- frame_rate (decoded Swift-side from AgVideoInfo's hex QuickTime
        -- rationals), has_audio, video_kind (mov/mp4/mpeg by extension).
        -- Populated by merge_lightroom_videos (same ON CONFLICT(file_path) upsert
        -- pattern as the images merge). Sequence PK because DuckDB 1.2.2 drops
        -- GENERATED ... IDENTITY (same reason as images_id_seq).
        CREATE SEQUENCE IF NOT EXISTS videos_id_seq START 1;

        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY DEFAULT nextval('videos_id_seq'),

            -- File-system properties (mirror images)
            file_path TEXT NOT NULL UNIQUE,
            file_size BIGINT NOT NULL,
            file_name TEXT NOT NULL,
            file_extension TEXT,
            directory_path VARCHAR,
            created_timestamp INTEGER NOT NULL,   -- Unix epoch seconds
            modified_timestamp INTEGER NOT NULL,  -- Unix epoch seconds

            -- Capture / media metadata
            capture_datetime TEXT,                -- ISO 8601
            pixel_width INTEGER,
            pixel_height INTEGER,
            duration_seconds DOUBLE,              -- AgVideoInfo.duration (decoded)
            frame_rate DOUBLE,                    -- AgVideoInfo.frame_rate (decoded)
            has_audio BOOLEAN,
            video_kind TEXT,                      -- 'mov' / 'mp4' / 'mpeg' (by extension)

            -- Curation (LR rates/flags/labels videos too)
            rating INTEGER,
            flag TEXT,
            color_label TEXT,

            -- Audit
            indexed_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_videos_capture_datetime ON videos(capture_datetime);
        CREATE INDEX IF NOT EXISTS idx_videos_rating ON videos(rating);
        CREATE INDEX IF NOT EXISTS idx_videos_directory_path ON videos(directory_path);

        -- === Saved Queries (Session 63; Docs/DESIGN-Saved-Queries.md) ===
        -- A saved Find in Gallery query: the RECIPE (criterion rows), never the
        -- results. `saved_query_criterion` mirrors the QueryPredicate wire
        -- struct exactly, plus placement: `position` (1-based slot in the
        -- sentence) and `connector` (how the row joins everything to its LEFT
        -- in the builder's left-to-right fold; NULL at position 1). One row per
        -- criterion — repeated subjects are simply more rows. Both tables are
        -- fresh CREATEs: no ALTER migration, ever (the S62 WAL lesson).
        CREATE SEQUENCE IF NOT EXISTS saved_query_id_seq START 1;

        CREATE TABLE IF NOT EXISTS saved_query (
            id INTEGER PRIMARY KEY DEFAULT nextval('saved_query_id_seq'),
            name TEXT NOT NULL,                 -- unique by construction (save_query suffixes collisions)
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS saved_query_criterion (
            query_id INTEGER NOT NULL,          -- -> saved_query.id
            position INTEGER NOT NULL,          -- 1-based slot in the sentence
            connector TEXT,                     -- 'and' / 'or' / 'xor'; NULL at position 1
            kind TEXT NOT NULL,                 -- QueryPredicate.kind (the subject discriminator)
            op TEXT,                            -- QueryPredicate.op
            value TEXT,                         -- QueryPredicate.value
            day TEXT,                           -- QueryPredicate.day ("YYYY:MM:DD")
            day_end TEXT,                       -- QueryPredicate.day_end
            stars INTEGER                       -- QueryPredicate.stars
        );

        CREATE INDEX IF NOT EXISTS idx_saved_query_criterion_query ON saved_query_criterion(query_id);
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

    // --- Backfill migration (DESIGN-Duplicate-Consolidation.md §8;
    // --- DESIGN-Filter-Aware-Pagination.md §4 added Step 3 in Session 19) ----
    //
    // Runs once at app launch, after schema creation (CREATE TABLE + ALTER
    // TABLE IF NOT EXISTS), before the catalogue connection is stored in
    // the CATALOGUE singleton. Atomic-with-init from the app's perspective —
    // no UI query can race the migration because the singleton is the only
    // public access path and is set only after this block returns.
    //
    // Three responsibilities, all wrapped in a single transaction:
    //   1. Populate file_stem and image_kind for any row where either column
    //      is NULL. On a fresh database this loop is a no-op (no rows). On
    //      an existing 32,855-record catalogue this is a one-time cost.
    //      The estimate per Section 14 is well under a second.
    //   2. Idempotent lowercase normalization of file_extension. The current
    //      catalogue spot-check shows file_extension is already lowercase,
    //      but the design wants a belt-and-braces UPDATE to enforce the
    //      lowercase-canonical convention even on data ingested before
    //      it was a stated invariant.
    //   3. Populate directory_path for any row where it is NULL, using the
    //      canonical SUBSTRING/INSTR/REVERSE extraction (mirrors
    //      get_distinct_directory_paths). Pure SQL UPDATE — no Rust
    //      round-trip per row. On a fresh database this is a no-op.
    //
    // Idempotency: the WHERE clauses on all three steps mean that running this
    // block a second time has no work to do — rows with non-NULL file_stem
    // and image_kind are skipped, rows whose file_extension already matches
    // LOWER(file_extension) are skipped, and rows with non-NULL directory_path
    // are skipped.
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
            ImageKind::Heif => "heif",
            ImageKind::Dng  => "dng",
            ImageKind::Psd  => "psd",
            ImageKind::Tiff => "tiff",
            ImageKind::Png  => "png",
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

    // Step 3: backfill directory_path for any row where it is NULL.
    // Pure SQL UPDATE — directory_path is derived directly from file_path
    // via the canonical SUBSTRING/INSTR/REVERSE extraction (mirrors
    // get_distinct_directory_paths; see also the "Directory-Path Extraction
    // SQL — Gotcha" note in CLAUDE.md for the left-anchored form that must
    // NOT be used). Idempotent via WHERE directory_path IS NULL; the
    // file_path LIKE '%/%' guard mirrors get_distinct_directory_paths'
    // safety convention against pathological rows lacking a slash.
    match conn.execute(
        "UPDATE images \
         SET directory_path = SUBSTRING(file_path, 1, LENGTH(file_path) - INSTR(REVERSE(file_path), '/')) \
         WHERE directory_path IS NULL AND file_path LIKE '%/%'",
        [],
    )
    {
        Ok(changed) => eprintln!("[migration] backfilled directory_path for {} rows", changed),
        Err(e) => eprintln!("[migration] Failed to backfill directory_path: {}", e),
    }

    // Step 4: reclassify formats promoted to their own ImageKind
    // (Docs/DESIGN-Lightroom-Catalog-Import.md §7a). DNG left the Raw class;
    // PSD/TIFF/PNG left Other. Rows ingested under the old kind are moved here.
    // file_extension is already lowercased by Step 2, so plain equality hits
    // idx_file_extension. Idempotent: the `image_kind <> …` guard no-ops re-runs.
    // (Rows with a NULL file_extension are skipped — they keep their parse-time
    // kind; this matches the §7a documented edge.)
    match conn.execute(
        "UPDATE images SET image_kind = 'dng' WHERE file_extension = 'dng' AND image_kind <> 'dng'",
        [],
    )
    {
        Ok(changed) => eprintln!("[migration] reclassified {} dng rows", changed),
        Err(e) => eprintln!("[migration] Failed to reclassify dng: {}", e),
    }
    match conn.execute(
        "UPDATE images SET image_kind = 'psd' WHERE file_extension = 'psd' AND image_kind <> 'psd'",
        [],
    )
    {
        Ok(changed) => eprintln!("[migration] reclassified {} psd rows", changed),
        Err(e) => eprintln!("[migration] Failed to reclassify psd: {}", e),
    }
    match conn.execute(
        "UPDATE images SET image_kind = 'tiff' WHERE file_extension IN ('tif','tiff') AND image_kind <> 'tiff'",
        [],
    )
    {
        Ok(changed) => eprintln!("[migration] reclassified {} tiff rows", changed),
        Err(e) => eprintln!("[migration] Failed to reclassify tiff: {}", e),
    }
    match conn.execute(
        "UPDATE images SET image_kind = 'png' WHERE file_extension = 'png' AND image_kind <> 'png'",
        [],
    )
    {
        Ok(changed) => eprintln!("[migration] reclassified {} png rows", changed),
        Err(e) => eprintln!("[migration] Failed to reclassify png: {}", e),
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
    //
    // directory_path (Session 19 Step 1, DESIGN-Filter-Aware-Pagination.md §4)
    // sits adjacent to image_kind for the same visual-grouping reason — it is
    // another stored derivation consumed by the filter-aware SQL added later
    // in Session 19. Unlike file_stem / image_kind, its value is derived
    // entirely from file_path with no Rust-side enrichment, so the VALUES
    // clause embeds the canonical SUBSTRING/INSTR/REVERSE expression directly
    // on ?1 (file_path) rather than binding a separately-computed parameter.
    // This guarantees byte-for-byte identical logic between ingest and the
    // backfill UPDATE in initialize_catalogue's migration block.
    let insert_sql = r#"
        INSERT OR IGNORE INTO images (
            file_path, file_size, file_name, file_extension,
            file_stem, image_kind, directory_path,
            created_timestamp, modified_timestamp,
            camera_make, camera_model, lens_model,
            focal_length, aperture, shutter_speed, iso,
            capture_datetime,
            pixel_width, pixel_height, color_space, bit_depth,
            gps_latitude, gps_longitude, gps_altitude,
            copyright, creator, description,
            rating, flag, color_label
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6,
            SUBSTRING(?1, 1, LENGTH(?1) - INSTR(REVERSE(?1), '/')),
            ?7, ?8, ?9, ?10,
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
            ImageKind::Heif => "heif",
            ImageKind::Dng  => "dng",
            ImageKind::Psd  => "psd",
            ImageKind::Tiff => "tiff",
            ImageKind::Png  => "png",
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
/// Step 4a (Session 20) — Filter-Aware Pagination: accepts the two
/// filter booleans (`apply_duplicate_filter`, `apply_raw_jpeg_collapse`)
/// and forwards them to `execute_image_count_query`. When both are
/// false the helper emits the unfiltered `SELECT COUNT(*) FROM images`
/// form, preserving the previous behavior bit-for-bit.
///
/// Data flow:
/// Swift calls this after ingestion or on app launch to populate UI statistics
///
/// Returns:
/// - Total number of image records in the catalogue (subject to the
///   two filter booleans)
/// - 0 if catalogue not initialized or query fails
pub async fn get_image_count(
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
) -> u64 {
    // Acquire lock and validate connection
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    // Forward predicate-only WHERE (empty) and the two filter booleans.
    // DuckDB returns i64; cast to u64 at the call site for the unsigned
    // count semantic (decision C5).
    execute_image_count_query(
        conn,
        "",
        apply_duplicate_filter,
        apply_raw_jpeg_collapse,
    ) as u64
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

/// Window-function `CASE` expression that computes `duplicate_group_id` for
/// each row in the result set.
///
/// Returns `NULL` for singletons (groups of size 1) and for rows whose
/// `capture_datetime` IS NULL (exempt from deduplication). For multi-row
/// groups, returns the `id` of the row with the lexicographically smallest
/// `file_path` — the canonical "group winner" referenced by the duplicate
/// filter predicate.
///
/// The 6-field partition (capture_datetime, camera_model, pixel_width,
/// pixel_height, image_kind, LOWER(file_stem)) is ordered high-to-low per
/// the Session 18 partition-ordering principle so looser future variants
/// remain structural prefixes of the strict current variant.
///
/// Embedded into the inner SELECT projection of both
/// `execute_image_record_query` (always) and `execute_image_count_query`
/// (only when the duplicate filter is active). Single source of truth —
/// both helpers reference this constant rather than duplicating the
/// expression inline.
const DUPLICATE_GROUP_ID_CASE: &str = "\
            CASE \
                WHEN capture_datetime IS NULL THEN NULL \
                WHEN COUNT(*) OVER ( \
                    PARTITION BY \
                        capture_datetime, \
                        camera_model, \
                        pixel_width, \
                        pixel_height, \
                        image_kind, \
                        LOWER(file_stem) \
                ) = 1 THEN NULL \
                ELSE FIRST_VALUE(id) OVER ( \
                    PARTITION BY \
                        capture_datetime, \
                        camera_model, \
                        pixel_width, \
                        pixel_height, \
                        image_kind, \
                        LOWER(file_stem) \
                    ORDER BY file_path ASC \
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING \
                ) \
            END AS duplicate_group_id";

/// Outer-WHERE predicate that keeps duplicate-group winners and singletons.
///
/// Applied at the OUTER WHERE level (after the projection wraps the inner
/// SELECT in a subquery) when `apply_duplicate_filter == true`. Swift sets
/// the flag from `!showDuplicates`. References the `duplicate_group_id`
/// projection alias produced by `DUPLICATE_GROUP_ID_CASE`.
///
/// Semantics: keep rows where the group_id is NULL (singletons or
/// capture_datetime-NULL exempts) OR where the row's `id` equals the
/// group winner. Non-winner duplicates are dropped.
///
/// Single source of truth — both helpers reference this constant. See
/// DESIGN-Filter-Aware-Pagination.md §6.
const DUPLICATE_FILTER_PREDICATE: &str =
    "(duplicate_group_id IS NULL OR id = duplicate_group_id)";

/// Inner-WHERE predicate that collapses RAW+JPEG counterpart pairs to the
/// JPEG.
///
/// Applied at the INNER WHERE level (alongside the existing path/date
/// predicate, where `images` is the FROM table and the column references
/// resolve directly) when `apply_raw_jpeg_collapse == true`. Swift sets
/// the flag from `showRawJpegPairsAsOne`. References only stored columns
/// (`image_kind`, `file_stem`, `directory_path`); no projection-alias
/// dependency.
///
/// Semantics: drop a record if it is a RAW and a JPEG **or HEIF** counterpart
/// exists in the same directory with the same stem. The lightweight sibling
/// (JPEG or HEIF) is kept; the RAW counterpart is hidden. (Session 41
/// generalized this from JPEG-only — Nikon/Canon/Sony RAW+HEIF shooting
/// modes produce e.g. NEF + HIF, which should collapse the same as RAW+JPEG.
/// If a RAW happens to have BOTH a JPEG and a HEIF sibling — rare — both
/// siblings remain visible; only the RAW is hidden.)
///
/// Note: `image_kind` literal comparisons use lowercase per the project-
/// wide lowercase canonical-case convention (Session 18). Verified in
/// `ingest_metadata` — image_kind is always stored as "jpeg" / "raw" /
/// "heif" / "other".
///
/// Single source of truth — both helpers reference this constant. See
/// DESIGN-Filter-Aware-Pagination.md §6.
const RAW_JPEG_COLLAPSE_PREDICATE: &str = "\
    NOT (image_kind = 'raw' AND EXISTS ( \
        SELECT 1 FROM images j \
        WHERE j.image_kind IN ('jpeg', 'heif') \
          AND j.file_stem = images.file_stem \
          AND j.directory_path = images.directory_path \
    ))";

/// Build the path/date predicate text from the (path_prefix, date_prefix)
/// pair.
///
/// Single source of truth for the four-arm composition used by
/// `get_images_for_path_prefix`, `get_image_count_for_filters`, and (A9)
/// `get_file_paths_for_filters`. Returns predicate text WITHOUT the
/// "WHERE" keyword (predicate-only convention, decision C3 — the query
/// helpers prepend the keyword).
///
/// Single-quote escape (replace `'` with `''`) is preserved verbatim from
/// the pre-extraction call sites — Queue item 4 (Chunk 6) will parameterize
/// this as a separate change. Both inputs are treated as untrusted strings
/// that may legitimately contain single quotes (paths with apostrophes,
/// for instance).
///
/// Arms:
/// - (empty,  empty)  → empty predicate (no path/date filter)
/// - (path,   empty)  → `file_path LIKE 'PATH%'`
/// - (empty,  date)   → `capture_datetime LIKE 'DATE%'`
/// - (path,   date)   → both, AND-joined
///
/// Composes safely with the other inner-WHERE predicates
/// (`RAW_JPEG_COLLAPSE_PREDICATE`) and with the outer duplicate-filter
/// wrapper — see `execute_image_record_query` /
/// `execute_image_count_query` / `execute_file_path_projection_query`.
fn build_path_date_predicate(path_prefix: &str, date_prefix: &str) -> String
{
    let escaped_path = path_prefix.replace("'", "''");
    let escaped_date = date_prefix.replace("'", "''");

    match (path_prefix.is_empty(), date_prefix.is_empty())
    {
        (true, true) => String::new(),
        (false, true) => format!("file_path LIKE '{}%'", escaped_path),
        (true, false) => format!("capture_datetime LIKE '{}%'", escaped_date),
        (false, false) => format!(
            "file_path LIKE '{}%' AND capture_datetime LIKE '{}%'",
            escaped_path, escaped_date
        ),
    }
}

/// Escape a string for use as a literal inside a DuckDB `SIMILAR TO`
/// regex pattern.
///
/// `SIMILAR TO` uses POSIX-style regex semantics. The characters that
/// have regex meaning and therefore need a leading `\` to be matched
/// literally are: `\ . ^ $ * + ? ( ) [ ] { } |`. Notably, underscore
/// (`_`) is NOT a wildcard in regex (that is a `LIKE`-ism) — so an
/// underscore in a filename like `RSW_0001.NEF` is literal under
/// `SIMILAR TO` without escaping. This is the precise property that
/// motivated choosing `SIMILAR TO` over `LIKE` for the destination-
/// family predicate (see `build_destination_family_predicate`): only
/// the two leading digits in the `NN_version_` prefix should be
/// wildcards; everything else — including every underscore in the
/// canonical and in "_version_" itself — must match literally.
///
/// Does NOT perform SQL single-quote escaping; callers wrap the result
/// in `'…'` and must follow with the usual `.replace("'", "''")` at
/// the assembly site (`build_destination_family_predicate` does both).
fn regex_escape_for_similar_to(s: &str) -> String
{
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars()
    {
        match c
        {
            '\\' | '.' | '^' | '$' | '*' | '+' | '?'
            | '(' | ')' | '[' | ']' | '{' | '}' | '|' =>
            {
                out.push('\\');
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

/// Build the predicate text for the destination-family catalogue query
/// used by the cross-plan overwrite-gap fix (Session 30).
///
/// The fix turns the destination probe from "is the on-disk file
/// identical?" into "what does the catalogue already record at this
/// basePath?" so that `versionCountByBasePath` can be pre-seeded
/// whenever the canonical destination path is occupied by a DISTINCT
/// image — not only on the suppression-identical branch as before.
///
/// The "family" at a given basePath is the union of:
/// - the canonical row at `<dir>/<canonical_file_name>`, and
/// - any version-prefixed rows at `<dir>/NN_version_<canonical_file_name>`
///   for two leading digits.
///
/// **`directory_path` match-by-construction (silent-no-op guard).**
/// The predicate pivots on the stored `directory_path` column with an
/// EQUALITY comparison against the SAME `SUBSTRING / LENGTH / INSTR /
/// REVERSE` expression used at ingest (`INSERT OR IGNORE INTO images
/// (…, directory_path, …) VALUES (…, SUBSTRING(?1, 1, LENGTH(?1) -
/// INSTR(REVERSE(?1), '/')), …)` — see the ingest block above). The
/// caller passes a SAMPLE destination `file_path` (e.g. the planner's
/// computed `destinationRoot + '/' + 'YYYY/MM_monthname/DD/' +
/// canonical_file_name`); the helper derives `directory_path` from
/// THAT string via the same SQL expression. The two strings are
/// produced by the same expression on inputs that are byte-equal by
/// construction (the orchestrator catalogues exactly what it wrote),
/// so the equality cannot silently return empty on a directory_path
/// drift — there is no Swift-side directory string in flight to drift.
///
/// **`SIMILAR TO` with strict digit-only wildcards.** The version-
/// prefix arm uses `file_name SIMILAR TO '[0-9][0-9]_version_<regex-
/// escaped-canonical>'`. ONLY the two leading digits are wildcards;
/// every literal `_` (in `_version_` and inside the canonical) and the
/// canonical's `.` are matched literally — the latter via
/// `regex_escape_for_similar_to`. `LIKE` would over-match here because
/// every `_` in the pattern is a single-char wildcard.
///
/// **Returned text (predicate-only convention, decision C3):** WITHOUT
/// the `WHERE` keyword — `execute_image_record_projection_query`
/// inserts the keyword when any predicate is active. SQL single-quote
/// escape (`'` → `''`) is applied to both inputs, matching the call-
/// site quoting style of `build_path_date_predicate`.
///
/// Example with `canonical_file_name = "RSW_0001.NEF"` and
/// `sample_file_path = "/Volumes/Photos/Library/2026/01_january/15/RSW_0001.NEF"`:
/// ```sql
/// directory_path = SUBSTRING('/Volumes/Photos/Library/2026/01_january/15/RSW_0001.NEF', 1,
///                            LENGTH('/Volumes/Photos/Library/2026/01_january/15/RSW_0001.NEF')
///                            - INSTR(REVERSE('/Volumes/Photos/Library/2026/01_january/15/RSW_0001.NEF'), '/'))
///   AND (file_name = 'RSW_0001.NEF' OR file_name SIMILAR TO '[0-9][0-9]_version_RSW_0001\.NEF')
/// ```
fn build_destination_family_predicate(sample_file_path: &str, canonical_file_name: &str) -> String
{
    let escaped_sample_path = sample_file_path.replace("'", "''");
    let escaped_canonical_sql = canonical_file_name.replace("'", "''");

    // Regex-escape first (chars with regex meaning), THEN SQL-escape
    // the result (any embedded single quotes in the canonical). The
    // two escape layers are independent: regex escaping protects the
    // SIMILAR TO engine; SQL escaping protects the string-literal
    // syntax around it.
    let regex_escaped = regex_escape_for_similar_to(canonical_file_name);
    let regex_escaped_sql = regex_escaped.replace("'", "''");

    format!(
        "directory_path = SUBSTRING('{}', 1, LENGTH('{}') - INSTR(REVERSE('{}'), '/')) \
         AND (file_name = '{}' OR file_name SIMILAR TO '[0-9][0-9]_version_{}')",
        escaped_sample_path,
        escaped_sample_path,
        escaped_sample_path,
        escaped_canonical_sql,
        regex_escaped_sql
    )
}

/// Execute a paginated `ImageRecord`-returning query against the catalogue.
///
/// Single source of truth for the 32-column SELECT projection (including
/// the window-function `CASE` from `DUPLICATE_GROUP_ID_CASE` that emits
/// `duplicate_group_id`) and the row decode shared by the four paginating
/// `ImageRecord` query functions: `get_all_images`, `get_images_sorted`,
/// `get_images_filtered`, and `get_images_for_path_prefix`. Each caller
/// builds its own predicate text and ORDER BY expression and delegates the
/// projection, prepare, bind, row decode, and error logging here.
///
/// `find_counterpart_image` is deliberately NOT routed through this helper
/// (Session 19 Step 2 decision C1): it returns `Option<ImageRecord>`, has
/// no LIMIT/OFFSET, uses parameterized binds, sorts by `file_extension
/// ASC`, and iterates Rust-side with classify-based early-exit.
///
/// **Filter composition (Session 20 Step 4a, design doc §5–§6):**
///
/// Two filter booleans gate the new WHERE-clause fragments:
///
/// - `apply_raw_jpeg_collapse` (true when Swift's `showRawJpegPairsAsOne`
///   is on): appends `RAW_JPEG_COLLAPSE_PREDICATE` to the inner WHERE
///   alongside the caller-supplied predicate. Both reference stored
///   columns on `images` and compose with `AND`.
///
/// - `apply_duplicate_filter` (true when Swift's `showDuplicates` is off):
///   wraps the inner projection in a subquery and applies
///   `DUPLICATE_FILTER_PREDICATE` at the OUTER WHERE level, where the
///   `duplicate_group_id` alias is in scope. When the filter is off, the
///   projection is emitted at the top level without subquery wrapping
///   (the alias is still present on the returned `ImageRecord`s for
///   future Filter Builder use).
///
/// **Caller convention (Step 4a change):** `where_clause` is now predicate
/// text WITHOUT the "WHERE" keyword (empty string for no filter). The
/// helper assembles WHERE clauses internally. This is a convention change
/// from Steps 2/3 where callers included the keyword.
///
/// WHERE-clause `format!` interpolation with manual single-quote escape is
/// preserved at call sites; Queue item 4 (Chunk 6) will parameterize these
/// as a separate change.
///
/// Parameters:
/// - conn: borrowed catalogue connection (caller holds the MutexGuard)
/// - where_clause: predicate text only (no "WHERE" keyword); empty string
///   for no caller-supplied filter
/// - order_by: ORDER BY expression text only (no "ORDER BY" keyword);
///   helper prepends the keyword
/// - limit: LIMIT value, bound as ?1
/// - offset: OFFSET value, bound as ?2
/// - apply_duplicate_filter: when true, wrap projection in subquery and
///   apply `DUPLICATE_FILTER_PREDICATE` at outer level
/// - apply_raw_jpeg_collapse: when true, AND-in
///   `RAW_JPEG_COLLAPSE_PREDICATE` at inner WHERE level
///
/// Returns:
/// - Vec of `ImageRecord` structs in the order specified by `order_by`.
/// - Empty Vec if prepare or query_map fails; errors logged via eprintln!.
fn execute_image_record_query(
    conn: &Connection,
    where_clause: &str,
    order_by: &str,
    limit: i64,
    offset: i64,
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
) -> Vec<ImageRecord>
{
    // Assemble the inner WHERE from the caller-supplied predicate text
    // and the RAW+JPEG collapse predicate (both stored-column references;
    // both safely composable with AND at the inner level).
    let mut inner_predicates: Vec<&str> = Vec::new();
    if !where_clause.is_empty()
    {
        inner_predicates.push(where_clause);
    }
    if apply_raw_jpeg_collapse
    {
        inner_predicates.push(RAW_JPEG_COLLAPSE_PREDICATE);
    }
    let inner_where = if inner_predicates.is_empty()
    {
        String::new()
    }
    else
    {
        format!("WHERE {}", inner_predicates.join(" AND "))
    };

    // Branch on apply_duplicate_filter (decision C1). When active, wrap
    // the projection-with-duplicate_group_id in a subquery so the outer
    // WHERE can reference the alias. When inactive, emit the same
    // projection at the top level — the alias is still surfaced on the
    // returned ImageRecord (column 30) for the existing Swift filter and
    // future Filter Builder consumers.
    let query_sql = if apply_duplicate_filter
    {
        format!(r#"
            SELECT * FROM (
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
                    {}
                FROM images
                {}
            )
            WHERE {}
            ORDER BY {}
            LIMIT ?1 OFFSET ?2
        "#,
            DUPLICATE_GROUP_ID_CASE,
            inner_where,
            DUPLICATE_FILTER_PREDICATE,
            order_by)
    }
    else
    {
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
                {}
            FROM images
            {}
            ORDER BY {}
            LIMIT ?1 OFFSET ?2
        "#,
            DUPLICATE_GROUP_ID_CASE,
            inner_where,
            order_by)
    };

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

/// Execute a `COUNT(*)` query against the `images` table.
///
/// Single source of truth for the COUNT shape shared by the four count
/// functions: `get_image_count`, `get_filtered_image_count`,
/// `get_image_count_for_path_prefix`, and `get_image_count_for_filters`.
/// Each caller passes its WHERE predicate text (without the "WHERE"
/// keyword) plus the two filter booleans; this helper assembles the
/// final SQL and delegates the prepare, query, and error logging.
///
/// Symmetrical with `execute_image_record_query` but lighter: COUNT has
/// no projection, no ORDER BY, no LIMIT/OFFSET, and no bound parameters.
/// WHERE-clause parameterization is preserved as-is at the call sites
/// (`format!` interpolation with manual single-quote escape). Queue
/// item 4 (Chunk 6) will parameterize these as a separate change.
///
/// Step 4a (Session 20) — Filter-Aware Pagination:
/// - `apply_duplicate_filter` (decision C2): When false, the helper
///   emits the simple `SELECT COUNT(*) FROM images {where}` form — no
///   subquery, no window function. When true, it wraps a projection
///   carrying `duplicate_group_id` (via DUPLICATE_GROUP_ID_CASE) in a
///   subquery and applies DUPLICATE_FILTER_PREDICATE at the outer WHERE,
///   matching the wrap-in-subquery shape of the record helper so the
///   two functions produce a consistent total/page relationship.
/// - `apply_raw_jpeg_collapse`: When true, RAW_JPEG_COLLAPSE_PREDICATE
///   is composed with the caller-supplied predicate at the inner WHERE
///   (AND-joined). Independent of the duplicate filter.
///
/// Predicate-only convention (decision C3): callers must NOT include
/// the "WHERE" keyword in `where_clause`; this helper inserts the
/// keyword when any predicate is active.
///
/// Implementation-log note: Session 19 Step 3 decision C1 — STATUS.md's
/// prior framing of "three count functions" was imprecise. The helper
/// covers four count functions; `get_image_count` was overlooked because
/// it is the only one returning `u64` (cast at the call site) rather
/// than `i64`.
///
/// Parameters:
/// - conn: borrowed catalogue connection (caller holds the MutexGuard)
/// - where_clause: caller-supplied predicate text WITHOUT the "WHERE"
///   keyword; empty string for no caller predicate.
/// - apply_duplicate_filter: when true, restrict to one row per
///   duplicate cluster via the wrap-in-subquery pattern.
/// - apply_raw_jpeg_collapse: when true, suppress RAW siblings of JPEGs
///   sharing file_stem within the same directory_path.
///
/// Returns:
/// - i64 count of matching rows.
/// - 0 if prepare or query fails; errors logged via eprintln! to match
///   the pre-extraction call-site behavior.
fn execute_image_count_query(
    conn: &Connection,
    where_clause: &str,
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
) -> i64
{
    let mut inner_predicates: Vec<&str> = Vec::new();
    if !where_clause.is_empty()
    {
        inner_predicates.push(where_clause);
    }
    if apply_raw_jpeg_collapse
    {
        inner_predicates.push(RAW_JPEG_COLLAPSE_PREDICATE);
    }
    let inner_where = if inner_predicates.is_empty()
    {
        String::new()
    }
    else
    {
        format!("WHERE {}", inner_predicates.join(" AND "))
    };

    // Branch on apply_duplicate_filter (decision C2). When inactive,
    // emit the minimal COUNT form — no subquery, no window function.
    // When active, wrap a projection carrying duplicate_group_id in a
    // subquery so the outer WHERE can reference the alias; this matches
    // the record helper's wrap-in-subquery shape so totals stay
    // consistent with the paginated page contents.
    let query_sql = if apply_duplicate_filter
    {
        format!(r#"
            SELECT COUNT(*) FROM (
                SELECT
                    id,
                    {}
                FROM images
                {}
            )
            WHERE {}
        "#,
            DUPLICATE_GROUP_ID_CASE,
            inner_where,
            DUPLICATE_FILTER_PREDICATE)
    }
    else
    {
        format!("SELECT COUNT(*) FROM images {}", inner_where)
    };

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
            eprintln!("Failed to query image count: {}", e);
            0
        }
    }
}

/// Execute a `file_path` projection query against the `images` table.
///
/// Third sibling to `execute_image_record_query` and
/// `execute_image_count_query` — projection-only enumeration of the
/// `file_path` column for callers that need the set of files matching
/// a filter state without materializing full `ImageRecord` rows.
///
/// Shape parallels the count helper: same inner-WHERE assembly
/// (caller-supplied predicate AND-joined with `RAW_JPEG_COLLAPSE_PREDICATE`
/// when active), same branch on `apply_duplicate_filter` (subquery wrap
/// with `DUPLICATE_GROUP_ID_CASE` + outer `DUPLICATE_FILTER_PREDICATE`),
/// no LIMIT/OFFSET, no ORDER BY. Sharing the filter constants
/// structurally with the other two helpers is what guarantees parity
/// with the gallery's loaded set by construction.
///
/// Path contract: returns `file_path` values EXACTLY as stored in the
/// catalogue (no normalization, no transformation). Order is unspecified
/// — callers that need a total order must sort. Empty Vec is a valid
/// success result (zero rows matched); use the outer
/// `Result`/sentinel-bundle to distinguish from a true failure.
///
/// Predicate-only convention (decision C3): `where_clause` is predicate
/// text WITHOUT the "WHERE" keyword; the helper inserts the keyword when
/// any predicate is active.
///
/// Parameters:
/// - conn: borrowed catalogue connection (caller holds the MutexGuard)
/// - where_clause: caller-supplied predicate text WITHOUT the "WHERE"
///   keyword; empty string for no caller predicate.
/// - apply_duplicate_filter: when true, restrict to one row per
///   duplicate cluster via the wrap-in-subquery pattern.
/// - apply_raw_jpeg_collapse: when true, AND-in
///   `RAW_JPEG_COLLAPSE_PREDICATE` at the inner WHERE level.
///
/// Returns:
/// - `Ok(Vec<String>)` of `file_path` values (possibly empty) on success.
/// - `Err(String)` on prepare/query failure, with the underlying error
///   message; errors also logged via `eprintln!` for parity with the
///   sibling helpers' diagnostic style.
fn execute_file_path_projection_query(
    conn: &Connection,
    where_clause: &str,
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
) -> Result<Vec<String>, String>
{
    let mut inner_predicates: Vec<&str> = Vec::new();
    if !where_clause.is_empty()
    {
        inner_predicates.push(where_clause);
    }
    if apply_raw_jpeg_collapse
    {
        inner_predicates.push(RAW_JPEG_COLLAPSE_PREDICATE);
    }
    let inner_where = if inner_predicates.is_empty()
    {
        String::new()
    }
    else
    {
        format!("WHERE {}", inner_predicates.join(" AND "))
    };

    // Branch on apply_duplicate_filter — mirrors the count helper's
    // shape. When inactive, emit the minimal projection. When active,
    // wrap an (id, file_path, duplicate_group_id) projection in a
    // subquery so the outer WHERE can apply DUPLICATE_FILTER_PREDICATE
    // against the alias.
    let query_sql = if apply_duplicate_filter
    {
        format!(r#"
            SELECT file_path FROM (
                SELECT
                    id,
                    file_path,
                    {}
                FROM images
                {}
            )
            WHERE {}
        "#,
            DUPLICATE_GROUP_ID_CASE,
            inner_where,
            DUPLICATE_FILTER_PREDICATE)
    }
    else
    {
        format!("SELECT file_path FROM images {}", inner_where)
    };

    let mut stmt = match conn.prepare(&query_sql)
    {
        Ok(s) => s,
        Err(e) =>
        {
            eprintln!("Failed to prepare file_path projection query: {}", e);
            return Err(format!("prepare failed: {}", e));
        }
    };

    let rows = match stmt.query_map([], |row| row.get::<_, String>(0))
    {
        Ok(r) => r,
        Err(e) =>
        {
            eprintln!("Failed to execute file_path projection query: {}", e);
            return Err(format!("query failed: {}", e));
        }
    };

    let mut paths: Vec<String> = Vec::new();
    for row_result in rows
    {
        match row_result
        {
            Ok(path) => paths.push(path),
            Err(e) => eprintln!("Failed to read file_path row: {}", e),
        }
    }

    Ok(paths)
}

/// Execute an unbounded `ImageRecord` query against the `images` table.
///
/// Records analogue of `execute_file_path_projection_query` (A9) — same
/// filter machinery, but selects the full `ImageRecord` column set
/// instead of just `file_path`. Used by `get_image_records_for_filters`
/// (A10) to enumerate every record matching a node's filter state in
/// one shot.
///
/// Diverges from `execute_image_record_query` (the paginated helper)
/// on three axes:
/// 1. **No LIMIT / OFFSET.** Returns every matching record in one
///    pass. The caller (A10's sidebar bulk-copy path) feeds the result
///    into `CopyPlanner`, which sorts and groups in Swift — bringing
///    the full set across the FFI in one call is cheaper than
///    paginating from Swift, and avoids any "did the page boundary
///    drop something" failure mode.
/// 2. **No ORDER BY.** The planner owns sort order (the Copy To
///    total order: `capture_datetime DESC NULLS LAST, created_timestamp
///    DESC, file_path ASC`). Adding a SQL ORDER BY here would be wasted
///    work — the planner re-sorts anyway.
/// 3. **No bound parameters.** The simplified prepare/query path uses
///    `query_map([], …)` directly.
///
/// Filter assembly is IDENTICAL to the other two helpers (`inner_where`
/// composition, branch on `apply_duplicate_filter` with the same
/// subquery-wrap + outer `DUPLICATE_FILTER_PREDICATE` shape). This is
/// the parity-by-construction guarantee: a sidebar bulk-copy enumerates
/// the same set of rows that the gallery would display for the same
/// `(path_prefix, date_prefix, apply_duplicate_filter,
/// apply_raw_jpeg_collapse)` tuple, because both routes flow through
/// the same predicate constants and the same composition order.
///
/// Predicate-only convention (decision C3): `where_clause` is predicate
/// text WITHOUT the "WHERE" keyword; the helper inserts the keyword
/// when any predicate is active.
///
/// Parameters:
/// - conn: borrowed catalogue connection (caller holds the MutexGuard)
/// - where_clause: caller-supplied predicate text WITHOUT the "WHERE"
///   keyword; empty string for no caller predicate.
/// - apply_duplicate_filter: when true, restrict to one row per
///   duplicate cluster via the wrap-in-subquery pattern.
/// - apply_raw_jpeg_collapse: when true, AND-in
///   `RAW_JPEG_COLLAPSE_PREDICATE` at the inner WHERE level.
///
/// Returns:
/// - `Vec<ImageRecord>` of matching rows in DuckDB-natural order
///   (unspecified — caller MUST sort). Empty vec on prepare/query
///   failure, with errors logged via `eprintln!` to match the sibling
///   helpers' diagnostic style and the established
///   record-returning-function convention (no `[Throws]`, no wrapper
///   dictionary — see `get_all_images`, `get_images_for_path_prefix`).
fn execute_image_record_projection_query(
    conn: &Connection,
    where_clause: &str,
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
) -> Vec<ImageRecord>
{
    let mut inner_predicates: Vec<&str> = Vec::new();
    if !where_clause.is_empty()
    {
        inner_predicates.push(where_clause);
    }
    if apply_raw_jpeg_collapse
    {
        inner_predicates.push(RAW_JPEG_COLLAPSE_PREDICATE);
    }
    let inner_where = if inner_predicates.is_empty()
    {
        String::new()
    }
    else
    {
        format!("WHERE {}", inner_predicates.join(" AND "))
    };

    // Branch on apply_duplicate_filter — mirrors the other two helpers
    // structurally. When active, wrap the projection-with-
    // duplicate_group_id in a subquery so the outer WHERE can reference
    // the alias. When inactive, emit the projection at the top level
    // (the alias is still surfaced on the returned ImageRecord, column
    // 30).
    let query_sql = if apply_duplicate_filter
    {
        format!(r#"
            SELECT * FROM (
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
                    {}
                FROM images
                {}
            )
            WHERE {}
        "#,
            DUPLICATE_GROUP_ID_CASE,
            inner_where,
            DUPLICATE_FILTER_PREDICATE)
    }
    else
    {
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
                {}
            FROM images
            {}
        "#,
            DUPLICATE_GROUP_ID_CASE,
            inner_where)
    };

    let mut stmt = match conn.prepare(&query_sql)
    {
        Ok(s) => s,
        Err(e) =>
        {
            eprintln!("Failed to prepare image record projection query: {}", e);
            return Vec::new();
        }
    };

    let rows = match stmt.query_map([], |row|
    {
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
            eprintln!("Failed to execute image record projection query: {}", e);
            return Vec::new();
        }
    };

    let mut records = Vec::new();
    for row_result in rows
    {
        match row_result
        {
            Ok(record) => records.push(record),
            Err(e) => eprintln!("Failed to parse image record row: {}", e),
        }
    }

    records
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
pub async fn get_all_images(
    limit: u32,
    offset: u32,
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
) -> Vec<ImageRecord> {
    // Acquire lock and validate connection
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    // Delegate projection, prepare, bind, row decode, and error logging to
    // execute_image_record_query. No caller predicate; ORDER BY id
    // (insertion order, not the date-sorted order used by the other three
    // paginating callers — get_all_images preserves the original
    // "browse by id" semantic for the Browse view). The two filter
    // booleans are forwarded as-is (Step 4a, Session 20).
    execute_image_record_query(
        conn,
        "",
        "id",
        limit as i64,
        offset as i64,
        apply_duplicate_filter,
        apply_raw_jpeg_collapse,
    )
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
pub async fn get_images_sorted(
    limit: u32,
    offset: u32,
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
) -> Vec<ImageRecord> {
    // Acquire lock and validate connection
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    // Delegate to execute_image_record_query. No caller predicate;
    // ORDER BY capture_datetime DESC NULLS LAST then created_timestamp
    // DESC — the "global sort" semantic used by the Photos view (images
    // with dates first, newest to oldest, with undated images at the
    // end). The two filter booleans are forwarded as-is (Step 4a,
    // Session 20).
    execute_image_record_query(
        conn,
        "",
        "capture_datetime DESC NULLS LAST, created_timestamp DESC",
        limit as i64,
        offset as i64,
        apply_duplicate_filter,
        apply_raw_jpeg_collapse,
    )
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

/// Update the pick/reject flag for an image
///
/// Sets the flag ("pick" or "reject") for an image identified by its file path.
/// Mirrors `update_image_rating`. Passing None clears the flag (sets it to NULL).
///
/// A non-null value OUTSIDE the canonical set {"pick", "reject"} is REJECTED
/// (returns false) rather than silently written, so the indexed `flag` column
/// can never hold a garbage value. This canonical set is also the stable
/// interchange vocabulary for a future Lightroom-catalogue import.
///
/// Returns:
/// - true if update succeeded
/// - false if catalogue not initialized, file not found, query failed, or an
///   invalid non-null value was supplied
pub async fn update_image_flag(file_path: String, flag: Option<String>) -> bool {
    // Acquire lock and validate connection
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return false;
        }
    };

    // Validate against the canonical set; None clears.
    let flag_value: Option<String> = match flag.as_deref() {
        None => None,
        Some(v @ ("pick" | "reject")) => Some(v.to_string()),
        Some(other) => {
            eprintln!("Rejected invalid flag value '{}' for {}", other, file_path);
            return false;
        }
    };

    let update_sql = "UPDATE images SET flag = ? WHERE file_path = ?";

    match conn.execute(update_sql, params![flag_value, file_path]) {
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
            eprintln!("Failed to update flag for {}: {}", file_path, e);
            false
        }
    }
}

/// Result of `relocate_file_path_prefix`. `ok` = the rewrite committed (or
/// there was nothing to move); `updated` = rows whose `file_path` and stored
/// `directory_path` were rewritten; `message` carries a short diagnostic on
/// failure (e.g. a UNIQUE collision when the new location overlaps another
/// cataloged root), empty on success.
pub struct RelocateResult
{
    pub ok: bool,
    pub updated: u64,
    pub message: String,
}

/// Re-point every catalogued row under `old_prefix` to `new_prefix` — a bulk
/// path-prefix rewrite for the Source-panel relocate feature (no re-scan).
/// Rewrites BOTH `file_path` AND the stored `directory_path` (the canonical
/// SUBSTRING/INSTR/REVERSE idiom) inside a transaction; any error — notably a
/// UNIQUE collision when `new_prefix` overlaps an existing cataloged root —
/// rolls the whole thing back and reports `ok = false`.
///
/// `old_prefix` / `new_prefix` are absolute root paths WITHOUT a trailing
/// slash; only rows strictly under them (`LIKE old_prefix || '/%'`) move, so a
/// sibling like `.../InPutTest2` is never caught by `.../InPutTest`. The SQL
/// was validated against a real 1,587-row catalogue copy before shipping
/// (S51 Gate 2).
pub async fn relocate_file_path_prefix(old_prefix: String, new_prefix: String) -> RelocateResult
{
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref()
    {
        Some(c) => c,
        None =>
        {
            eprintln!("relocate_file_path_prefix: catalogue not initialized");
            return RelocateResult { ok: false, updated: 0, message: "Catalogue not initialized".to_string() };
        }
    };

    if old_prefix.is_empty() || new_prefix.is_empty()
    {
        return RelocateResult { ok: false, updated: 0, message: "Empty source or destination prefix".to_string() };
    }
    if old_prefix == new_prefix
    {
        return RelocateResult { ok: true, updated: 0, message: String::new() };
    }

    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;")
    {
        eprintln!("relocate_file_path_prefix: begin failed: {}", e);
        return RelocateResult { ok: false, updated: 0, message: format!("begin failed: {}", e) };
    }

    // Rewrite file_path AND directory_path from the OLD file_path. The new-path
    // subexpression is repeated because a SET clause cannot reference a column
    // value being assigned in the same statement. ?1 = new_prefix, ?2 =
    // old_prefix (LENGTH(?2) drives the tail split; ?2 is reused in the WHERE).
    let update_sql = "UPDATE images \
        SET file_path = ?1 || SUBSTR(file_path, LENGTH(?2) + 1), \
            directory_path = SUBSTRING( \
                ?1 || SUBSTR(file_path, LENGTH(?2) + 1), 1, \
                LENGTH(?1 || SUBSTR(file_path, LENGTH(?2) + 1)) \
                    - INSTR(REVERSE(?1 || SUBSTR(file_path, LENGTH(?2) + 1)), '/')) \
        WHERE file_path LIKE ?2 || '/%'";

    let changed = match conn.execute(update_sql, params![new_prefix, old_prefix])
    {
        Ok(n) => n as u64,
        Err(e) =>
        {
            eprintln!("relocate_file_path_prefix: update failed: {}", e);
            let _ = conn.execute_batch("ROLLBACK;");
            return RelocateResult { ok: false, updated: 0, message: format!("rewrite failed (possible path collision): {}", e) };
        }
    };

    if let Err(e) = conn.execute_batch("COMMIT;")
    {
        eprintln!("relocate_file_path_prefix: commit failed: {}", e);
        let _ = conn.execute_batch("ROLLBACK;");
        return RelocateResult { ok: false, updated: 0, message: format!("commit failed: {}", e) };
    }

    eprintln!("relocate_file_path_prefix: moved {} rows '{}' -> '{}'", changed, old_prefix, new_prefix);
    RelocateResult { ok: true, updated: changed, message: String::new() }
}

/// Update the color label for an image
///
/// Sets the color label for an image identified by its file path. Mirrors
/// `update_image_rating`. Passing None clears the label (sets it to NULL).
///
/// A non-null value OUTSIDE the canonical set
/// {"red", "yellow", "green", "blue", "purple"} is REJECTED (returns false)
/// rather than silently written, so the indexed `color_label` column can never
/// hold a garbage value. This canonical set is also the stable interchange
/// vocabulary for a future Lightroom-catalogue import.
///
/// Returns:
/// - true if update succeeded
/// - false if catalogue not initialized, file not found, query failed, or an
///   invalid non-null value was supplied
pub async fn update_image_color_label(file_path: String, color_label: Option<String>) -> bool {
    // Acquire lock and validate connection
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return false;
        }
    };

    // Validate against the canonical set; None clears.
    let label_value: Option<String> = match color_label.as_deref() {
        None => None,
        Some(v @ ("red" | "yellow" | "green" | "blue" | "purple")) => Some(v.to_string()),
        Some(other) => {
            eprintln!("Rejected invalid color label '{}' for {}", other, file_path);
            return false;
        }
    };

    let update_sql = "UPDATE images SET color_label = ? WHERE file_path = ?";

    match conn.execute(update_sql, params![label_value, file_path]) {
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
            eprintln!("Failed to update color label for {}: {}", file_path, e);
            false
        }
    }
}

// ============================================================================
// Filter / Query Builder — structured query (see DESIGN-Filter-Query-Builder.md)
// ============================================================================

/// A boolean connector between two filter segments, applied LEFT-TO-RIGHT.
/// Variant ORDER must match the UDL `enum Connector` (UniFFI maps by position).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Connector
{
    And,
    Or,
    Xor,
}

/// One filter segment, flattened for the FFI. `kind` selects which fields are
/// meaningful; the Swift side holds the strong `FilterSegment` type and
/// serializes to this. See the UDL `dictionary QueryPredicate` for the
/// kind→fields map. (`PartialEq` is for the saved-query round-trip tests;
/// it is not part of the FFI surface.)
#[derive(Debug, Clone, PartialEq)]
pub struct QueryPredicate
{
    pub kind: String,
    pub day: Option<String>,
    pub day_end: Option<String>,
    pub op: Option<String>,
    pub stars: Option<u8>,
    pub value: Option<String>,
}

/// Validate a day string as exactly `YYYY:MM:DD` (10 chars; colons at index 4
/// and 7; digits elsewhere) — the form produced by `SUBSTRING(capture_datetime,
/// 1, 10)`. Because only digits + colons can pass, a validated day cannot carry
/// a SQL-injection payload (defense-in-depth on top of Swift-side validation).
fn is_valid_day(s: &str) -> bool
{
    let b = s.as_bytes();
    if b.len() != 10
    {
        return false;
    }
    for (i, &c) in b.iter().enumerate()
    {
        let ok = if i == 4 || i == 7 { c == b':' } else { c.is_ascii_digit() };
        if !ok
        {
            return false;
        }
    }
    true
}

fn is_valid_flag(s: &str) -> bool
{
    matches!(s, "pick" | "reject")
}

fn is_valid_color(s: &str) -> bool
{
    matches!(s, "red" | "yellow" | "green" | "blue" | "purple")
}

/// Escape a user-supplied string for safe use INSIDE a DuckDB `ILIKE` pattern
/// whose `ESCAPE` character is backslash: neutralize the `%` and `_` wildcards
/// (and the escape backslash itself) so typed text matches LITERALLY rather than
/// acting as a wildcard. The single-quote doubling for the surrounding SQL
/// string literal is applied separately by `filename_ilike_atom`.
fn escape_for_ilike(s: &str) -> String
{
    s.replace('\\', "\\\\").replace('%', "\\%").replace('_', "\\_")
}

/// Build a parenthesized, case-insensitive file-name match atom from an already
/// wildcard-escaped `ILIKE` pattern. Doubles single quotes for the SQL literal
/// and pins the `ESCAPE` character to backslash. `ILIKE` gives the case-
/// insensitivity the four File Name modes all require.
fn filename_ilike_atom(pattern: &str) -> String
{
    format!("(file_name ILIKE '{}' ESCAPE '\\')", pattern.replace('\'', "''"))
}

/// Map a rating comparison op token to its SQL symbol. Unknown → None.
fn sql_compare_op(op: &str) -> Option<&'static str>
{
    match op
    {
        "eq" => Some("="),
        "gt" => Some(">"),
        "lt" => Some("<"),
        "gte" => Some(">="),
        "lte" => Some("<="),
        _ => None,
    }
}

/// SQL for a connector. XOR is boolean inequality (`<>`) — exactly-one-true.
fn connector_sql(c: &Connector) -> &'static str
{
    match c
    {
        Connector::And => "AND",
        Connector::Or => "OR",
        Connector::Xor => "<>",
    }
}

/// Translate ONE filter segment into a parenthesized boolean SQL atom.
///
/// Every value is validated against its canonical set / format and single-
/// quote-escaped before interpolation; an invalid or malformed segment becomes
/// `(FALSE)` (matches nothing) rather than risking malformed or unsafe SQL.
/// Swift validates before sending, so `(FALSE)` is a defensive backstop.
fn predicate_to_sql(p: &QueryPredicate) -> String
{
    // A segment that should match nothing (invalid input backstop).
    let bad = || "(FALSE)".to_string();

    match p.kind.as_str()
    {
        "date_equals" => match p.day.as_deref()
        {
            Some(d) if is_valid_day(d) =>
                format!("(SUBSTRING(capture_datetime, 1, 10) = '{}')", d.replace('\'', "''")),
            _ => bad(),
        },
        "date_between" => match (p.day.as_deref(), p.day_end.as_deref())
        {
            (Some(a), Some(b)) if is_valid_day(a) && is_valid_day(b) => format!(
                "(SUBSTRING(capture_datetime, 1, 10) BETWEEN '{}' AND '{}')",
                a.replace('\'', "''"),
                b.replace('\'', "''")
            ),
            _ => bad(),
        },
        "date_after" => match p.day.as_deref() // on or after
        {
            Some(d) if is_valid_day(d) =>
                format!("(SUBSTRING(capture_datetime, 1, 10) >= '{}')", d.replace('\'', "''")),
            _ => bad(),
        },
        "date_before" => match p.day.as_deref() // on or before (<=)
        {
            Some(d) if is_valid_day(d) =>
                format!("(SUBSTRING(capture_datetime, 1, 10) <= '{}')", d.replace('\'', "''")),
            _ => bad(),
        },
        "date_gt" => match p.day.as_deref() // strictly after (>)
        {
            Some(d) if is_valid_day(d) =>
                format!("(SUBSTRING(capture_datetime, 1, 10) > '{}')", d.replace('\'', "''")),
            _ => bad(),
        },
        "date_lt" => match p.day.as_deref() // strictly before (<)
        {
            Some(d) if is_valid_day(d) =>
                format!("(SUBSTRING(capture_datetime, 1, 10) < '{}')", d.replace('\'', "''")),
            _ => bad(),
        },
        "rating" => match (p.op.as_deref(), p.stars)
        {
            (Some(op), Some(stars)) if (1..=5).contains(&stars) => match sql_compare_op(op)
            {
                Some(sym) => format!("(rating {} {})", sym, stars),
                None => bad(),
            },
            _ => bad(),
        },
        "rating_unrated" => "(rating IS NULL)".to_string(),
        "flag" => match p.value.as_deref()
        {
            Some(v) if is_valid_flag(v) => format!("(flag = '{}')", v.replace('\'', "''")),
            _ => bad(),
        },
        "flag_or_unflagged" => match p.value.as_deref()
        {
            Some(v) if is_valid_flag(v) =>
                format!("(flag = '{}' OR flag IS NULL)", v.replace('\'', "''")),
            _ => bad(),
        },
        "unflagged" => "(flag IS NULL)".to_string(),
        "color" => match p.value.as_deref()
        {
            Some(v) if is_valid_color(v) => format!("(color_label = '{}')", v.replace('\'', "''")),
            _ => bad(),
        },
        "no_color" => "(color_label IS NULL)".to_string(),
        // Keyword subject (Session 45). Label-equality is automatically
        // subtree-inclusive: every ancestor is its own materialized row carrying
        // its label, so `label = 'Animals'` matches everything beneath Animals.
        // Correlated on images.id against the active-only view. Isolation-first:
        // this pair of arms is the ONLY touch to the existing query engine.
        "keyword_has" => match p.value.as_deref()
        {
            Some(v) if !v.is_empty() => format!(
                "(EXISTS (SELECT 1 FROM keyword_visible k WHERE k.image_id = images.id AND k.label = '{}'))",
                v.replace('\'', "''")
            ),
            _ => bad(),
        },
        "keyword_not" => match p.value.as_deref()
        {
            Some(v) if !v.is_empty() => format!(
                "(NOT EXISTS (SELECT 1 FROM keyword_visible k WHERE k.image_id = images.id AND k.label = '{}'))",
                v.replace('\'', "''")
            ),
            _ => bad(),
        },
        // Collection subject (Session 63 — the gallery "Select Collection"
        // scope). Membership = a row in the RAW `keyword` table carrying this
        // label with the `collection` switch ON. Deliberately NOT the
        // keyword_visible view: collection and visibility are independent
        // switches on the row (the two-switch doctrine), so a search-hidden
        // keyword still counts as a collection member. Label equality is
        // exact-case ('Dogs' ≠ 'dogs' — identity is case-sensitive); the Swift
        // picker supplies exact stored labels. Multiple selected collections
        // arrive as one collection_is predicate each, joined with OR (union).
        "collection_is" => match p.value.as_deref()
        {
            Some(v) if !v.is_empty() => format!(
                "(EXISTS (SELECT 1 FROM keyword k WHERE k.image_id = images.id AND k.label = '{}' AND k.collection = TRUE))",
                v.replace('\'', "''")
            ),
            _ => bad(),
        },
        // File Name subject (Session 58). Four match modes against `file_name`
        // (the basename WITH extension), all case-insensitive via `ILIKE`; the
        // typed text rides in `value`. The text is wildcard-escaped so a literal
        // `%`/`_` matches itself, then quote-escaped for the SQL literal.
        "filename_contains" => match p.value.as_deref()
        {
            Some(v) if !v.is_empty() => filename_ilike_atom(&format!("%{}%", escape_for_ilike(v))),
            _ => bad(),
        },
        "filename_starts" => match p.value.as_deref()
        {
            Some(v) if !v.is_empty() => filename_ilike_atom(&format!("{}%", escape_for_ilike(v))),
            _ => bad(),
        },
        "filename_ends" => match p.value.as_deref()
        {
            Some(v) if !v.is_empty() => filename_ilike_atom(&format!("%{}", escape_for_ilike(v))),
            _ => bad(),
        },
        "filename_exact" => match p.value.as_deref()
        {
            Some(v) if !v.is_empty() => filename_ilike_atom(&escape_for_ilike(v)),
            _ => bad(),
        },
        other =>
        {
            eprintln!("Unknown query predicate kind '{}'", other);
            bad()
        }
    }
}

/// Assemble the filter segments into ONE predicate string, folding LEFT-TO-
/// RIGHT: `((A op B) op C) …` — no operator precedence, exactly as the filter
/// sentence reads. Empty when there are no predicates (→ all rows).
///
/// The whole accumulation is wrapped in an outer paren so the record/count
/// helpers can AND it with the RAW+JPEG-collapse predicate without precedence
/// surprises (AND binds tighter than OR; an unwrapped trailing OR would wrongly
/// bind the collapse predicate to only the last branch).
fn build_filter_predicate(predicates: &[QueryPredicate], connectors: &[Connector]) -> String
{
    if predicates.is_empty()
    {
        return String::new();
    }

    let mut acc = predicate_to_sql(&predicates[0]);
    for i in 1..predicates.len()
    {
        // connectors[i-1] joins the running result with segment i. Swift
        // sends predicates.len()-1 connectors; default to AND if short.
        let op = connectors.get(i - 1).map(connector_sql).unwrap_or("AND");
        let next = predicate_to_sql(&predicates[i]);
        acc = format!("({}) {} ({})", acc, op, next);
    }

    format!("({})", acc)
}

/// The Browse default row order: capture date newest-first. Used for the empty
/// filter and for any first-subject that doesn't define its own order.
const DEFAULT_FILTER_ORDER_BY: &str = "capture_datetime DESC NULLS LAST, created_timestamp DESC";

/// First-subject-rating order: stars high-to-low, then newest-first as the
/// tie-break (and a stable final key).
const RATING_FILTER_ORDER_BY: &str =
    "rating DESC NULLS LAST, capture_datetime DESC NULLS LAST, created_timestamp DESC";

/// Choose the `ORDER BY` for a filtered query from the FIRST predicate's
/// subject — the Session-44 "first filter subject drives the sort" rule:
///   - rating-first  → stars best-to-worst (`rating DESC`)
///   - date-first    → newest-first (the default — `capture_datetime DESC`)
///   - flag/color-first, or no filter → the default (newest-first)
///
/// This only affects WHICH rows land on each page and the default display
/// order; the Swift side still allows page-local column re-sorting on top.
/// `count_query_images` needs no order, so it is unaffected (count/page parity
/// is about the WHERE, not the ORDER BY).
fn order_by_for_filter(predicates: &[QueryPredicate]) -> &'static str
{
    match predicates.first()
    {
        Some(p) => match p.kind.as_str()
        {
            "rating" | "rating_unrated" => RATING_FILTER_ORDER_BY,
            _ => DEFAULT_FILTER_ORDER_BY,
        },
        None => DEFAULT_FILTER_ORDER_BY,
    }
}

/// Filter / Query Builder — paginated records matching a structured filter.
///
/// Builds ONE predicate string (left-to-right connectors) and delegates to the
/// shared `execute_image_record_query` helper — the SAME helper every other
/// record query uses — so count/page parity with `count_query_images` holds by
/// construction. Default sort. The two filter booleans compose orthogonally,
/// unchanged. Empty predicate list → all rows.
pub async fn query_images(
    predicates: Vec<QueryPredicate>,
    connectors: Vec<Connector>,
    limit: u32,
    offset: u32,
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
) -> Vec<ImageRecord>
{
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

    let where_clause = build_filter_predicate(&predicates, &connectors);
    let order_by = order_by_for_filter(&predicates);

    execute_image_record_query(
        conn,
        &where_clause,
        order_by,
        limit as i64,
        offset as i64,
        apply_duplicate_filter,
        apply_raw_jpeg_collapse,
    )
}

/// Filter / Query Builder — total matching count for the SAME filter, via the
/// shared `execute_image_count_query` helper (parity by construction).
pub async fn count_query_images(
    predicates: Vec<QueryPredicate>,
    connectors: Vec<Connector>,
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
) -> u64
{
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

    let where_clause = build_filter_predicate(&predicates, &connectors);

    let count = execute_image_count_query(
        conn,
        &where_clause,
        apply_duplicate_filter,
        apply_raw_jpeg_collapse,
    );

    if count < 0 { 0 } else { count as u64 }
}

// ============================================================================
// Browse multi-select — selection + bulk actions (Session 44)
//   - query_image_ids:        the IDs of every row matching a filter (⌘A)
//   - get_images_by_ids:      resolve a selection (any IDs, even cross-page or
//                             whole-query) to full records for Copy / Reveal
//   - update_*_for_ids:       bulk curation in ONE statement (so "select the
//                             whole query → Set Flag" isn't N round-trips)
// ============================================================================

/// Build a safe `id IN (...)` predicate from i64 IDs (predicate text only — no
/// "WHERE", per the projection-helper convention). The IDs are integers, so
/// direct interpolation carries no injection risk. `None` for an empty slice —
/// the caller MUST short-circuit, since `IN ()` is a syntax error.
fn id_in_list(ids: &[i64]) -> Option<String>
{
    if ids.is_empty()
    {
        return None;
    }
    let joined = ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(", ");
    Some(format!("id IN ({})", joined))
}

/// ID projection — the `id` analogue of `execute_file_path_projection_query`.
/// Same filter machinery (inner WHERE + the duplicate-filter subquery wrap), but
/// SELECTs `id`, so ⌘A can enumerate every matching row's ID cheaply without
/// materializing full records.
fn execute_id_projection_query(
    conn: &Connection,
    where_clause: &str,
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
) -> Vec<i64>
{
    let mut inner_predicates: Vec<&str> = Vec::new();
    if !where_clause.is_empty()
    {
        inner_predicates.push(where_clause);
    }
    if apply_raw_jpeg_collapse
    {
        inner_predicates.push(RAW_JPEG_COLLAPSE_PREDICATE);
    }
    let inner_where = if inner_predicates.is_empty()
    {
        String::new()
    }
    else
    {
        format!("WHERE {}", inner_predicates.join(" AND "))
    };

    let query_sql = if apply_duplicate_filter
    {
        format!(r#"
            SELECT id FROM (
                SELECT
                    id,
                    {}
                FROM images
                {}
            )
            WHERE {}
        "#,
            DUPLICATE_GROUP_ID_CASE,
            inner_where,
            DUPLICATE_FILTER_PREDICATE)
    }
    else
    {
        format!("SELECT id FROM images {}", inner_where)
    };

    let mut stmt = match conn.prepare(&query_sql)
    {
        Ok(s) => s,
        Err(e) =>
        {
            eprintln!("Failed to prepare id projection query: {}", e);
            return Vec::new();
        }
    };

    let rows = match stmt.query_map([], |row| row.get::<_, i64>(0))
    {
        Ok(r) => r,
        Err(e) =>
        {
            eprintln!("Failed to execute id projection query: {}", e);
            return Vec::new();
        }
    };

    let mut ids: Vec<i64> = Vec::new();
    for row_result in rows
    {
        match row_result
        {
            Ok(id) => ids.push(id),
            Err(e) => eprintln!("Failed to read id row: {}", e),
        }
    }

    ids
}

/// All matching record IDs for a structured filter — powers ⌘A "select the
/// whole query" on Browse. Same predicate + two-boolean machinery as
/// `query_images`, so the selected set is EXACTLY the rows the filtered table
/// shows. Empty predicate list → every row (subject to the toggles).
pub async fn query_image_ids(
    predicates: Vec<QueryPredicate>,
    connectors: Vec<Connector>,
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
) -> Vec<i64>
{
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

    let where_clause = build_filter_predicate(&predicates, &connectors);
    execute_id_projection_query(conn, &where_clause, apply_duplicate_filter, apply_raw_jpeg_collapse)
}

/// Resolve record IDs to full `ImageRecord`s — turns a Browse selection (which
/// may span pages, or be the whole query) into records for Copy / Reveal.
/// The IDs ARE the exact selection, so NO duplicate / raw-collapse filtering is
/// applied. Order is unspecified (the copy planner re-sorts). Empty → empty.
pub async fn get_images_by_ids(ids: Vec<i64>) -> Vec<ImageRecord>
{
    let where_clause = match id_in_list(&ids)
    {
        Some(w) => w,
        None => return Vec::new(),
    };

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

    execute_image_record_projection_query(conn, &where_clause, false, false)
}

/// Expand a set of visible record IDs to their RAW+JPEG/HEIF collapse-group:
/// the input IDs PLUS any hidden RAW siblings sharing `(file_stem,
/// directory_path)`. Used when the collapse toggle is ON so an action on a
/// collapsed row (curation, copy) reaches the hidden RAW too.
///
/// Adds ONLY RAW rows — exactly what `RAW_JPEG_COLLAPSE_PREDICATE` hides — so
/// two same-stem JPEGs with no RAW are never falsely merged. Deduped by UNION;
/// order unspecified. On any error it falls back to the input IDs (degraded:
/// the action still hits the visible rows, just not the RAW). Empty → empty.
pub async fn expand_collapse_group_ids(ids: Vec<i64>) -> Vec<i64>
{
    if ids.is_empty()
    {
        return Vec::new();
    }
    let csv = ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(", ");

    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref()
    {
        Some(c) => c,
        None =>
        {
            eprintln!("Catalogue not initialized");
            return ids;
        }
    };

    let query_sql = format!(
        "SELECT id FROM images WHERE id IN ({csv}) \
         UNION \
         SELECT r.id FROM images r \
         WHERE r.image_kind = 'raw' AND EXISTS ( \
             SELECT 1 FROM images s \
             WHERE s.id IN ({csv}) \
               AND s.file_stem = r.file_stem \
               AND s.directory_path = r.directory_path \
         )",
        csv = csv
    );

    let mut stmt = match conn.prepare(&query_sql)
    {
        Ok(s) => s,
        Err(e) =>
        {
            eprintln!("Failed to prepare collapse-group expansion: {}", e);
            return ids;
        }
    };

    let rows = match stmt.query_map([], |row| row.get::<_, i64>(0))
    {
        Ok(r) => r,
        Err(e) =>
        {
            eprintln!("Failed to execute collapse-group expansion: {}", e);
            return ids;
        }
    };

    let mut result: Vec<i64> = Vec::new();
    for row_result in rows
    {
        if let Ok(id) = row_result
        {
            result.push(id);
        }
    }

    if result.is_empty() { ids } else { result }
}

// === Keyword system (Session 45) ===
//
// A hierarchical keyword subsystem in a SINGLE `keyword` table (see
// Docs/DESIGN-Keyword-System.md). Each applied keyword path is materialized as
// one row per ancestor level; each row's `label` is that node and `path` is the
// chain root->that node, joined by U+001F. Soft-hide via `status` (1 active, 0
// hidden). Reads go through the `keyword_visible` view; the raw table is the
// recovery surface. Isolation-first: these are all brand-new functions; the
// only touch to existing code is the additive `keyword_has`/`keyword_not` arm
// in `predicate_to_sql`.

/// The path-segment separator: U+001F (ASCII Unit Separator). Non-printing, so
/// it can never collide with human-typed keyword text — no escaping needed. The
/// UI renders a visible glyph (e.g. "›") in its place.
const KEYWORD_PATH_SEPARATOR: &str = "\u{001F}";

/// A single materialized keyword row, as returned to Swift.
#[derive(Debug, Clone)]
pub struct KeywordRow
{
    pub label: String,
    pub path: String,
    pub status: i32,
    pub created_at: String,
    pub hidden_at: Option<String>,
}

/// A distinct (label, path) node — the vocabulary, for autocomplete + browsing.
#[derive(Debug, Clone)]
pub struct KeywordNode
{
    pub label: String,
    pub path: String,
}

/// Materialize an ordered segment list into one (label, path) pair per ancestor
/// depth. `["Animals","Dog","Lab"]` -> `[("Animals","Animals"),
/// ("Dog","Animals␟Dog"), ("Lab","Animals␟Dog␟Lab")]`. Returns empty if any
/// segment is blank or itself contains the separator (-> caller no-ops).
fn keyword_materialized_rows(segments: &[String]) -> Vec<(String, String)>
{
    let mut rows: Vec<(String, String)> = Vec::new();
    let mut prefix: Vec<String> = Vec::new();
    for seg in segments
    {
        let trimmed = seg.trim();
        if trimmed.is_empty() || trimmed.contains(KEYWORD_PATH_SEPARATOR)
        {
            return Vec::new();
        }
        prefix.push(trimmed.to_string());
        rows.push((trimmed.to_string(), prefix.join(KEYWORD_PATH_SEPARATOR)));
    }
    rows
}

/// `image_id IN (...)` for the keyword table — sibling of `id_in_list`, which is
/// hard-coded to the `images.id` column. `None` on empty.
fn keyword_image_id_in_list(ids: &[i64]) -> Option<String>
{
    if ids.is_empty()
    {
        return None;
    }
    let joined = ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(", ");
    Some(format!("image_id IN ({})", joined))
}

/// Assign a keyword PATH to many images. Rust materializes the ancestor chain
/// and inserts one row per depth for each image. Blind-insert, except it skips a
/// row byte-identical to an already-ACTIVE row for that image (so a double-apply
/// doesn't spam duplicates). A previously-removed (hidden) identical row is NOT
/// resurrected — a fresh active row is inserted, preserving history. One
/// transaction. Returns the number of rows inserted.
pub async fn assign_keyword_for_ids(ids: Vec<i64>, segments: Vec<String>) -> u64
{
    if ids.is_empty()
    {
        return 0;
    }
    let rows = keyword_materialized_rows(&segments);
    if rows.is_empty()
    {
        eprintln!("assign_keyword_for_ids: empty or invalid segments");
        return 0;
    }

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

    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;")
    {
        eprintln!("assign_keyword_for_ids: begin failed: {}", e);
        return 0;
    }

    let mut inserted: u64 = 0;
    for id in &ids
    {
        for (label, path) in &rows
        {
            let existing: Result<i64, _> = conn.query_row(
                "SELECT 1 FROM keyword WHERE image_id = ? AND path = ? AND status = 1 LIMIT 1",
                params![id, path],
                |r| r.get(0),
            );
            if existing.is_ok()
            {
                continue;
            }
            match conn.execute(
                "INSERT INTO keyword (image_id, label, path, status, created_at) \
                 VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)",
                params![id, label, path],
            )
            {
                Ok(_) => inserted += 1,
                Err(e) =>
                {
                    eprintln!("assign_keyword_for_ids: insert failed: {}", e);
                    let _ = conn.execute_batch("ROLLBACK;");
                    return 0;
                }
            }
        }
    }

    if let Err(e) = conn.execute_batch("COMMIT;")
    {
        eprintln!("assign_keyword_for_ids: commit failed: {}", e);
        return 0;
    }
    inserted
}

/// Remove a keyword from many images — soft-hide the node AND its descendants
/// (`path = ? OR starts_with(path, ?␟)`) for those images. Ancestors are LEFT
/// intact (a lone parent is a valid flat keyword). Returns rows hidden.
pub async fn remove_keyword_for_ids(ids: Vec<i64>, path: String) -> u64
{
    if ids.is_empty() || path.is_empty()
    {
        return 0;
    }
    let where_ids = match keyword_image_id_in_list(&ids)
    {
        Some(w) => w,
        None => return 0,
    };

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

    let prefix = format!("{}{}", path, KEYWORD_PATH_SEPARATOR);
    let sql = format!(
        "UPDATE keyword SET status = 0, hidden_at = CURRENT_TIMESTAMP \
         WHERE status = 1 AND {} AND (path = ? OR starts_with(path, ?))",
        where_ids
    );
    match conn.execute(&sql, params![path, prefix])
    {
        Ok(changed) => changed as u64,
        Err(e) =>
        {
            eprintln!("remove_keyword_for_ids: {}", e);
            0
        }
    }
}

/// Restore (un-hide) a previously removed keyword node + descendants for many
/// images — undo of `remove_keyword_for_ids` and the recovery-screen action.
pub async fn restore_keyword_for_ids(ids: Vec<i64>, path: String) -> u64
{
    if ids.is_empty() || path.is_empty()
    {
        return 0;
    }
    let where_ids = match keyword_image_id_in_list(&ids)
    {
        Some(w) => w,
        None => return 0,
    };

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

    let prefix = format!("{}{}", path, KEYWORD_PATH_SEPARATOR);
    let sql = format!(
        "UPDATE keyword SET status = 1, hidden_at = NULL \
         WHERE status = 0 AND {} AND (path = ? OR starts_with(path, ?))",
        where_ids
    );
    match conn.execute(&sql, params![path, prefix])
    {
        Ok(changed) => changed as u64,
        Err(e) =>
        {
            eprintln!("restore_keyword_for_ids: {}", e);
            0
        }
    }
}

/// All ACTIVE keyword rows for one image, ordered by path (root->leaf within a
/// branch). For the detail-panel reconstruction.
pub async fn keywords_for_image(image_id: i64) -> Vec<KeywordRow>
{
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

    let mut stmt = match conn.prepare(
        "SELECT label, path, status, CAST(created_at AS VARCHAR), CAST(hidden_at AS VARCHAR) \
         FROM keyword_visible WHERE image_id = ? ORDER BY path",
    )
    {
        Ok(s) => s,
        Err(e) =>
        {
            eprintln!("keywords_for_image: prepare {}", e);
            return Vec::new();
        }
    };

    let mapped = stmt.query_map(params![image_id], |row|
    {
        Ok(KeywordRow {
            label: row.get(0)?,
            path: row.get(1)?,
            status: row.get(2)?,
            created_at: row.get(3)?,
            hidden_at: row.get(4)?,
        })
    });

    match mapped
    {
        Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
        Err(e) =>
        {
            eprintln!("keywords_for_image: query {}", e);
            Vec::new()
        }
    }
}

/// The DISTINCT (label, path) keyword vocabulary over the active view — for the
/// assignment-panel autocomplete and (future) tree browser. Ordered by path.
pub async fn keyword_vocabulary() -> Vec<KeywordNode>
{
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

    let mut stmt = match conn.prepare("SELECT DISTINCT label, path FROM keyword_visible ORDER BY path")
    {
        Ok(s) => s,
        Err(e) =>
        {
            eprintln!("keyword_vocabulary: prepare {}", e);
            return Vec::new();
        }
    };

    let mapped = stmt.query_map([], |row|
    {
        Ok(KeywordNode { label: row.get(0)?, path: row.get(1)? })
    });

    match mapped
    {
        Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
        Err(e) =>
        {
            eprintln!("keyword_vocabulary: query {}", e);
            Vec::new()
        }
    }
}

/// Distinct VISIBLE keyword labels (case-sensitive, alphabetical) — the source for
/// the Collection "Add" dialog's autofill + dropdown. Deliberately ALL labels,
/// regardless of the `collection` flag, so any keyword can seed a collection and a
/// dead collection's name still suggests itself. (The Collection TAB picker filters
/// to `collection = TRUE` instead — a separate read, added with the tab.)
pub async fn keyword_labels() -> Vec<String>
{
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

    let mut stmt = match conn.prepare("SELECT DISTINCT label FROM keyword_visible ORDER BY label")
    {
        Ok(s) => s,
        Err(e) =>
        {
            eprintln!("keyword_labels: prepare {}", e);
            return Vec::new();
        }
    };

    let mapped = stmt.query_map([], |row|
    {
        let label: String = row.get(0)?;
        Ok(label)
    });

    match mapped
    {
        Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
        Err(e) =>
        {
            eprintln!("keyword_labels: query {}", e);
            Vec::new()
        }
    }
}

// === Saved Queries (Session 63; Docs/DESIGN-Saved-Queries.md) =============
// The recipe, never the results: a header row (saved_query) + one
// saved_query_criterion row per segment, mirroring the QueryPredicate wire
// struct plus placement (position + the connector joining the row leftward).
// Impl fns take &Connection so the unit tests can drive them on an in-memory
// database; the pub async FFI wrappers below just lock CATALOGUE.

/// One saved query's identity (header row), returned to Swift.
#[derive(Debug, Clone, PartialEq)]
pub struct SavedQueryInfo
{
    pub id: i64,
    pub name: String,
}

/// A loaded saved query: the same two arrays `query_images` consumes — load,
/// hand to the sheet, run. Parity by construction with the live builder.
#[derive(Debug, Clone, PartialEq)]
pub struct SavedQueryPayload
{
    pub predicates: Vec<QueryPredicate>,
    pub connectors: Vec<Connector>,
}

/// Storage text for a connector (saved_query_criterion.connector).
fn connector_to_text(c: &Connector) -> &'static str
{
    match c
    {
        Connector::And => "and",
        Connector::Or => "or",
        Connector::Xor => "xor",
    }
}

/// Inverse of `connector_to_text`. Unknown/garbled → AND (the builder's
/// default — same forgiveness as `build_filter_predicate`'s short-array rule).
fn connector_from_text(s: &str) -> Connector
{
    match s
    {
        "or" => Connector::Or,
        "xor" => Connector::Xor,
        _ => Connector::And,
    }
}

/// Save a Find in Gallery sentence under `name`. Collision policy (Richard's
/// rule, S63): an existing name gains a numeric suffix — "Dogs" → "Dogs-01" →
/// "Dogs-02" … — never a replace, never a prompt. The suffixing lives HERE so
/// there is exactly one source of truth for it (and it runs under the
/// CATALOGUE lock, so two saves can't race to the same name). Returns the
/// header carrying the FINAL (possibly suffixed) name, or None on invalid
/// input / DB failure. One transaction.
fn save_query_impl(conn: &Connection, name: &str,
                   predicates: &[QueryPredicate], connectors: &[Connector]) -> Option<SavedQueryInfo>
{
    let base = name.trim();
    if base.is_empty() || predicates.is_empty()
    {
        return None;
    }

    let exists = |n: &str| -> bool
    {
        conn.query_row("SELECT COUNT(*) FROM saved_query WHERE name = ?", [n],
                       |row| row.get::<_, i64>(0))
            .unwrap_or(0) > 0
    };

    let mut final_name = base.to_string();
    if exists(&final_name)
    {
        let mut n = 1;
        loop
        {
            if n > 999
            {
                eprintln!("save_query: suffix space exhausted for '{}'", base);
                return None;
            }
            let candidate = format!("{}-{:02}", base, n);
            if !exists(&candidate)
            {
                final_name = candidate;
                break;
            }
            n += 1;
        }
    }

    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;")
    {
        eprintln!("save_query: begin failed: {}", e);
        return None;
    }

    if let Err(e) = conn.execute("INSERT INTO saved_query (name) VALUES (?)", [&final_name])
    {
        eprintln!("save_query: insert header failed: {}", e);
        let _ = conn.execute_batch("ROLLBACK;");
        return None;
    }

    // The name is unique by construction (checked above, under the lock), so
    // it safely keys the id read-back.
    let id: i64 = match conn.query_row("SELECT id FROM saved_query WHERE name = ?",
                                       [&final_name], |row| row.get(0))
    {
        Ok(v) => v,
        Err(e) =>
        {
            eprintln!("save_query: id read-back failed: {}", e);
            let _ = conn.execute_batch("ROLLBACK;");
            return None;
        }
    };

    for (i, p) in predicates.iter().enumerate()
    {
        // Row 1 has nothing to its left; row i (1-based) joins via connectors[i-2]
        // (Swift sends predicates.len()-1 connectors; default AND if short —
        // the same rule build_filter_predicate applies at query time).
        let connector_text: Option<&str> = if i == 0
        {
            None
        }
        else
        {
            Some(connector_to_text(connectors.get(i - 1).unwrap_or(&Connector::And)))
        };

        if let Err(e) = conn.execute(
            "INSERT INTO saved_query_criterion \
             (query_id, position, connector, kind, op, value, day, day_end, stars) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![id, (i + 1) as i64, connector_text, p.kind, p.op, p.value,
                    p.day, p.day_end, p.stars.map(|s| s as i32)],
        )
        {
            eprintln!("save_query: insert criterion failed: {}", e);
            let _ = conn.execute_batch("ROLLBACK;");
            return None;
        }
    }

    if let Err(e) = conn.execute_batch("COMMIT;")
    {
        eprintln!("save_query: commit failed: {}", e);
        let _ = conn.execute_batch("ROLLBACK;");
        return None;
    }

    Some(SavedQueryInfo { id, name: final_name })
}

/// All saved queries, ordered by name (case-folded) for the picker list.
fn list_saved_queries_impl(conn: &Connection) -> Vec<SavedQueryInfo>
{
    let mut stmt = match conn.prepare("SELECT id, name FROM saved_query ORDER BY LOWER(name), id")
    {
        Ok(s) => s,
        Err(e) =>
        {
            eprintln!("list_saved_queries: prepare {}", e);
            return Vec::new();
        }
    };

    let mapped = stmt.query_map([], |row|
    {
        Ok(SavedQueryInfo { id: row.get(0)?, name: row.get(1)? })
    });

    match mapped
    {
        Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
        Err(e) =>
        {
            eprintln!("list_saved_queries: query {}", e);
            Vec::new()
        }
    }
}

/// Load a saved query's criterion rows back into the two arrays the builder
/// (and `query_images`) consume. None for an unknown id or an empty recipe.
fn load_saved_query_impl(conn: &Connection, id: i64) -> Option<SavedQueryPayload>
{
    let mut stmt = match conn.prepare(
        "SELECT position, connector, kind, op, value, day, day_end, stars \
         FROM saved_query_criterion WHERE query_id = ? ORDER BY position")
    {
        Ok(s) => s,
        Err(e) =>
        {
            eprintln!("load_saved_query: prepare {}", e);
            return None;
        }
    };

    let mapped = stmt.query_map([id], |row|
    {
        Ok((
            row.get::<_, i64>(0)?,            // position
            row.get::<_, Option<String>>(1)?, // connector
            row.get::<_, String>(2)?,         // kind
            row.get::<_, Option<String>>(3)?, // op
            row.get::<_, Option<String>>(4)?, // value
            row.get::<_, Option<String>>(5)?, // day
            row.get::<_, Option<String>>(6)?, // day_end
            row.get::<_, Option<i32>>(7)?,    // stars
        ))
    });

    let rows = match mapped
    {
        Ok(iter) => iter.filter_map(|r| r.ok()),
        Err(e) =>
        {
            eprintln!("load_saved_query: query {}", e);
            return None;
        }
    };

    let mut predicates: Vec<QueryPredicate> = Vec::new();
    let mut connectors: Vec<Connector> = Vec::new();
    for (position, connector, kind, op, value, day, day_end, stars) in rows
    {
        if position > 1
        {
            connectors.push(connector_from_text(connector.as_deref().unwrap_or("and")));
        }
        predicates.push(QueryPredicate {
            kind,
            day,
            day_end,
            op,
            stars: stars.map(|s| s as u8),
            value,
        });
    }

    if predicates.is_empty()
    {
        return None;
    }
    Some(SavedQueryPayload { predicates, connectors })
}

/// Delete a saved query (header + criterion rows, one transaction). Returns
/// whether a header row was actually removed.
fn delete_saved_query_impl(conn: &Connection, id: i64) -> bool
{
    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;")
    {
        eprintln!("delete_saved_query: begin failed: {}", e);
        return false;
    }

    if let Err(e) = conn.execute("DELETE FROM saved_query_criterion WHERE query_id = ?", [id])
    {
        eprintln!("delete_saved_query: criteria delete failed: {}", e);
        let _ = conn.execute_batch("ROLLBACK;");
        return false;
    }

    let removed = match conn.execute("DELETE FROM saved_query WHERE id = ?", [id])
    {
        Ok(n) => n,
        Err(e) =>
        {
            eprintln!("delete_saved_query: header delete failed: {}", e);
            let _ = conn.execute_batch("ROLLBACK;");
            return false;
        }
    };

    if let Err(e) = conn.execute_batch("COMMIT;")
    {
        eprintln!("delete_saved_query: commit failed: {}", e);
        return false;
    }

    removed > 0
}

/// FFI: save the current Find in Gallery sentence under `name` (suffixing a
/// colliding name per the S63 policy). Returns the header with the FINAL name.
pub async fn save_query(name: String, predicates: Vec<QueryPredicate>,
                        connectors: Vec<Connector>) -> Option<SavedQueryInfo>
{
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
    save_query_impl(conn, &name, &predicates, &connectors)
}

/// FFI: all saved queries (id + name), name-ordered, for the picker.
pub async fn list_saved_queries() -> Vec<SavedQueryInfo>
{
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
    list_saved_queries_impl(conn)
}

/// FFI: load a saved query back into builder arrays. None if id unknown.
pub async fn load_saved_query(id: i64) -> Option<SavedQueryPayload>
{
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
    load_saved_query_impl(conn, id)
}

/// FFI: delete a saved query. True if it existed.
pub async fn delete_saved_query(id: i64) -> bool
{
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
    delete_saved_query_impl(conn, id)
}

/// Distinct collection names — labels carrying `collection = TRUE` on any row,
/// read from the RAW `keyword` table (membership is independent of search
/// visibility). The gallery "Select Collection" picker's autofill/dropdown —
/// collections ONLY, the deliberate opposite of `keyword_labels` (the Add
/// dialog's flag-agnostic list). A collection whose members were all removed
/// (every `collection` switch flipped back FALSE) simply doesn't appear here;
/// its label still re-suggests in the Add dialog and re-creates by name.
pub async fn collection_labels() -> Vec<String>
{
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

    let mut stmt = match conn.prepare("SELECT DISTINCT label FROM keyword WHERE collection = TRUE ORDER BY label")
    {
        Ok(s) => s,
        Err(e) =>
        {
            eprintln!("collection_labels: prepare {}", e);
            return Vec::new();
        }
    };

    let mapped = stmt.query_map([], |row|
    {
        let label: String = row.get(0)?;
        Ok(label)
    });

    match mapped
    {
        Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
        Err(e) =>
        {
            eprintln!("collection_labels: query {}", e);
            Vec::new()
        }
    }
}

/// Add images to one or more collections — the right-click → Collection "Apply".
/// A collection IS a `keyword` label carrying `collection = TRUE`, so for each
/// image x label: flip the `collection` switch ON for any existing VISIBLE row with
/// that label (a flat keyword OR the leaf of a hierarchical one — the same record
/// serves both keyword and collection); if the image has no visible row with that
/// label, insert a FLAT row (path = label) with `collection = TRUE`. Idempotent
/// (already a member -> no-op). `status`/visibility is never touched — `collection`
/// and visible are independent switches on the row. One transaction. Returns the
/// number of rows changed (flipped + inserted).
pub async fn add_images_to_collections(ids: Vec<i64>, labels: Vec<String>) -> u64
{
    if ids.is_empty()
    {
        return 0;
    }

    // A collection name is a single flat segment — trim, drop empties and any that
    // carry the path separator (mirrors keyword_materialized_rows), then de-dupe.
    let mut clean: Vec<String> = Vec::new();
    for label in &labels
    {
        let trimmed = label.trim();
        if trimmed.is_empty() || trimmed.contains(KEYWORD_PATH_SEPARATOR)
        {
            continue;
        }
        let s = trimmed.to_string();
        if !clean.contains(&s)
        {
            clean.push(s);
        }
    }
    if clean.is_empty()
    {
        return 0;
    }

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

    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;")
    {
        eprintln!("add_images_to_collections: begin failed: {}", e);
        return 0;
    }

    let mut changed: u64 = 0;
    for id in &ids
    {
        for label in &clean
        {
            // Flip ON any existing visible row(s) with this label that aren't
            // already a collection (FALSE or, on migrated catalogues, NULL).
            let flipped = match conn.execute(
                "UPDATE keyword SET collection = TRUE \
                 WHERE image_id = ? AND label = ? AND status = 1 \
                   AND (collection = FALSE OR collection IS NULL)",
                params![id, label],
            )
            {
                Ok(n) => n as u64,
                Err(e) =>
                {
                    eprintln!("add_images_to_collections: update failed: {}", e);
                    let _ = conn.execute_batch("ROLLBACK;");
                    return 0;
                }
            };
            if flipped > 0
            {
                changed += flipped;
                continue;
            }

            // Already a member (a visible row with collection already TRUE)? no-op.
            let already: Result<i64, _> = conn.query_row(
                "SELECT 1 FROM keyword \
                 WHERE image_id = ? AND label = ? AND status = 1 AND collection = TRUE LIMIT 1",
                params![id, label],
                |r| r.get(0),
            );
            if already.is_ok()
            {
                continue;
            }

            // No visible row with this label → insert a flat collection row.
            match conn.execute(
                "INSERT INTO keyword (image_id, label, path, status, created_at, collection) \
                 VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP, TRUE)",
                params![id, label, label],
            )
            {
                Ok(_) => changed += 1,
                Err(e) =>
                {
                    eprintln!("add_images_to_collections: insert failed: {}", e);
                    let _ = conn.execute_batch("ROLLBACK;");
                    return 0;
                }
            }
        }
    }

    if let Err(e) = conn.execute_batch("COMMIT;")
    {
        eprintln!("add_images_to_collections: commit failed: {}", e);
        return 0;
    }
    changed
}

/// Hidden (removed) keyword rows for one image — the recovery surface. Reads the
/// RAW table (not the view), newest-hidden first.
pub async fn hidden_keywords_for_image(image_id: i64) -> Vec<KeywordRow>
{
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

    let mut stmt = match conn.prepare(
        "SELECT label, path, status, CAST(created_at AS VARCHAR), CAST(hidden_at AS VARCHAR) \
         FROM keyword WHERE image_id = ? AND status = 0 ORDER BY hidden_at DESC",
    )
    {
        Ok(s) => s,
        Err(e) =>
        {
            eprintln!("hidden_keywords_for_image: prepare {}", e);
            return Vec::new();
        }
    };

    let mapped = stmt.query_map(params![image_id], |row|
    {
        Ok(KeywordRow {
            label: row.get(0)?,
            path: row.get(1)?,
            status: row.get(2)?,
            created_at: row.get(3)?,
            hidden_at: row.get(4)?,
        })
    });

    match mapped
    {
        Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
        Err(e) =>
        {
            eprintln!("hidden_keywords_for_image: query {}", e);
            Vec::new()
        }
    }
}

/// Re-parent a keyword node (and its whole subtree) GLOBALLY — for every image
/// that carries it. (1) Materializes the new-parent ancestor chain for affected
/// images; (2) re-roots the moved subtree under `<new_parent> ␟ <last source
/// segment>` (labels unchanged, paths re-rooted); (3) hides the old subtree.
/// Empty `new_parent` moves the node to the top level. One transaction. Returns
/// the number of old rows hidden.
pub async fn reparent_keyword(source_path: Vec<String>, new_parent: Vec<String>) -> u64
{
    if source_path.is_empty()
    {
        return 0;
    }
    for s in source_path.iter().chain(new_parent.iter())
    {
        if s.trim().is_empty() || s.contains(KEYWORD_PATH_SEPARATOR)
        {
            eprintln!("reparent_keyword: invalid segment");
            return 0;
        }
    }

    let source_joined = source_path.join(KEYWORD_PATH_SEPARATOR);
    let source_prefix = format!("{}{}", source_joined, KEYWORD_PATH_SEPARATOR);
    let last_seg = source_path.last().unwrap().trim().to_string();
    let new_root = if new_parent.is_empty()
    {
        last_seg
    }
    else
    {
        format!("{}{}{}", new_parent.join(KEYWORD_PATH_SEPARATOR), KEYWORD_PATH_SEPARATOR, last_seg)
    };
    // char count + 1 = 1-indexed position of the first char AFTER source_joined
    // ("" for the subtree root, "␟Yellow" for a descendant).
    let suffix_start = (source_joined.chars().count() + 1) as i64;
    let new_parent_rows = keyword_materialized_rows(&new_parent);

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

    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;")
    {
        eprintln!("reparent_keyword: begin failed: {}", e);
        return 0;
    }

    // (1) Ensure the new-parent ancestor chain exists for every affected image.
    for (label, path) in &new_parent_rows
    {
        let sql = "INSERT INTO keyword (image_id, label, path, status, created_at) \
                   SELECT DISTINCT image_id, ?, ?, 1, CURRENT_TIMESTAMP FROM keyword \
                   WHERE status = 1 AND (path = ? OR starts_with(path, ?)) \
                   AND image_id NOT IN (SELECT image_id FROM keyword WHERE status = 1 AND path = ?)";
        if let Err(e) = conn.execute(sql, params![label, path, source_joined, source_prefix, path])
        {
            eprintln!("reparent_keyword: ancestor insert failed: {}", e);
            let _ = conn.execute_batch("ROLLBACK;");
            return 0;
        }
    }

    // (2) Re-root the moved subtree (label unchanged; path = new_root || suffix).
    let move_sql = "INSERT INTO keyword (image_id, label, path, status, created_at) \
                    SELECT image_id, label, ? || substr(path, ?), 1, CURRENT_TIMESTAMP FROM keyword \
                    WHERE status = 1 AND (path = ? OR starts_with(path, ?))";
    if let Err(e) = conn.execute(move_sql, params![new_root, suffix_start, source_joined, source_prefix])
    {
        eprintln!("reparent_keyword: move insert failed: {}", e);
        let _ = conn.execute_batch("ROLLBACK;");
        return 0;
    }

    // (3) Hide the old subtree.
    let hide_sql = "UPDATE keyword SET status = 0, hidden_at = CURRENT_TIMESTAMP \
                    WHERE status = 1 AND (path = ? OR starts_with(path, ?))";
    let changed = match conn.execute(hide_sql, params![source_joined, source_prefix])
    {
        Ok(c) => c as u64,
        Err(e) =>
        {
            eprintln!("reparent_keyword: hide failed: {}", e);
            let _ = conn.execute_batch("ROLLBACK;");
            return 0;
        }
    };

    if let Err(e) = conn.execute_batch("COMMIT;")
    {
        eprintln!("reparent_keyword: commit failed: {}", e);
        return 0;
    }
    changed
}

/// Rename a keyword node GLOBALLY (its label + the corresponding path segment),
/// cascading to descendants. Same machinery as reparent, but the subtree is
/// re-rooted under the SAME parent with the new label (so no ancestor insert is
/// needed — the parent already exists). One transaction. Returns rows hidden.
pub async fn rename_keyword(target_path: Vec<String>, new_label: String) -> u64
{
    if target_path.is_empty()
    {
        return 0;
    }
    let new_label = new_label.trim().to_string();
    if new_label.is_empty() || new_label.contains(KEYWORD_PATH_SEPARATOR)
    {
        eprintln!("rename_keyword: invalid new label");
        return 0;
    }
    for s in target_path.iter()
    {
        if s.trim().is_empty() || s.contains(KEYWORD_PATH_SEPARATOR)
        {
            eprintln!("rename_keyword: invalid segment");
            return 0;
        }
    }

    let target_joined = target_path.join(KEYWORD_PATH_SEPARATOR);
    let target_prefix = format!("{}{}", target_joined, KEYWORD_PATH_SEPARATOR);
    let suffix_start = (target_joined.chars().count() + 1) as i64;
    let parent = &target_path[..target_path.len() - 1];
    let new_root = if parent.is_empty()
    {
        new_label.clone()
    }
    else
    {
        format!("{}{}{}", parent.join(KEYWORD_PATH_SEPARATOR), KEYWORD_PATH_SEPARATOR, new_label)
    };

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

    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;")
    {
        eprintln!("rename_keyword: begin failed: {}", e);
        return 0;
    }

    // Re-root the subtree; the root row's label becomes new_label (descendants
    // keep theirs), path re-rooted for the whole subtree.
    let move_sql = "INSERT INTO keyword (image_id, label, path, status, created_at) \
                    SELECT image_id, \
                           CASE WHEN path = ? THEN ? ELSE label END, \
                           ? || substr(path, ?), 1, CURRENT_TIMESTAMP FROM keyword \
                    WHERE status = 1 AND (path = ? OR starts_with(path, ?))";
    if let Err(e) = conn.execute(
        move_sql,
        params![target_joined, new_label, new_root, suffix_start, target_joined, target_prefix],
    )
    {
        eprintln!("rename_keyword: move insert failed: {}", e);
        let _ = conn.execute_batch("ROLLBACK;");
        return 0;
    }

    let hide_sql = "UPDATE keyword SET status = 0, hidden_at = CURRENT_TIMESTAMP \
                    WHERE status = 1 AND (path = ? OR starts_with(path, ?))";
    let changed = match conn.execute(hide_sql, params![target_joined, target_prefix])
    {
        Ok(c) => c as u64,
        Err(e) =>
        {
            eprintln!("rename_keyword: hide failed: {}", e);
            let _ = conn.execute_batch("ROLLBACK;");
            return 0;
        }
    };

    if let Err(e) = conn.execute_batch("COMMIT;")
    {
        eprintln!("rename_keyword: commit failed: {}", e);
        return 0;
    }
    changed
}

// === Lightroom catalog import (Docs/DESIGN-Lightroom-Catalog-Import.md) ===

/// Per-chunk merge result. `image_ids` is aligned to the INPUT order so the
/// Swift keyword pass can attach keywords by id. On any row error the chunk is
/// rolled back and a zeroed result is returned (inserted + updated == 0 on a
/// non-empty input signals a failed chunk to the orchestrator).
#[derive(Debug, Clone)]
pub struct MergeChunkResult
{
    pub inserted: u64,
    pub updated: u64,
    pub image_ids: Vec<i64>,
}

/// The image-merge logic, on a borrowed connection (so it is testable against an
/// in-memory catalogue). Wrapped by `merge_lightroom_records`, which locks the
/// global CATALOGUE. Per-row **check-then-UPDATE-or-INSERT** — see §4 for WHY
/// this is NOT `ON CONFLICT`: `images.id` is a sequence-default PK, and DuckDB's
/// `ON CONFLICT DO UPDATE` fires `nextval()` for the proposed tuple on the
/// conflict path (advancing the sequence and confusing `RETURNING id`). The
/// explicit path is id-stable by construction (an UPDATE never touches `id` —
/// `keyword.image_id` FKs depend on it) and yields the inserted/updated tally
/// for free.
fn merge_records_into(conn: &Connection, records: &[ImageMetadata]) -> MergeChunkResult
{
    let mut out = MergeChunkResult { inserted: 0, updated: 0, image_ids: Vec::with_capacity(records.len()) };
    if records.is_empty()
    {
        return out;
    }

    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;")
    {
        eprintln!("merge_records_into: begin failed: {}", e);
        return out;
    }

    for record in records
    {
        // Pre-check: an existing row keeps its id (UPDATE); else a fresh INSERT.
        let existing: Result<i64, _> = conn.query_row(
            "SELECT id FROM images WHERE file_path = ?1",
            params![record.file_path],
            |r| r.get(0),
        );

        // Each arm yields Result<(was_insert, id), Error>.
        let row_result: Result<(bool, i64), _> = if let Ok(id) = existing
        {
            // UPDATE — facts fill-if-missing, curation Lightroom-wins (§4/§5).
            conn.execute(
                "UPDATE images SET \
                    capture_datetime = COALESCE(capture_datetime, ?2), \
                    pixel_width      = COALESCE(pixel_width, ?3), \
                    pixel_height     = COALESCE(pixel_height, ?4), \
                    camera_make      = COALESCE(camera_make, ?5), \
                    camera_model     = COALESCE(camera_model, ?6), \
                    lens_model       = COALESCE(lens_model, ?7), \
                    focal_length     = COALESCE(focal_length, ?8), \
                    aperture         = COALESCE(aperture, ?9), \
                    shutter_speed    = COALESCE(shutter_speed, ?10), \
                    iso              = COALESCE(iso, ?11), \
                    bit_depth        = COALESCE(bit_depth, ?12), \
                    gps_latitude     = COALESCE(gps_latitude, ?13), \
                    gps_longitude    = COALESCE(gps_longitude, ?14), \
                    rating      = COALESCE(?15, rating), \
                    flag        = COALESCE(?16, flag), \
                    color_label = COALESCE(?17, color_label) \
                 WHERE id = ?1",
                params![
                    id,
                    record.capture_datetime,
                    record.pixel_width.map(|v| v as i64),
                    record.pixel_height.map(|v| v as i64),
                    record.camera_make,
                    record.camera_model,
                    record.lens_model,
                    record.focal_length,
                    record.aperture,
                    record.shutter_speed,
                    record.iso.map(|v| v as i64),
                    record.bit_depth.map(|v| v as i64),
                    record.gps_latitude,
                    record.gps_longitude,
                    record.rating.map(|v| v as i64),
                    record.flag,
                    record.color_label,
                ],
            ).map(|_| (false, id))
        }
        else
        {
            // INSERT — mirror ingest_metadata's column set + the canonical
            // directory_path expression; RETURNING id (a plain insert is reliable).
            let parsed = parse_filename(record.file_name.clone());
            let image_kind_str = match parsed.kind
            {
                ImageKind::Jpeg => "jpeg",
                ImageKind::Raw  => "raw",
                ImageKind::Other => "other",
                ImageKind::Heif => "heif",
                ImageKind::Dng  => "dng",
                ImageKind::Psd  => "psd",
                ImageKind::Tiff => "tiff",
                ImageKind::Png  => "png",
            };
            conn.query_row(
                "INSERT INTO images ( \
                    file_path, file_size, file_name, file_extension, \
                    file_stem, image_kind, directory_path, \
                    created_timestamp, modified_timestamp, \
                    camera_make, camera_model, lens_model, \
                    focal_length, aperture, shutter_speed, iso, \
                    capture_datetime, pixel_width, pixel_height, color_space, bit_depth, \
                    gps_latitude, gps_longitude, gps_altitude, \
                    copyright, creator, description, \
                    rating, flag, color_label \
                 ) VALUES ( \
                    ?1, ?2, ?3, ?4, ?5, ?6, \
                    SUBSTRING(?1, 1, LENGTH(?1) - INSTR(REVERSE(?1), '/')), \
                    ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, \
                    ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29 \
                 ) RETURNING id",
                params![
                    record.file_path,
                    record.file_size as i64,
                    record.file_name,
                    record.file_extension,
                    parsed.stem,
                    image_kind_str,
                    record.created_timestamp,
                    record.modified_timestamp,
                    record.camera_make,
                    record.camera_model,
                    record.lens_model,
                    record.focal_length,
                    record.aperture,
                    record.shutter_speed,
                    record.iso.map(|v| v as i64),
                    record.capture_datetime,
                    record.pixel_width.map(|v| v as i64),
                    record.pixel_height.map(|v| v as i64),
                    record.color_space,
                    record.bit_depth.map(|v| v as i64),
                    record.gps_latitude,
                    record.gps_longitude,
                    record.gps_altitude,
                    record.copyright,
                    record.creator,
                    record.description,
                    record.rating.map(|v| v as i64),
                    record.flag,
                    record.color_label,
                ],
                |r| r.get::<_, i64>(0),
            ).map(|new_id| (true, new_id))
        };

        match row_result
        {
            Ok((true, id))  => { out.inserted += 1; out.image_ids.push(id); }
            Ok((false, id)) => { out.updated  += 1; out.image_ids.push(id); }
            Err(e) =>
            {
                eprintln!("merge_records_into: row failed for {}: {}", record.file_path, e);
                let _ = conn.execute_batch("ROLLBACK;");
                return MergeChunkResult { inserted: 0, updated: 0, image_ids: Vec::new() };
            }
        }
    }

    if let Err(e) = conn.execute_batch("COMMIT;")
    {
        eprintln!("merge_records_into: commit failed: {}", e);
        return MergeChunkResult { inserted: 0, updated: 0, image_ids: Vec::new() };
    }
    out
}

/// FFI entry: merge a chunk of Lightroom-sourced image records into the
/// catalogue (matched on file_path). Reuses `ImageMetadata` as the input — it is
/// an exact superset of what LR provides (§10). Returns per-chunk stats + the
/// resulting catalogue ids (aligned to input order) for the keyword pass.
pub async fn merge_lightroom_records(records: Vec<ImageMetadata>) -> MergeChunkResult
{
    if records.is_empty()
    {
        return MergeChunkResult { inserted: 0, updated: 0, image_ids: Vec::new() };
    }
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref()
    {
        Some(c) => c,
        None =>
        {
            eprintln!("merge_lightroom_records: catalogue not initialized");
            return MergeChunkResult { inserted: 0, updated: 0, image_ids: Vec::new() };
        }
    };
    merge_records_into(conn, &records)
}

/// One Lightroom-sourced VIDEO record (input to merge_lightroom_videos). No
/// `ImageMetadata` analogue — videos carry duration/frame_rate/has_audio/
/// video_kind and no EXIF. `directory_path` is derived Rust-side (like images).
#[derive(Debug, Clone)]
pub struct LightroomVideoRecord
{
    pub file_path: String,
    pub file_size: u64,
    pub file_name: String,
    pub file_extension: Option<String>,
    pub created_timestamp: i64,
    pub modified_timestamp: i64,
    pub capture_datetime: Option<String>,
    pub pixel_width: Option<u32>,
    pub pixel_height: Option<u32>,
    pub duration_seconds: Option<f64>,
    pub frame_rate: Option<f64>,
    pub has_audio: Option<bool>,
    pub video_kind: Option<String>,
    pub rating: Option<u8>,
    pub flag: Option<String>,
    pub color_label: Option<String>,
}

/// Video-merge logic on a borrowed connection (testable). Same
/// check-then-UPDATE-or-INSERT pattern as `merge_records_into` (§4), into the
/// `videos` table. The result's `image_ids` holds the VIDEO-row ids (aligned to
/// input) — the field name is shared for one `MergeChunkResult` shape.
fn merge_videos_into(conn: &Connection, records: &[LightroomVideoRecord]) -> MergeChunkResult
{
    let mut out = MergeChunkResult { inserted: 0, updated: 0, image_ids: Vec::with_capacity(records.len()) };
    if records.is_empty()
    {
        return out;
    }

    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;")
    {
        eprintln!("merge_videos_into: begin failed: {}", e);
        return out;
    }

    for record in records
    {
        let existing: Result<i64, _> = conn.query_row(
            "SELECT id FROM videos WHERE file_path = ?1",
            params![record.file_path],
            |r| r.get(0),
        );

        let row_result: Result<(bool, i64), _> = if let Ok(id) = existing
        {
            // UPDATE — facts fill-if-missing, curation Lightroom-wins.
            conn.execute(
                "UPDATE videos SET \
                    capture_datetime = COALESCE(capture_datetime, ?2), \
                    pixel_width      = COALESCE(pixel_width, ?3), \
                    pixel_height     = COALESCE(pixel_height, ?4), \
                    duration_seconds = COALESCE(duration_seconds, ?5), \
                    frame_rate       = COALESCE(frame_rate, ?6), \
                    has_audio        = COALESCE(has_audio, ?7), \
                    video_kind       = COALESCE(video_kind, ?8), \
                    rating      = COALESCE(?9, rating), \
                    flag        = COALESCE(?10, flag), \
                    color_label = COALESCE(?11, color_label) \
                 WHERE id = ?1",
                params![
                    id,
                    record.capture_datetime,
                    record.pixel_width.map(|v| v as i64),
                    record.pixel_height.map(|v| v as i64),
                    record.duration_seconds,
                    record.frame_rate,
                    record.has_audio,
                    record.video_kind,
                    record.rating.map(|v| v as i64),
                    record.flag,
                    record.color_label,
                ],
            ).map(|_| (false, id))
        }
        else
        {
            conn.query_row(
                "INSERT INTO videos ( \
                    file_path, file_size, file_name, file_extension, directory_path, \
                    created_timestamp, modified_timestamp, \
                    capture_datetime, pixel_width, pixel_height, \
                    duration_seconds, frame_rate, has_audio, video_kind, \
                    rating, flag, color_label \
                 ) VALUES ( \
                    ?1, ?2, ?3, ?4, \
                    SUBSTRING(?1, 1, LENGTH(?1) - INSTR(REVERSE(?1), '/')), \
                    ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16 \
                 ) RETURNING id",
                params![
                    record.file_path,
                    record.file_size as i64,
                    record.file_name,
                    record.file_extension,
                    record.created_timestamp,
                    record.modified_timestamp,
                    record.capture_datetime,
                    record.pixel_width.map(|v| v as i64),
                    record.pixel_height.map(|v| v as i64),
                    record.duration_seconds,
                    record.frame_rate,
                    record.has_audio,
                    record.video_kind,
                    record.rating.map(|v| v as i64),
                    record.flag,
                    record.color_label,
                ],
                |r| r.get::<_, i64>(0),
            ).map(|new_id| (true, new_id))
        };

        match row_result
        {
            Ok((true, id))  => { out.inserted += 1; out.image_ids.push(id); }
            Ok((false, id)) => { out.updated  += 1; out.image_ids.push(id); }
            Err(e) =>
            {
                eprintln!("merge_videos_into: row failed for {}: {}", record.file_path, e);
                let _ = conn.execute_batch("ROLLBACK;");
                return MergeChunkResult { inserted: 0, updated: 0, image_ids: Vec::new() };
            }
        }
    }

    if let Err(e) = conn.execute_batch("COMMIT;")
    {
        eprintln!("merge_videos_into: commit failed: {}", e);
        return MergeChunkResult { inserted: 0, updated: 0, image_ids: Vec::new() };
    }
    out
}

/// FFI entry: merge a chunk of Lightroom-sourced VIDEO records into the `videos`
/// table (matched on file_path). Returns per-chunk stats + the video-row ids.
pub async fn merge_lightroom_videos(records: Vec<LightroomVideoRecord>) -> MergeChunkResult
{
    if records.is_empty()
    {
        return MergeChunkResult { inserted: 0, updated: 0, image_ids: Vec::new() };
    }
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref()
    {
        Some(c) => c,
        None =>
        {
            eprintln!("merge_lightroom_videos: catalogue not initialized");
            return MergeChunkResult { inserted: 0, updated: 0, image_ids: Vec::new() };
        }
    };
    merge_videos_into(conn, &records)
}

#[cfg(test)]
mod keyword_tests
{
    use super::*;

    #[test]
    fn materialized_rows_builds_ancestor_chain()
    {
        let rows = keyword_materialized_rows(&vec![
            "Animals".to_string(),
            "Dog".to_string(),
            "Lab".to_string(),
        ]);
        let sep = KEYWORD_PATH_SEPARATOR;
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], ("Animals".to_string(), "Animals".to_string()));
        assert_eq!(rows[1], ("Dog".to_string(), format!("Animals{sep}Dog")));
        assert_eq!(rows[2], ("Lab".to_string(), format!("Animals{sep}Dog{sep}Lab")));
    }

    #[test]
    fn materialized_rows_trims_and_rejects_bad_segments()
    {
        // Blank segment -> whole path rejected (empty).
        assert!(keyword_materialized_rows(&vec!["Animals".to_string(), "  ".to_string()]).is_empty());
        // Segment containing the separator -> rejected.
        assert!(keyword_materialized_rows(&vec![format!("a{KEYWORD_PATH_SEPARATOR}b")]).is_empty());
        // Trimming.
        let rows = keyword_materialized_rows(&vec![" Animals ".to_string()]);
        assert_eq!(rows[0], ("Animals".to_string(), "Animals".to_string()));
    }

    #[test]
    fn keyword_image_id_in_list_assembly()
    {
        assert_eq!(keyword_image_id_in_list(&[]), None);
        assert_eq!(keyword_image_id_in_list(&[5, 9, 12]), Some("image_id IN (5, 9, 12)".to_string()));
    }

    #[test]
    fn keyword_predicate_sql()
    {
        let has = QueryPredicate {
            kind: "keyword_has".to_string(),
            day: None, day_end: None, op: None, stars: None,
            value: Some("Wagner".to_string()),
        };
        assert_eq!(
            predicate_to_sql(&has),
            "(EXISTS (SELECT 1 FROM keyword_visible k WHERE k.image_id = images.id AND k.label = 'Wagner'))"
        );

        let not = QueryPredicate {
            kind: "keyword_not".to_string(),
            day: None, day_end: None, op: None, stars: None,
            value: Some("snapshot".to_string()),
        };
        assert_eq!(
            predicate_to_sql(&not),
            "(NOT EXISTS (SELECT 1 FROM keyword_visible k WHERE k.image_id = images.id AND k.label = 'snapshot'))"
        );

        // Empty value -> backstop.
        let empty = QueryPredicate {
            kind: "keyword_has".to_string(),
            day: None, day_end: None, op: None, stars: None,
            value: Some(String::new()),
        };
        assert_eq!(predicate_to_sql(&empty), "(FALSE)");
    }

    #[test]
    fn filename_predicate_sql()
    {
        let fname = |kind: &str, v: &str| QueryPredicate {
            kind: kind.to_string(),
            day: None, day_end: None, op: None, stars: None,
            value: Some(v.to_string()),
        };

        // The four modes, all case-insensitive (ILIKE) with backslash ESCAPE.
        assert_eq!(
            predicate_to_sql(&fname("filename_contains", "somers")),
            "(file_name ILIKE '%somers%' ESCAPE '\\')"
        );
        assert_eq!(
            predicate_to_sql(&fname("filename_starts", "RSW")),
            "(file_name ILIKE 'RSW%' ESCAPE '\\')"
        );
        assert_eq!(
            predicate_to_sql(&fname("filename_ends", ".nef")),
            "(file_name ILIKE '%.nef' ESCAPE '\\')"
        );
        assert_eq!(
            predicate_to_sql(&fname("filename_exact", "RSW_0001.NEF")),
            // The underscore is a LIKE wildcard → escaped so it matches a literal '_'.
            "(file_name ILIKE 'RSW\\_0001.NEF' ESCAPE '\\')"
        );

        // Wildcard + quote injection are neutralized: '%' and '_' are escaped,
        // the apostrophe is doubled for the SQL literal.
        assert_eq!(
            predicate_to_sql(&fname("filename_contains", "50% O'Neil_")),
            "(file_name ILIKE '%50\\% O''Neil\\_%' ESCAPE '\\')"
        );

        // Empty value -> backstop (matches nothing).
        assert_eq!(predicate_to_sql(&fname("filename_contains", "")), "(FALSE)");
    }

    #[test]
    fn collection_predicate_sql()
    {
        let coll = |v: &str| QueryPredicate {
            kind: "collection_is".to_string(),
            day: None, day_end: None, op: None, stars: None,
            value: Some(v.to_string()),
        };

        // Membership probes the RAW keyword table (not keyword_visible) with
        // the collection switch ON — collection and visibility are independent.
        assert_eq!(
            predicate_to_sql(&coll("Dogs")),
            "(EXISTS (SELECT 1 FROM keyword k WHERE k.image_id = images.id AND k.label = 'Dogs' AND k.collection = TRUE))"
        );

        // The apostrophe doubles for the SQL literal.
        assert_eq!(
            predicate_to_sql(&coll("Richard's Picks")),
            "(EXISTS (SELECT 1 FROM keyword k WHERE k.image_id = images.id AND k.label = 'Richard''s Picks' AND k.collection = TRUE))"
        );

        // Empty value -> backstop (matches nothing).
        assert_eq!(predicate_to_sql(&coll("")), "(FALSE)");
    }
}

#[cfg(test)]
mod saved_query_tests
{
    use super::*;
    use duckdb::Connection;

    /// The saved-query DDL, as in the main schema (fresh CREATEs, no ALTERs).
    fn setup() -> Connection
    {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE SEQUENCE saved_query_id_seq START 1;
             CREATE TABLE saved_query (
                 id INTEGER PRIMARY KEY DEFAULT nextval('saved_query_id_seq'),
                 name TEXT NOT NULL,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
             );
             CREATE TABLE saved_query_criterion (
                 query_id INTEGER NOT NULL,
                 position INTEGER NOT NULL,
                 connector TEXT,
                 kind TEXT NOT NULL,
                 op TEXT,
                 value TEXT,
                 day TEXT,
                 day_end TEXT,
                 stars INTEGER
             );",
        )
        .expect("saved-query DDL");
        conn
    }

    fn pred(kind: &str, day: Option<&str>, day_end: Option<&str>,
            op: Option<&str>, stars: Option<u8>, value: Option<&str>) -> QueryPredicate
    {
        QueryPredicate {
            kind: kind.to_string(),
            day: day.map(str::to_string),
            day_end: day_end.map(str::to_string),
            op: op.map(str::to_string),
            stars,
            value: value.map(str::to_string),
        }
    }

    #[test]
    fn round_trip_preserves_sentence()
    {
        let conn = setup();

        // The design doc's "Spring picks": two date ranges EITHER, color AND,
        // color EITHER, rating AND — repeated subjects + every connector slot.
        let predicates = vec![
            pred("date_between", Some("2024:01:01"), Some("2024:02:15"), None, None, None),
            pred("date_between", Some("2025:06:01"), Some("2025:07:04"), None, None, None),
            pred("color", None, None, None, None, Some("blue")),
            pred("color", None, None, None, None, Some("green")),
            pred("rating", None, None, Some("gte"), Some(3), None),
        ];
        let connectors = vec![Connector::Or, Connector::And, Connector::Or, Connector::And];

        let info = save_query_impl(&conn, "Spring picks", &predicates, &connectors)
            .expect("save succeeds");
        assert_eq!(info.name, "Spring picks");

        let payload = load_saved_query_impl(&conn, info.id).expect("load succeeds");
        assert_eq!(payload.predicates, predicates);
        assert_eq!(payload.connectors, connectors);
    }

    #[test]
    fn name_collisions_gain_numeric_suffixes()
    {
        let conn = setup();
        let predicates = vec![pred("rating", None, None, Some("gte"), Some(1), None)];

        let a = save_query_impl(&conn, "Dogs", &predicates, &[]).expect("first save");
        let b = save_query_impl(&conn, "Dogs", &predicates, &[]).expect("second save");
        let c = save_query_impl(&conn, "Dogs", &predicates, &[]).expect("third save");
        assert_eq!(a.name, "Dogs");
        assert_eq!(b.name, "Dogs-01");
        assert_eq!(c.name, "Dogs-02");

        // The list shows all three, name-ordered.
        let names: Vec<String> = list_saved_queries_impl(&conn).into_iter().map(|q| q.name).collect();
        assert_eq!(names, vec!["Dogs", "Dogs-01", "Dogs-02"]);
    }

    #[test]
    fn empty_name_or_empty_sentence_rejected()
    {
        let conn = setup();
        let predicates = vec![pred("rating", None, None, Some("gte"), Some(1), None)];
        assert!(save_query_impl(&conn, "   ", &predicates, &[]).is_none());
        assert!(save_query_impl(&conn, "Fine", &[], &[]).is_none());
    }

    #[test]
    fn delete_removes_header_and_criteria()
    {
        let conn = setup();
        let predicates = vec![
            pred("flag", None, None, None, None, Some("pick")),
            pred("color", None, None, None, None, Some("red")),
        ];
        let info = save_query_impl(&conn, "Culls", &predicates, &[Connector::And]).expect("save");

        assert!(delete_saved_query_impl(&conn, info.id));
        assert!(load_saved_query_impl(&conn, info.id).is_none());
        assert!(list_saved_queries_impl(&conn).is_empty());

        // Deleting again reports false (nothing there).
        assert!(!delete_saved_query_impl(&conn, info.id));

        // No orphaned criterion rows.
        let leftover: i64 = conn
            .query_row("SELECT COUNT(*) FROM saved_query_criterion", [], |r| r.get(0))
            .unwrap();
        assert_eq!(leftover, 0);
    }
}

#[cfg(test)]
mod lightroom_import_tests
{
    use super::*;
    use duckdb::{Connection, params};

    // ---- classify_extension: the new ImageKind promotions (§7) ----

    #[test]
    fn classify_promotes_new_kinds()
    {
        // DNG out of Raw (its own kind, checked BEFORE the RAW table).
        assert_eq!(classify_extension("dng".to_string()), ImageKind::Dng);
        // PSD / TIFF / PNG out of the Other bucket.
        assert_eq!(classify_extension("psd".to_string()), ImageKind::Psd);
        assert_eq!(classify_extension("tif".to_string()), ImageKind::Tiff);
        assert_eq!(classify_extension("tiff".to_string()), ImageKind::Tiff);
        assert_eq!(classify_extension("png".to_string()), ImageKind::Png);
    }

    #[test]
    fn classify_preserves_existing_kinds()
    {
        // The promotions must not disturb the existing classifications.
        assert_eq!(classify_extension("nef".to_string()), ImageKind::Raw);
        assert_eq!(classify_extension("cr3".to_string()), ImageKind::Raw);
        assert_eq!(classify_extension("jpg".to_string()), ImageKind::Jpeg);
        assert_eq!(classify_extension("jpeg".to_string()), ImageKind::Jpeg);
        assert_eq!(classify_extension("heic".to_string()), ImageKind::Heif);
        assert_eq!(classify_extension("xyz".to_string()), ImageKind::Other);
        assert_eq!(classify_extension(String::new()), ImageKind::Other);
    }

    #[test]
    fn classify_is_case_insensitive_for_new_kinds()
    {
        assert_eq!(classify_extension("DNG".to_string()), ImageKind::Dng);
        assert_eq!(classify_extension("Psd".to_string()), ImageKind::Psd);
        assert_eq!(classify_extension("TIFF".to_string()), ImageKind::Tiff);
        assert_eq!(classify_extension("PNG".to_string()), ImageKind::Png);
    }

    // ---- DuckDB ON CONFLICT upsert probe (de-risks §4 before the merge) ----
    //
    // Proves the BUNDLED DuckDB engine supports the exact upsert the Lightroom
    // merge relies on: INSERT ... ON CONFLICT(unique) DO UPDATE SET, referencing
    // both `excluded.<col>` (the would-be-inserted row) and the target table's
    // own columns, in BOTH COALESCE directions:
    //   - CURATION (Lightroom-wins):  COALESCE(excluded.x, t.x)
    //   - FACTS    (fill-if-missing): COALESCE(t.x, excluded.x)
    // and that an LR-NULL value never erases an existing curation value.
    #[test]
    fn duckdb_on_conflict_upsert_behaves()
    {
        let conn = Connection::open_in_memory().expect("open in-memory duckdb");
        conn.execute_batch("CREATE TABLE t (fp TEXT UNIQUE, rating INTEGER, cam TEXT);")
            .expect("create table");

        // One upsert mirroring the merge: rating = LR-wins, cam = fill-if-missing.
        let upsert = "INSERT INTO t (fp, rating, cam) VALUES (?1, ?2, ?3) \
                      ON CONFLICT (fp) DO UPDATE SET \
                        rating = COALESCE(excluded.rating, t.rating), \
                        cam    = COALESCE(t.cam, excluded.cam)";

        // 1. First insert (no conflict): the row is created.
        conn.execute(upsert, params!["a", 3i32, "CanonX"]).expect("insert 1");

        // 2. Conflict with a NEW rating + a different cam:
        //    rating -> 5 (Lightroom-wins), cam stays CanonX (fact, fill-if-missing).
        conn.execute(upsert, params!["a", 5i32, "CanonY"]).expect("insert 2");
        let (rating, cam): (i64, String) = conn.query_row(
            "SELECT rating, cam FROM t WHERE fp = 'a'", [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        ).expect("select 2");
        assert_eq!(rating, 5, "Lightroom-wins should overwrite rating");
        assert_eq!(cam, "CanonX", "fact should be fill-if-missing (keep existing)");

        // 3. Conflict with a NULL rating must NOT erase the existing 5.
        conn.execute(upsert, params!["a", Option::<i32>::None, "CanonZ"]).expect("insert 3");
        let rating_after: Option<i64> = conn.query_row(
            "SELECT rating FROM t WHERE fp = 'a'", [], |r| r.get(0),
        ).expect("select 3");
        assert_eq!(rating_after, Some(5), "LR-null must not erase existing curation");

        // Upserts, not duplicate inserts: exactly one row.
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM t", [], |r| r.get(0)).expect("count");
        assert_eq!(count, 1);
    }

    // ---- videos table DDL probe (de-risks §8 before merge_lightroom_videos) ----
    //
    // Validates the new `videos` schema against the BUNDLED engine: the
    // sequence-default PK (DuckDB drops IDENTITY), and the DOUBLE / BOOLEAN /
    // BIGINT column types not otherwise exercised by the images schema.
    #[test]
    fn videos_table_ddl_and_insert()
    {
        let conn = Connection::open_in_memory().expect("open in-memory duckdb");
        conn.execute_batch(
            "CREATE SEQUENCE videos_id_seq START 1;
             CREATE TABLE videos (
                id INTEGER PRIMARY KEY DEFAULT nextval('videos_id_seq'),
                file_path TEXT NOT NULL UNIQUE,
                file_size BIGINT NOT NULL,
                file_name TEXT NOT NULL,
                file_extension TEXT,
                directory_path VARCHAR,
                created_timestamp INTEGER NOT NULL,
                modified_timestamp INTEGER NOT NULL,
                capture_datetime TEXT,
                pixel_width INTEGER,
                pixel_height INTEGER,
                duration_seconds DOUBLE,
                frame_rate DOUBLE,
                has_audio BOOLEAN,
                video_kind TEXT,
                rating INTEGER,
                flag TEXT,
                color_label TEXT,
                indexed_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
             );"
        ).expect("create videos table");

        // Insert omitting id (sequence default) + exercising DOUBLE/BOOLEAN/BIGINT.
        conn.execute(
            "INSERT INTO videos
                (file_path, file_size, file_name, file_extension, directory_path,
                 created_timestamp, modified_timestamp, capture_datetime,
                 pixel_width, pixel_height, duration_seconds, frame_rate, has_audio,
                 video_kind, rating, flag, color_label)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
            params![
                "/v/clip.mov", 4_087_595_000i64, "clip.mov", "mov", "/v",
                1_700_000_000i64, 1_700_000_000i64, "2025-01-02T03:04:05",
                1920i32, 1080i32, 204.4f64, 59.94f64, true,
                "mov", 4i32, "pick", "blue"
            ],
        ).expect("insert video");

        let (id, dur, fps, audio, kind): (i64, f64, f64, bool, String) = conn.query_row(
            "SELECT id, duration_seconds, frame_rate, has_audio, video_kind \
             FROM videos WHERE file_path = '/v/clip.mov'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
        ).expect("read video");

        assert_eq!(id, 1, "sequence default should auto-assign id = 1");
        assert!((dur - 204.4).abs() < 1e-6, "DOUBLE round-trips");
        assert!((fps - 59.94).abs() < 1e-6, "DOUBLE round-trips");
        assert!(audio, "BOOLEAN round-trips");
        assert_eq!(kind, "mov");
    }

    // ---- ON CONFLICT + sequence-PK id behavior (CRITICAL — drives the merge) ----
    //
    // The merge matches on file_path and must keep each row's `id` STABLE across
    // re-import — `keyword.image_id` FKs depend on it. DuckDB evaluates the
    // sequence DEFAULT for the proposed insert tuple even on the conflict path,
    // so `RETURNING id` after DO UPDATE returns the PROPOSED (advanced) id, not
    // the existing one. What actually matters is whether the STORED id changes.
    // This test pins that down. (The merge retrieves ids via SELECT, never
    // RETURNING; if the STORED id were unstable we'd abandon ON CONFLICT for an
    // explicit check-then-UPDATE-or-INSERT, which never touches id.)
    #[test]
    fn duckdb_upsert_keeps_stored_id_stable()
    {
        let conn = Connection::open_in_memory().expect("open in-memory duckdb");
        conn.execute_batch(
            "CREATE SEQUENCE s START 1;
             CREATE TABLE m (id INTEGER PRIMARY KEY DEFAULT nextval('s'), fp TEXT UNIQUE, rating INTEGER);"
        ).expect("create");

        conn.execute("INSERT INTO m (fp, rating) VALUES ('a', 3)", []).expect("insert a");
        let id_before: i64 = conn.query_row("SELECT id FROM m WHERE fp = 'a'", [], |r| r.get(0)).expect("id before");

        // Conflicting upsert (curation COALESCE direction).
        conn.execute(
            "INSERT INTO m (fp, rating) VALUES ('a', 5) \
             ON CONFLICT (fp) DO UPDATE SET rating = COALESCE(excluded.rating, m.rating)",
            [],
        ).expect("upsert a");

        let id_after: i64 = conn.query_row("SELECT id FROM m WHERE fp = 'a'", [], |r| r.get(0)).expect("id after");
        let rating_after: i64 = conn.query_row("SELECT rating FROM m WHERE fp = 'a'", [], |r| r.get(0)).expect("rating after");

        assert_eq!(id_before, id_after, "STORED id must be stable across ON CONFLICT update (keyword FKs depend on it)");
        assert_eq!(rating_after, 5, "curation still updates (Lightroom-wins)");
    }

    // ---- merge_records_into: insert / update / policies / id-stability ----

    // A minimal `images` table covering exactly the columns the merge touches.
    fn create_images(conn: &Connection)
    {
        conn.execute_batch(
            "CREATE SEQUENCE images_id_seq START 1;
             CREATE TABLE images (
                id INTEGER PRIMARY KEY DEFAULT nextval('images_id_seq'),
                file_path TEXT NOT NULL UNIQUE,
                file_size BIGINT NOT NULL,
                file_name TEXT NOT NULL,
                file_extension TEXT,
                file_stem VARCHAR,
                image_kind VARCHAR,
                directory_path VARCHAR,
                created_timestamp INTEGER NOT NULL,
                modified_timestamp INTEGER NOT NULL,
                camera_make TEXT, camera_model TEXT, lens_model TEXT,
                focal_length REAL, aperture REAL, shutter_speed REAL, iso INTEGER,
                capture_datetime TEXT, pixel_width INTEGER, pixel_height INTEGER,
                color_space TEXT, bit_depth INTEGER,
                gps_latitude REAL, gps_longitude REAL, gps_altitude REAL,
                copyright TEXT, creator TEXT, description TEXT,
                rating INTEGER, flag TEXT, color_label TEXT
             );"
        ).expect("create images");
    }

    // A default ImageMetadata; override fields per test.
    fn img(file_path: &str, file_name: &str) -> ImageMetadata
    {
        ImageMetadata {
            file_path: file_path.to_string(),
            file_size: 1000,
            file_name: file_name.to_string(),
            file_extension: Some("nef".to_string()),
            created_timestamp: 1_700_000_000,
            modified_timestamp: 1_700_000_000,
            camera_make: None, camera_model: None, lens_model: None,
            focal_length: None, aperture: None, shutter_speed: None, iso: None,
            capture_datetime: None,
            pixel_width: None, pixel_height: None, color_space: None, bit_depth: None,
            gps_latitude: None, gps_longitude: None, gps_altitude: None,
            copyright: None, creator: None, description: None,
            rating: None, flag: None, color_label: None,
        }
    }

    #[test]
    fn merge_inserts_then_updates_with_policies()
    {
        let conn = Connection::open_in_memory().expect("open in-memory duckdb");
        create_images(&conn);

        // 1. First merge: two NEW rows.
        let mut a = img("/p/a.nef", "a.nef");
        a.rating = Some(3);
        a.camera_model = Some("Nikon Z8".to_string());
        let b = img("/p/b.nef", "b.nef");
        let r1 = merge_records_into(&conn, &[a, b]);
        assert_eq!(r1.inserted, 2);
        assert_eq!(r1.updated, 0);
        assert_eq!(r1.image_ids.len(), 2);
        let id_a = r1.image_ids[0];

        // Derived columns: nef -> 'raw'; directory_path from file_path.
        let (kind, dir): (String, String) = conn.query_row(
            "SELECT image_kind, directory_path FROM images WHERE file_path = '/p/a.nef'", [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        ).expect("derived cols");
        assert_eq!(kind, "raw");
        assert_eq!(dir, "/p");

        // 2. Re-merge 'a': new rating (LR-wins), a fact that was missing (fills),
        //    and a fact already set (must NOT overwrite).
        let mut a2 = img("/p/a.nef", "a.nef");
        a2.rating = Some(5);                                            // LR-wins -> 5
        a2.capture_datetime = Some("2025-01-01T00:00:00".to_string()); // fact was NULL -> fills
        a2.camera_model = Some("WRONG".to_string());                   // fact set -> keep "Nikon Z8"
        let r2 = merge_records_into(&conn, &[a2]);
        assert_eq!(r2.inserted, 0);
        assert_eq!(r2.updated, 1);
        assert_eq!(r2.image_ids[0], id_a, "id stable across re-merge (FK safety)");

        let (rating, cap, cam): (i64, String, String) = conn.query_row(
            "SELECT rating, capture_datetime, camera_model FROM images WHERE file_path = '/p/a.nef'", [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        ).expect("post-update");
        assert_eq!(rating, 5, "curation: Lightroom wins");
        assert_eq!(cap, "2025-01-01T00:00:00", "fact: filled when missing");
        assert_eq!(cam, "Nikon Z8", "fact: existing value kept, not overwritten");

        // 3. LR-null rating must NOT erase the existing 5.
        let a3 = img("/p/a.nef", "a.nef"); // rating None
        let r3 = merge_records_into(&conn, &[a3]);
        assert_eq!(r3.updated, 1);
        let rating2: i64 = conn.query_row(
            "SELECT rating FROM images WHERE file_path = '/p/a.nef'", [], |r| r.get(0),
        ).expect("rating after null");
        assert_eq!(rating2, 5, "LR-null must not erase existing curation");
    }

    // ---- merge_videos_into: insert / update / policies (the videos table) ----

    fn create_videos(conn: &Connection)
    {
        conn.execute_batch(
            "CREATE SEQUENCE videos_id_seq START 1;
             CREATE TABLE videos (
                id INTEGER PRIMARY KEY DEFAULT nextval('videos_id_seq'),
                file_path TEXT NOT NULL UNIQUE, file_size BIGINT NOT NULL,
                file_name TEXT NOT NULL, file_extension TEXT, directory_path VARCHAR,
                created_timestamp INTEGER NOT NULL, modified_timestamp INTEGER NOT NULL,
                capture_datetime TEXT, pixel_width INTEGER, pixel_height INTEGER,
                duration_seconds DOUBLE, frame_rate DOUBLE, has_audio BOOLEAN, video_kind TEXT,
                rating INTEGER, flag TEXT, color_label TEXT,
                indexed_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
             );"
        ).expect("create videos");
    }

    fn vid(file_path: &str, file_name: &str) -> LightroomVideoRecord
    {
        LightroomVideoRecord {
            file_path: file_path.to_string(),
            file_size: 5000,
            file_name: file_name.to_string(),
            file_extension: Some("mov".to_string()),
            created_timestamp: 1_700_000_000,
            modified_timestamp: 1_700_000_000,
            capture_datetime: None, pixel_width: None, pixel_height: None,
            duration_seconds: None, frame_rate: None, has_audio: None,
            video_kind: Some("mov".to_string()),
            rating: None, flag: None, color_label: None,
        }
    }

    #[test]
    fn merge_videos_inserts_then_updates()
    {
        let conn = Connection::open_in_memory().expect("open in-memory duckdb");
        create_videos(&conn);

        // 1. Insert a new video with a fact (duration) + curation (rating).
        let mut a = vid("/v/a.mov", "a.mov");
        a.duration_seconds = Some(204.4);
        a.rating = Some(2);
        let r1 = merge_videos_into(&conn, &[a]);
        assert_eq!(r1.inserted, 1);
        let id = r1.image_ids[0];
        let dir: String = conn.query_row(
            "SELECT directory_path FROM videos WHERE file_path = '/v/a.mov'", [], |r| r.get(0),
        ).expect("dir");
        assert_eq!(dir, "/v");

        // 2. Re-merge: rating LR-wins; duration (fact, already set) must NOT change.
        let mut a2 = vid("/v/a.mov", "a.mov");
        a2.rating = Some(4);
        a2.duration_seconds = Some(999.0);
        let r2 = merge_videos_into(&conn, &[a2]);
        assert_eq!(r2.updated, 1);
        assert_eq!(r2.image_ids[0], id, "video id stable across re-merge");

        let (rating, dur): (i64, f64) = conn.query_row(
            "SELECT rating, duration_seconds FROM videos WHERE file_path = '/v/a.mov'", [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        ).expect("post-update");
        assert_eq!(rating, 4, "curation: Lightroom wins");
        assert!((dur - 204.4).abs() < 1e-6, "fact: existing duration kept, not overwritten");
    }
}

/// Bulk-set the pick/reject flag on many records in ONE statement (Browse
/// "Set Flag" on a selection / the whole query). Mirrors `update_image_flag`'s
/// allow-list guard: `None` clears; any value outside {pick, reject} rejects
/// the WHOLE update (→ 0). Returns the number of rows changed.
pub async fn update_flag_for_ids(ids: Vec<i64>, flag: Option<String>) -> u64
{
    let flag_value: Option<String> = match flag.as_deref()
    {
        None => None,
        Some(v @ ("pick" | "reject")) => Some(v.to_string()),
        Some(other) =>
        {
            eprintln!("Rejected invalid flag value '{}' for bulk update", other);
            return 0;
        }
    };

    let where_clause = match id_in_list(&ids)
    {
        Some(w) => w,
        None => return 0,
    };

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

    let update_sql = format!("UPDATE images SET flag = ? WHERE {}", where_clause);
    match conn.execute(&update_sql, params![flag_value])
    {
        Ok(changed) => changed as u64,
        Err(e) =>
        {
            eprintln!("Failed to bulk-update flag: {}", e);
            0
        }
    }
}

/// Bulk-set the color label on many records in ONE statement. Mirrors
/// `update_image_color_label`'s allow-list guard. Returns rows changed.
pub async fn update_color_label_for_ids(ids: Vec<i64>, color_label: Option<String>) -> u64
{
    let label_value: Option<String> = match color_label.as_deref()
    {
        None => None,
        Some(v @ ("red" | "yellow" | "green" | "blue" | "purple")) => Some(v.to_string()),
        Some(other) =>
        {
            eprintln!("Rejected invalid color label '{}' for bulk update", other);
            return 0;
        }
    };

    let where_clause = match id_in_list(&ids)
    {
        Some(w) => w,
        None => return 0,
    };

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

    let update_sql = format!("UPDATE images SET color_label = ? WHERE {}", where_clause);
    match conn.execute(&update_sql, params![label_value])
    {
        Ok(changed) => changed as u64,
        Err(e) =>
        {
            eprintln!("Failed to bulk-update color label: {}", e);
            0
        }
    }
}

/// Bulk-set the star rating on many records in ONE statement. Rating 0 clears
/// (NULL), mirroring `update_image_rating`. Returns rows changed.
pub async fn update_rating_for_ids(ids: Vec<i64>, rating: u32) -> u64
{
    let rating_value: Option<i64> = if rating == 0 { None } else { Some(rating as i64) };

    let where_clause = match id_in_list(&ids)
    {
        Some(w) => w,
        None => return 0,
    };

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

    let update_sql = format!("UPDATE images SET rating = ? WHERE {}", where_clause);
    match conn.execute(&update_sql, params![rating_value])
    {
        Ok(changed) => changed as u64,
        Err(e) =>
        {
            eprintln!("Failed to bulk-update rating: {}", e);
            0
        }
    }
}

#[cfg(test)]
mod query_builder_tests
{
    use super::*;

    fn qp(kind: &str) -> QueryPredicate
    {
        QueryPredicate
        {
            kind: kind.to_string(),
            day: None,
            day_end: None,
            op: None,
            stars: None,
            value: None,
        }
    }

    fn rating(op: &str, stars: u8) -> QueryPredicate
    {
        let mut p = qp("rating");
        p.op = Some(op.to_string());
        p.stars = Some(stars);
        p
    }

    fn flag(value: &str) -> QueryPredicate
    {
        let mut p = qp("flag");
        p.value = Some(value.to_string());
        p
    }

    fn color(value: &str) -> QueryPredicate
    {
        let mut p = qp("color");
        p.value = Some(value.to_string());
        p
    }

    #[test]
    fn day_validation()
    {
        assert!(is_valid_day("2026:05:15"));
        assert!(!is_valid_day("2026-05-15")); // dashes, not colons
        assert!(!is_valid_day("2026:5:15")); // wrong length
        assert!(!is_valid_day("abcd:ef:gh")); // non-digits
        assert!(!is_valid_day(""));
    }

    #[test]
    fn atom_sql()
    {
        assert_eq!(predicate_to_sql(&rating("gte", 4)), "(rating >= 4)");
        assert_eq!(predicate_to_sql(&flag("pick")), "(flag = 'pick')");
        assert_eq!(predicate_to_sql(&color("red")), "(color_label = 'red')");
        assert_eq!(predicate_to_sql(&qp("rating_unrated")), "(rating IS NULL)");
        assert_eq!(predicate_to_sql(&qp("unflagged")), "(flag IS NULL)");
        let mut fou = qp("flag_or_unflagged");
        fou.value = Some("pick".to_string());
        assert_eq!(predicate_to_sql(&fou), "(flag = 'pick' OR flag IS NULL)");
    }

    #[test]
    fn date_atoms()
    {
        let mut eq = qp("date_equals");
        eq.day = Some("2026:05:15".to_string());
        assert_eq!(predicate_to_sql(&eq), "(SUBSTRING(capture_datetime, 1, 10) = '2026:05:15')");

        let mut ge = qp("date_after");
        ge.day = Some("2026:05:15".to_string());
        assert_eq!(predicate_to_sql(&ge), "(SUBSTRING(capture_datetime, 1, 10) >= '2026:05:15')");

        let mut le = qp("date_before");
        le.day = Some("2026:05:15".to_string());
        assert_eq!(predicate_to_sql(&le), "(SUBSTRING(capture_datetime, 1, 10) <= '2026:05:15')");

        let mut gt = qp("date_gt");
        gt.day = Some("2026:05:15".to_string());
        assert_eq!(predicate_to_sql(&gt), "(SUBSTRING(capture_datetime, 1, 10) > '2026:05:15')");

        let mut lt = qp("date_lt");
        lt.day = Some("2026:05:15".to_string());
        assert_eq!(predicate_to_sql(&lt), "(SUBSTRING(capture_datetime, 1, 10) < '2026:05:15')");

        let mut bt = qp("date_between");
        bt.day = Some("2026:01:01".to_string());
        bt.day_end = Some("2026:03:31".to_string());
        assert_eq!(predicate_to_sql(&bt), "(SUBSTRING(capture_datetime, 1, 10) BETWEEN '2026:01:01' AND '2026:03:31')");
    }

    #[test]
    fn invalid_atoms_become_false()
    {
        assert_eq!(predicate_to_sql(&flag("bogus")), "(FALSE)");
        assert_eq!(predicate_to_sql(&rating("gte", 6)), "(FALSE)"); // stars out of range
        assert_eq!(predicate_to_sql(&color("teal")), "(FALSE)");
        let mut bad_date = qp("date_equals");
        bad_date.day = Some("2026-05-15".to_string()); // dashes → rejected
        assert_eq!(predicate_to_sql(&bad_date), "(FALSE)");
    }

    #[test]
    fn empty_predicates_no_where()
    {
        assert_eq!(build_filter_predicate(&[], &[]), "");
    }

    #[test]
    fn id_in_list_assembly()
    {
        assert_eq!(id_in_list(&[]), None);
        assert_eq!(id_in_list(&[5]), Some("id IN (5)".to_string()));
        assert_eq!(id_in_list(&[1, 2, 3]), Some("id IN (1, 2, 3)".to_string()));
    }

    #[test]
    fn order_by_follows_first_subject()
    {
        // Rating-first → stars best-to-worst.
        assert_eq!(order_by_for_filter(&[rating("gte", 4)]), RATING_FILTER_ORDER_BY);
        assert_eq!(order_by_for_filter(&[qp("rating_unrated")]), RATING_FILTER_ORDER_BY);
        // The FIRST subject wins even when rating appears later.
        assert_eq!(
            order_by_for_filter(&[flag("pick"), rating("gte", 4)]),
            DEFAULT_FILTER_ORDER_BY
        );
        // Date-, flag-, color-first, and empty → the default newest-first.
        let mut d = qp("date_after");
        d.day = Some("2026:05:15".to_string());
        assert_eq!(order_by_for_filter(&[d]), DEFAULT_FILTER_ORDER_BY);
        assert_eq!(order_by_for_filter(&[flag("pick")]), DEFAULT_FILTER_ORDER_BY);
        assert_eq!(order_by_for_filter(&[color("red")]), DEFAULT_FILTER_ORDER_BY);
        assert_eq!(order_by_for_filter(&[]), DEFAULT_FILTER_ORDER_BY);
    }

    #[test]
    fn single_predicate_wrapped()
    {
        assert_eq!(build_filter_predicate(&[rating("gte", 4)], &[]), "((rating >= 4))");
    }

    #[test]
    fn left_to_right_accumulation()
    {
        // "A and B or C" → ((A AND B) OR C), left-to-right, NO precedence.
        let preds = vec![rating("gte", 4), flag("pick"), color("red")];
        let conns = vec![Connector::And, Connector::Or];
        assert_eq!(
            build_filter_predicate(&preds, &conns),
            "((((rating >= 4)) AND ((flag = 'pick'))) OR ((color_label = 'red')))"
        );
    }

    #[test]
    fn xor_is_boolean_inequality()
    {
        let preds = vec![flag("pick"), color("red")];
        let conns = vec![Connector::Xor];
        assert_eq!(
            build_filter_predicate(&preds, &conns),
            "(((flag = 'pick')) <> ((color_label = 'red')))"
        );
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
pub async fn get_images_filtered(
    limit: i64,
    offset: i64,
    date_prefix: String,
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
) -> Vec<ImageRecord> {
    // Acquire lock and validate connection
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    // Build optional predicate (predicate-only convention per decision
    // C3 — no "WHERE" keyword). Preserves the pre-extraction format!
    // interpolation pattern verbatim (decision C6) — Queue item 4
    // (Chunk 6) will parameterize this as a separate change. Empty
    // date_prefix → empty predicate; non-empty → LIKE filter.
    let predicate = if date_prefix.is_empty() {
        String::new()
    } else {
        format!("capture_datetime LIKE '{}%'", date_prefix)
    };

    execute_image_record_query(
        conn,
        &predicate,
        "capture_datetime DESC NULLS LAST, created_timestamp DESC",
        limit,
        offset,
        apply_duplicate_filter,
        apply_raw_jpeg_collapse,
    )
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
pub async fn get_filtered_image_count(
    date_prefix: String,
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
) -> i64 {
    // Acquire lock and validate connection
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    // Build optional predicate (predicate-only convention per decision
    // C3 — no "WHERE" keyword). Single-quote escape preserved verbatim
    // from the pre-extraction code (decision C6) — Queue item 4 (Chunk
    // 6) will parameterize this as a separate change.
    let predicate = if date_prefix.is_empty() {
        String::new()
    } else {
        format!("capture_datetime LIKE '{}%'", date_prefix)
    };

    execute_image_count_query(
        conn,
        &predicate,
        apply_duplicate_filter,
        apply_raw_jpeg_collapse,
    )
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
pub async fn get_image_count_for_path_prefix(
    path_prefix: String,
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
) -> i64
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

    // Build predicate (predicate-only convention per decision C3 — no
    // "WHERE" keyword). Single-quote escape preserved verbatim from the
    // pre-extraction code (decision C6) — Queue item 4 (Chunk 6) will
    // parameterize this as a separate change.
    let escaped_prefix = path_prefix.replace("'", "''");
    let predicate = format!("file_path LIKE '{}%'", escaped_prefix);

    execute_image_count_query(
        conn,
        &predicate,
        apply_duplicate_filter,
        apply_raw_jpeg_collapse,
    )
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
pub async fn get_images_for_path_prefix(
    limit: i64,
    offset: i64,
    path_prefix: String,
    date_prefix: String,
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
) -> Vec<ImageRecord>
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

    // Build predicate via shared helper (single source of truth for the
    // four-arm path/date composition — see `build_path_date_predicate`).
    let predicate = build_path_date_predicate(&path_prefix, &date_prefix);

    execute_image_record_query(
        conn,
        &predicate,
        "capture_datetime DESC NULLS LAST, created_timestamp DESC",
        limit,
        offset,
        apply_duplicate_filter,
        apply_raw_jpeg_collapse,
    )
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
pub async fn get_image_count_for_filters(
    path_prefix: String,
    date_prefix: String,
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
) -> i64
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

    // Build predicate via shared helper (single source of truth for the
    // four-arm path/date composition — see `build_path_date_predicate`).
    let predicate = build_path_date_predicate(&path_prefix, &date_prefix);

    execute_image_count_query(
        conn,
        &predicate,
        apply_duplicate_filter,
        apply_raw_jpeg_collapse,
    )
}

/// Result wrapper for `get_file_paths_for_filters` (A9).
///
/// Mirrors the UDL dictionary `FilePathsResult`. Distinguishes a genuine
/// failure (catalogue not initialized, prepare/query error) from a
/// legitimate empty-result success — required because this codebase
/// signals errors via sentinel/wrapper structs rather than `[Throws]`.
///
/// Contract:
/// - `ok == true`: query succeeded. `paths` holds the matching
///   `file_path` values (possibly empty); `error_message` is `None`.
/// - `ok == false`: query failed. `paths` is empty; `error_message`
///   carries a human-readable diagnostic (also logged via `eprintln!`
///   inside the helpers for parallel observability).
#[derive(Debug, Clone)]
pub struct FilePathsResult {
    pub ok: bool,
    pub paths: Vec<String>,
    pub error_message: Option<String>,
}

/// Return the set of `file_path` values matching the catalogue filter
/// state — projection-only enumeration of paths.
///
/// **STATUS (Session 24, A10 landing):** CURRENTLY UNUSED. Originally
/// designed (Session 24, A9) as the enumeration foundation for the
/// sidebar bulk-copy feature, but A10's actual consumer needs records,
/// not paths: `CopyPlanner.planGroupsWithoutDuplicates` /
/// `planGroupsWithDuplicates` consume `[ImageRecord]` because they read
/// `fileName`, `captureDatetime`, `createdTimestamp`, `filePath`, and
/// the `(file_size, file_name, capture_datetime)` three-field equality
/// tuple. So A10 calls the records sibling `get_image_records_for_filters`
/// instead, and this function ships unused.
///
/// **Why it stays in the build**: the natural future consumer is the
/// **Query / Filter Builder** (the v1 late-capstone feature, briefly
/// described in CLAUDE.md §"Three-Surface Architectural Philosophy"
/// point 3). A query builder that wants to enumerate the file_paths of
/// an arbitrary result set — for export, scripting, "open in Finder",
/// or any path-only consumer — can take this function as-is. The
/// projection-only design saves the ~94MB → ~4MB hit at catalogue scale
/// when only paths are needed.
///
/// **MARK FOR POTENTIAL DELETION (Session 24)**: if the Query Builder
/// is designed and turns out NOT to need path-only enumeration (or to
/// need it through a different surface — e.g., result-set-id-based
/// rather than predicate-based), this function AND `FilePathsResult`
/// AND `execute_file_path_projection_query` become safe to remove. Do
/// not delete blindly — confirm no other callers first.
///
/// ----
///
/// **Why** (original A9 rationale, retained for context): copying every
/// file in the current filtered view would require dragging the full
/// ~94MB `ImageRecord` payload across the FFI just to read one column.
/// This function returns the same set of matching rows projected down
/// to `file_path` only (~4MB at catalogue scale).
///
/// **Parity with the gallery (by construction, not parallel
/// reimplementation)**: this function builds its predicate via the
/// shared `build_path_date_predicate` helper and executes via
/// `execute_file_path_projection_query`, which composes the same
/// `RAW_JPEG_COLLAPSE_PREDICATE` and `DUPLICATE_GROUP_ID_CASE` /
/// `DUPLICATE_FILTER_PREDICATE` constants used by
/// `get_images_for_path_prefix` and `get_image_count_for_filters`.
/// The set returned here equals the set the gallery would load for
/// the same `(path_prefix, date_prefix, apply_duplicate_filter,
/// apply_raw_jpeg_collapse)` tuple.
///
/// **Argument polarity**: the caller is responsible for translating
/// UI booleans to filter booleans (e.g., pass `!show_duplicates` for
/// `apply_duplicate_filter`). This function does NOT bake that polarity
/// in — matching the convention of every other filter-aware FFI here.
///
/// **Path contract**: `file_path` values are returned EXACTLY as stored
/// in the catalogue. No normalization, no canonicalization. Order is
/// UNSPECIFIED — caller must sort if a total order is required.
///
/// **Error vs empty**: see `FilePathsResult` doc.
///
/// Parameters:
/// - `path_prefix`: Directory path prefix (empty string for no filter).
/// - `date_prefix`: Date prefix in `YYYY[:MM[:DD]]` form (empty for no
///   filter).
/// - `apply_duplicate_filter`: When true, restrict to one row per
///   duplicate cluster (caller passes `!show_duplicates`).
/// - `apply_raw_jpeg_collapse`: When true, suppress RAW siblings of
///   JPEGs sharing `file_stem` within the same `directory_path`.
pub async fn get_file_paths_for_filters(
    path_prefix: String,
    date_prefix: String,
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
) -> FilePathsResult
{
    // Acquire lock and validate connection. Catalogue not initialized
    // is a hard failure — caller must distinguish from "zero matches".
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref()
    {
        Some(c) => c,
        None =>
        {
            eprintln!("Catalogue not initialized");
            return FilePathsResult {
                ok: false,
                paths: Vec::new(),
                error_message: Some("catalogue not initialized".to_string()),
            };
        }
    };

    // Build predicate via shared helper (single source of truth for the
    // four-arm path/date composition — see `build_path_date_predicate`).
    let predicate = build_path_date_predicate(&path_prefix, &date_prefix);

    match execute_file_path_projection_query(
        conn,
        &predicate,
        apply_duplicate_filter,
        apply_raw_jpeg_collapse,
    )
    {
        Ok(paths) => FilePathsResult {
            ok: true,
            paths,
            error_message: None,
        },
        Err(msg) => FilePathsResult {
            ok: false,
            paths: Vec::new(),
            error_message: Some(msg),
        },
    }
}

/// A10 (sidebar bulk-copy records enumeration): return the full
/// `ImageRecord` set matching the gallery's current filter state via
/// the records-projection helper — no LIMIT/OFFSET, no ORDER BY.
///
/// Records-returning sibling of A9's `get_file_paths_for_filters`.
/// Chosen over A9's path-projection output because the copy planner
/// (`CopyPlanner.planGroupsWithoutDuplicates` /
/// `planGroupsWithDuplicates`) consumes `[ImageRecord]` directly —
/// reusing A9 would require a second round trip to inflate paths back
/// into records. See A9 doc for the path-projection rationale and its
/// future-consumer (Query/Filter Builder) marker.
///
/// **Parity-by-construction**: shares the same `build_path_date_predicate`
/// helper and the same filter constants as
/// `get_images_for_path_prefix` / `get_image_count_for_filters` /
/// `get_file_paths_for_filters`. A change to any predicate constant
/// updates every consumer in lockstep — preventing the four-arm match
/// drift that motivated the Layer B refactor.
///
/// **Argument polarity**: caller translates UI booleans to filter
/// booleans (e.g., pass `!show_duplicates` for `apply_duplicate_filter`).
/// A10's sidebar callers pass `false, false` — RAW CATALOGUE TRUTH,
/// independent of the gallery's view preferences.
///
/// **Order**: UNSPECIFIED. The planner owns sort order (selection-scoped
/// dedup + collision naming pass), so returning unsorted rows is correct
/// and avoids paying for an ORDER BY the consumer would override.
///
/// **Error vs empty**: mirrors the established record-returning
/// convention (NOT the A9 wrapper-dict idiom). Failure returns an empty
/// `Vec<ImageRecord>` with `eprintln!` diagnostics on the Rust side.
/// The caller cannot distinguish failure from zero matches, but for the
/// sidebar bulk-copy flow this is acceptable: an empty result correctly
/// surfaces in the UI as "nothing to copy" regardless of root cause,
/// and the diagnostic trail lives in the console log.
pub async fn get_image_records_for_filters(
    path_prefix: String,
    date_prefix: String,
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
) -> Vec<ImageRecord>
{
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

    let predicate = build_path_date_predicate(&path_prefix, &date_prefix);

    // `execute_image_record_projection_query` returns Vec<ImageRecord>
    // directly (NOT Result) — matching the established convention of
    // every other record-returning helper in this file. SQL errors are
    // logged inside the helper and surface here as an empty Vec, which
    // the sidebar UI correctly renders as "nothing to copy".
    execute_image_record_projection_query(
        conn,
        &predicate,
        apply_duplicate_filter,
        apply_raw_jpeg_collapse,
    )
}

/// Session 31: catalogue-only bulk DELETE for the Sources sidebar's
/// "Remove from Catalogue" context-menu action. Removes every row
/// matching the same path/date predicate the gallery reader uses —
/// touches NO files and NO thumbnails.
///
/// **Predicate parity-by-construction.** Reuses
/// `build_path_date_predicate`, the same builder consumed by
/// `get_image_records_for_filters` and the rest of the A10 family.
/// The set of rows deleted is, by construction, the set of rows that
/// would have been listed at that sidebar node. A future change to
/// the predicate updates reader and remover in lockstep.
///
/// **Safety guard: empty-predicate refusal.** When both prefixes are
/// empty, `build_path_date_predicate` returns the empty string. We
/// refuse the call and return 0 rather than emit
/// `DELETE FROM images WHERE ` (syntax error) — and, more importantly,
/// we never want a future caller change to silently produce
/// `DELETE FROM images` (full catalogue wipe). The current Sources
/// caller always passes a non-empty path_prefix, so this guard never
/// fires in practice; it exists to bound the blast radius.
///
/// **Return.** Number of rows deleted, as i64. Returns 0 on every
/// failure mode (catalogue not initialized, empty-predicate refusal,
/// SQL error) — mirroring the failure-as-zero convention of
/// `get_image_count_for_path_prefix`. Diagnostic detail goes to
/// stderr. The caller cannot distinguish "zero matched" from "error";
/// the Swift trace and the post-remove notification carry the user-
/// facing surface, and a zero return correctly produces no UI change.
pub async fn remove_images_for_filters(
    path_prefix: String,
    date_prefix: String,
) -> i64
{
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

    let predicate = build_path_date_predicate(&path_prefix, &date_prefix);
    if predicate.is_empty()
    {
        eprintln!(
            "remove_images_for_filters refused: empty predicate \
             (path_prefix and date_prefix both empty)"
        );
        return 0;
    }

    let delete_sql = format!("DELETE FROM images WHERE {}", predicate);
    match conn.execute(&delete_sql, [])
    {
        Ok(rows_changed) => rows_changed as i64,
        Err(e) =>
        {
            eprintln!("remove_images_for_filters DELETE failed: {}", e);
            0
        }
    }
}

/// Session 30 (cross-plan overwrite-gap fix): return every catalogued
/// `ImageRecord` in the destination "family" at a given basePath —
/// i.e. the canonical row at `<dir>/<canonical_file_name>` plus any
/// version-prefixed rows at `<dir>/NN_version_<canonical_file_name>`.
///
/// **Why this exists.** Two sequential removable-storage card imports
/// of cards each containing distinct-content `RSW_0001.NEF` are two
/// SEPARATE plans; Card B's planner has no in-plan knowledge of Card
/// A's already-written file. The shipped `suppressIdenticalAtDestination`
/// pre-seeded `versionCountByBasePath` ONLY on the suppression
/// (identical-match) branch — when the on-disk file was a DISTINCT
/// image (size mismatch → not identical → no pre-seed), the next
/// distinct tuple got the clean canonical path and the engine's
/// residual `removeItem`-then-copy SILENTLY OVERWROTE it. See the
/// "Copy To — Cross-Plan Overwrite Gap" section in CLAUDE.md.
///
/// **Design (catalogue-as-truth).** The planner queries the catalogue
/// for the family at each basePath BEFORE collision naming. The
/// family count → pre-seed `versionCountByBasePath` (the count is the
/// number of slots already reserved on disk; a new distinct tuple
/// must be forced to `(N+1)_version_`). The pre-seed fires
/// UNCONDITIONALLY of identical-match — no disk-confirm — because the
/// stakes are asymmetric: an extra seed only forces a `NN_version_`
/// prefix on a tuple, harmless; a missed seed re-introduces the
/// overwrite bug. The disk-confirm stays on the SUPPRESSION decision
/// (planner's existing identical-at-destination check), where stale-
/// catalogue stakes are real ("skip a copy that should occur").
///
/// **Parity-by-construction with ingest.** The predicate built by
/// `build_destination_family_predicate` pivots on `directory_path`
/// EQUALITY against `SUBSTRING(?1, 1, LENGTH(?1) - INSTR(REVERSE(?1),
/// '/'))` applied to a sample destination `file_path` — the SAME
/// expression used at ingest time (see the `insert_sql` block above,
/// `directory_path` column). The caller passes a sample destination
/// `file_path` (e.g. the planner's computed `destinationRoot + '/' +
/// 'YYYY/MM_monthname/DD/' + canonical_file_name`); Rust derives
/// `directory_path` from THAT string via the same SQL. There is no
/// Swift-computed directory string in flight to drift; the equality
/// cannot silently return empty. (Item #3 in the Session-30 read-and-
/// confirm — match-by-construction or no fix.)
///
/// **Filters: catalogue truth, not gallery view.** Passes `false,
/// false` to `execute_image_record_projection_query` — neither the
/// duplicate filter nor the RAW+JPEG collapse predicate may suppress
/// rows here. The family is the catalogue's complete record of what
/// occupies the basePath; gallery view preferences are irrelevant to
/// "what does disk hold?". Matches the A10 sidebar-copy convention.
///
/// **Order**: UNSPECIFIED. The planner counts (and inspects for
/// existing `NN_version_*` to pre-seed HIGHER than 1 — handled Swift-
/// side); sort order is not part of the contract.
///
/// **Error vs empty**: mirrors the established record-returning
/// convention (NOT the A9 wrapper-dict idiom — see
/// `get_image_records_for_filters`). Failure returns an empty
/// `Vec<ImageRecord>` with `eprintln!` diagnostics on the Rust side.
/// For the planner's pre-seed flow, empty-on-failure degrades
/// gracefully to "no family found" → no pre-seed (the pre-fix
/// behavior) — the planner does not get worse than before in the
/// failure mode.
///
/// Parameters:
/// - sample_file_path: a sample destination `file_path` whose
///   `directory_path` (via SUBSTRING/INSTR/REVERSE on this string) is
///   the basePath directory to query. Typically the planner's
///   computed canonical destination path; need not refer to an
///   actually-catalogued row.
/// - canonical_file_name: the canonical (un-prefixed) filename for
///   the basePath collision group, e.g. `RSW_0001.NEF`.
pub async fn get_destination_family_records(
    sample_file_path: String,
    canonical_file_name: String,
) -> Vec<ImageRecord>
{
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

    let predicate = build_destination_family_predicate(
        &sample_file_path,
        &canonical_file_name,
    );

    execute_image_record_projection_query(
        conn,
        &predicate,
        false,
        false,
    )
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
        // Session 41: HEIF has no RAW↔HEIF counterpart lookup yet (the
        // "Open RAW / Open JPEG" submenu enhancement is deliberately
        // deferred). HEIF still participates in grid pair-COLLAPSE via
        // RAW_JPEG_COLLAPSE_PREDICATE; only this menu-counterpart path
        // returns None for now.
        ImageKind::Heif => return None,
        // Lightroom import: DNG/PSD/TIFF/PNG are standalone kinds with no
        // JPEG↔RAW menu-counterpart lookup (Docs/DESIGN-Lightroom-Catalog-Import.md §7).
        ImageKind::Dng | ImageKind::Psd | ImageKind::Tiff | ImageKind::Png => return None,
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
