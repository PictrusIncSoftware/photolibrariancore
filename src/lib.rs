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
            file_size INTEGER NOT NULL,
            file_name TEXT NOT NULL,
            file_extension TEXT,
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

            -- Audit timestamp: when this record was added to the catalogue
            indexed_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

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
    let insert_sql = r#"
        INSERT OR IGNORE INTO images (
            file_path, file_size, file_name, file_extension,
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
            ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18,
            ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27
        )
    "#;

    let mut inserted_count = 0u32;

    // Iterate through each metadata record and insert individually
    // Per-record approach: Better error isolation at the cost of slightly lower throughput
    for record in metadata {
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
                record.created_timestamp,                   // ?5
                record.modified_timestamp,                  // ?6
                record.camera_make,                         // ?7
                record.camera_model,                        // ?8
                record.lens_model,                          // ?9
                record.focal_length,                        // ?10
                record.aperture,                            // ?11
                record.shutter_speed,                       // ?12
                record.iso.map(|v| v as i64),              // ?13 (u32 → i64)
                record.capture_datetime,                    // ?14
                record.pixel_width.map(|v| v as i64),      // ?15 (u32 → i64)
                record.pixel_height.map(|v| v as i64),     // ?16 (u32 → i64)
                record.color_space,                         // ?17
                record.bit_depth.map(|v| v as i64),        // ?18 (u32 → i64)
                record.gps_latitude,                        // ?19
                record.gps_longitude,                       // ?20
                record.gps_altitude,                        // ?21
                record.copyright,                           // ?22
                record.creator,                             // ?23
                record.description,                         // ?24
                record.rating.map(|v| v as i64),           // ?25 (u8 → i64)
                record.flag,                                // ?26
                record.color_label,                         // ?27
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

/// Get all images from the catalogue
///
/// Returns a complete list of all image records in the catalogue, ordered by ID.
/// This is a development/debugging function for viewing catalogue contents.
///
/// Design decision: Returns all records with no LIMIT for v1. The function signature
/// could be extended with limit/offset parameters in the future if needed for pagination.
///
/// Data flow:
/// - Swift calls this function to populate a browse/debug view
/// - Rust queries all records from the images table
/// - Returns full ImageRecord structs including database-generated fields
///
/// Returns:
/// - Vec of ImageRecord structs, empty vec if catalogue is empty or not initialized
pub async fn get_all_images() -> Vec<ImageRecord> {
    // Acquire lock and validate connection
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    // Query all images, ordered by ID
    // Note: indexed_timestamp is excluded because DuckDB's TIMESTAMP type
    // doesn't have a direct String conversion in the FFI layer
    let query_sql = r#"
        SELECT
            id, file_path, file_size, file_name, file_extension,
            created_timestamp, modified_timestamp,
            camera_make, camera_model, lens_model,
            focal_length, aperture, shutter_speed, iso,
            capture_datetime,
            pixel_width, pixel_height, color_space, bit_depth,
            gps_latitude, gps_longitude, gps_altitude,
            copyright, creator, description,
            rating, flag, color_label
        FROM images
        ORDER BY id
    "#;

    // Execute query and collect results
    let mut stmt = match conn.prepare(query_sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to prepare query: {}", e);
            return Vec::new();
        }
    };

    let rows = match stmt.query_map([], |row| {
        // Extract all columns from the row
        // Type conversions: i64 from DuckDB → u64/u32/u8 for Rust
        // Note: indexed_timestamp is set to empty string as placeholder since
        // we don't query it (TIMESTAMP type doesn't have direct String conversion)
        Ok(ImageRecord {
            id: row.get(0)?,
            indexed_timestamp: String::new(),  // Placeholder - not queried from DB
            file_path: row.get(1)?,
            file_size: row.get::<_, i64>(2)? as u64,
            file_name: row.get(3)?,
            file_extension: row.get(4)?,
            created_timestamp: row.get(5)?,
            modified_timestamp: row.get(6)?,
            camera_make: row.get(7)?,
            camera_model: row.get(8)?,
            lens_model: row.get(9)?,
            focal_length: row.get(10)?,
            aperture: row.get(11)?,
            shutter_speed: row.get(12)?,
            iso: row.get::<_, Option<i64>>(13)?.map(|v| v as u32),
            capture_datetime: row.get(14)?,
            pixel_width: row.get::<_, Option<i64>>(15)?.map(|v| v as u32),
            pixel_height: row.get::<_, Option<i64>>(16)?.map(|v| v as u32),
            color_space: row.get(17)?,
            bit_depth: row.get::<_, Option<i64>>(18)?.map(|v| v as u32),
            gps_latitude: row.get(19)?,
            gps_longitude: row.get(20)?,
            gps_altitude: row.get(21)?,
            copyright: row.get(22)?,
            creator: row.get(23)?,
            description: row.get(24)?,
            rating: row.get::<_, Option<i64>>(25)?.map(|v| v as u8),
            flag: row.get(26)?,
            color_label: row.get(27)?,
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
