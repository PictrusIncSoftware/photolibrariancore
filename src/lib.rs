// Import DuckDB for embedded database operations
// DuckDB is used instead of SQLite for better analytical query performance on large image catalogues
use arrow_array::types::Float32Type;
use arrow_array::{
    Array, FixedSizeListArray, Float64Array, Int64Array, RecordBatch, RecordBatchIterator,
    StringArray, UInt32Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use duckdb::{params, Connection};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

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

static CATALOGUE_PATH: once_cell::sync::Lazy<Arc<Mutex<Option<PathBuf>>>> =
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
    pub file_path: String, // Absolute path; passed from Swift's sandbox-legal picker
    pub file_size: u64,
    pub file_name: String,              // Basename extracted from path
    pub file_extension: Option<String>, // Lowercase extension (e.g., "jpg", "cr3")
    pub created_timestamp: i64,         // Unix epoch seconds
    pub modified_timestamp: i64,        // Unix epoch seconds

    // Camera/capture metadata (from EXIF, may be absent)
    pub camera_make: Option<String>,  // e.g., "Canon", "Nikon"
    pub camera_model: Option<String>, // e.g., "EOS R5", "D850"
    pub lens_model: Option<String>,
    pub focal_length: Option<f64>,        // Millimeters
    pub aperture: Option<f64>,            // F-stop value
    pub shutter_speed: Option<f64>,       // Seconds (e.g., 0.0005 for 1/2000)
    pub iso: Option<u32>,                 // ISO speed rating
    pub capture_datetime: Option<String>, // ISO 8601 string from EXIF

    // Image properties (resolution and color)
    pub pixel_width: Option<u32>,
    pub pixel_height: Option<u32>,
    pub color_space: Option<String>, // e.g., "sRGB", "Adobe RGB"
    pub bit_depth: Option<u32>,      // Bits per channel

    // GPS coordinates (from geotagged images)
    pub gps_latitude: Option<f64>,  // Decimal degrees
    pub gps_longitude: Option<f64>, // Decimal degrees
    pub gps_altitude: Option<f64>,  // Meters above sea level

    // IPTC/copyright metadata (user-editable in many cameras/software)
    pub copyright: Option<String>,
    pub creator: Option<String>,     // Photographer name
    pub description: Option<String>, // Caption/alt text

    // Organization metadata (user-assigned within PhotoLibrarian)
    // These fields are initially None and populated by user interactions in the app
    pub rating: Option<u8>,          // 0-5 stars
    pub flag: Option<String>,        // "pick", "reject", or None
    pub color_label: Option<String>, // "red", "green", "blue", etc.

    // S67 (Copy and Import): the source record's in-app rotation, inherited by
    // a catalogued copy so it displays like its original (the aliased
    // thumbnail is baked-rotated — a 0 here would contradict it). APPENDED
    // LAST with a UDL `= null` default (the S65 wire-struct growth rule) so
    // every existing Swift construction site compiles untouched; None at
    // INSERT takes the schema default 0, None at UPDATE preserves the row's
    // value (a Lightroom re-import can never clobber an in-app rotation).
    pub rotation: Option<i32>,

    // === Video / unified-media (Step 2a; DESIGN-Video-Schema-Unified-Table.md §4b).
    // Appended LAST, each with a UDL default (`= false` / `= null`), so every
    // existing Swift construction site (makeImageMetadata, the LR import) compiles
    // untouched and produces stills (is_video=false, the rest None). The
    // AVFoundation extractor (Step 2b) is the only producer that populates these. ===
    pub is_video: bool,                  // discriminator; false for every still
    pub duration_seconds: Option<f64>,   // seconds
    pub frame_rate: Option<f64>,         // fps
    pub video_kind: Option<String>,      // container: "mov" / "mp4" / "mxf"
    pub video_codec: Option<String>,     // "hevc" / "prores" / "h264"
    pub video_bitrate: Option<i64>,      // bits/sec
    pub color_primaries: Option<String>, // CICP canonical strings (bt2020 / smpte432 / bt709)
    pub color_transfer: Option<String>,  // arib-std-b67 (HLG) / smpte2084 (PQ) / bt709
    pub color_matrix: Option<String>,    // bt2020nc / smpte170m / bt709
    pub color_range: Option<String>,     // "tv" / "pc"
    pub dv_profile: Option<i32>,         // Dolby Vision profile (8 on iPhone); None = none
    pub has_audio: Option<bool>,
    pub audio_codec: Option<String>, // aac / pcm_s16le / pcm_s24le
    pub audio_channels: Option<i32>,
    pub audio_sample_rate: Option<i32>,
    pub audio_bitrate: Option<i64>,    // bits/sec
    pub live_photo_id: Option<String>, // QuickTime content.identifier; pair on equality

    // Apple Photos import (in-place catalogue-everything model). The asset's
    // Apple identity (ZASSET.ZUUID) — durable provenance and the PhotoKit handle
    // for the future on-demand materialize of an iCloud-only original. Non-null
    // marks an Apple-managed row. APPENDED LAST with a UDL `= null` default (the
    // S65 wire-struct growth rule) so every existing construction site (scan
    // ingest, the Lightroom import) compiles untouched and writes NULL.
    pub external_source_id: Option<String>,
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
    pub id: i64,                   // Auto-generated primary key
    pub indexed_timestamp: String, // When record was added to catalogue (ISO 8601)

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

/// One still-image row that needs focus analysis.
///
/// This deliberately avoids `ImageRecord`: focus analysis only needs a stable id
/// and path. Keeping the carrier narrow avoids the hot bulk-record lift while
/// giving Swift enough information to run a cloud-safe, off-main analyzer.
#[derive(Debug, Clone)]
pub struct FocusAnalysisCandidate {
    pub id: i64,
    pub file_path: String,
    pub file_size: u64,
}

/// One still-image row eligible for similar-photo grouping.
///
/// Similar grouping is computed in Swift because Vision owns the feature-print
/// observations. The core supplies a narrow, ordered candidate carrier so the
/// app does not need to lift full ImageRecord rows just to read paths.
#[derive(Debug, Clone)]
pub struct SimilarPhotoCandidate {
    pub id: i64,
    pub file_path: String,
    pub file_size: u64,
    pub created_timestamp: i64,
    pub capture_datetime: Option<String>,
    pub directory_path: Option<String>,
    pub camera_model: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SimilarPhotoFeatureprint {
    pub image_id: i64,
    pub source_stamp: String,
    pub featureprint_blob: Vec<u8>,
}

/// One durable similar-photo stack membership row.
///
/// The current production representation stores only group membership, not the
/// Vision feature vector. A row exists only for photos that belong to a
/// multi-photo group; singletons have no membership row.
#[derive(Debug, Clone)]
pub struct SimilarPhotoGroupMember {
    pub image_id: i64,
    pub group_id: i64,
    pub representative_id: i64,
    pub member_rank: u32,
    pub distance_to_representative: Option<f64>,
    pub threshold: f64,
}

#[derive(Debug, Clone)]
pub struct SimilarPhotoWorkUnit {
    pub unit_index: i64,
    pub start_image_id: i64,
    pub end_image_id: i64,
    pub candidate_count: i64,
    pub member_count: i64,
}

/// Compact gallery metadata for a visible similar-photo representative.
///
/// `logical_count` collapses RAW/JPEG siblings with the same directory/stem;
/// `physical_count` is the underlying row count available to a future expanded
/// stack/deck view.
#[derive(Debug, Clone)]
pub struct SimilarPhotoStackSummary {
    pub image_id: i64,
    pub group_id: i64,
    pub logical_count: u32,
    pub physical_count: u32,
}

/// Expanded physical membership for one or more selected visible stack rows.
#[derive(Debug, Clone)]
pub struct SimilarPhotoStackMember {
    pub image_id: i64,
    pub group_id: i64,
    pub representative_id: i64,
    pub member_rank: u32,
}

/// One focus-analysis writeback row.
///
/// `status` is an allow-listed token owned by the analyzer:
/// - `complete`: score is present
/// - `online_only`: source bytes are unavailable without a cloud download
/// - `unreadable`: source could not be decoded/read
/// - `failed`: unexpected analyzer failure
#[derive(Debug, Clone)]
pub struct FocusAnalysisResult {
    pub id: i64,
    pub focus_score: Option<f64>,
    pub focus_basis: Option<String>,
    pub algorithm_version: String,
    pub analysis_run_id: String,
    pub status: String,
    pub focus_human_score: Option<f64>,
    pub focus_animal_score: Option<f64>,
    pub focus_foreground_score: Option<f64>,
    pub focus_saliency_score: Option<f64>,
    pub focus_animal_pose_score: Option<f64>,
    pub focus_whole_image_score: Option<f64>,
    pub face_count: Option<i32>,
    pub face_quality_best: Option<f64>,
    pub face_quality_average: Option<f64>,
    pub face_quality_min: Option<f64>,
    pub face_eyes_open_count: Option<i32>,
    pub face_blink_risk_count: Option<i32>,
    pub auto_keywords: Vec<String>,
    pub face_observations: Vec<FaceObservationResult>,
}

/// One detected human face from the Vision enrichment pass.
///
/// These rows are scalar geometry/quality facts stored in DuckDB. Future
/// AuraFace embeddings should live in the vector store keyed by the durable
/// `face_observation.id`.
#[derive(Debug, Clone)]
pub struct FaceObservationResult {
    pub face_index: u32,
    pub bounding_box_x: f64,
    pub bounding_box_y: f64,
    pub bounding_box_width: f64,
    pub bounding_box_height: f64,
    pub detection_confidence: Option<f64>,
    pub face_capture_quality: Option<f64>,
    pub face_focus_score: Option<f64>,
    pub left_eye_open_score: Option<f64>,
    pub right_eye_open_score: Option<f64>,
    pub eyes_open_score: Option<f64>,
    pub blink_risk_score: Option<f64>,
    pub left_eye_x: Option<f64>,
    pub left_eye_y: Option<f64>,
    pub right_eye_x: Option<f64>,
    pub right_eye_y: Option<f64>,
    pub nose_x: Option<f64>,
    pub nose_y: Option<f64>,
    pub mouth_left_x: Option<f64>,
    pub mouth_left_y: Option<f64>,
    pub mouth_right_x: Option<f64>,
    pub mouth_right_y: Option<f64>,
}

/// Durable face-observation row read back from the catalogue.
///
/// This is the production read model for the `face_observation` table. Swift
/// can use it to crop/align faces for the recognition pass and to display
/// diagnostics without querying DuckDB directly.
#[derive(Debug, Clone)]
pub struct FaceObservationRecord {
    pub id: i64,
    pub image_id: i64,
    pub analyzed_image_id: i64,
    pub face_index: u32,
    pub algorithm_version: String,
    pub analysis_run_id: String,
    pub bounding_box_x: f64,
    pub bounding_box_y: f64,
    pub bounding_box_width: f64,
    pub bounding_box_height: f64,
    pub detection_confidence: Option<f64>,
    pub face_capture_quality: Option<f64>,
    pub face_focus_score: Option<f64>,
    pub left_eye_open_score: Option<f64>,
    pub right_eye_open_score: Option<f64>,
    pub eyes_open_score: Option<f64>,
    pub blink_risk_score: Option<f64>,
    pub left_eye_x: Option<f64>,
    pub left_eye_y: Option<f64>,
    pub right_eye_x: Option<f64>,
    pub right_eye_y: Option<f64>,
    pub nose_x: Option<f64>,
    pub nose_y: Option<f64>,
    pub mouth_left_x: Option<f64>,
    pub mouth_left_y: Option<f64>,
    pub mouth_right_x: Option<f64>,
    pub mouth_right_y: Option<f64>,
    pub created_at: String,
}

#[derive(Debug, Clone)]
pub struct FaceEmbeddingVectorRecord {
    pub face_observation_id: i64,
    pub image_id: i64,
    pub analyzed_image_id: i64,
    pub face_index: u32,
    pub model_name: String,
    pub model_version: String,
    pub preprocessing_version: String,
    pub input_size: u32,
    pub color_order: String,
    pub normalization: String,
    pub embedding_dimension: u32,
    pub embedding_l2_norm: f64,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct FaceRecognitionMenuState {
    pub image_id: i64,
    pub analysis_status: Option<String>,
    pub face_observation_count: u32,
    pub indexed_face_count: u32,
}

#[derive(Debug, Clone)]
pub struct FaceEmbeddingStoreResult {
    pub requested_count: u64,
    pub stored_count: u64,
    pub total_count: u64,
    pub status: String,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct FaceEmbeddingNeighborRecord {
    pub query_face_observation_id: i64,
    pub neighbor_face_observation_id: i64,
    pub query_image_id: i64,
    pub neighbor_image_id: i64,
    pub query_analyzed_image_id: i64,
    pub neighbor_analyzed_image_id: i64,
    pub query_face_index: u32,
    pub neighbor_face_index: u32,
    pub cosine: f64,
}

#[derive(Debug, Clone)]
pub struct FaceSearchMatchRecord {
    pub image_id: i64,
    pub face_observation_id: i64,
    pub seed_face_observation_id: i64,
    pub face_index: u32,
    pub cosine: f64,
}

#[derive(Debug, Clone)]
pub struct FaceClusterRunRecord {
    pub run_id: String,
    pub face_algorithm_version: String,
    pub model_version: String,
    pub preprocessing_version: String,
    pub threshold: f64,
    pub cluster_count: u32,
    pub member_count: u32,
}

#[derive(Debug, Clone)]
pub struct FaceClusterRunSummary {
    pub run_id: String,
    pub face_algorithm_version: String,
    pub model_version: String,
    pub preprocessing_version: String,
    pub threshold: f64,
    pub cluster_count: u32,
    pub member_count: u32,
    pub created_at: String,
}

#[derive(Debug, Clone)]
pub struct FaceClusterMemberRecord {
    pub run_id: String,
    pub cluster_id: i64,
    pub face_observation_id: i64,
    pub image_id: i64,
    pub analyzed_image_id: i64,
    pub face_index: u32,
    pub member_rank: u32,
    pub cluster_size: u32,
    pub nearest_neighbor_face_observation_id: Option<i64>,
    pub nearest_neighbor_cosine: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct PersonClusterAcceptResult {
    pub person_id: i64,
    pub person_name: String,
    pub assigned_face_count: u64,
    pub keyword_image_count: u64,
    pub keyword_row_count: u64,
    pub keyword_path: String,
    pub status: String,
    pub message: String,
}

/// Durable coordinator row for enrichment / intelligent-culling work.
///
/// The current focus analyzer still runs in Swift, but the job identity and
/// counters live in DuckDB so a foreground run and a future helper process can
/// observe, cancel, and resume the same unit of work.
#[derive(Debug, Clone)]
pub struct AnalysisJob {
    pub id: i64,
    pub job_kind: String,
    pub scope_kind: String,
    pub scope_value: Option<String>,
    pub algorithm_version: String,
    pub analysis_run_id: String,
    pub status: String,
    pub total_candidate_count: u64,
    pub processed_count: u64,
    pub completed_count: u64,
    pub skipped_count: u64,
    pub failed_count: u64,
    pub updated_count: u64,
    pub cancel_requested: bool,
    pub created_at: String,
    pub started_at: Option<String>,
    pub updated_at: Option<String>,
    pub finished_at: Option<String>,
    pub last_error: Option<String>,
    pub current_image_id: Option<i64>,
    pub current_file_path: Option<String>,
    pub current_started_at: Option<String>,
    pub last_timeout_image_id: Option<i64>,
    pub last_timeout_file_path: Option<String>,
    pub last_timeout_at: Option<String>,
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
    "nef", // Nikon
    "cr2", // Canon (older)
    "cr3", // Canon (newer)
    "arw", // Sony
    // NOTE: "dng" intentionally NOT here — DNG is its own ImageKind::Dng
    // (Lightroom import, Docs/DESIGN-Lightroom-Catalog-Import.md §7), so it does
    // NOT pair-collapse as a RAW. Re-adding it here would regress that. See
    // DNG_EXTENSIONS below.
    "raf", // Fujifilm
    "rw2", // Panasonic
    "orf", // Olympus / OM System
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
pub enum ImageKind {
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
pub fn classify_extension(ext: String) -> ImageKind {
    let lower = ext.to_lowercase();
    if JPEG_EXTENSIONS.contains(&lower.as_str()) {
        ImageKind::Jpeg
    } else if HEIF_EXTENSIONS.contains(&lower.as_str()) {
        ImageKind::Heif
    } else if DNG_EXTENSIONS.contains(&lower.as_str()) {
        // Checked before RAW: DNG is its own kind, not a RAW (see DNG_EXTENSIONS).
        ImageKind::Dng
    } else if RAW_EXTENSIONS.contains(&lower.as_str()) {
        ImageKind::Raw
    } else if PSD_EXTENSIONS.contains(&lower.as_str()) {
        ImageKind::Psd
    } else if TIFF_EXTENSIONS.contains(&lower.as_str()) {
        ImageKind::Tiff
    } else if PNG_EXTENSIONS.contains(&lower.as_str()) {
        ImageKind::Png
    } else {
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
pub fn get_raw_extensions() -> Vec<String> {
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
pub fn get_jpeg_extensions() -> Vec<String> {
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
pub struct ParsedFilename {
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
pub fn parse_filename(file_name: String) -> ParsedFilename {
    // Empty filename: synthetic stem so the stored column is never empty.
    if file_name.is_empty() {
        return ParsedFilename {
            stem: "metype".to_string(),
            extension_lower: "metype".to_string(),
            kind: ImageKind::Other,
        };
    }

    match file_name.rfind('.') {
        // No dot at all — case 2.
        None => ParsedFilename {
            stem: file_name,
            extension_lower: "metype".to_string(),
            kind: ImageKind::Other,
        },
        // Leading dot, e.g. ".DS_Store" — case 2.
        // The whole name is treated as the stem; there is no extension to parse.
        Some(0) => ParsedFilename {
            stem: file_name,
            extension_lower: "metype".to_string(),
            kind: ImageKind::Other,
        },
        // Trailing dot, e.g. "foo." — case 2.
        // Note: idx is a byte offset; for the trailing-dot test we compare against
        // file_name.len() - 1. Byte offsets are safe here because '.' is single-byte
        // ASCII; rfind on a multi-byte string still returns a byte offset on a code
        // unit boundary.
        Some(idx) if idx == file_name.len() - 1 => ParsedFilename {
            stem: file_name,
            extension_lower: "metype".to_string(),
            kind: ImageKind::Other,
        },
        // Normal case: split at the last dot.
        Some(idx) => {
            let stem = file_name[..idx].to_string();
            let ext = file_name[idx + 1..].to_lowercase();
            let kind = classify_extension(ext.clone());
            ParsedFilename {
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

            -- === Intelligent culling / focus analysis (Session 76; Docs/DESIGN-Intelligent-Culling.md) ===
            -- Scalar machine-derived facts. User curation fields above remain
            -- untouched; focus analysis is advisory and filterable, never an
            -- automatic rating/flag/color write.
            focus_score DOUBLE,              -- Laplacian-variance score; higher = sharper
            focus_basis TEXT,                -- human_face / animal / foreground / saliency / animal_pose / whole_image / unknown
            focus_human_score DOUBLE,
            focus_animal_score DOUBLE,
            focus_foreground_score DOUBLE,
            focus_saliency_score DOUBLE,
            focus_animal_pose_score DOUBLE,
            focus_whole_image_score DOUBLE,
            focus_algorithm_version TEXT,    -- e.g. "laplacian-v1"
            focus_analysis_status TEXT,      -- complete / online_only / unreadable / failed
            focus_analysis_attempt_id TEXT,
            focus_scored_at TIMESTAMP,
            face_count INTEGER,              -- Vision-detected people count (0 = no faces detected)
            face_quality_best DOUBLE,        -- Best Apple Vision faceCaptureQuality in image
            face_quality_average DOUBLE,     -- Average faceCaptureQuality across scored faces
            face_quality_min DOUBLE,         -- Lowest faceCaptureQuality across scored faces
            face_eyes_open_count INTEGER,    -- Heuristic count of faces whose eyes appear open
            face_blink_risk_count INTEGER,   -- Heuristic count of faces with likely blink/closed eyes

            -- === Video / unified-media support (Session 70; Docs/DESIGN-Video-Schema-Unified-Table.md) ===
            -- is_video discriminates stills (FALSE) from video (TRUE). CREATE-time
            -- default is safe; the ALTER path below adds it WITHOUT a default then
            -- backfills FALSE (S62 WAL rule). Video rows leave the EXIF columns NULL
            -- and stills leave these NULL — columnar storage makes that nearly free.
            is_video BOOLEAN NOT NULL DEFAULT FALSE,

            -- Video stream (NULL for stills)
            duration_seconds DOUBLE,             -- container duration, seconds
            frame_rate DOUBLE,                   -- nominal fps (e.g. 29.97)
            video_kind TEXT,                     -- container: 'mov' / 'mp4' / 'mxf'
            video_codec TEXT,                    -- 'hevc' / 'prores' / 'h264'
            video_bitrate BIGINT,                -- bits per second

            -- Color science (CICP — applies to HDR stills too; distinct from
            -- color_space above, which is the still's ICC profile name)
            color_primaries TEXT,                -- 'bt2020' / 'smpte432' (P3) / 'bt709'
            color_transfer TEXT,                 -- 'arib-std-b67' (HLG) / 'smpte2084' (PQ) / 'bt709'
            color_matrix TEXT,                   -- 'bt2020nc' / 'smpte170m' / 'bt709'
            color_range TEXT,                    -- 'tv' (limited) / 'pc' (full)
            dv_profile INTEGER,                  -- Dolby Vision profile (8 on iPhone); NULL = none

            -- Audio stream (NULL for stills / silent video)
            has_audio BOOLEAN,
            audio_codec TEXT,                    -- 'aac' / 'pcm_s16le' / 'pcm_s24le'
            audio_channels INTEGER,              -- 1 / 2
            audio_sample_rate INTEGER,           -- 44100 / 48000
            audio_bitrate BIGINT,                -- bits per second

            -- Live Photo: QuickTime content.identifier UUID, shared by the still +
            -- motion pair. NULL = not a Live Photo. Pair = rows with equal value.
            live_photo_id TEXT,

            -- Apple Photos import provenance (Apple Photos in-place catalogue).
            -- ZASSET.ZUUID of the source asset; non-NULL marks an Apple-managed
            -- row and is the PhotoKit handle for on-demand materialize. NULL for
            -- scanned / Lightroom-imported rows.
            external_source_id TEXT,

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

        -- Intelligent culling / focus-analysis columns (Session 76). ADD COLUMN
        -- with NO default (S62 WAL rule). Existing rows stay NULL and therefore
        -- naturally form the pending analysis queue.
        ALTER TABLE images ADD COLUMN IF NOT EXISTS focus_score DOUBLE;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS focus_basis TEXT;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS focus_human_score DOUBLE;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS focus_animal_score DOUBLE;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS focus_foreground_score DOUBLE;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS focus_saliency_score DOUBLE;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS focus_animal_pose_score DOUBLE;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS focus_whole_image_score DOUBLE;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS focus_algorithm_version TEXT;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS focus_analysis_status TEXT;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS focus_analysis_attempt_id TEXT;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS focus_scored_at TIMESTAMP;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS face_count INTEGER;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS face_quality_best DOUBLE;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS face_quality_average DOUBLE;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS face_quality_min DOUBLE;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS face_eyes_open_count INTEGER;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS face_blink_risk_count INTEGER;

        -- Video / unified-media columns (Session 70). ADD COLUMN IF NOT EXISTS with
        -- NO default (an ALTER ... ADD COLUMN ... DEFAULT <expr> does not survive
        -- WAL replay — S62), then backfill only the is_video discriminator; the
        -- rest are correctly NULL on existing stills.
        ALTER TABLE images ADD COLUMN IF NOT EXISTS is_video BOOLEAN;
        UPDATE images SET is_video = FALSE WHERE is_video IS NULL;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS duration_seconds DOUBLE;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS frame_rate DOUBLE;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS video_kind TEXT;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS video_codec TEXT;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS video_bitrate BIGINT;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS color_primaries TEXT;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS color_transfer TEXT;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS color_matrix TEXT;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS color_range TEXT;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS dv_profile INTEGER;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS has_audio BOOLEAN;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS audio_codec TEXT;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS audio_channels INTEGER;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS audio_sample_rate INTEGER;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS audio_bitrate BIGINT;
        ALTER TABLE images ADD COLUMN IF NOT EXISTS live_photo_id TEXT;

        -- Apple Photos import provenance (Apple Photos in-place catalogue).
        -- ADD COLUMN with NO default (S62 WAL rule); existing rows stay NULL,
        -- which correctly means "not an Apple-managed asset".
        ALTER TABLE images ADD COLUMN IF NOT EXISTS external_source_id TEXT;

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
        CREATE INDEX IF NOT EXISTS idx_focus_score ON images(focus_score);
        CREATE INDEX IF NOT EXISTS idx_focus_human_score ON images(focus_human_score);
        CREATE INDEX IF NOT EXISTS idx_focus_animal_score ON images(focus_animal_score);
        CREATE INDEX IF NOT EXISTS idx_focus_foreground_score ON images(focus_foreground_score);
        CREATE INDEX IF NOT EXISTS idx_focus_saliency_score ON images(focus_saliency_score);
        CREATE INDEX IF NOT EXISTS idx_focus_animal_pose_score ON images(focus_animal_pose_score);
        CREATE INDEX IF NOT EXISTS idx_focus_whole_image_score ON images(focus_whole_image_score);
        CREATE INDEX IF NOT EXISTS idx_focus_analysis_status ON images(focus_analysis_status);
        CREATE INDEX IF NOT EXISTS idx_face_count ON images(face_count);
        CREATE INDEX IF NOT EXISTS idx_face_quality_best ON images(face_quality_best);
        CREATE INDEX IF NOT EXISTS idx_face_quality_average ON images(face_quality_average);
        CREATE INDEX IF NOT EXISTS idx_face_quality_min ON images(face_quality_min);
        CREATE INDEX IF NOT EXISTS idx_face_eyes_open_count ON images(face_eyes_open_count);
        CREATE INDEX IF NOT EXISTS idx_face_blink_risk_count ON images(face_blink_risk_count);

        -- === Face observations (Vision detection, pre-recognition) ===
        -- One row per detected human face per image/algorithm version. This is
        -- the durable scalar side of face detection: geometry, capture quality,
        -- focus, eye-state measurements, and coarse alignment landmarks. Future
        -- AuraFace embeddings belong in LanceDB keyed by face_observation.id.
        CREATE SEQUENCE IF NOT EXISTS face_observation_id_seq START 1;

        CREATE TABLE IF NOT EXISTS face_observation (
            id INTEGER PRIMARY KEY DEFAULT nextval('face_observation_id_seq'),
            image_id INTEGER NOT NULL,
            analyzed_image_id INTEGER NOT NULL,
            face_index INTEGER NOT NULL,
            algorithm_version TEXT NOT NULL,
            analysis_run_id TEXT NOT NULL,
            bounding_box_x DOUBLE NOT NULL,
            bounding_box_y DOUBLE NOT NULL,
            bounding_box_width DOUBLE NOT NULL,
            bounding_box_height DOUBLE NOT NULL,
            detection_confidence DOUBLE,
            face_capture_quality DOUBLE,
            face_focus_score DOUBLE,
            left_eye_open_score DOUBLE,
            right_eye_open_score DOUBLE,
            eyes_open_score DOUBLE,
            blink_risk_score DOUBLE,
            left_eye_x DOUBLE,
            left_eye_y DOUBLE,
            right_eye_x DOUBLE,
            right_eye_y DOUBLE,
            nose_x DOUBLE,
            nose_y DOUBLE,
            mouth_left_x DOUBLE,
            mouth_left_y DOUBLE,
            mouth_right_x DOUBLE,
            mouth_right_y DOUBLE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (image_id, algorithm_version, face_index)
        );

        CREATE INDEX IF NOT EXISTS idx_face_observation_image ON face_observation(image_id);
        CREATE INDEX IF NOT EXISTS idx_face_observation_analyzed_image ON face_observation(analyzed_image_id);
        CREATE INDEX IF NOT EXISTS idx_face_observation_algorithm ON face_observation(algorithm_version);
        CREATE INDEX IF NOT EXISTS idx_face_observation_run ON face_observation(analysis_run_id);

        -- === Face recognition cluster diagnostics (developer-only first pass) ===
        -- AuraFace vectors live in LanceDB keyed by face_observation.id. These
        -- DuckDB tables persist the scalar result of one clustering run so
        -- diagnostics can prove readback before any person/naming UI exists.
        CREATE TABLE IF NOT EXISTS face_cluster_run (
            run_id TEXT PRIMARY KEY,
            face_algorithm_version TEXT NOT NULL,
            model_version TEXT NOT NULL,
            preprocessing_version TEXT NOT NULL,
            threshold DOUBLE NOT NULL,
            cluster_count INTEGER NOT NULL,
            member_count INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS face_cluster_member (
            run_id TEXT NOT NULL,
            cluster_id INTEGER NOT NULL,
            face_observation_id INTEGER NOT NULL,
            image_id INTEGER NOT NULL,
            analyzed_image_id INTEGER NOT NULL,
            face_index INTEGER NOT NULL,
            member_rank INTEGER NOT NULL,
            cluster_size INTEGER NOT NULL,
            nearest_neighbor_face_observation_id INTEGER,
            nearest_neighbor_cosine DOUBLE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (run_id, face_observation_id)
        );

        CREATE INDEX IF NOT EXISTS idx_face_cluster_member_run ON face_cluster_member(run_id);
        CREATE INDEX IF NOT EXISTS idx_face_cluster_member_cluster ON face_cluster_member(run_id, cluster_id);
        CREATE INDEX IF NOT EXISTS idx_face_cluster_member_face ON face_cluster_member(face_observation_id);
        CREATE INDEX IF NOT EXISTS idx_face_cluster_run_versions ON face_cluster_run(model_version, preprocessing_version, threshold);

        -- === People / face identity (production source of truth) ===
        -- Named people are durable identities. Keywords are a search/export
        -- projection, not the authority. Face assignments preserve the exact
        -- face observation and the cluster/model provenance that suggested it.
        CREATE SEQUENCE IF NOT EXISTS person_id_seq START 1;

        CREATE TABLE IF NOT EXISTS person (
            id INTEGER PRIMARY KEY DEFAULT nextval('person_id_seq'),
            display_name TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_person_normalized_name ON person(normalized_name);

        CREATE TABLE IF NOT EXISTS person_face_assignment (
            face_observation_id INTEGER PRIMARY KEY,
            person_id INTEGER NOT NULL,
            image_id INTEGER NOT NULL,
            analyzed_image_id INTEGER NOT NULL,
            face_index INTEGER NOT NULL,
            assignment_source TEXT NOT NULL,
            face_cluster_run_id TEXT,
            face_cluster_id INTEGER,
            model_version TEXT,
            preprocessing_version TEXT,
            threshold DOUBLE,
            confidence_cosine DOUBLE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_person_face_assignment_person ON person_face_assignment(person_id);
        CREATE INDEX IF NOT EXISTS idx_person_face_assignment_image ON person_face_assignment(image_id);
        CREATE INDEX IF NOT EXISTS idx_person_face_assignment_cluster ON person_face_assignment(face_cluster_run_id, face_cluster_id);

        -- === Durable analysis jobs (background intelligent culling coordinator) ===
        -- One row per user/requested enrichment run. The current focus pass uses
        -- this as durable foreground state; the future helper/agent will claim
        -- and update the same rows after the app exits. This table is separate
        -- from image-level scalar facts so curation and analysis remain
        -- disentangled. New table only: no ALTER ... ADD COLUMN ... DEFAULT path.
        CREATE SEQUENCE IF NOT EXISTS analysis_job_id_seq START 1;

        CREATE TABLE IF NOT EXISTS analysis_jobs (
            id INTEGER PRIMARY KEY DEFAULT nextval('analysis_job_id_seq'),
            job_kind TEXT NOT NULL,              -- focus_quality / subject_detection / ...
            scope_kind TEXT NOT NULL,            -- whole_catalogue / selection / path_prefix / ...
            scope_value TEXT,                    -- optional serialized scope payload
            algorithm_version TEXT NOT NULL,
            analysis_run_id TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL,                -- queued / running / cancelling / cancelled / completed / failed
            total_candidate_count BIGINT NOT NULL,
            processed_count BIGINT NOT NULL,
            completed_count BIGINT NOT NULL,
            skipped_count BIGINT NOT NULL,
            failed_count BIGINT NOT NULL,
            updated_count BIGINT NOT NULL,
            cancel_requested BOOLEAN NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            updated_at TIMESTAMP,
            finished_at TIMESTAMP,
            last_error TEXT,
            current_image_id BIGINT,
            current_file_path TEXT,
            current_started_at TIMESTAMP,
            last_timeout_image_id BIGINT,
            last_timeout_file_path TEXT,
            last_timeout_at TIMESTAMP
        );

        ALTER TABLE analysis_jobs ADD COLUMN IF NOT EXISTS current_image_id BIGINT;
        ALTER TABLE analysis_jobs ADD COLUMN IF NOT EXISTS current_file_path TEXT;
        ALTER TABLE analysis_jobs ADD COLUMN IF NOT EXISTS current_started_at TIMESTAMP;
        ALTER TABLE analysis_jobs ADD COLUMN IF NOT EXISTS last_timeout_image_id BIGINT;
        ALTER TABLE analysis_jobs ADD COLUMN IF NOT EXISTS last_timeout_file_path TEXT;
        ALTER TABLE analysis_jobs ADD COLUMN IF NOT EXISTS last_timeout_at TIMESTAMP;

        CREATE INDEX IF NOT EXISTS idx_analysis_jobs_kind_status ON analysis_jobs(job_kind, status);
        CREATE INDEX IF NOT EXISTS idx_analysis_jobs_run_id ON analysis_jobs(analysis_run_id);

        -- === Similar-photo grouping (Intelligent Culling, S94) ===
        -- Vision feature-print comparison runs in Swift, then writes only durable
        -- group membership here. Singletons have no row. `group_id` is the
        -- representative image id, which keeps groups stable and readable without
        -- a separate sequence/table for v1.
        CREATE TABLE IF NOT EXISTS similar_photo_group_member (
            image_id INTEGER PRIMARY KEY,
            group_id INTEGER NOT NULL,
            representative_id INTEGER NOT NULL,
            member_rank INTEGER NOT NULL,
            distance_to_representative DOUBLE,
            algorithm_version TEXT NOT NULL,
            threshold DOUBLE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_similar_photo_group ON similar_photo_group_member(group_id);
        CREATE INDEX IF NOT EXISTS idx_similar_photo_rep ON similar_photo_group_member(representative_id);
        CREATE INDEX IF NOT EXISTS idx_similar_photo_algorithm ON similar_photo_group_member(algorithm_version);

        -- Durable Vision featureprints for resumable similar-photo grouping.
        -- The Swift runner owns the Vision observation bytes; core stores them
        -- by image/algorithm/source stamp so cancelled runs do not regenerate
        -- the expensive featureprint work.
        CREATE TABLE IF NOT EXISTS similar_photo_featureprint (
            image_id INTEGER NOT NULL,
            algorithm_version TEXT NOT NULL,
            source_stamp TEXT NOT NULL,
            featureprint_blob BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (image_id, algorithm_version)
        );

        CREATE INDEX IF NOT EXISTS idx_similar_featureprint_algorithm ON similar_photo_featureprint(algorithm_version);

        -- Resumable/progressive grouping checkpoints. A unit is complete only
        -- for the deterministic candidate boundary that wrote it; callers match
        -- unit index plus start/end ids before skipping.
        CREATE TABLE IF NOT EXISTS similar_photo_group_work_unit (
            algorithm_version TEXT NOT NULL,
            scope_key TEXT NOT NULL,
            unit_index BIGINT NOT NULL,
            start_image_id INTEGER NOT NULL,
            end_image_id INTEGER NOT NULL,
            candidate_count BIGINT NOT NULL,
            member_count BIGINT NOT NULL,
            status TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (algorithm_version, scope_key, unit_index)
        );

        CREATE INDEX IF NOT EXISTS idx_similar_work_unit_scope ON similar_photo_group_work_unit(algorithm_version, scope_key, status);

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
            origin INTEGER NOT NULL DEFAULT 1,  -- bitmask: 1 = user/import, 2 = auto-categorization, 3 = both
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            hidden_at TIMESTAMP,                -- set when status->0; NULL while active
            collection BOOLEAN NOT NULL DEFAULT FALSE,  -- Collections-panel membership; orthogonal to status (keyword search stays oblivious)
            color BOOLEAN NOT NULL DEFAULT FALSE,       -- S66: this label is a COLOR (custom Lightroom color-label text); third independent switch
            is_video BOOLEAN NOT NULL DEFAULT FALSE     -- S70: denormalized media-type discriminator (= images.is_video of the row this points to); join-free media filtering on keyword
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

        -- Same migration shape for the color switch (S66). A pre-existing
        -- catalogue's color-derived rows can't be retro-identified (that's the
        -- gap this column closes) — they backfill FALSE and a re-import marks
        -- them.
        ALTER TABLE keyword ADD COLUMN IF NOT EXISTS color BOOLEAN;
        UPDATE keyword SET color = FALSE WHERE color IS NULL;

        -- Keyword provenance (S93): separate from `status`, which remains the
        -- active/hidden switch. Bitmask: 1 = user/import-applied, 2 =
        -- auto-categorization, 3 = both. Existing rows backfill user-origin;
        -- development catalogues are normally rebuilt fresh, but this keeps the
        -- schema reopen path total.
        ALTER TABLE keyword ADD COLUMN IF NOT EXISTS origin INTEGER;
        UPDATE keyword SET origin = 1 WHERE origin IS NULL;
        CREATE INDEX IF NOT EXISTS idx_keyword_origin ON keyword(origin);

        -- S70: denormalized media-type discriminator on each keyword row — a copy
        -- of images.is_video for the row it points to. is_video is immutable, so
        -- the copy never drifts; backfilled FROM the referenced image (correlated
        -- subquery), idempotent via the IS NULL guard so it runs once. Lets the
        -- keyword-vocabulary / media-count queries filter without a keyword-images
        -- join on the highest-cardinality table.
        ALTER TABLE keyword ADD COLUMN IF NOT EXISTS is_video BOOLEAN;
        UPDATE keyword SET is_video =
            COALESCE((SELECT i.is_video FROM images i WHERE i.id = keyword.image_id), FALSE)
            WHERE is_video IS NULL;

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
        -- criterion — repeated subjects are simply more rows. Migrations stay
        -- WAL-safe: bare ALTER ADD COLUMN only, never with a DEFAULT (the S62
        -- lesson) — see the S65 num/num_end ALTERs below.
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
            stars INTEGER,                      -- QueryPredicate.stars
            num DOUBLE,                         -- QueryPredicate.num (S65 numeric subjects)
            num_end DOUBLE                      -- QueryPredicate.num_end ("between" upper bound)
        );

        -- S65: catalogues whose saved_query_criterion predates the numeric
        -- subjects gain the two columns here — bare ALTER, NO DEFAULT (the S62
        -- WAL-replay lesson), and no backfill: NULL is the correct resting
        -- value for every pre-numeric row.
        ALTER TABLE saved_query_criterion ADD COLUMN IF NOT EXISTS num DOUBLE;
        ALTER TABLE saved_query_criterion ADD COLUMN IF NOT EXISTS num_end DOUBLE;

        CREATE INDEX IF NOT EXISTS idx_saved_query_criterion_query ON saved_query_criterion(query_id);

        -- === Folder Sync (Session 68; Docs/DESIGN-Folder-Sync.md) ===
        -- One row per catalogued directory: that directory's on-disk mtime
        -- (epoch seconds, as stat reports it) recorded when the directory was
        -- last scanned or synced. The SourceLocationMonitor sweep compares a
        -- fresh stat against last_sync_mtime — INEQUALITY, not greater-than
        -- (a folder restored from backup moves BACKWARD and is still changed).
        -- A brand-new table needs no ALTER migration, so the S62 WAL-replay
        -- hazard cannot apply here; CREATE-time shape is final.
        CREATE TABLE IF NOT EXISTS directory_sync_state (
            directory_path  TEXT PRIMARY KEY,
            last_sync_mtime BIGINT NOT NULL
        );
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
        |row| row.get::<_, String>(0),
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
    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;") {
        eprintln!("[migration] Failed to begin transaction: {}", e);
        return false;
    }

    // Step 1: backfill file_stem and image_kind.
    let rows_needing_backfill: Vec<(i64, String)> = match conn
        .prepare("SELECT id, file_name FROM images WHERE file_stem IS NULL OR image_kind IS NULL")
    {
        Ok(mut stmt) => {
            match stmt.query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            }) {
                Ok(mapped) => mapped.filter_map(Result::ok).collect(),
                Err(e) => {
                    eprintln!("[migration] Failed to query rows needing backfill: {}", e);
                    Vec::new()
                }
            }
        }
        Err(e) => {
            eprintln!("[migration] Failed to prepare backfill query: {}", e);
            Vec::new()
        }
    };

    let mut backfilled_count = 0u64;
    for (row_id, file_name) in &rows_needing_backfill {
        let parsed = parse_filename(file_name.clone());
        let kind_str = match parsed.kind {
            ImageKind::Jpeg => "jpeg",
            ImageKind::Raw => "raw",
            ImageKind::Other => "other",
            ImageKind::Heif => "heif",
            ImageKind::Dng => "dng",
            ImageKind::Psd => "psd",
            ImageKind::Tiff => "tiff",
            ImageKind::Png => "png",
        };
        match conn.execute(
            "UPDATE images SET file_stem = ?1, image_kind = ?2 WHERE id = ?3",
            params![parsed.stem, kind_str, row_id],
        ) {
            Ok(_) => backfilled_count += 1,
            Err(e) => eprintln!("[migration] Failed to update row id={}: {}", row_id, e),
        }
    }
    eprintln!(
        "[migration] backfilled file_stem/image_kind for {} rows",
        backfilled_count
    );

    // Step 2: idempotent lowercase normalization of file_extension.
    // The WHERE clause restricts the UPDATE to rows that actually differ;
    // DuckDB returns the count of changed rows in `changed`.
    match conn.execute(
        "UPDATE images SET file_extension = LOWER(file_extension) \
         WHERE file_extension IS NOT NULL AND file_extension != LOWER(file_extension)",
        [],
    ) {
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
    ) {
        Ok(changed) => eprintln!("[migration] reclassified {} dng rows", changed),
        Err(e) => eprintln!("[migration] Failed to reclassify dng: {}", e),
    }
    match conn.execute(
        "UPDATE images SET image_kind = 'psd' WHERE file_extension = 'psd' AND image_kind <> 'psd'",
        [],
    ) {
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
    ) {
        Ok(changed) => eprintln!("[migration] reclassified {} png rows", changed),
        Err(e) => eprintln!("[migration] Failed to reclassify png: {}", e),
    }

    if let Err(e) = conn.execute_batch("COMMIT;") {
        eprintln!("[migration] Failed to commit migration transaction: {}", e);
        return false;
    }

    // DuckDB 1.2.2 can fail WAL replay when a pending ALTER TABLE ADD COLUMN is
    // replayed against a table whose primary key default references a sequence.
    // Checkpoint after startup schema/migration work so future launches do not
    // depend on replaying those DDL records.
    if let Err(e) = conn.execute_batch("CHECKPOINT;") {
        eprintln!(
            "[migration] Failed to checkpoint catalogue after schema migration: {}",
            e
        );
        return false;
    }
    // --- End backfill migration --------------------------------------------

    // Store the connection in the global state
    // This connection will be reused by all subsequent catalogue operations
    // Mutex ensures thread-safe access when called from multiple Swift async tasks
    let mut catalogue = CATALOGUE.lock().unwrap();
    *catalogue = Some(conn);

    let mut stored_path = CATALOGUE_PATH.lock().unwrap();
    *stored_path = Some(path);

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
            rating, flag, color_label,
            is_video,
            duration_seconds, frame_rate, video_kind, video_codec, video_bitrate,
            color_primaries, color_transfer, color_matrix, color_range, dv_profile,
            has_audio, audio_codec, audio_channels, audio_sample_rate, audio_bitrate,
            live_photo_id,
            rotation
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6,
            SUBSTRING(?1, 1, LENGTH(?1) - INSTR(REVERSE(?1), '/')),
            ?7, ?8, ?9, ?10,
            ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20,
            ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29,
            ?30,
            ?31, ?32, ?33, ?34, ?35,
            ?36, ?37, ?38, ?39, ?40,
            ?41, ?42, ?43, ?44, ?45,
            ?46,
            COALESCE(?47, 0)
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
        let image_kind_str = match parsed.kind {
            ImageKind::Jpeg => "jpeg",
            ImageKind::Raw => "raw",
            ImageKind::Other => "other",
            ImageKind::Heif => "heif",
            ImageKind::Dng => "dng",
            ImageKind::Psd => "psd",
            ImageKind::Tiff => "tiff",
            ImageKind::Png => "png",
        };

        // Execute the prepared statement with positional parameters
        // Type conversions: u32/u64 → i64 for DuckDB INTEGER columns
        // Option<T> fields are passed directly — DuckDB handles NULL for None
        let result = conn.execute(
            insert_sql,
            params![
                record.file_path,                      // ?1
                record.file_size as i64, // ?2  (u64 → i64 cast safe for file sizes < 9 exabytes)
                record.file_name,        // ?3
                record.file_extension,   // ?4
                parsed.stem,             // ?5  file_stem (original case preserved)
                image_kind_str,          // ?6  image_kind (always lowercase: "jpeg"/"raw"/"other")
                record.created_timestamp, // ?7
                record.modified_timestamp, // ?8
                record.camera_make,      // ?9
                record.camera_model,     // ?10
                record.lens_model,       // ?11
                record.focal_length,     // ?12
                record.aperture,         // ?13
                record.shutter_speed,    // ?14
                record.iso.map(|v| v as i64), // ?15 (u32 → i64)
                record.capture_datetime, // ?16
                record.pixel_width.map(|v| v as i64), // ?17 (u32 → i64)
                record.pixel_height.map(|v| v as i64), // ?18 (u32 → i64)
                record.color_space,      // ?19
                record.bit_depth.map(|v| v as i64), // ?20 (u32 → i64)
                record.gps_latitude,     // ?21
                record.gps_longitude,    // ?22
                record.gps_altitude,     // ?23
                record.copyright,        // ?24
                record.creator,          // ?25
                record.description,      // ?26
                record.rating.map(|v| v as i64), // ?27 (u8 → i64)
                record.flag,             // ?28
                record.color_label,      // ?29
                // --- Video / unified-media (Step 2a) — false/None for stills ---
                record.is_video,          // ?30 (bool)
                record.duration_seconds,  // ?31
                record.frame_rate,        // ?32
                record.video_kind,        // ?33
                record.video_codec,       // ?34
                record.video_bitrate,     // ?35
                record.color_primaries,   // ?36
                record.color_transfer,    // ?37
                record.color_matrix,      // ?38
                record.color_range,       // ?39
                record.dv_profile,        // ?40
                record.has_audio,         // ?41
                record.audio_codec,       // ?42
                record.audio_channels,    // ?43
                record.audio_sample_rate, // ?44
                record.audio_bitrate,     // ?45
                record.live_photo_id,     // ?46
                record.rotation,          // ?47 (COALESCE(?,0): None/stills -> 0)
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
    media_type: MediaType,
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
        false,
        "",
        media_type,
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

    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        year, month, day, hours, minutes, seconds
    )
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
const DUPLICATE_FILTER_PREDICATE: &str = "(duplicate_group_id IS NULL OR id = duplicate_group_id)";

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

/// Similar-photo collapse is an OUTER filter, not an inner `WHERE id =
/// representative_id` check. The durable grouping table's representative can be
/// a RAW row; after RAW/JPEG collapse the visible gallery row may instead be
/// the JPEG/HEIF sibling. Ranking all rows that survive the normal inner
/// filters gives the page/count helpers one visible representative per stack
/// without making a RAW representative erase the whole group.
const SIMILAR_VISIBLE_ID_PROJECTION: &str = "\
            FIRST_VALUE(id) OVER ( \
                PARTITION BY similar_group_id \
                ORDER BY \
                    CASE \
                        WHEN similar_image_kind IN ('jpeg', 'heif') THEN 0 \
                        WHEN similar_image_kind = 'raw' THEN 1 \
                        ELSE 2 \
                    END, \
                    id ASC \
                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING \
            ) AS similar_visible_id";

const SIMILAR_COLLAPSE_PREDICATE: &str = "(similar_group_id IS NULL OR id = similar_visible_id)";

fn sql_string_literal(value: &str) -> String {
    format!("'{}'", value.replace("'", "''"))
}

fn similar_photo_join_clause(
    apply_similar_photo_collapse: bool,
    algorithm_version: &str,
) -> String {
    if !apply_similar_photo_collapse || algorithm_version.trim().is_empty() {
        String::new()
    } else {
        format!(
            "LEFT JOIN similar_photo_group_member spgm \
             ON spgm.image_id = images.id AND spgm.algorithm_version = {}",
            sql_string_literal(algorithm_version.trim())
        )
    }
}

/// Inner-WHERE predicate restricting a result set to STILLS (non-video).
///
/// S70 folded video into `images`, discriminated by `is_video`. Until
/// poster-frame thumbnails + playback exist (roadmap Stage 6), the product
/// stance is "video is catalogued but not shown"
/// (DESIGN-Video-Schema-Unified-Table.md §7). This constant enforces that at
/// the shared query chokepoint: it is pushed UNCONDITIONALLY into the
/// `inner_predicates` of every paginating / count / projection helper, so the
/// gallery, Browse, ⌘A, filtered counts, and the path/date-prefix queries all
/// gate on it identically. The two sidebar-count GROUP-BYs
/// (`directory_image_counts`, `capture_day_image_counts`) reference it too, so
/// Dates/Sources counts match what the gallery shows.
///
/// `IS NOT TRUE` (not `= FALSE`) is deliberate and NULL-safe — equivalent to
/// `(is_video = FALSE OR is_video IS NULL)`: a row is hidden only when
/// `is_video` is definitively TRUE, so a still can never be wrongly suppressed.
///
/// As of the Stage-6 media-type control (DESIGN §11) this is no longer pushed
/// unconditionally — `media_predicate` selects it / `VIDEOS_ONLY_PREDICATE` /
/// neither, threaded through the five query helpers + the two sidebar GROUP-BYs.
const STILLS_ONLY_PREDICATE: &str = "is_video IS NOT TRUE";

/// The videos-only stance — the complement of `STILLS_ONLY_PREDICATE`. `IS TRUE`
/// is NULL-safe the same way: only a definitively `is_video = TRUE` row passes,
/// so a still (FALSE after the backfill) can never leak into a videos-only view.
const VIDEOS_ONLY_PREDICATE: &str = "is_video IS TRUE";

/// The three-state media-type view stance (DESIGN-Video-Schema-Unified-Table.md
/// §11) — the three-state sibling of the `apply_raw_jpeg_collapse` /
/// `apply_duplicate_filter` view booleans, threaded through every shared query
/// helper and the two sidebar-count GROUP-BYs. Order MUST match the UDL `enum
/// MediaType` (UniFFI maps by position).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaType {
    StillsOnly,
    VideosOnly,
    Both,
}

/// Map a `MediaType` to its WHERE-clause fragment, or `None` for `Both` (no
/// media predicate — stills and video together). The single place the stance
/// becomes SQL; every query helper calls this instead of pushing a constant.
fn media_predicate(media_type: MediaType) -> Option<&'static str> {
    match media_type {
        MediaType::StillsOnly => Some(STILLS_ONLY_PREDICATE),
        MediaType::VideosOnly => Some(VIDEOS_ONLY_PREDICATE),
        MediaType::Both => None,
    }
}

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
fn build_path_date_predicate(path_prefix: &str, date_prefix: &str) -> String {
    let escaped_path = path_prefix.replace("'", "''");
    let escaped_date = date_prefix.replace("'", "''");

    // Apple Photos library originals live INSIDE a `.photoslibrary` package and are
    // surfaced under their own "Apple Libraries" sidebar scope — the Sources folder
    // tree (SourceLocationsView) and its counts exclude them. But a folder node's
    // path PREFIX still string-matches them when the library is nested under that
    // folder (e.g. `~/Pictures/X.photoslibrary` under `/Users/<name>`), which wrongly
    // pulls the whole Apple library into a folder scope (its count says 1,148, the
    // query returned 45k). So a NON-EMPTY folder prefix excludes Apple-package files —
    // UNLESS the prefix itself points into a `.photoslibrary` (the Apple Library node's
    // own scope, which legitimately shows only those files). An empty prefix
    // (All Sources) excludes nothing. Mirrors `ScanService.isInsideApplePhotosLibrary`.
    let apple_exclusion = if !path_prefix.is_empty() && !path_prefix.contains(".photoslibrary") {
        " AND file_path NOT LIKE '%.photoslibrary/%'"
    } else {
        ""
    };

    match (path_prefix.is_empty(), date_prefix.is_empty()) {
        (true, true) => String::new(),
        (false, true) => format!("file_path LIKE '{}%'{}", escaped_path, apple_exclusion),
        (true, false) => format!("capture_datetime LIKE '{}%'", escaped_date),
        (false, false) => format!(
            "file_path LIKE '{}%' AND capture_datetime LIKE '{}%'{}",
            escaped_path, escaped_date, apple_exclusion
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
fn regex_escape_for_similar_to(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '\\' | '.' | '^' | '$' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' => {
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
fn build_destination_family_predicate(sample_file_path: &str, canonical_file_name: &str) -> String {
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
    apply_similar_photo_collapse: bool,
    similar_algorithm_version: &str,
    media_type: MediaType,
) -> Vec<ImageRecord> {
    // Assemble the inner WHERE from the caller-supplied predicate text
    // and the RAW+JPEG collapse predicate (both stored-column references;
    // both safely composable with AND at the inner level).
    let mut inner_predicates: Vec<&str> = Vec::new();
    if !where_clause.is_empty() {
        inner_predicates.push(where_clause);
    }
    if apply_raw_jpeg_collapse {
        inner_predicates.push(RAW_JPEG_COLLAPSE_PREDICATE);
    }
    // Media-type stance (DESIGN-Video-Schema-Unified-Table.md §11): gallery,
    // Browse, ⌘A, filtered counts, and path/date-prefix queries all gate through
    // this one seam. `media_predicate` maps the caller's MediaType to its WHERE
    // fragment — None for Both (stills + video together), so nothing is pushed.
    if let Some(media_pred) = media_predicate(media_type) {
        inner_predicates.push(media_pred);
    }
    let inner_where = if inner_predicates.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", inner_predicates.join(" AND "))
    };

    let similar_join =
        similar_photo_join_clause(apply_similar_photo_collapse, similar_algorithm_version);
    let effective_similar_collapse = apply_similar_photo_collapse && !similar_join.is_empty();
    let needs_outer_filter = apply_duplicate_filter || effective_similar_collapse;
    let duplicate_filter = if apply_duplicate_filter {
        Some(DUPLICATE_FILTER_PREDICATE)
    } else {
        None
    };
    let similar_filter = if effective_similar_collapse {
        Some(SIMILAR_COLLAPSE_PREDICATE)
    } else {
        None
    };
    let outer_filters = [duplicate_filter, similar_filter]
        .into_iter()
        .flatten()
        .collect::<Vec<&str>>()
        .join(" AND ");
    let similar_inner_projection = if effective_similar_collapse {
        ", spgm.group_id AS similar_group_id, image_kind AS similar_image_kind"
    } else {
        ""
    };

    // Branch on outer filters. Duplicate and similar-photo collapse both need
    // projection aliases, so they share the same wrap shape. When inactive,
    // emit the minimal top-level projection.
    let query_sql = if needs_outer_filter {
        let inner_select = format!(
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
                {},
                focus_score
                {}
            FROM images
            {}
            {}
        "#,
            DUPLICATE_GROUP_ID_CASE, similar_inner_projection, similar_join, inner_where
        );
        let filtered_source = if effective_similar_collapse {
            format!(
                r#"
                SELECT
                    *,
                    {}
                FROM (
                    {}
                )
            "#,
                SIMILAR_VISIBLE_ID_PROJECTION, inner_select
            )
        } else {
            inner_select
        };
        format!(
            r#"
            SELECT * FROM (
                {}
            )
            WHERE {}
            ORDER BY {}
            LIMIT ?1 OFFSET ?2
        "#,
            filtered_source, outer_filters, order_by
        )
    } else {
        format!(
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
                {}
            FROM images
            {}
            ORDER BY {}
            LIMIT ?1 OFFSET ?2
        "#,
            DUPLICATE_GROUP_ID_CASE, inner_where, order_by
        )
    };

    let mut stmt = match conn.prepare(&query_sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to prepare query: {}", e);
            return Vec::new();
        }
    };

    let rows = match stmt.query_map(params![limit, offset], |row| {
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

    let mut records = Vec::new();
    for row_result in rows {
        match row_result {
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
    apply_similar_photo_collapse: bool,
    similar_algorithm_version: &str,
    media_type: MediaType,
) -> i64 {
    let mut inner_predicates: Vec<&str> = Vec::new();
    if !where_clause.is_empty() {
        inner_predicates.push(where_clause);
    }
    if apply_raw_jpeg_collapse {
        inner_predicates.push(RAW_JPEG_COLLAPSE_PREDICATE);
    }
    // Media-type stance (DESIGN-Video-Schema-Unified-Table.md §11): gallery,
    // Browse, ⌘A, filtered counts, and path/date-prefix queries all gate through
    // this one seam. `media_predicate` maps the caller's MediaType to its WHERE
    // fragment — None for Both (stills + video together), so nothing is pushed.
    if let Some(media_pred) = media_predicate(media_type) {
        inner_predicates.push(media_pred);
    }
    let inner_where = if inner_predicates.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", inner_predicates.join(" AND "))
    };

    let similar_join =
        similar_photo_join_clause(apply_similar_photo_collapse, similar_algorithm_version);
    let effective_similar_collapse = apply_similar_photo_collapse && !similar_join.is_empty();
    let needs_outer_filter = apply_duplicate_filter || effective_similar_collapse;
    let duplicate_filter = if apply_duplicate_filter {
        Some(DUPLICATE_FILTER_PREDICATE)
    } else {
        None
    };
    let similar_filter = if effective_similar_collapse {
        Some(SIMILAR_COLLAPSE_PREDICATE)
    } else {
        None
    };
    let outer_filters = [duplicate_filter, similar_filter]
        .into_iter()
        .flatten()
        .collect::<Vec<&str>>()
        .join(" AND ");
    let similar_inner_projection = if effective_similar_collapse {
        ", spgm.group_id AS similar_group_id, image_kind AS similar_image_kind"
    } else {
        ""
    };

    // Duplicate and similar-photo collapse both operate on projection aliases.
    // Keep the COUNT shape in lockstep with `execute_image_record_query` so
    // pages count visible representatives, not hidden member rows.
    let query_sql = if needs_outer_filter {
        let inner_select = format!(
            r#"
            SELECT
                id,
                {}
                {}
            FROM images
            {}
            {}
        "#,
            DUPLICATE_GROUP_ID_CASE, similar_inner_projection, similar_join, inner_where
        );
        let filtered_source = if effective_similar_collapse {
            format!(
                r#"
                SELECT
                    *,
                    {}
                FROM (
                    {}
                )
            "#,
                SIMILAR_VISIBLE_ID_PROJECTION, inner_select
            )
        } else {
            inner_select
        };
        format!(
            r#"
            SELECT COUNT(*) FROM (
                {}
            )
            WHERE {}
        "#,
            filtered_source, outer_filters
        )
    } else {
        format!("SELECT COUNT(*) FROM images {}", inner_where)
    };

    let count_result: Result<i64, _> = conn.query_row(&query_sql, [], |row| row.get(0));

    match count_result {
        Ok(count) => count,
        Err(e) => {
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
    media_type: MediaType,
) -> Result<Vec<String>, String> {
    let mut inner_predicates: Vec<&str> = Vec::new();
    if !where_clause.is_empty() {
        inner_predicates.push(where_clause);
    }
    if apply_raw_jpeg_collapse {
        inner_predicates.push(RAW_JPEG_COLLAPSE_PREDICATE);
    }
    // Media-type stance (DESIGN §11). folder-sync's computeDiff passes Both so
    // its disk-vs-catalogue diff sees video on BOTH sides (the disk listing uses
    // allMediaExtensions); an asymmetric gate would read every catalogued video
    // as a phantom new arrival. `media_predicate` maps the stance to its WHERE
    // fragment — None for Both, so nothing is pushed.
    if let Some(media_pred) = media_predicate(media_type) {
        inner_predicates.push(media_pred);
    }
    let inner_where = if inner_predicates.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", inner_predicates.join(" AND "))
    };

    // Branch on apply_duplicate_filter — mirrors the count helper's
    // shape. When inactive, emit the minimal projection. When active,
    // wrap an (id, file_path, duplicate_group_id) projection in a
    // subquery so the outer WHERE can apply DUPLICATE_FILTER_PREDICATE
    // against the alias.
    let query_sql = if apply_duplicate_filter {
        format!(
            r#"
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
            DUPLICATE_GROUP_ID_CASE, inner_where, DUPLICATE_FILTER_PREDICATE
        )
    } else {
        format!("SELECT file_path FROM images {}", inner_where)
    };

    let mut stmt = match conn.prepare(&query_sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to prepare file_path projection query: {}", e);
            return Err(format!("prepare failed: {}", e));
        }
    };

    let rows = match stmt.query_map([], |row| row.get::<_, String>(0)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to execute file_path projection query: {}", e);
            return Err(format!("query failed: {}", e));
        }
    };

    let mut paths: Vec<String> = Vec::new();
    for row_result in rows {
        match row_result {
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
    media_type: MediaType,
) -> Vec<ImageRecord> {
    let mut inner_predicates: Vec<&str> = Vec::new();
    if !where_clause.is_empty() {
        inner_predicates.push(where_clause);
    }
    if apply_raw_jpeg_collapse {
        inner_predicates.push(RAW_JPEG_COLLAPSE_PREDICATE);
    }
    // Media-type stance (DESIGN §11). folder-sync's REMOVAL path resolves missing
    // video paths → ids through here passing Both, so it MUST see video —
    // otherwise it finds 0 ids for a vanished clip and can never delete it (the
    // S72 −N loop). `media_predicate` maps the stance to its WHERE fragment —
    // None for Both, so nothing is pushed.
    if let Some(media_pred) = media_predicate(media_type) {
        inner_predicates.push(media_pred);
    }
    let inner_where = if inner_predicates.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", inner_predicates.join(" AND "))
    };

    // Branch on apply_duplicate_filter — mirrors the other two helpers
    // structurally. When active, wrap the projection-with-
    // duplicate_group_id in a subquery so the outer WHERE can reference
    // the alias. When inactive, emit the projection at the top level
    // (the alias is still surfaced on the returned ImageRecord, column
    // 30).
    let query_sql = if apply_duplicate_filter {
        format!(
            r#"
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
            DUPLICATE_GROUP_ID_CASE, inner_where, DUPLICATE_FILTER_PREDICATE
        )
    } else {
        format!(
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
                {}
            FROM images
            {}
        "#,
            DUPLICATE_GROUP_ID_CASE, inner_where
        )
    };

    let mut stmt = match conn.prepare(&query_sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to prepare image record projection query: {}", e);
            return Vec::new();
        }
    };

    let rows = match stmt.query_map([], |row| {
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
            eprintln!("Failed to execute image record projection query: {}", e);
            return Vec::new();
        }
    };

    let mut records = Vec::new();
    for row_result in rows {
        match row_result {
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
    media_type: MediaType,
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
        false,
        "",
        media_type,
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
        false,
        "",
        MediaType::StillsOnly,
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
pub struct RelocateResult {
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
pub async fn relocate_file_path_prefix(old_prefix: String, new_prefix: String) -> RelocateResult {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("relocate_file_path_prefix: catalogue not initialized");
            return RelocateResult {
                ok: false,
                updated: 0,
                message: "Catalogue not initialized".to_string(),
            };
        }
    };

    if old_prefix.is_empty() || new_prefix.is_empty() {
        return RelocateResult {
            ok: false,
            updated: 0,
            message: "Empty source or destination prefix".to_string(),
        };
    }
    if old_prefix == new_prefix {
        return RelocateResult {
            ok: true,
            updated: 0,
            message: String::new(),
        };
    }

    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;") {
        eprintln!("relocate_file_path_prefix: begin failed: {}", e);
        return RelocateResult {
            ok: false,
            updated: 0,
            message: format!("begin failed: {}", e),
        };
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

    let changed = match conn.execute(update_sql, params![new_prefix, old_prefix]) {
        Ok(n) => n as u64,
        Err(e) => {
            eprintln!("relocate_file_path_prefix: update failed: {}", e);
            let _ = conn.execute_batch("ROLLBACK;");
            return RelocateResult {
                ok: false,
                updated: 0,
                message: format!("rewrite failed (possible path collision): {}", e),
            };
        }
    };

    if let Err(e) = conn.execute_batch("COMMIT;") {
        eprintln!("relocate_file_path_prefix: commit failed: {}", e);
        let _ = conn.execute_batch("ROLLBACK;");
        return RelocateResult {
            ok: false,
            updated: 0,
            message: format!("commit failed: {}", e),
        };
    }

    eprintln!(
        "relocate_file_path_prefix: moved {} rows '{}' -> '{}'",
        changed, old_prefix, new_prefix
    );
    RelocateResult {
        ok: true,
        updated: changed,
        message: String::new(),
    }
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
pub enum Connector {
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
pub struct QueryPredicate {
    pub kind: String,
    pub day: Option<String>,
    pub day_end: Option<String>,
    pub op: Option<String>,
    pub stars: Option<u8>,
    pub value: Option<String>,
    // Numeric subjects (S65 — ISO / aperture / shutter / focal length): the
    // bound(s). `num` carries the value (or the lower bound), `num_end` the
    // upper bound for op "between". APPENDED LAST — UniFFI maps dictionary
    // fields by position (the ImageKind lesson); both carry UDL `= null`
    // defaults so existing Swift construction sites compile unchanged.
    pub num: Option<f64>,
    pub num_end: Option<f64>,
}

/// Validate a day string as exactly `YYYY:MM:DD` (10 chars; colons at index 4
/// and 7; digits elsewhere) — the form produced by `SUBSTRING(capture_datetime,
/// 1, 10)`. Because only digits + colons can pass, a validated day cannot carry
/// a SQL-injection payload (defense-in-depth on top of Swift-side validation).
fn is_valid_day(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() != 10 {
        return false;
    }
    for (i, &c) in b.iter().enumerate() {
        let ok = if i == 4 || i == 7 {
            c == b':'
        } else {
            c.is_ascii_digit()
        };
        if !ok {
            return false;
        }
    }
    true
}

fn is_valid_flag(s: &str) -> bool {
    matches!(s, "pick" | "reject")
}

fn is_valid_color(s: &str) -> bool {
    matches!(s, "red" | "yellow" | "green" | "blue" | "purple")
}

/// Escape a user-supplied string for safe use INSIDE a DuckDB `ILIKE` pattern
/// whose `ESCAPE` character is backslash: neutralize the `%` and `_` wildcards
/// (and the escape backslash itself) so typed text matches LITERALLY rather than
/// acting as a wildcard. The single-quote doubling for the surrounding SQL
/// string literal is applied separately by `filename_ilike_atom`.
fn escape_for_ilike(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

/// Build a parenthesized, case-insensitive file-name match atom from an already
/// wildcard-escaped `ILIKE` pattern. Doubles single quotes for the SQL literal
/// and pins the `ESCAPE` character to backslash. `ILIKE` gives the case-
/// insensitivity the four File Name modes all require.
fn filename_ilike_atom(pattern: &str) -> String {
    format!(
        "(file_name ILIKE '{}' ESCAPE '\\')",
        pattern.replace('\'', "''")
    )
}

/// `filename_ilike_atom`'s twin, pointed at `lens_model` — the Lens subject's
/// "contains" mode (Session 63). The lens string is composite (maker + focal +
/// aperture + line), so a case-insensitive fragment like "Viltrox" or "85mm"
/// matches the whole family.
fn lens_ilike_atom(pattern: &str) -> String {
    format!(
        "(lens_model ILIKE '{}' ESCAPE '\\')",
        pattern.replace('\'', "''")
    )
}

/// Map a rating comparison op token to its SQL symbol. Unknown → None.
fn sql_compare_op(op: &str) -> Option<&'static str> {
    match op {
        "eq" => Some("="),
        "gt" => Some(">"),
        "lt" => Some("<"),
        "gte" => Some(">="),
        "lte" => Some("<="),
        _ => None,
    }
}

/// Map a `date_in_last` unit token to its DuckDB INTERVAL keyword (S66 —
/// Gate 3). An allow-list, so unit text can never reach the SQL verbatim;
/// the tokens are the wire vocabulary the Swift `DateUnit` enum emits (and
/// Lightroom's own smart-collection unit words). Unknown → None.
fn interval_unit_sql(unit: &str) -> Option<&'static str> {
    match unit {
        "days" => Some("DAY"),
        "weeks" => Some("WEEK"),
        "months" => Some("MONTH"),
        "years" => Some("YEAR"),
        _ => None,
    }
}

/// One numeric-subject atom (S65 — ISO / aperture / shutter_speed /
/// focal_length). `column` is fixed by the calling arm (never caller text);
/// the numbers are formatted with Rust's shortest-round-trip f64 Display, so
/// the SQL literal parses back to the exact stored double — equality on a
/// dropdown-picked value is exact, not approximate. Non-finite numbers and
/// unknown ops fall to the `(FALSE)` backstop via None. `between` requires
/// both bounds; the other ops reuse the rating token vocabulary.
fn numeric_atom(column: &str, op: &str, num: f64, num_end: Option<f64>) -> Option<String> {
    if !num.is_finite() {
        return None;
    }
    if op == "between" {
        let end = num_end?;
        if !end.is_finite() {
            return None;
        }
        return Some(format!("({} BETWEEN {} AND {})", column, num, end));
    }
    let sym = sql_compare_op(op)?;
    Some(format!("({} {} {})", column, sym, num))
}

/// The shared arm body for the four numeric kinds: op + num are required
/// (num_end only for "between"); anything malformed matches nothing.
fn numeric_predicate_sql(p: &QueryPredicate, column: &str) -> String {
    match (p.op.as_deref(), p.num) {
        (Some(op), Some(num)) => {
            numeric_atom(column, op, num, p.num_end).unwrap_or_else(|| "(FALSE)".to_string())
        }
        _ => "(FALSE)".to_string(),
    }
}

fn people_count_predicate_sql(p: &QueryPredicate) -> String {
    let Some(op) = p.op.as_deref() else {
        return "(FALSE)".to_string();
    };
    let Some(num) = p.num else {
        return "(FALSE)".to_string();
    };
    if !num.is_finite() || num.fract() != 0.0 {
        return "(FALSE)".to_string();
    }

    let count = num as i32;
    match op {
        "eq" if (1..=3).contains(&count) => format!("(face_count = {})", count),
        "gt" if (1..=3).contains(&count) => format!("(face_count > {})", count),
        "lt" if (2..=3).contains(&count) => format!("(face_count >= 1 AND face_count < {})", count),
        _ => "(FALSE)".to_string(),
    }
}

fn keyword_origin_clause(op: Option<&str>) -> &'static str {
    match op {
        Some("user") => " AND (k.origin & 1) <> 0",
        Some("auto") => " AND (k.origin & 2) <> 0",
        Some("both") | None => "",
        _ => "",
    }
}

fn focus_quality_column_for_basis(basis: &str) -> Option<&'static str> {
    match basis {
        "human_face" | "face" => Some("focus_human_score"),
        "animal" | "dog_cat" => Some("focus_animal_score"),
        "foreground" | "subject" => Some("focus_foreground_score"),
        "saliency" => Some("focus_saliency_score"),
        "animal_pose" => Some("focus_animal_pose_score"),
        "whole_image" => Some("focus_whole_image_score"),
        _ => None,
    }
}

fn focus_quality_default_columns() -> Vec<&'static str> {
    vec![
        "focus_human_score",
        "focus_animal_score",
        "focus_foreground_score",
        "focus_saliency_score",
        "focus_animal_pose_score",
        "focus_whole_image_score",
    ]
}

fn focus_quality_columns(value: Option<&str>) -> Option<Vec<&'static str>> {
    let Some(value) = value else {
        return Some(focus_quality_default_columns());
    };
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Some(focus_quality_default_columns());
    }

    let mut columns = Vec::new();
    for token in trimmed.split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        let column = focus_quality_column_for_basis(token)?;
        if !columns.contains(&column) {
            columns.push(column);
        }
    }

    if columns.is_empty() {
        None
    } else {
        Some(columns)
    }
}

fn focus_quality_bucket_from_num(num: f64) -> Option<u8> {
    if !num.is_finite() || num.fract() != 0.0 {
        return None;
    }
    if (1.0..=10.0).contains(&num) {
        return Some(num as u8);
    }
    if (1.0..=100.0).contains(&num) {
        return Some(((num / 10.0).ceil() as u8).clamp(1, 10));
    }
    None
}

fn focus_quality_bucket_threshold_sql(bucket: u8, column: &str) -> Option<String> {
    if !(1..=10).contains(&bucket) {
        return None;
    }
    let position = f64::from(bucket - 1) / 10.0;
    Some(format!(
        "(SELECT quantile_cont(score, {position}) \
         FROM (SELECT {column} AS score FROM images WHERE {column} IS NOT NULL) focus_values)",
        position = position,
        column = column
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PredicateGroupMode {
    Any,
    All,
    ExactlyOne,
}

fn predicate_group_mode(token: &str) -> Option<PredicateGroupMode> {
    match token {
        "any" | "or" => Some(PredicateGroupMode::Any),
        "all" | "and" => Some(PredicateGroupMode::All),
        "one" | "xor" | "exactly_one" => Some(PredicateGroupMode::ExactlyOne),
        _ => None,
    }
}

fn split_group_mode_prefix(value: &str) -> Option<(PredicateGroupMode, &str)> {
    let trimmed = value.trim();
    let Some((prefix, rest)) = trimmed.split_once(';') else {
        return Some((PredicateGroupMode::Any, trimmed));
    };
    let Some(mode_token) = prefix.trim().strip_prefix("mode=") else {
        return None;
    };
    let mode = predicate_group_mode(mode_token.trim())?;
    Some((mode, rest.trim()))
}

fn combine_group_clauses(clauses: Vec<String>, mode: PredicateGroupMode) -> String {
    if clauses.is_empty() {
        return "(FALSE)".to_string();
    }

    match mode {
        PredicateGroupMode::Any => format!("({})", clauses.join(" OR ")),
        PredicateGroupMode::All => format!("({})", clauses.join(" AND ")),
        PredicateGroupMode::ExactlyOne => {
            let sum = clauses
                .into_iter()
                .map(|clause| format!("(CASE WHEN {} THEN 1 ELSE 0 END)", clause))
                .collect::<Vec<_>>()
                .join(" + ");
            format!("(({}) = 1)", sum)
        }
    }
}

fn focus_quality_threshold_criteria(
    value: &str,
) -> Option<(PredicateGroupMode, Vec<(&'static str, u8)>)> {
    let trimmed = value.trim();
    if trimmed.is_empty() || !trimmed.contains('=') {
        return None;
    }
    let (mode, criteria_value) = split_group_mode_prefix(trimmed)?;

    let mut criteria = Vec::new();
    for token in criteria_value.split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        let mut parts = token.splitn(2, '=');
        let basis = parts.next()?.trim();
        let threshold = parts.next()?.trim().parse::<u8>().ok()?;
        if !(1..=10).contains(&threshold) {
            return None;
        }
        let column = focus_quality_column_for_basis(basis)?;
        if !criteria.iter().any(|(existing, _)| existing == &column) {
            criteria.push((column, threshold));
        }
    }

    if criteria.is_empty() {
        None
    } else {
        Some((mode, criteria))
    }
}

fn focus_quality_compare_sql(column: &str, op: &str, bucket: u8) -> Option<String> {
    let threshold = focus_quality_bucket_threshold_sql(bucket, column)?;
    let sym = sql_compare_op(op)?;
    Some(format!(
        "({column} IS NOT NULL AND {column} {sym} {threshold})",
        column = column,
        sym = sym,
        threshold = threshold
    ))
}

fn focus_quality_between_sql(column: &str, start: u8, end: u8) -> Option<String> {
    let a = focus_quality_bucket_threshold_sql(start, column)?;
    let b = focus_quality_bucket_threshold_sql(end, column)?;
    Some(format!(
        "({column} IS NOT NULL AND {column} BETWEEN LEAST({a}, {b}) AND GREATEST({a}, {b}))",
        column = column,
        a = a,
        b = b
    ))
}

fn focus_quality_predicate_sql(p: &QueryPredicate) -> String {
    if let Some(value) = p.value.as_deref() {
        if let Some((mode, criteria)) = focus_quality_threshold_criteria(value) {
            let clauses = criteria
                .into_iter()
                .filter_map(|(column, bucket)| focus_quality_compare_sql(column, "gte", bucket))
                .collect::<Vec<_>>();
            return combine_group_clauses(clauses, mode);
        }
    }

    let Some(columns) = focus_quality_columns(p.value.as_deref()) else {
        return "(FALSE)".to_string();
    };

    match (p.op.as_deref(), p.num) {
        (Some("between"), Some(num)) => {
            let Some(end) = p.num_end else {
                return "(FALSE)".to_string();
            };
            let Some(start_bucket) = focus_quality_bucket_from_num(num) else {
                return "(FALSE)".to_string();
            };
            let Some(end_bucket) = focus_quality_bucket_from_num(end) else {
                return "(FALSE)".to_string();
            };
            let clauses = columns
                .iter()
                .filter_map(|column| focus_quality_between_sql(column, start_bucket, end_bucket))
                .collect::<Vec<_>>();
            if clauses.is_empty() {
                "(FALSE)".to_string()
            } else {
                format!("({})", clauses.join(" OR "))
            }
        }
        (Some(op), Some(num)) => {
            let Some(bucket) = focus_quality_bucket_from_num(num) else {
                return "(FALSE)".to_string();
            };
            let clauses = columns
                .iter()
                .filter_map(|column| focus_quality_compare_sql(column, op, bucket))
                .collect::<Vec<_>>();
            if clauses.is_empty() {
                "(FALSE)".to_string()
            } else {
                format!("({})", clauses.join(" OR "))
            }
        }
        _ => "(FALSE)".to_string(),
    }
}

fn face_quality_column_for_metric(metric: &str) -> Option<&'static str> {
    match metric {
        "best" | "face" => Some("face_quality_best"),
        "average" | "avg" => Some("face_quality_average"),
        "lowest" | "min" | "minimum" => Some("face_quality_min"),
        _ => None,
    }
}

fn face_quality_threshold_criteria(
    value: &str,
) -> Option<(PredicateGroupMode, Vec<(&'static str, u8)>)> {
    let trimmed = value.trim();
    if trimmed.is_empty() || !trimmed.contains('=') {
        return None;
    }
    let (mode, criteria_value) = split_group_mode_prefix(trimmed)?;

    let mut criteria = Vec::new();
    for token in criteria_value.split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        let mut parts = token.splitn(2, '=');
        let metric = parts.next()?.trim();
        let threshold = parts.next()?.trim().parse::<u8>().ok()?;
        if threshold > 100 {
            return None;
        }
        let column = face_quality_column_for_metric(metric)?;
        if !criteria.iter().any(|(existing, _)| existing == &column) {
            criteria.push((column, threshold));
        }
    }

    if criteria.is_empty() {
        None
    } else {
        Some((mode, criteria))
    }
}

fn face_quality_compare_sql(column: &str, percent: u8) -> String {
    let threshold = f64::from(percent) / 100.0;
    format!(
        "({column} IS NOT NULL AND {column} >= {threshold})",
        column = column,
        threshold = threshold
    )
}

fn face_quality_predicate_sql(p: &QueryPredicate) -> String {
    let Some(value) = p.value.as_deref() else {
        return "(FALSE)".to_string();
    };
    let Some((mode, criteria)) = face_quality_threshold_criteria(value) else {
        return "(FALSE)".to_string();
    };
    let clauses = criteria
        .into_iter()
        .map(|(column, percent)| face_quality_compare_sql(column, percent))
        .collect::<Vec<_>>();
    combine_group_clauses(clauses, mode)
}

/// SQL for a connector. XOR is boolean inequality (`<>`) — exactly-one-true.
fn connector_sql(c: &Connector) -> &'static str {
    match c {
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
fn predicate_to_sql(p: &QueryPredicate) -> String {
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
        // Dynamic date (Session 66 — Gate 3 of the smart-collections plan):
        // "in the last N days/weeks/months/years". THE DATABASE resolves the
        // cutoff at EXECUTION time — `CURRENT_DATE - INTERVAL` arithmetic at
        // this one chokepoint — so every consumer (paging, counts, ⌘A, saved
        // queries, future surfaces) stays honest by construction, and a saved
        // "Past Month" never freezes into the month it was saved (the S65
        // decision: never snapshot a relative date). The count rides `value`,
        // the unit token rides `op` (zero schema change). This is the arm
        // Lightroom's `captureTime inLast N <unit>` smart-collection rule
        // maps to. Day-granular like every other date arm: strftime renders
        // the cutoff in the stored colon form and the comparison is `>=`.
        "date_in_last" => match (p.value.as_deref(), p.op.as_deref())
        {
            (Some(v), Some(unit)) => match (v.parse::<u32>(), interval_unit_sql(unit))
            {
                (Ok(n), Some(u)) if (1..=9999).contains(&n) => format!(
                    "(SUBSTRING(capture_datetime, 1, 10) >= strftime(CURRENT_DATE - INTERVAL {} {}, '%Y:%m:%d'))",
                    n, u
                ),
                _ => bad(),
            },
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
        // The color subject knows BOTH storage halves (S66): the five standard
        // names live in `images.color_label`; custom Lightroom color-label text
        // lives as a keyword row with the `color` switch ON. Color-ness reads
        // the RAW keyword table (like collection_is — hiding a keyword doesn't
        // un-color the photo), so any_color/no_color stay exact complements.
        "no_color" => "(color_label IS NULL AND NOT EXISTS \
                       (SELECT 1 FROM keyword k WHERE k.image_id = images.id AND k.color = TRUE))"
            .to_string(),
        "any_color" => "(color_label IS NOT NULL OR EXISTS \
                        (SELECT 1 FROM keyword k WHERE k.image_id = images.id AND k.color = TRUE))"
            .to_string(),
        // Keyword subject (Session 45). Label-equality is automatically
        // subtree-inclusive: every ancestor is its own materialized row carrying
        // its label, so `label = 'Animals'` matches everything beneath Animals.
        // Correlated on images.id against the active-only view. Isolation-first:
        // this pair of arms is the ONLY touch to the existing query engine.
        "keyword_has" => match p.value.as_deref()
        {
            Some(v) if !v.is_empty() => format!(
                "(EXISTS (SELECT 1 FROM keyword_visible k WHERE k.image_id = images.id AND k.label = '{}'{}))",
                v.replace('\'', "''"),
                keyword_origin_clause(p.op.as_deref())
            ),
            _ => bad(),
        },
        "keyword_not" => match p.value.as_deref()
        {
            Some(v) if !v.is_empty() => format!(
                "(NOT EXISTS (SELECT 1 FROM keyword_visible k WHERE k.image_id = images.id AND k.label = '{}'{}))",
                v.replace('\'', "''"),
                keyword_origin_clause(p.op.as_deref())
            ),
            _ => bad(),
        },
        // "Has no keywords" (Session 66 — Gate 2 of the smart-collections
        // plan): true when the image carries NO visible NON-COLOR keyword row.
        // The label-free sibling of keyword_not — `value` is ignored. Rows
        // whose `color` switch is ON are custom Lightroom color-label text,
        // not subject keywords, so they don't count — which keeps this arm in
        // exact parity with Lightroom's "Without Keywords" rule (Lightroom
        // never counts color labels as keywords). NULL-tolerant for the
        // instant between a migration ALTER and its backfill.
        "keyword_none" =>
            format!(
                "(NOT EXISTS (SELECT 1 FROM keyword_visible k WHERE k.image_id = images.id \
                  AND (k.color = FALSE OR k.color IS NULL){}))",
                keyword_origin_clause(p.op.as_deref())
            ),
        // Sidebar multi-select scopes (S67): one predicate per selected
        // sidebar node, OR-joined by the caller — the union of the selected
        // folders/dates rides the same paged query machinery every other
        // scope uses. `path_prefix` carries a slash-terminated directory
        // prefix (the caller normalizes); `capture_prefix` carries a
        // capture_datetime prefix ("2026:", "2026:06:", "2026:06:06") —
        // the same prefix semantics as the single-node sidebar load, so a
        // one-node selection and a union behave identically per node.
        // starts_with(NULL, …) is NULL → undated rows never match a capture
        // prefix (parity with the single-node date load).
        "path_prefix" => match p.value.as_deref()
        {
            Some(v) if !v.is_empty() =>
            {
                // Folder-scope prefixes exclude Apple Photos library originals
                // (their own "Apple Libraries" scope; nested under a folder path
                // they'd otherwise be string-matched in — see
                // build_path_date_predicate). A prefix that IS inside a
                // `.photoslibrary` (the Apple Library node) is exempt. The whole
                // atom stays parenthesized so OR-joining several is safe.
                let apple_exclusion = if v.contains(".photoslibrary")
                {
                    ""
                }
                else
                {
                    " AND file_path NOT LIKE '%.photoslibrary/%'"
                };
                format!(
                    "(starts_with(file_path, '{}'){})",
                    v.replace('\'', "''"),
                    apple_exclusion
                )
            },
            _ => bad(),
        },
        "capture_prefix" => match p.value.as_deref()
        {
            Some(v) if !v.is_empty() => format!(
                "(starts_with(capture_datetime, '{}'))",
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
        // Metadata subjects (Session 63 — the dropdown doctrine; Docs/
        // DESIGN-Saved-Queries.md §3): exact equality against the PICKED
        // catalogue value. The Swift type-ahead supplies values that exist
        // verbatim (and file_extension is lowercase-canonical by migration
        // invariant), so no case folding; a NULL column (no EXIF) correctly
        // never matches.
        "extension_is" => match p.value.as_deref()
        {
            Some(v) if !v.is_empty() => format!("(file_extension = '{}')", v.replace('\'', "''")),
            _ => bad(),
        },
        "kind_is" => match p.value.as_deref()
        {
            Some(v) if !v.is_empty() => format!("(image_kind = '{}')", v.replace('\'', "''")),
            _ => bad(),
        },
        "camera_make_is" => match p.value.as_deref()
        {
            Some(v) if !v.is_empty() => format!("(camera_make = '{}')", v.replace('\'', "''")),
            _ => bad(),
        },
        "camera_model_is" => match p.value.as_deref()
        {
            Some(v) if !v.is_empty() => format!("(camera_model = '{}')", v.replace('\'', "''")),
            _ => bad(),
        },
        "lens_is" => match p.value.as_deref()
        {
            Some(v) if !v.is_empty() => format!("(lens_model = '{}')", v.replace('\'', "''")),
            _ => bad(),
        },
        "creator_is" => match p.value.as_deref()
        {
            Some(v) if !v.is_empty() => format!("(creator = '{}')", v.replace('\'', "''")),
            _ => bad(),
        },
        // Codec (S75 — Query Builder video fields): exact match against the
        // canonical codec string ("hevc" / "h264" / "prores"), dropdown-sourced
        // like the other metadata subjects. A NULL column (a still) never matches.
        "codec_is" => match p.value.as_deref()
        {
            Some(v) if !v.is_empty() => format!("(video_codec = '{}')", v.replace('\'', "''")),
            _ => bad(),
        },
        // Lens "contains" — free text, case-INDEPENDENT (ILIKE), wildcard-
        // escaped so a literal %/_ matches itself. No autofill by design:
        // the user types a fragment, not an existing value (Richard, S63).
        "lens_contains" => match p.value.as_deref()
        {
            Some(v) if !v.is_empty() => lens_ilike_atom(&format!("%{}%", escape_for_ilike(v))),
            _ => bad(),
        },
        // Numeric subjects (S65 — Gate 1 of the smart-collections plan): the
        // four exposure columns, op ∈ eq/gte/lte/between (UI sends those; the
        // remaining rating tokens gt/lt also work — the LR import may use
        // them). The bound(s) ride `num`/`num_end`; a NULL column (no EXIF)
        // correctly never matches. ISO is INTEGER, the rest REAL — DuckDB
        // compares either against the f64 literal directly.
        "iso_num" => numeric_predicate_sql(p, "iso"),
        "aperture_num" => numeric_predicate_sql(p, "aperture"),
        "shutter_num" => numeric_predicate_sql(p, "shutter_speed"),
        "focal_num" => numeric_predicate_sql(p, "focal_length"),
        "focus_num" => numeric_predicate_sql(p, "focus_score"),
        "focus_quality" => focus_quality_predicate_sql(p),
        "people_count" => people_count_predicate_sql(p),
        "face_quality" => face_quality_predicate_sql(p),
        "face_quality_best_num" => numeric_predicate_sql(p, "face_quality_best"),
        "face_quality_average_num" => numeric_predicate_sql(p, "face_quality_average"),
        "face_quality_min_num" => numeric_predicate_sql(p, "face_quality_min"),
        "eyes_open_count_num" => numeric_predicate_sql(p, "face_eyes_open_count"),
        "blink_risk_count_num" => numeric_predicate_sql(p, "face_blink_risk_count"),
        // Video numeric subjects (S75) — same machinery, new columns. Duration
        // is seconds (DOUBLE); frame_rate is fps (DOUBLE). A NULL column (a
        // still) never matches, exactly like the exposure numerics above.
        "duration_num" => numeric_predicate_sql(p, "duration_seconds"),
        "framerate_num" => numeric_predicate_sql(p, "frame_rate"),
        // Video fixed-choice subjects (S75) — the `kind` alone is the predicate
        // (no value/op), like rating_unrated / unflagged. Dynamic range is derived
        // from the CICP transfer + Dolby Vision profile: HLG/HDR10 are transfer
        // matches and DV is the presence of a profile (a still's NULL columns
        // never match those). SDR = a video that is none of the HDR forms, so it
        // carries an explicit `is_video` guard. Audio reads the tri-state
        // has_audio (a still is NULL → excluded by both = TRUE / = FALSE). Live
        // Photo is video-scoped on the shared content.identifier.
        "dynrange_hlg" => "(color_transfer = 'arib-std-b67')".to_string(),
        "dynrange_hdr10" => "(color_transfer = 'smpte2084')".to_string(),
        "dynrange_dv" => "(dv_profile IS NOT NULL)".to_string(),
        "dynrange_sdr" =>
            "(is_video IS TRUE AND dv_profile IS NULL \
              AND (color_transfer IS NULL OR color_transfer NOT IN ('arib-std-b67', 'smpte2084')))"
                .to_string(),
        "audio_present" => "(has_audio = TRUE)".to_string(),
        "audio_absent" => "(has_audio = FALSE)".to_string(),
        "livephoto_yes" => "(is_video IS TRUE AND live_photo_id IS NOT NULL)".to_string(),
        "livephoto_no" => "(is_video IS TRUE AND live_photo_id IS NULL)".to_string(),
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
fn build_filter_predicate(predicates: &[QueryPredicate], connectors: &[Connector]) -> String {
    if predicates.is_empty() {
        return String::new();
    }

    let mut acc = predicate_to_sql(&predicates[0]);
    for i in 1..predicates.len() {
        // connectors[i-1] joins the running result with segment i. Swift
        // sends predicates.len()-1 connectors; default to AND if short.
        let op = connectors.get(i - 1).map(connector_sql).unwrap_or("AND");
        let next = predicate_to_sql(&predicates[i]);
        acc = format!("({}) {} ({})", acc, op, next);
    }

    format!("({})", acc)
}

/// Compose a normal Query Builder sentence with an independent gallery scope
/// (Sources/Dates). The flat `connectors` array is intentionally left-to-right,
/// so appending scope predicates directly would let OR clauses leak out of the
/// selected sidebar scope. Keep the two groups parenthesized and AND them here.
fn build_scoped_filter_predicate(
    predicates: &[QueryPredicate],
    connectors: &[Connector],
    scope_predicates: &[QueryPredicate],
    scope_connectors: &[Connector],
) -> String {
    let filter = build_filter_predicate(predicates, connectors);
    let scope = build_filter_predicate(scope_predicates, scope_connectors);

    match (filter.is_empty(), scope.is_empty()) {
        (true, true) => String::new(),
        (false, true) => filter,
        (true, false) => scope,
        (false, false) => format!("({}) AND ({})", filter, scope),
    }
}

/// The Browse default row order: capture date newest-first. Used for the empty
/// filter and for any first-subject that doesn't define its own order.
const DEFAULT_FILTER_ORDER_BY: &str = "capture_datetime DESC NULLS LAST, created_timestamp DESC";

/// First-subject-rating order: stars high-to-low, then newest-first as the
/// tie-break (and a stable final key).
const RATING_FILTER_ORDER_BY: &str =
    "rating DESC NULLS LAST, capture_datetime DESC NULLS LAST, created_timestamp DESC";

/// First-subject-focus order: sharpest analyzed images first, then newest-first
/// as the ordinary tie-break. NULLS LAST keeps not-yet-analyzed files honest.
const FOCUS_FILTER_ORDER_BY: &str =
    "focus_score DESC NULLS LAST, capture_datetime DESC NULLS LAST, created_timestamp DESC";

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
fn order_by_for_filter(predicates: &[QueryPredicate]) -> &'static str {
    match predicates.first() {
        Some(p) => match p.kind.as_str() {
            "rating" | "rating_unrated" => RATING_FILTER_ORDER_BY,
            "focus_num" | "focus_quality" => FOCUS_FILTER_ORDER_BY,
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
    media_type: MediaType,
) -> Vec<ImageRecord> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
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
        false,
        "",
        media_type,
    )
}

/// Filter / Query Builder — total matching count for the SAME filter, via the
/// shared `execute_image_count_query` helper (parity by construction).
pub async fn count_query_images(
    predicates: Vec<QueryPredicate>,
    connectors: Vec<Connector>,
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
    media_type: MediaType,
) -> u64 {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
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
        false,
        "",
        media_type,
    );

    if count < 0 {
        0
    } else {
        count as u64
    }
}

/// Gallery-specific Query Builder variant that can collapse durable
/// similar-photo stacks before pagination. Browse intentionally stays on
/// `query_images` so its long-standing physical-row semantics do not change.
pub async fn query_images_gallery(
    predicates: Vec<QueryPredicate>,
    connectors: Vec<Connector>,
    limit: u32,
    offset: u32,
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
    apply_similar_photo_collapse: bool,
    similar_algorithm_version: String,
    media_type: MediaType,
) -> Vec<ImageRecord> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
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
        apply_similar_photo_collapse,
        &similar_algorithm_version,
        media_type,
    )
}

pub async fn count_query_images_gallery(
    predicates: Vec<QueryPredicate>,
    connectors: Vec<Connector>,
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
    apply_similar_photo_collapse: bool,
    similar_algorithm_version: String,
    media_type: MediaType,
) -> u64 {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
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
        apply_similar_photo_collapse,
        &similar_algorithm_version,
        media_type,
    );

    if count < 0 {
        0
    } else {
        count as u64
    }
}

/// Gallery Query Builder inside the currently viewed Sources/Dates scope.
/// Browse deliberately keeps using `query_images`; this variant exists because
/// Gallery needs `(query sentence) AND (sidebar scope)` grouping, which the flat
/// left-to-right connector model cannot express safely when either side has ORs.
pub async fn query_images_scoped(
    predicates: Vec<QueryPredicate>,
    connectors: Vec<Connector>,
    scope_predicates: Vec<QueryPredicate>,
    scope_connectors: Vec<Connector>,
    limit: u32,
    offset: u32,
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
    media_type: MediaType,
) -> Vec<ImageRecord> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    let where_clause = build_scoped_filter_predicate(
        &predicates,
        &connectors,
        &scope_predicates,
        &scope_connectors,
    );
    let order_by = order_by_for_filter(&predicates);

    execute_image_record_query(
        conn,
        &where_clause,
        order_by,
        limit as i64,
        offset as i64,
        apply_duplicate_filter,
        apply_raw_jpeg_collapse,
        false,
        "",
        media_type,
    )
}

/// Count counterpart for `query_images_scoped`; uses the exact same WHERE
/// assembly so page/count parity holds.
pub async fn count_query_images_scoped(
    predicates: Vec<QueryPredicate>,
    connectors: Vec<Connector>,
    scope_predicates: Vec<QueryPredicate>,
    scope_connectors: Vec<Connector>,
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
    media_type: MediaType,
) -> u64 {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    let where_clause = build_scoped_filter_predicate(
        &predicates,
        &connectors,
        &scope_predicates,
        &scope_connectors,
    );

    let count = execute_image_count_query(
        conn,
        &where_clause,
        apply_duplicate_filter,
        apply_raw_jpeg_collapse,
        false,
        "",
        media_type,
    );

    if count < 0 {
        0
    } else {
        count as u64
    }
}

pub async fn query_images_scoped_gallery(
    predicates: Vec<QueryPredicate>,
    connectors: Vec<Connector>,
    scope_predicates: Vec<QueryPredicate>,
    scope_connectors: Vec<Connector>,
    limit: u32,
    offset: u32,
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
    apply_similar_photo_collapse: bool,
    similar_algorithm_version: String,
    media_type: MediaType,
) -> Vec<ImageRecord> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    let where_clause = build_scoped_filter_predicate(
        &predicates,
        &connectors,
        &scope_predicates,
        &scope_connectors,
    );
    let order_by = order_by_for_filter(&predicates);

    execute_image_record_query(
        conn,
        &where_clause,
        order_by,
        limit as i64,
        offset as i64,
        apply_duplicate_filter,
        apply_raw_jpeg_collapse,
        apply_similar_photo_collapse,
        &similar_algorithm_version,
        media_type,
    )
}

pub async fn count_query_images_scoped_gallery(
    predicates: Vec<QueryPredicate>,
    connectors: Vec<Connector>,
    scope_predicates: Vec<QueryPredicate>,
    scope_connectors: Vec<Connector>,
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
    apply_similar_photo_collapse: bool,
    similar_algorithm_version: String,
    media_type: MediaType,
) -> u64 {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    let where_clause = build_scoped_filter_predicate(
        &predicates,
        &connectors,
        &scope_predicates,
        &scope_connectors,
    );

    let count = execute_image_count_query(
        conn,
        &where_clause,
        apply_duplicate_filter,
        apply_raw_jpeg_collapse,
        apply_similar_photo_collapse,
        &similar_algorithm_version,
        media_type,
    );

    if count < 0 {
        0
    } else {
        count as u64
    }
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
fn id_in_list(ids: &[i64]) -> Option<String> {
    if ids.is_empty() {
        return None;
    }
    let joined = ids
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(", ");
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
    media_type: MediaType,
) -> Vec<i64> {
    let mut inner_predicates: Vec<&str> = Vec::new();
    if !where_clause.is_empty() {
        inner_predicates.push(where_clause);
    }
    if apply_raw_jpeg_collapse {
        inner_predicates.push(RAW_JPEG_COLLAPSE_PREDICATE);
    }
    // Media-type stance (DESIGN-Video-Schema-Unified-Table.md §11): gallery,
    // Browse, ⌘A, filtered counts, and path/date-prefix queries all gate through
    // this one seam. `media_predicate` maps the caller's MediaType to its WHERE
    // fragment — None for Both (stills + video together), so nothing is pushed.
    if let Some(media_pred) = media_predicate(media_type) {
        inner_predicates.push(media_pred);
    }
    let inner_where = if inner_predicates.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", inner_predicates.join(" AND "))
    };

    let query_sql = if apply_duplicate_filter {
        format!(
            r#"
            SELECT id FROM (
                SELECT
                    id,
                    {}
                FROM images
                {}
            )
            WHERE {}
        "#,
            DUPLICATE_GROUP_ID_CASE, inner_where, DUPLICATE_FILTER_PREDICATE
        )
    } else {
        format!("SELECT id FROM images {}", inner_where)
    };

    let mut stmt = match conn.prepare(&query_sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to prepare id projection query: {}", e);
            return Vec::new();
        }
    };

    let rows = match stmt.query_map([], |row| row.get::<_, i64>(0)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to execute id projection query: {}", e);
            return Vec::new();
        }
    };

    let mut ids: Vec<i64> = Vec::new();
    for row_result in rows {
        match row_result {
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
    media_type: MediaType,
) -> Vec<i64> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    let where_clause = build_filter_predicate(&predicates, &connectors);
    execute_id_projection_query(
        conn,
        &where_clause,
        apply_duplicate_filter,
        apply_raw_jpeg_collapse,
        media_type,
    )
}

/// Resolve record IDs to full `ImageRecord`s — turns a Browse selection (which
/// may span pages, or be the whole query) into records for Copy / Reveal.
/// The IDs ARE the exact selection, so NO duplicate / raw-collapse filtering is
/// applied. Order is unspecified (the copy planner re-sorts). Empty → empty.
pub async fn get_images_by_ids(ids: Vec<i64>) -> Vec<ImageRecord> {
    let where_clause = match id_in_list(&ids) {
        Some(w) => w,
        None => return Vec::new(),
    };

    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    // Resolve an explicit id selection to records (Copy / Reveal). Catalogue
    // truth — `Both`, so a video the user selected once Stage 6 display is live
    // still resolves; in stills-only mode the selected ids are all stills, so
    // the result is unchanged from the former stills-only gate.
    execute_image_record_projection_query(conn, &where_clause, false, false, MediaType::Both)
}

fn project_raw_jpeg_visible_ids_impl(
    conn: &Connection,
    ids: &[i64],
    apply_raw_jpeg_collapse: bool,
) -> Vec<i64> {
    if ids.is_empty() {
        return Vec::new();
    }

    if !apply_raw_jpeg_collapse {
        return ids.to_vec();
    }

    let values = ids
        .iter()
        .enumerate()
        .map(|(index, id)| format!("({}, {})", *id, index))
        .collect::<Vec<_>>()
        .join(", ");
    let query_sql = format!(
        r#"
        WITH input(id, ord) AS (VALUES {})
        SELECT
            input.ord,
            COALESCE(
                (
                    SELECT sibling.id
                    FROM images sibling
                    WHERE image.image_kind = 'raw'
                      AND sibling.image_kind IN ('jpeg', 'heif')
                      AND sibling.file_stem = image.file_stem
                      AND sibling.directory_path = image.directory_path
                    ORDER BY
                        CASE
                            WHEN sibling.image_kind = 'jpeg' THEN 0
                            WHEN sibling.image_kind = 'heif' THEN 1
                            ELSE 2
                        END,
                        sibling.file_path ASC
                    LIMIT 1
                ),
                image.id
            ) AS visible_id
        FROM input
        JOIN images image ON image.id = input.id
        ORDER BY input.ord
    "#,
        values
    );

    let mut stmt = match conn.prepare(&query_sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to prepare RAW/JPEG projection query: {}", e);
            return Vec::new();
        }
    };

    let rows = match stmt.query_map([], |row| row.get::<_, i64>(1)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to execute RAW/JPEG projection query: {}", e);
            return Vec::new();
        }
    };

    let mut seen = std::collections::HashSet::<i64>::new();
    let mut projected = Vec::<i64>::new();
    for row_result in rows {
        match row_result {
            Ok(id) => {
                if seen.insert(id) {
                    projected.push(id);
                }
            }
            Err(e) => eprintln!("Failed to read RAW/JPEG projection row: {}", e),
        }
    }

    projected
}

/// Project an ordered image-id list through the Gallery's RAW/JPEG visibility
/// rule. Hidden RAW hits become their visible JPEG/HEIF sibling, and duplicate
/// projected IDs are removed while preserving the original score/order.
pub async fn project_raw_jpeg_visible_ids(
    ids: Vec<i64>,
    apply_raw_jpeg_collapse: bool,
) -> Vec<i64> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    project_raw_jpeg_visible_ids_impl(conn, &ids, apply_raw_jpeg_collapse)
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
pub async fn expand_collapse_group_ids(ids: Vec<i64>) -> Vec<i64> {
    if ids.is_empty() {
        return Vec::new();
    }
    let csv = ids
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
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

    let mut stmt = match conn.prepare(&query_sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to prepare collapse-group expansion: {}", e);
            return ids;
        }
    };

    let rows = match stmt.query_map([], |row| row.get::<_, i64>(0)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to execute collapse-group expansion: {}", e);
            return ids;
        }
    };

    let mut result: Vec<i64> = Vec::new();
    for row_result in rows {
        if let Ok(id) = row_result {
            result.push(id);
        }
    }

    if result.is_empty() {
        ids
    } else {
        result
    }
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
const KEYWORD_ORIGIN_USER: i32 = 1;
const KEYWORD_ORIGIN_AUTO: i32 = 2;

/// A single materialized keyword row, as returned to Swift.
#[derive(Debug, Clone)]
pub struct KeywordRow {
    pub label: String,
    pub path: String,
    pub status: i32,
    pub origin: i32,
    pub created_at: String,
    pub hidden_at: Option<String>,
}

/// A distinct (label, path) node — the vocabulary, for autocomplete + browsing.
#[derive(Debug, Clone)]
pub struct KeywordNode {
    pub label: String,
    pub path: String,
}

/// Settings → Keywords management row. One row per distinct keyword path,
/// aggregated across the raw keyword table so cleanup can see collection-backed
/// rows and hidden orphan rows that normal vocabulary reads intentionally hide.
#[derive(Debug, Clone)]
pub struct KeywordManagementRow {
    pub label: String,
    pub path: String,
    pub origin: i32,
    pub visible_count: i64,
    pub hidden_count: i64,
    pub collection_count: i64,
    pub total_count: i64,
}

/// Materialize an ordered segment list into one (label, path) pair per ancestor
/// depth. `["Animals","Dog","Lab"]` -> `[("Animals","Animals"),
/// ("Dog","Animals␟Dog"), ("Lab","Animals␟Dog␟Lab")]`. Returns empty if any
/// segment is blank or itself contains the separator (-> caller no-ops).
fn keyword_materialized_rows(segments: &[String]) -> Vec<(String, String)> {
    let mut rows: Vec<(String, String)> = Vec::new();
    let mut prefix: Vec<String> = Vec::new();
    for seg in segments {
        let trimmed = seg.trim();
        if trimmed.is_empty() || trimmed.contains(KEYWORD_PATH_SEPARATOR) {
            return Vec::new();
        }
        prefix.push(trimmed.to_string());
        rows.push((trimmed.to_string(), prefix.join(KEYWORD_PATH_SEPARATOR)));
    }
    rows
}

/// `image_id IN (...)` predicate — sibling of `id_in_list`, which is hard-coded
/// to the `images.id` column. `None` on empty.
fn image_id_in_list(ids: &[i64]) -> Option<String> {
    if ids.is_empty() {
        return None;
    }
    let joined = ids
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    Some(format!("image_id IN ({})", joined))
}

fn merge_active_keyword_origin(
    conn: &Connection,
    image_id: i64,
    path: &str,
    origin: i32,
) -> Result<bool, duckdb::Error> {
    let changed = conn.execute(
        "UPDATE keyword SET origin = (origin | ?) \
         WHERE image_id = ? AND path = ? AND status = 1 AND (origin & ?) = 0",
        params![origin, image_id, path, origin],
    )?;
    Ok(changed > 0)
}

fn insert_or_merge_active_keyword_for_image(
    conn: &Connection,
    image_id: i64,
    segments: &[String],
    origin: i32,
) -> Result<u64, duckdb::Error> {
    let rows = keyword_materialized_rows(segments);
    if rows.is_empty() {
        return Ok(0);
    }

    let mut changed: u64 = 0;
    for (label, path) in rows {
        let existing: Result<i64, _> = conn.query_row(
            "SELECT 1 FROM keyword WHERE image_id = ? AND path = ? AND status = 1 LIMIT 1",
            params![image_id, path],
            |r| r.get(0),
        );
        if existing.is_ok() {
            if merge_active_keyword_origin(conn, image_id, &path, origin)? {
                changed += 1;
            }
            continue;
        }

        conn.execute(
            "INSERT INTO keyword (image_id, label, path, status, origin, created_at, is_video) \
             VALUES (?, ?, ?, 1, ?, CURRENT_TIMESTAMP, COALESCE((SELECT is_video FROM images WHERE id = ?), FALSE))",
            params![image_id, label, path, origin, image_id],
        )?;
        changed += 1;
    }

    Ok(changed)
}

/// Assign a keyword PATH to many images. Rust materializes the ancestor chain
/// and inserts one row per depth for each image. Blind-insert, except it skips a
/// row byte-identical to an already-ACTIVE row for that image (so a double-apply
/// doesn't spam duplicates). A previously-removed (hidden) identical row is NOT
/// resurrected — a fresh active row is inserted, preserving history. One
/// transaction. Returns the number of rows inserted.
pub async fn assign_keyword_for_ids(ids: Vec<i64>, segments: Vec<String>) -> u64 {
    if ids.is_empty() {
        return 0;
    }
    let rows = keyword_materialized_rows(&segments);
    if rows.is_empty() {
        eprintln!("assign_keyword_for_ids: empty or invalid segments");
        return 0;
    }

    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;") {
        eprintln!("assign_keyword_for_ids: begin failed: {}", e);
        return 0;
    }

    let mut inserted: u64 = 0;
    for id in &ids {
        match insert_or_merge_active_keyword_for_image(conn, *id, &segments, KEYWORD_ORIGIN_USER) {
            Ok(changed) => inserted += changed,
            Err(e) => {
                eprintln!("assign_keyword_for_ids: insert failed: {}", e);
                let _ = conn.execute_batch("ROLLBACK;");
                return 0;
            }
        }
    }

    if let Err(e) = conn.execute_batch("COMMIT;") {
        eprintln!("assign_keyword_for_ids: commit failed: {}", e);
        return 0;
    }
    inserted
}

/// Remove a keyword from many images — soft-hide the node AND its descendants
/// (`path = ? OR starts_with(path, ?␟)`) for those images. Ancestors are LEFT
/// intact (a lone parent is a valid flat keyword). Returns rows hidden.
pub async fn remove_keyword_for_ids(ids: Vec<i64>, path: String) -> u64 {
    if ids.is_empty() || path.is_empty() {
        return 0;
    }
    let where_ids = match image_id_in_list(&ids) {
        Some(w) => w,
        None => return 0,
    };

    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
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
    match conn.execute(&sql, params![path, prefix]) {
        Ok(changed) => changed as u64,
        Err(e) => {
            eprintln!("remove_keyword_for_ids: {}", e);
            0
        }
    }
}

/// Restore (un-hide) a previously removed keyword node + descendants for many
/// images — undo of `remove_keyword_for_ids` and the recovery-screen action.
pub async fn restore_keyword_for_ids(ids: Vec<i64>, path: String) -> u64 {
    if ids.is_empty() || path.is_empty() {
        return 0;
    }
    let where_ids = match image_id_in_list(&ids) {
        Some(w) => w,
        None => return 0,
    };

    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
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
    match conn.execute(&sql, params![path, prefix]) {
        Ok(changed) => changed as u64,
        Err(e) => {
            eprintln!("restore_keyword_for_ids: {}", e);
            0
        }
    }
}

/// The body of `mirror_keyword_rows_across_pairs`, against an explicit
/// connection (the S66 impl/wrapper pattern for live-testability).
///
/// Copies keyword rows across RAW <-> JPEG/HEIF pair siblings (same
/// `file_stem` + same `directory_path` — the pair-collapse identity) so both
/// halves carry identical keyword state. The collapse predicate hides a RAW
/// whenever a JPEG/HEIF twin exists TABLE-WIDE, so a keyword query matching
/// only the RAW would otherwise collapse the photo out of its own results
/// (the kept twin doesn't match). UI keywording never needs this — it
/// expands pairs before assigning (S44) — but import-SYNTHESIZED siblings
/// (S67, the Lightroom sidecar pass) are born after the import's keyword
/// pass and start bare; this heals them in one set-based statement per
/// direction. Whole rows ride: status, collection, color, hidden_at — the
/// three-switch model copies intact. Idempotent: NOT EXISTS on
/// (image_id, path) skips anything already there (including a row the user
/// hid on one half — their history wins); DISTINCT collapses double-sources
/// (a .nef and a .dng sharing one stem are BOTH raw twins of one JPEG).
fn mirror_keyword_rows_across_pairs_impl(conn: &Connection) -> u64 {
    // (dst kinds, src kinds) — both directions of the pair.
    const DIRECTIONS: [(&str, &str); 2] = [
        (
            "dst.image_kind IN ('jpeg', 'heif') AND src.image_kind = 'raw'",
            "raw -> jpeg/heif",
        ),
        (
            "dst.image_kind = 'raw' AND src.image_kind IN ('jpeg', 'heif')",
            "jpeg/heif -> raw",
        ),
    ];

    let mut copied: u64 = 0;
    for (kind_filter, direction_label) in DIRECTIONS {
        let sql = format!(
            "INSERT INTO keyword (image_id, label, path, status, origin, created_at, hidden_at, collection, color) \
             SELECT DISTINCT dst.id, k.label, k.path, k.status, k.origin, CURRENT_TIMESTAMP, k.hidden_at, k.collection, k.color \
             FROM images dst \
             JOIN images src ON src.file_stem = dst.file_stem \
                            AND src.directory_path = dst.directory_path \
             JOIN keyword k ON k.image_id = src.id \
             WHERE {} \
               AND NOT EXISTS (SELECT 1 FROM keyword k2 \
                               WHERE k2.image_id = dst.id AND k2.path = k.path)",
            kind_filter
        );
        match conn.execute(&sql, []) {
            Ok(changed) => copied += changed as u64,
            Err(e) => {
                eprintln!(
                    "mirror_keyword_rows_across_pairs ({}): {}",
                    direction_label, e
                );
            }
        }
    }
    copied
}

/// Mirror keyword rows across RAW+JPEG/HEIF pair siblings — see the impl
/// above. Called by the Lightroom sidecar pass (S67) after synthesizing the
/// sidecar JPEG records; safe (and a no-op) any other time.
pub async fn mirror_keyword_rows_across_pairs() -> u64 {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };
    mirror_keyword_rows_across_pairs_impl(conn)
}

/// The body of `copy_keyword_rows_for_image_pairs`, against an explicit
/// connection (the impl/wrapper pattern for live-testability).
///
/// Copies every keyword row from each source image to its paired destination
/// image — the Copy-and-Import inheritance step (S67): a catalogued copy
/// arrives wearing its original's keywords, collection memberships, and
/// color marks (all three switches ride the same rows). Unlike the
/// pair-sibling MIRROR above, the pairing here is EXPLICIT (parallel id
/// arrays from the copy plan) — the copy lives at a different path, so no
/// stem/directory identity exists to infer. Idempotent: NOT EXISTS on
/// (image_id, path) makes a re-copy a no-op. One transaction.
fn copy_keyword_rows_for_image_pairs_impl(
    conn: &Connection,
    source_ids: &[i64],
    destination_ids: &[i64],
) -> u64 {
    if source_ids.is_empty() || source_ids.len() != destination_ids.len() {
        if source_ids.len() != destination_ids.len() {
            eprintln!(
                "copy_keyword_rows_for_image_pairs: id arrays differ in length ({} vs {})",
                source_ids.len(),
                destination_ids.len()
            );
        }
        return 0;
    }

    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;") {
        eprintln!("copy_keyword_rows_for_image_pairs: begin failed: {}", e);
        return 0;
    }

    let mut copied: u64 = 0;
    for (src, dst) in source_ids.iter().zip(destination_ids.iter()) {
        if src == dst {
            continue;
        }
        match conn.execute(
            // Carry is_video from the source keyword rows — a byte copy keeps the media
            // type, so the source image's is_video is the copy's (was defaulting to
            // FALSE) — S72 closeout.
            "INSERT INTO keyword (image_id, label, path, status, origin, created_at, hidden_at, collection, color, is_video) \
             SELECT ?2, k.label, k.path, k.status, k.origin, CURRENT_TIMESTAMP, k.hidden_at, k.collection, k.color, k.is_video \
             FROM keyword k \
             WHERE k.image_id = ?1 \
               AND NOT EXISTS (SELECT 1 FROM keyword k2 \
                               WHERE k2.image_id = ?2 AND k2.path = k.path)",
            params![src, dst],
        )
        {
            Ok(changed) => copied += changed as u64,
            Err(e) =>
            {
                eprintln!("copy_keyword_rows_for_image_pairs ({} -> {}): {}", src, dst, e);
                let _ = conn.execute_batch("ROLLBACK;");
                return 0;
            }
        }
    }

    if let Err(e) = conn.execute_batch("COMMIT;") {
        eprintln!("copy_keyword_rows_for_image_pairs: commit failed: {}", e);
        return 0;
    }
    copied
}

/// Copy keyword rows from source images to their catalogued copies — see the
/// impl above. Called by the Copy-and-Import pass (S67) with the copy plan's
/// (source id, destination id) pairs, aligned by index.
pub async fn copy_keyword_rows_for_image_pairs(
    source_ids: Vec<i64>,
    destination_ids: Vec<i64>,
) -> u64 {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };
    copy_keyword_rows_for_image_pairs_impl(conn, &source_ids, &destination_ids)
}

/// All ACTIVE keyword rows for one image, ordered by path (root->leaf within a
/// branch). For the detail-panel reconstruction.
pub async fn keywords_for_image(image_id: i64) -> Vec<KeywordRow> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    let mut stmt = match conn.prepare(
        "SELECT label, path, status, origin, CAST(created_at AS VARCHAR), CAST(hidden_at AS VARCHAR) \
         FROM keyword_visible WHERE image_id = ? ORDER BY path",
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("keywords_for_image: prepare {}", e);
            return Vec::new();
        }
    };

    let mapped = stmt.query_map(params![image_id], |row| {
        Ok(KeywordRow {
            label: row.get(0)?,
            path: row.get(1)?,
            status: row.get(2)?,
            origin: row.get(3)?,
            created_at: row.get(4)?,
            hidden_at: row.get(5)?,
        })
    });

    match mapped {
        Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
        Err(e) => {
            eprintln!("keywords_for_image: query {}", e);
            Vec::new()
        }
    }
}

fn keyword_vocabulary_origin_where(origin: &str) -> Option<&'static str> {
    match origin {
        "both" | "" => Some(""),
        "user" => Some(" WHERE (origin & 1) <> 0"),
        "auto" => Some(" WHERE (origin & 2) <> 0"),
        _ => None,
    }
}

fn keyword_vocabulary_impl(conn: &Connection, origin: &str) -> Vec<KeywordNode> {
    let Some(where_clause) = keyword_vocabulary_origin_where(origin) else {
        return Vec::new();
    };
    let sql = format!(
        "SELECT DISTINCT label, path FROM keyword_visible{} ORDER BY path",
        where_clause
    );

    let mut stmt = match conn.prepare(&sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("keyword_vocabulary: prepare {}", e);
            return Vec::new();
        }
    };

    let mapped = stmt.query_map([], |row| {
        Ok(KeywordNode {
            label: row.get(0)?,
            path: row.get(1)?,
        })
    });

    match mapped {
        Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
        Err(e) => {
            eprintln!("keyword_vocabulary: query {}", e);
            Vec::new()
        }
    }
}

/// The DISTINCT (label, path) keyword vocabulary over the active view — for the
/// assignment-panel autocomplete and (future) tree browser. Ordered by path.
pub async fn keyword_vocabulary() -> Vec<KeywordNode> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    keyword_vocabulary_impl(conn, "both")
}

/// Provenance-filtered keyword vocabulary for Query Builder keyword predicates.
/// `both` deliberately means all active keyword rows, not only origin == 3.
pub async fn keyword_vocabulary_for_origin(origin: String) -> Vec<KeywordNode> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    keyword_vocabulary_impl(conn, origin.as_str())
}

fn keyword_management_origin_clause(origin: &str) -> Option<&'static str> {
    match origin {
        "both" | "" => Some(""),
        "user" => Some("WHERE (origin & 1) <> 0"),
        "auto" => Some("WHERE (origin & 2) <> 0"),
        _ => None,
    }
}

fn keyword_management_rows_impl(
    conn: &Connection,
    origin: &str,
    include_collections: bool,
    include_orphaned: bool,
) -> Vec<KeywordManagementRow> {
    let Some(where_clause) = keyword_management_origin_clause(origin) else {
        return Vec::new();
    };
    let collection_filter = if include_collections {
        ""
    } else {
        "AND collection_count = 0"
    };
    let orphan_filter = if include_orphaned {
        ""
    } else {
        "AND NOT (visible_count = 0 AND collection_count = 0)"
    };
    let sql = format!(
        "WITH scoped AS (
             SELECT label,
                    path,
                    origin,
                    status,
                    COALESCE(collection, FALSE) AS collection
             FROM keyword
             {}
         ),
         grouped AS (
             SELECT label,
                    path,
                    CAST(bit_or(origin) AS INTEGER) AS origin,
                    SUM(CASE WHEN status = 1 THEN 1 ELSE 0 END) AS visible_count,
                    SUM(CASE WHEN status <> 1 THEN 1 ELSE 0 END) AS hidden_count,
                    SUM(CASE WHEN collection THEN 1 ELSE 0 END) AS collection_count,
                    COUNT(*) AS total_count
             FROM scoped
             GROUP BY label, path
         )
         SELECT label, path, origin, visible_count, hidden_count, collection_count, total_count
         FROM grouped
         WHERE 1 = 1 {} {}
         ORDER BY path",
        where_clause, collection_filter, orphan_filter
    );

    let mut stmt = match conn.prepare(&sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("keyword_management_rows: prepare {}", e);
            return Vec::new();
        }
    };

    let mapped = stmt.query_map([], |row| {
        Ok(KeywordManagementRow {
            label: row.get(0)?,
            path: row.get(1)?,
            origin: row.get(2)?,
            visible_count: row.get(3)?,
            hidden_count: row.get(4)?,
            collection_count: row.get(5)?,
            total_count: row.get(6)?,
        })
    });

    match mapped {
        Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
        Err(e) => {
            eprintln!("keyword_management_rows: query {}", e);
            Vec::new()
        }
    }
}

/// Settings → Keywords management read. Unlike keyword_vocabulary, this reads
/// the raw keyword table so orphaned hidden rows are visible for cleanup.
pub async fn keyword_management_rows(
    origin: String,
    include_collections: bool,
    include_orphaned: bool,
) -> Vec<KeywordManagementRow> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    keyword_management_rows_impl(conn, origin.as_str(), include_collections, include_orphaned)
}

fn delete_keyword_paths_impl(conn: &Connection, paths: &[String]) -> u64 {
    if paths.is_empty() {
        return 0;
    }

    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;") {
        eprintln!("delete_keyword_paths: begin failed: {}", e);
        return 0;
    }

    let mut deleted: u64 = 0;
    for path in paths {
        let trimmed = path.trim();
        if trimmed.is_empty() {
            continue;
        }
        let prefix = format!("{}{}", trimmed, KEYWORD_PATH_SEPARATOR);
        match conn.execute(
            "DELETE FROM keyword WHERE path = ? OR starts_with(path, ?)",
            params![trimmed, prefix],
        ) {
            Ok(changed) => deleted += changed as u64,
            Err(e) => {
                eprintln!("delete_keyword_paths: delete failed: {}", e);
                let _ = conn.execute_batch("ROLLBACK;");
                return 0;
            }
        }
    }

    if let Err(e) = conn.execute_batch("COMMIT;") {
        eprintln!("delete_keyword_paths: commit failed: {}", e);
        return 0;
    }
    deleted
}

/// Physically purge keyword rows for the selected path(s), including descendants.
/// This is intentionally separate from remove_keyword_for_ids, which is the
/// normal user-facing soft-hide operation.
pub async fn delete_keyword_paths(paths: Vec<String>) -> u64 {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    delete_keyword_paths_impl(conn, &paths)
}

/// Distinct VISIBLE keyword labels (case-sensitive, alphabetical) — the source for
/// the Collection "Add" dialog's autofill + dropdown. Deliberately ALL labels,
/// regardless of the `collection` flag, so any keyword can seed a collection and a
/// dead collection's name still suggests itself. (The Collection TAB picker filters
/// to `collection = TRUE` instead — a separate read, added with the tab.)
pub async fn keyword_labels() -> Vec<String> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    let mut stmt = match conn.prepare("SELECT DISTINCT label FROM keyword_visible ORDER BY label") {
        Ok(s) => s,
        Err(e) => {
            eprintln!("keyword_labels: prepare {}", e);
            return Vec::new();
        }
    };

    let mapped = stmt.query_map([], |row| {
        let label: String = row.get(0)?;
        Ok(label)
    });

    match mapped {
        Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
        Err(e) => {
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
pub struct SavedQueryInfo {
    pub id: i64,
    pub name: String,
}

/// A loaded saved query: the same two arrays `query_images` consumes — load,
/// hand to the sheet, run. Parity by construction with the live builder.
#[derive(Debug, Clone, PartialEq)]
pub struct SavedQueryPayload {
    pub predicates: Vec<QueryPredicate>,
    pub connectors: Vec<Connector>,
}

/// Storage text for a connector (saved_query_criterion.connector).
fn connector_to_text(c: &Connector) -> &'static str {
    match c {
        Connector::And => "and",
        Connector::Or => "or",
        Connector::Xor => "xor",
    }
}

/// Inverse of `connector_to_text`. Unknown/garbled → AND (the builder's
/// default — same forgiveness as `build_filter_predicate`'s short-array rule).
fn connector_from_text(s: &str) -> Connector {
    match s {
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
fn save_query_impl(
    conn: &Connection,
    name: &str,
    predicates: &[QueryPredicate],
    connectors: &[Connector],
) -> Option<SavedQueryInfo> {
    let base = name.trim();
    if base.is_empty() || predicates.is_empty() {
        return None;
    }

    let exists = |n: &str| -> bool {
        conn.query_row(
            "SELECT COUNT(*) FROM saved_query WHERE name = ?",
            [n],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0)
            > 0
    };

    let mut final_name = base.to_string();
    if exists(&final_name) {
        let mut n = 1;
        loop {
            if n > 999 {
                eprintln!("save_query: suffix space exhausted for '{}'", base);
                return None;
            }
            let candidate = format!("{}-{:02}", base, n);
            if !exists(&candidate) {
                final_name = candidate;
                break;
            }
            n += 1;
        }
    }

    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;") {
        eprintln!("save_query: begin failed: {}", e);
        return None;
    }

    if let Err(e) = conn.execute("INSERT INTO saved_query (name) VALUES (?)", [&final_name]) {
        eprintln!("save_query: insert header failed: {}", e);
        let _ = conn.execute_batch("ROLLBACK;");
        return None;
    }

    // The name is unique by construction (checked above, under the lock), so
    // it safely keys the id read-back.
    let id: i64 = match conn.query_row(
        "SELECT id FROM saved_query WHERE name = ?",
        [&final_name],
        |row| row.get(0),
    ) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("save_query: id read-back failed: {}", e);
            let _ = conn.execute_batch("ROLLBACK;");
            return None;
        }
    };

    for (i, p) in predicates.iter().enumerate() {
        // Row 1 has nothing to its left; row i (1-based) joins via connectors[i-2]
        // (Swift sends predicates.len()-1 connectors; default AND if short —
        // the same rule build_filter_predicate applies at query time).
        let connector_text: Option<&str> = if i == 0 {
            None
        } else {
            Some(connector_to_text(
                connectors.get(i - 1).unwrap_or(&Connector::And),
            ))
        };

        if let Err(e) = conn.execute(
            "INSERT INTO saved_query_criterion \
             (query_id, position, connector, kind, op, value, day, day_end, stars, num, num_end) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                id,
                (i + 1) as i64,
                connector_text,
                p.kind,
                p.op,
                p.value,
                p.day,
                p.day_end,
                p.stars.map(|s| s as i32),
                p.num,
                p.num_end
            ],
        ) {
            eprintln!("save_query: insert criterion failed: {}", e);
            let _ = conn.execute_batch("ROLLBACK;");
            return None;
        }
    }

    if let Err(e) = conn.execute_batch("COMMIT;") {
        eprintln!("save_query: commit failed: {}", e);
        let _ = conn.execute_batch("ROLLBACK;");
        return None;
    }

    Some(SavedQueryInfo {
        id,
        name: final_name,
    })
}

/// All saved queries, ordered by name (case-folded) for the picker list.
fn list_saved_queries_impl(conn: &Connection) -> Vec<SavedQueryInfo> {
    let mut stmt = match conn.prepare("SELECT id, name FROM saved_query ORDER BY LOWER(name), id") {
        Ok(s) => s,
        Err(e) => {
            eprintln!("list_saved_queries: prepare {}", e);
            return Vec::new();
        }
    };

    let mapped = stmt.query_map([], |row| {
        Ok(SavedQueryInfo {
            id: row.get(0)?,
            name: row.get(1)?,
        })
    });

    match mapped {
        Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
        Err(e) => {
            eprintln!("list_saved_queries: query {}", e);
            Vec::new()
        }
    }
}

/// Load a saved query's criterion rows back into the two arrays the builder
/// (and `query_images`) consume. None for an unknown id or an empty recipe.
fn load_saved_query_impl(conn: &Connection, id: i64) -> Option<SavedQueryPayload> {
    let mut stmt = match conn.prepare(
        "SELECT position, connector, kind, op, value, day, day_end, stars, num, num_end \
         FROM saved_query_criterion WHERE query_id = ? ORDER BY position",
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("load_saved_query: prepare {}", e);
            return None;
        }
    };

    let mapped = stmt.query_map([id], |row| {
        Ok((
            row.get::<_, i64>(0)?,            // position
            row.get::<_, Option<String>>(1)?, // connector
            row.get::<_, String>(2)?,         // kind
            row.get::<_, Option<String>>(3)?, // op
            row.get::<_, Option<String>>(4)?, // value
            row.get::<_, Option<String>>(5)?, // day
            row.get::<_, Option<String>>(6)?, // day_end
            row.get::<_, Option<i32>>(7)?,    // stars
            row.get::<_, Option<f64>>(8)?,    // num (S65)
            row.get::<_, Option<f64>>(9)?,    // num_end (S65)
        ))
    });

    let rows = match mapped {
        Ok(iter) => iter.filter_map(|r| r.ok()),
        Err(e) => {
            eprintln!("load_saved_query: query {}", e);
            return None;
        }
    };

    let mut predicates: Vec<QueryPredicate> = Vec::new();
    let mut connectors: Vec<Connector> = Vec::new();
    for (position, connector, kind, op, value, day, day_end, stars, num, num_end) in rows {
        if position > 1 {
            connectors.push(connector_from_text(connector.as_deref().unwrap_or("and")));
        }
        predicates.push(QueryPredicate {
            kind,
            day,
            day_end,
            op,
            stars: stars.map(|s| s as u8),
            value,
            num,
            num_end,
        });
    }

    if predicates.is_empty() {
        return None;
    }
    Some(SavedQueryPayload {
        predicates,
        connectors,
    })
}

/// Delete a saved query (header + criterion rows, one transaction). Returns
/// whether a header row was actually removed.
fn delete_saved_query_impl(conn: &Connection, id: i64) -> bool {
    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;") {
        eprintln!("delete_saved_query: begin failed: {}", e);
        return false;
    }

    if let Err(e) = conn.execute("DELETE FROM saved_query_criterion WHERE query_id = ?", [id]) {
        eprintln!("delete_saved_query: criteria delete failed: {}", e);
        let _ = conn.execute_batch("ROLLBACK;");
        return false;
    }

    let removed = match conn.execute("DELETE FROM saved_query WHERE id = ?", [id]) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("delete_saved_query: header delete failed: {}", e);
            let _ = conn.execute_batch("ROLLBACK;");
            return false;
        }
    };

    if let Err(e) = conn.execute_batch("COMMIT;") {
        eprintln!("delete_saved_query: commit failed: {}", e);
        return false;
    }

    removed > 0
}

/// FFI: save the current Find in Gallery sentence under `name` (suffixing a
/// colliding name per the S63 policy). Returns the header with the FINAL name.
pub async fn save_query(
    name: String,
    predicates: Vec<QueryPredicate>,
    connectors: Vec<Connector>,
) -> Option<SavedQueryInfo> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return None;
        }
    };
    save_query_impl(conn, &name, &predicates, &connectors)
}

/// FFI: all saved queries (id + name), name-ordered, for the picker.
pub async fn list_saved_queries() -> Vec<SavedQueryInfo> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };
    list_saved_queries_impl(conn)
}

/// FFI: load a saved query back into builder arrays. None if id unknown.
pub async fn load_saved_query(id: i64) -> Option<SavedQueryPayload> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return None;
        }
    };
    load_saved_query_impl(conn, id)
}

/// FFI: delete a saved query. True if it existed.
pub async fn delete_saved_query(id: i64) -> bool {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return false;
        }
    };
    delete_saved_query_impl(conn, id)
}

/// Distinct stored values of ONE allow-listed `images` column — the metadata
/// subjects' type-ahead source (Session 63; the dropdown doctrine: a query can
/// only ask for values that exist). The field token is matched against the
/// fixed allow-list and mapped to its column identifier HERE — caller strings
/// are never interpolated into SQL. Unknown token → empty.
pub async fn distinct_image_values(field: String) -> Vec<String> {
    let column = match field.as_str() {
        "file_extension" => "file_extension",
        "image_kind" => "image_kind",
        "camera_make" => "camera_make",
        "camera_model" => "camera_model",
        "lens_model" => "lens_model",
        "creator" => "creator",
        "video_codec" => "video_codec", // S75 — Codec subject dropdown
        other => {
            eprintln!("distinct_image_values: unknown field '{}'", other);
            return Vec::new();
        }
    };

    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    let sql = format!(
        "SELECT DISTINCT {col} FROM images WHERE {col} IS NOT NULL AND {col} <> '' ORDER BY LOWER({col})",
        col = column
    );
    let mut stmt = match conn.prepare(&sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("distinct_image_values: prepare {}", e);
            return Vec::new();
        }
    };

    let mapped = stmt.query_map([], |row| row.get::<_, String>(0));

    match mapped {
        Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
        Err(e) => {
            eprintln!("distinct_image_values: query {}", e);
            Vec::new()
        }
    }
}

/// `distinct_image_values`' numeric twin (S65) — the four exposure subjects'
/// type-ahead source, returned as the exact stored doubles (Swift formats the
/// photographer notation — f/2.8, 1/2000, 70 mm — and hands the picked double
/// straight back into the predicate, so "is exactly" equality never rides a
/// string round-trip). ISO is an INTEGER column → CAST keeps one return type.
/// Ascending numeric order. Unknown token → empty.
pub async fn distinct_numeric_values(field: String) -> Vec<f64> {
    let column = match field.as_str() {
        "iso" => "iso",
        "aperture" => "aperture",
        "shutter_speed" => "shutter_speed",
        "focal_length" => "focal_length",
        "focus_score" => "focus_score", // S76 — Focus Quality subject dropdown
        "face_count" => "face_count",   // Vision enrichment — People Present subject
        "face_quality_best" => "face_quality_best",
        "face_quality_average" => "face_quality_average",
        "face_quality_min" => "face_quality_min",
        "face_eyes_open_count" => "face_eyes_open_count",
        "face_blink_risk_count" => "face_blink_risk_count",
        "duration_seconds" => "duration_seconds", // S75 — Duration subject dropdown
        "frame_rate" => "frame_rate",             // S75 — Frame rate subject dropdown
        other => {
            eprintln!("distinct_numeric_values: unknown field '{}'", other);
            return Vec::new();
        }
    };

    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    let sql = format!(
        "SELECT DISTINCT CAST({col} AS DOUBLE) AS v FROM images WHERE {col} IS NOT NULL ORDER BY v",
        col = column
    );
    let mut stmt = match conn.prepare(&sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("distinct_numeric_values: prepare {}", e);
            return Vec::new();
        }
    };

    let mapped = stmt.query_map([], |row| row.get::<_, f64>(0));

    match mapped {
        Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
        Err(e) => {
            eprintln!("distinct_numeric_values: query {}", e);
            Vec::new()
        }
    }
}

fn is_valid_analysis_job_status(status: &str) -> bool {
    matches!(
        status,
        "queued" | "running" | "cancelling" | "cancelled" | "completed" | "failed"
    )
}

fn is_terminal_analysis_job_status(status: &str) -> bool {
    matches!(status, "cancelled" | "completed" | "failed")
}

fn analysis_job_token_is_valid(token: &str) -> bool {
    let trimmed = token.trim();
    !trimmed.is_empty()
        && trimmed.len() <= 96
        && trimmed
            .chars()
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
}

fn u64_to_i64_clamped(value: u64) -> i64 {
    value.min(i64::MAX as u64) as i64
}

fn i64_to_u64_floor(value: i64) -> u64 {
    value.max(0) as u64
}

fn row_to_analysis_job(row: &duckdb::Row) -> Result<AnalysisJob, duckdb::Error> {
    Ok(AnalysisJob {
        id: row.get(0)?,
        job_kind: row.get(1)?,
        scope_kind: row.get(2)?,
        scope_value: row.get(3)?,
        algorithm_version: row.get(4)?,
        analysis_run_id: row.get(5)?,
        status: row.get(6)?,
        total_candidate_count: i64_to_u64_floor(row.get(7)?),
        processed_count: i64_to_u64_floor(row.get(8)?),
        completed_count: i64_to_u64_floor(row.get(9)?),
        skipped_count: i64_to_u64_floor(row.get(10)?),
        failed_count: i64_to_u64_floor(row.get(11)?),
        updated_count: i64_to_u64_floor(row.get(12)?),
        cancel_requested: row.get(13)?,
        created_at: row.get(14)?,
        started_at: row.get(15)?,
        updated_at: row.get(16)?,
        finished_at: row.get(17)?,
        last_error: row.get(18)?,
        current_image_id: row.get(19)?,
        current_file_path: row.get(20)?,
        current_started_at: row.get(21)?,
        last_timeout_image_id: row.get(22)?,
        last_timeout_file_path: row.get(23)?,
        last_timeout_at: row.get(24)?,
    })
}

const ANALYSIS_JOB_SELECT_COLUMNS: &str = "\
    id, job_kind, scope_kind, scope_value, algorithm_version, analysis_run_id, status, \
    total_candidate_count, processed_count, completed_count, skipped_count, failed_count, \
    updated_count, cancel_requested, CAST(created_at AS VARCHAR), CAST(started_at AS VARCHAR), \
    CAST(updated_at AS VARCHAR), CAST(finished_at AS VARCHAR), last_error, \
    current_image_id, current_file_path, CAST(current_started_at AS VARCHAR), \
    last_timeout_image_id, last_timeout_file_path, CAST(last_timeout_at AS VARCHAR)";

fn analysis_job_by_id_impl(conn: &Connection, id: i64) -> Option<AnalysisJob> {
    let sql = format!(
        "SELECT {} FROM analysis_jobs WHERE id = ?1",
        ANALYSIS_JOB_SELECT_COLUMNS
    );
    conn.query_row(&sql, params![id], row_to_analysis_job).ok()
}

fn create_analysis_job_impl(
    conn: &Connection,
    job_kind: &str,
    scope_kind: &str,
    scope_value: Option<String>,
    algorithm_version: &str,
    analysis_run_id: &str,
    total_candidate_count: u64,
) -> Option<AnalysisJob> {
    let job_kind = job_kind.trim();
    let scope_kind = scope_kind.trim();
    let algorithm_version = algorithm_version.trim();
    let analysis_run_id = analysis_run_id.trim();

    if !analysis_job_token_is_valid(job_kind)
        || !analysis_job_token_is_valid(scope_kind)
        || algorithm_version.is_empty()
        || analysis_run_id.is_empty()
    {
        eprintln!("create_analysis_job: invalid job metadata");
        return None;
    }

    let total = u64_to_i64_clamped(total_candidate_count);
    let inserted = conn.execute(
        "INSERT INTO analysis_jobs (
             job_kind, scope_kind, scope_value, algorithm_version, analysis_run_id, status,
             total_candidate_count, processed_count, completed_count, skipped_count,
             failed_count, updated_count, cancel_requested
         )
         VALUES (?1, ?2, ?3, ?4, ?5, 'queued', ?6, 0, 0, 0, 0, 0, FALSE)",
        params![
            job_kind,
            scope_kind,
            scope_value,
            algorithm_version,
            analysis_run_id,
            total
        ],
    );

    if let Err(e) = inserted {
        eprintln!("create_analysis_job: insert failed: {}", e);
        return None;
    }

    let id = match conn.query_row(
        "SELECT id FROM analysis_jobs WHERE analysis_run_id = ?1",
        params![analysis_run_id],
        |row| row.get::<_, i64>(0),
    ) {
        Ok(id) => id,
        Err(e) => {
            eprintln!("create_analysis_job: id lookup failed: {}", e);
            return None;
        }
    };

    analysis_job_by_id_impl(conn, id)
}

fn active_analysis_job_impl(conn: &Connection, job_kind: &str) -> Option<AnalysisJob> {
    let job_kind = job_kind.trim();
    if !analysis_job_token_is_valid(job_kind) {
        return None;
    }

    let sql = format!(
        "SELECT {}
         FROM analysis_jobs
         WHERE job_kind = ?1 AND status IN ('queued', 'running', 'cancelling')
         ORDER BY id DESC
         LIMIT 1",
        ANALYSIS_JOB_SELECT_COLUMNS
    );
    conn.query_row(&sql, params![job_kind], row_to_analysis_job)
        .ok()
}

fn active_analysis_jobs_impl(conn: &Connection) -> Vec<AnalysisJob> {
    let sql = format!(
        "SELECT {}
         FROM analysis_jobs
         WHERE status IN ('queued', 'running', 'cancelling')
         ORDER BY id DESC",
        ANALYSIS_JOB_SELECT_COLUMNS
    );
    let mut stmt = match conn.prepare(&sql) {
        Ok(stmt) => stmt,
        Err(e) => {
            eprintln!("active_analysis_jobs: prepare failed: {}", e);
            return Vec::new();
        }
    };

    let mapped = stmt.query_map([], row_to_analysis_job);
    match mapped {
        Ok(iter) => iter.filter_map(|job| job.ok()).collect(),
        Err(e) => {
            eprintln!("active_analysis_jobs: query failed: {}", e);
            Vec::new()
        }
    }
}

fn update_analysis_job_progress_impl(
    conn: &Connection,
    id: i64,
    processed_delta: u64,
    completed_delta: u64,
    skipped_delta: u64,
    failed_delta: u64,
    updated_delta: u64,
    total_candidate_count: Option<u64>,
) -> Option<AnalysisJob> {
    let processed_delta = u64_to_i64_clamped(processed_delta);
    let completed_delta = u64_to_i64_clamped(completed_delta);
    let skipped_delta = u64_to_i64_clamped(skipped_delta);
    let failed_delta = u64_to_i64_clamped(failed_delta);
    let updated_delta = u64_to_i64_clamped(updated_delta);
    let total_candidate_count = total_candidate_count.map(u64_to_i64_clamped);

    let updated = conn.execute(
        "UPDATE analysis_jobs
         SET status = CASE WHEN status = 'queued' THEN 'running' ELSE status END,
             started_at = COALESCE(started_at, CURRENT_TIMESTAMP),
             updated_at = CURRENT_TIMESTAMP,
             total_candidate_count = COALESCE(?7, total_candidate_count),
             processed_count = processed_count + ?2,
             completed_count = completed_count + ?3,
             skipped_count = skipped_count + ?4,
             failed_count = failed_count + ?5,
             updated_count = updated_count + ?6
         WHERE id = ?1 AND status IN ('queued', 'running', 'cancelling')",
        params![
            id,
            processed_delta,
            completed_delta,
            skipped_delta,
            failed_delta,
            updated_delta,
            total_candidate_count
        ],
    );

    match updated {
        Ok(0) => None,
        Ok(_) => analysis_job_by_id_impl(conn, id),
        Err(e) => {
            eprintln!("update_analysis_job_progress: update failed: {}", e);
            None
        }
    }
}

fn update_analysis_job_breadcrumb_impl(
    conn: &Connection,
    id: i64,
    current_image_id: Option<i64>,
    current_file_path: Option<String>,
    timed_out: bool,
) -> Option<AnalysisJob> {
    if current_image_id.is_some() != current_file_path.is_some() {
        eprintln!("update_analysis_job_breadcrumb: image id/path mismatch");
        return None;
    }

    let updated = conn.execute(
        "UPDATE analysis_jobs
         SET updated_at = CURRENT_TIMESTAMP,
             current_image_id = ?2,
             current_file_path = ?3,
             current_started_at = CASE WHEN ?2 IS NULL THEN NULL ELSE CURRENT_TIMESTAMP END,
             last_timeout_image_id = CASE WHEN ?4 THEN ?2 ELSE last_timeout_image_id END,
             last_timeout_file_path = CASE WHEN ?4 THEN ?3 ELSE last_timeout_file_path END,
             last_timeout_at = CASE WHEN ?4 THEN CURRENT_TIMESTAMP ELSE last_timeout_at END
         WHERE id = ?1 AND status IN ('queued', 'running', 'cancelling')",
        params![id, current_image_id, current_file_path, timed_out],
    );

    match updated {
        Ok(0) => None,
        Ok(_) => analysis_job_by_id_impl(conn, id),
        Err(e) => {
            eprintln!("update_analysis_job_breadcrumb: update failed: {}", e);
            None
        }
    }
}

fn request_cancel_analysis_job_impl(conn: &Connection, id: i64) -> bool {
    match conn.execute(
        "UPDATE analysis_jobs
         SET cancel_requested = TRUE,
             status = CASE WHEN status IN ('queued', 'running') THEN 'cancelling' ELSE status END,
             updated_at = CURRENT_TIMESTAMP
         WHERE id = ?1 AND status IN ('queued', 'running', 'cancelling')",
        params![id],
    ) {
        Ok(n) => n > 0,
        Err(e) => {
            eprintln!("request_cancel_analysis_job: update failed: {}", e);
            false
        }
    }
}

fn finish_analysis_job_impl(
    conn: &Connection,
    id: i64,
    status: &str,
    last_error: Option<String>,
) -> Option<AnalysisJob> {
    let status = status.trim();
    if !is_valid_analysis_job_status(status) || !is_terminal_analysis_job_status(status) {
        eprintln!("finish_analysis_job: invalid terminal status '{}'", status);
        return None;
    }

    let updated = conn.execute(
        "UPDATE analysis_jobs
         SET status = ?2,
             cancel_requested = CASE WHEN ?2 = 'cancelled' THEN TRUE ELSE cancel_requested END,
             updated_at = CURRENT_TIMESTAMP,
             finished_at = CURRENT_TIMESTAMP,
             last_error = ?3,
             current_image_id = NULL,
             current_file_path = NULL,
             current_started_at = NULL
         WHERE id = ?1 AND status IN ('queued', 'running', 'cancelling')",
        params![id, status, last_error],
    );

    match updated {
        Ok(0) => None,
        Ok(_) => analysis_job_by_id_impl(conn, id),
        Err(e) => {
            eprintln!("finish_analysis_job: update failed: {}", e);
            None
        }
    }
}

fn recover_interrupted_analysis_jobs_impl(
    conn: &Connection,
    job_kind: &str,
    terminal_status: &str,
    last_error: Option<String>,
) -> u64 {
    let job_kind = job_kind.trim();
    let terminal_status = terminal_status.trim();
    if !analysis_job_token_is_valid(job_kind)
        || !is_valid_analysis_job_status(terminal_status)
        || !is_terminal_analysis_job_status(terminal_status)
    {
        eprintln!("recover_interrupted_analysis_jobs: invalid recovery metadata");
        return 0;
    }

    match conn.execute(
        "UPDATE analysis_jobs
         SET status = ?2,
             cancel_requested = CASE WHEN ?2 = 'cancelled' THEN TRUE ELSE cancel_requested END,
             updated_at = CURRENT_TIMESTAMP,
             finished_at = CURRENT_TIMESTAMP,
             last_error = ?3,
             current_image_id = NULL,
             current_file_path = NULL,
             current_started_at = NULL
         WHERE job_kind = ?1 AND status IN ('queued', 'running', 'cancelling')",
        params![job_kind, terminal_status, last_error],
    ) {
        Ok(n) => n as u64,
        Err(e) => {
            eprintln!("recover_interrupted_analysis_jobs: update failed: {}", e);
            0
        }
    }
}

/// Create a durable analysis/enrichment job. The first caller is foreground
/// focus analysis; the background helper will use the same row shape later.
pub async fn create_analysis_job(
    job_kind: String,
    scope_kind: String,
    scope_value: Option<String>,
    algorithm_version: String,
    analysis_run_id: String,
    total_candidate_count: u64,
) -> Option<AnalysisJob> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("create_analysis_job: catalogue not initialized");
            return None;
        }
    };

    create_analysis_job_impl(
        conn,
        &job_kind,
        &scope_kind,
        scope_value,
        &algorithm_version,
        &analysis_run_id,
        total_candidate_count,
    )
}

/// Return the newest non-terminal job for a kind, if any. Used by UI status
/// checks and, later, by the helper to avoid double-owning foreground work.
pub async fn active_analysis_job(job_kind: String) -> Option<AnalysisJob> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("active_analysis_job: catalogue not initialized");
            return None;
        }
    };

    active_analysis_job_impl(conn, &job_kind)
}

/// Return every non-terminal analysis job. This is the generic app-facing
/// status surface for background work; callers can filter by job kind for
/// command-specific actions.
pub async fn active_analysis_jobs() -> Vec<AnalysisJob> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("active_analysis_jobs: catalogue not initialized");
            return Vec::new();
        }
    };

    active_analysis_jobs_impl(conn)
}

/// Increment durable counters and heartbeat a job. Deltas are additive so the
/// caller can write after each batch without re-reading global state.
pub async fn update_analysis_job_progress(
    id: i64,
    processed_delta: u64,
    completed_delta: u64,
    skipped_delta: u64,
    failed_delta: u64,
    updated_delta: u64,
    total_candidate_count: Option<u64>,
) -> Option<AnalysisJob> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("update_analysis_job_progress: catalogue not initialized");
            return None;
        }
    };

    update_analysis_job_progress_impl(
        conn,
        id,
        processed_delta,
        completed_delta,
        skipped_delta,
        failed_delta,
        updated_delta,
        total_candidate_count,
    )
}

/// Heartbeat the per-image breadcrumb for the currently running enrichment item.
/// Passing nil id/path clears the current breadcrumb; `timed_out` also preserves
/// the timed-out item as the latest timeout diagnostic.
pub async fn update_analysis_job_breadcrumb(
    id: i64,
    current_image_id: Option<i64>,
    current_file_path: Option<String>,
    timed_out: bool,
) -> Option<AnalysisJob> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("update_analysis_job_breadcrumb: catalogue not initialized");
            return None;
        }
    };

    update_analysis_job_breadcrumb_impl(conn, id, current_image_id, current_file_path, timed_out)
}

/// Mark a running/queued job as cancellation-requested. Workers should poll
/// this row between chunks and finish as `cancelled` when teardown completes.
pub async fn request_cancel_analysis_job(id: i64) -> bool {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("request_cancel_analysis_job: catalogue not initialized");
            return false;
        }
    };

    request_cancel_analysis_job_impl(conn, id)
}

/// Close an analysis job with a terminal status.
pub async fn finish_analysis_job(
    id: i64,
    status: String,
    last_error: Option<String>,
) -> Option<AnalysisJob> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("finish_analysis_job: catalogue not initialized");
            return None;
        }
    };

    finish_analysis_job_impl(conn, id, &status, last_error)
}

/// Terminalize non-owned jobs left active after an app/helper process exits.
/// Foreground focus analysis currently recovers these as cancelled on launch;
/// future background work can use the same primitive with a stricter policy.
pub async fn recover_interrupted_analysis_jobs(
    job_kind: String,
    terminal_status: String,
    last_error: Option<String>,
) -> u64 {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("recover_interrupted_analysis_jobs: catalogue not initialized");
            return 0;
        }
    };

    recover_interrupted_analysis_jobs_impl(conn, &job_kind, &terminal_status, last_error)
}

fn is_valid_focus_status(status: &str) -> bool {
    matches!(status, "complete" | "online_only" | "unreadable" | "failed")
}

fn is_valid_focus_basis(basis: &str) -> bool {
    matches!(
        basis,
        "human_face"
            | "face"
            | "animal"
            | "foreground"
            | "subject"
            | "saliency"
            | "animal_pose"
            | "whole_image"
            | "unknown"
    )
}

fn focus_score_is_valid(score: Option<f64>) -> bool {
    score.map(|score| score.is_finite()).unwrap_or(true)
}

fn face_count_is_valid(count: Option<i32>) -> bool {
    count
        .map(|count| (0..=10_000).contains(&count))
        .unwrap_or(true)
}

fn face_quality_is_valid(score: Option<f64>) -> bool {
    score.map(|score| score.is_finite()).unwrap_or(true)
}

fn normalized_optional_score_is_valid(score: Option<f64>) -> bool {
    score
        .map(|score| score.is_finite() && (-0.01..=1.01).contains(&score))
        .unwrap_or(true)
}

fn optional_finite_is_valid(score: Option<f64>) -> bool {
    score.map(|score| score.is_finite()).unwrap_or(true)
}

fn landmark_pair_is_valid(x: Option<f64>, y: Option<f64>) -> bool {
    match (x, y) {
        (Some(x), Some(y)) => x.is_finite() && y.is_finite(),
        (None, None) => true,
        _ => false,
    }
}

fn face_bounding_box_is_valid(observation: &FaceObservationResult) -> bool {
    if !observation.bounding_box_x.is_finite()
        || !observation.bounding_box_y.is_finite()
        || !observation.bounding_box_width.is_finite()
        || !observation.bounding_box_height.is_finite()
        || observation.bounding_box_width <= 0.0
        || observation.bounding_box_height <= 0.0
    {
        return false;
    }

    let min_x = observation.bounding_box_x;
    let min_y = observation.bounding_box_y;
    let max_x = observation.bounding_box_x + observation.bounding_box_width;
    let max_y = observation.bounding_box_y + observation.bounding_box_height;

    max_x > 0.0 && max_y > 0.0 && min_x < 1.0 && min_y < 1.0
}

fn face_observation_is_valid(observation: &FaceObservationResult) -> bool {
    observation.face_index <= 10_000
        && face_bounding_box_is_valid(observation)
        && normalized_optional_score_is_valid(observation.detection_confidence)
        && normalized_optional_score_is_valid(observation.face_capture_quality)
        && optional_finite_is_valid(observation.face_focus_score)
        && optional_finite_is_valid(observation.left_eye_open_score)
        && optional_finite_is_valid(observation.right_eye_open_score)
        && optional_finite_is_valid(observation.eyes_open_score)
        && normalized_optional_score_is_valid(observation.blink_risk_score)
        && landmark_pair_is_valid(observation.left_eye_x, observation.left_eye_y)
        && landmark_pair_is_valid(observation.right_eye_x, observation.right_eye_y)
        && landmark_pair_is_valid(observation.nose_x, observation.nose_y)
        && landmark_pair_is_valid(observation.mouth_left_x, observation.mouth_left_y)
        && landmark_pair_is_valid(observation.mouth_right_x, observation.mouth_right_y)
}

fn sanitized_focus_result(mut result: FocusAnalysisResult) -> FocusAnalysisResult {
    let scores_are_valid = focus_score_is_valid(result.focus_score)
        && focus_score_is_valid(result.focus_human_score)
        && focus_score_is_valid(result.focus_animal_score)
        && focus_score_is_valid(result.focus_foreground_score)
        && focus_score_is_valid(result.focus_saliency_score)
        && focus_score_is_valid(result.focus_animal_pose_score)
        && focus_score_is_valid(result.focus_whole_image_score);
    let basis_is_valid = result
        .focus_basis
        .as_deref()
        .map(is_valid_focus_basis)
        .unwrap_or(true);
    let complete_has_score = result.status != "complete" || result.focus_score.is_some();

    if !is_valid_focus_status(&result.status)
        || !scores_are_valid
        || !basis_is_valid
        || !face_count_is_valid(result.face_count)
        || !face_quality_is_valid(result.face_quality_best)
        || !face_quality_is_valid(result.face_quality_average)
        || !face_quality_is_valid(result.face_quality_min)
        || !face_count_is_valid(result.face_eyes_open_count)
        || !face_count_is_valid(result.face_blink_risk_count)
        || !complete_has_score
    {
        eprintln!(
            "update_focus_analysis_results: quarantining invalid result for id {}",
            result.id
        );
        result.status = "failed".to_string();
        result.focus_score = None;
        result.focus_basis = Some("unknown".to_string());
        result.focus_human_score = None;
        result.focus_animal_score = None;
        result.focus_foreground_score = None;
        result.focus_saliency_score = None;
        result.focus_animal_pose_score = None;
        result.focus_whole_image_score = None;
        result.face_count = None;
        result.face_quality_best = None;
        result.face_quality_average = None;
        result.face_quality_min = None;
        result.face_eyes_open_count = None;
        result.face_blink_risk_count = None;
        result.auto_keywords.clear();
        result.face_observations.clear();
    } else {
        let original_observation_count = result.face_observations.len();
        result
            .face_observations
            .retain(|observation| face_observation_is_valid(observation));
        if result.face_observations.len() != original_observation_count {
            eprintln!(
                "update_focus_analysis_results: skipped {} invalid face observation(s) for id {}",
                original_observation_count - result.face_observations.len(),
                result.id
            );
        }
    }

    result
}

fn insert_active_keyword_for_image(conn: &Connection, image_id: i64, segments: &[String]) -> bool {
    match insert_or_merge_active_keyword_for_image(conn, image_id, segments, KEYWORD_ORIGIN_AUTO) {
        Ok(_) => true,
        Err(e) => {
            eprintln!("insert_active_keyword_for_image: insert failed: {}", e);
            false
        }
    }
}

fn focus_analysis_queue_where_clause(id_predicate: Option<&str>) -> String {
    let mut predicates = vec![
        "is_video IS NOT TRUE".to_string(),
        RAW_JPEG_COLLAPSE_PREDICATE.to_string(),
        "(
             focus_analysis_status IS NULL
             OR focus_algorithm_version IS NULL
             OR focus_algorithm_version <> ?1
             OR (
                 focus_analysis_status IN ('online_only', 'unreadable', 'failed')
                 AND (
                     focus_analysis_attempt_id IS NULL
                     OR focus_analysis_attempt_id <> ?2
                 )
             )
         )"
        .to_string(),
    ];
    if let Some(predicate) = id_predicate {
        predicates.push(format!("({})", predicate));
    }

    format!("WHERE {}", predicates.join(" AND "))
}

fn focus_analysis_scope_predicate(ids: &[i64]) -> Option<String> {
    if ids.is_empty() {
        return None;
    }

    let csv = ids
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    Some(format!(
        "(id IN ({csv}) \
          OR (image_kind IN ('raw', 'jpeg', 'heif') \
              AND EXISTS ( \
                  SELECT 1 FROM images selected \
                  WHERE selected.id IN ({csv}) \
                    AND selected.image_kind IN ('raw', 'jpeg', 'heif') \
                    AND selected.file_stem = images.file_stem \
                    AND selected.directory_path = images.directory_path \
              )))",
        csv = csv
    ))
}

fn focus_analysis_candidates_impl(
    conn: &Connection,
    limit: u32,
    algorithm_version: &str,
    analysis_run_id: &str,
    scoped_ids: Option<&[i64]>,
) -> Vec<FocusAnalysisCandidate> {
    let scope_filter = match scoped_ids {
        Some(ids) => match focus_analysis_scope_predicate(ids) {
            Some(filter) => Some(filter),
            None => return Vec::new(),
        },
        None => None,
    };
    let where_clause = focus_analysis_queue_where_clause(scope_filter.as_deref());
    let capped_limit = limit.clamp(1, 5000) as i64;
    let sql = format!(
        "SELECT id, file_path, file_size
         FROM images
         {}
         ORDER BY id
         LIMIT ?3",
        where_clause
    );
    let mut stmt = match conn.prepare(&sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("focus_analysis_candidates: prepare {}", e);
            return Vec::new();
        }
    };

    let mapped = stmt.query_map(
        params![algorithm_version, analysis_run_id, capped_limit],
        |row| {
            Ok(FocusAnalysisCandidate {
                id: row.get(0)?,
                file_path: row.get(1)?,
                file_size: row.get::<_, i64>(2)? as u64,
            })
        },
    );

    match mapped {
        Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
        Err(e) => {
            eprintln!("focus_analysis_candidates: query {}", e);
            Vec::new()
        }
    }
}

fn focus_analysis_candidate_count_impl(
    conn: &Connection,
    algorithm_version: &str,
    analysis_run_id: &str,
    scoped_ids: Option<&[i64]>,
) -> u64 {
    let scope_filter = match scoped_ids {
        Some(ids) => match focus_analysis_scope_predicate(ids) {
            Some(filter) => Some(filter),
            None => return 0,
        },
        None => None,
    };
    let where_clause = focus_analysis_queue_where_clause(scope_filter.as_deref());
    let sql = format!("SELECT COUNT(*) FROM images {}", where_clause);

    match conn.query_row(&sql, params![algorithm_version, analysis_run_id], |row| {
        row.get::<_, i64>(0)
    }) {
        Ok(count) => count.max(0) as u64,
        Err(e) => {
            eprintln!("focus_analysis_candidate_count: query {}", e);
            0
        }
    }
}

/// Return a narrow page of still-image rows whose focus analysis is missing or
/// stale for the requested algorithm version. The NULL columns are the queue; no
/// separate enrichment table is persisted.
pub async fn focus_analysis_candidates(
    limit: u32,
    algorithm_version: String,
    analysis_run_id: String,
) -> Vec<FocusAnalysisCandidate> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    focus_analysis_candidates_impl(conn, limit, &algorithm_version, &analysis_run_id, None)
}

/// Return focus-analysis candidates intersected with an explicit image-id
/// selection. Empty selection means empty queue.
pub async fn focus_analysis_candidates_for_ids(
    ids: Vec<i64>,
    limit: u32,
    algorithm_version: String,
    analysis_run_id: String,
) -> Vec<FocusAnalysisCandidate> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    focus_analysis_candidates_impl(
        conn,
        limit,
        &algorithm_version,
        &analysis_run_id,
        Some(&ids),
    )
}

/// Count still-image rows whose focus analysis is missing or stale for the
/// requested algorithm version. This lets the Swift status bar show a
/// determinate progress bar while analysis runs.
pub async fn focus_analysis_candidate_count(
    algorithm_version: String,
    analysis_run_id: String,
) -> u64 {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    focus_analysis_candidate_count_impl(conn, &algorithm_version, &analysis_run_id, None)
}

/// Count focus-analysis candidates intersected with an explicit image-id
/// selection. Empty selection means zero candidates.
pub async fn focus_analysis_candidate_count_for_ids(
    ids: Vec<i64>,
    algorithm_version: String,
    analysis_run_id: String,
) -> u64 {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    focus_analysis_candidate_count_impl(conn, &algorithm_version, &analysis_run_id, Some(&ids))
}

fn focus_analysis_writeback_target_ids(conn: &Connection, image_id: i64) -> Vec<i64> {
    let mut stmt = match conn.prepare(
        "SELECT target.id
         FROM images source
         JOIN images target
           ON target.id = source.id
           OR (
               source.image_kind IN ('raw', 'jpeg', 'heif')
               AND target.image_kind IN ('raw', 'jpeg', 'heif')
               AND target.file_stem = source.file_stem
               AND target.directory_path = source.directory_path
           )
         WHERE source.id = ?1
         ORDER BY target.id",
    ) {
        Ok(stmt) => stmt,
        Err(e) => {
            eprintln!("focus_analysis_writeback_target_ids: prepare {}", e);
            return vec![image_id];
        }
    };

    let rows = match stmt.query_map(params![image_id], |row| row.get::<_, i64>(0)) {
        Ok(rows) => rows,
        Err(e) => {
            eprintln!("focus_analysis_writeback_target_ids: query {}", e);
            return vec![image_id];
        }
    };

    let ids = rows.filter_map(|row| row.ok()).collect::<Vec<_>>();
    if ids.is_empty() {
        vec![image_id]
    } else {
        ids
    }
}

fn replace_face_observations_for_targets(
    conn: &Connection,
    result: &FocusAnalysisResult,
    target_ids: &[i64],
    is_complete: bool,
) -> bool {
    for target_id in target_ids {
        if let Err(e) = conn.execute(
            "DELETE FROM face_observation
             WHERE image_id = ?1
               AND algorithm_version = ?2",
            params![target_id, &result.algorithm_version],
        ) {
            eprintln!(
                "replace_face_observations_for_targets: delete image_id={} {}",
                target_id, e
            );
            return false;
        }
    }

    if !is_complete {
        return true;
    }

    for target_id in target_ids {
        for observation in &result.face_observations {
            if let Err(e) = conn.execute(
                "INSERT INTO face_observation (
                     image_id,
                     analyzed_image_id,
                     face_index,
                     algorithm_version,
                     analysis_run_id,
                     bounding_box_x,
                     bounding_box_y,
                     bounding_box_width,
                     bounding_box_height,
                     detection_confidence,
                     face_capture_quality,
                     face_focus_score,
                     left_eye_open_score,
                     right_eye_open_score,
                     eyes_open_score,
                     blink_risk_score,
                     left_eye_x,
                     left_eye_y,
                     right_eye_x,
                     right_eye_y,
                     nose_x,
                     nose_y,
                     mouth_left_x,
                     mouth_left_y,
                     mouth_right_x,
                     mouth_right_y
                 )
                 VALUES (
                     ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13,
                     ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24,
                     ?25, ?26
                 )",
                params![
                    target_id,
                    result.id,
                    observation.face_index as i64,
                    &result.algorithm_version,
                    &result.analysis_run_id,
                    observation.bounding_box_x,
                    observation.bounding_box_y,
                    observation.bounding_box_width,
                    observation.bounding_box_height,
                    observation.detection_confidence,
                    observation.face_capture_quality,
                    observation.face_focus_score,
                    observation.left_eye_open_score,
                    observation.right_eye_open_score,
                    observation.eyes_open_score,
                    observation.blink_risk_score,
                    observation.left_eye_x,
                    observation.left_eye_y,
                    observation.right_eye_x,
                    observation.right_eye_y,
                    observation.nose_x,
                    observation.nose_y,
                    observation.mouth_left_x,
                    observation.mouth_left_y,
                    observation.mouth_right_x,
                    observation.mouth_right_y,
                ],
            ) {
                eprintln!(
                    "replace_face_observations_for_targets: insert image_id={} face_index={} {}",
                    target_id, observation.face_index, e
                );
                return false;
            }
        }
    }

    true
}

/// Batch writeback for focus-analysis results. Returns the number of rows updated.
pub async fn update_focus_analysis_results(results: Vec<FocusAnalysisResult>) -> u64 {
    if results.is_empty() {
        return 0;
    }

    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;") {
        eprintln!("update_focus_analysis_results: begin {}", e);
        return 0;
    }

    let mut updated = 0u64;
    for result in results {
        let result = sanitized_focus_result(result);
        let is_complete = result.status == "complete";
        let target_ids = if is_complete {
            focus_analysis_writeback_target_ids(conn, result.id)
        } else {
            vec![result.id]
        };

        let score = if is_complete {
            result.focus_score
        } else {
            None
        };
        let human_score = if is_complete {
            result.focus_human_score
        } else {
            None
        };
        let animal_score = if is_complete {
            result.focus_animal_score
        } else {
            None
        };
        let foreground_score = if is_complete {
            result.focus_foreground_score
        } else {
            None
        };
        let saliency_score = if is_complete {
            result.focus_saliency_score
        } else {
            None
        };
        let animal_pose_score = if is_complete {
            result.focus_animal_pose_score
        } else {
            None
        };
        let whole_image_score = if is_complete {
            result.focus_whole_image_score
        } else {
            None
        };
        let face_count = if is_complete { result.face_count } else { None };
        let face_quality_best = if is_complete {
            result.face_quality_best
        } else {
            None
        };
        let face_quality_average = if is_complete {
            result.face_quality_average
        } else {
            None
        };
        let face_quality_min = if is_complete {
            result.face_quality_min
        } else {
            None
        };
        let face_eyes_open_count = if is_complete {
            result.face_eyes_open_count
        } else {
            None
        };
        let face_blink_risk_count = if is_complete {
            result.face_blink_risk_count
        } else {
            None
        };
        let basis = result
            .focus_basis
            .clone()
            .unwrap_or_else(|| "unknown".to_string());

        match conn.execute(
            "UPDATE images
             SET focus_score = ?2,
                 focus_basis = ?3,
                 focus_human_score = ?4,
                 focus_animal_score = ?5,
                 focus_foreground_score = ?6,
                 focus_saliency_score = ?7,
                 focus_animal_pose_score = ?8,
                 focus_whole_image_score = ?9,
                 focus_algorithm_version = ?10,
                 focus_analysis_status = ?11,
                 focus_analysis_attempt_id = ?12,
                 focus_scored_at = CURRENT_TIMESTAMP,
                 face_count = ?13,
                 face_quality_best = ?14,
                 face_quality_average = ?15,
                 face_quality_min = ?16,
                 face_eyes_open_count = ?17,
                 face_blink_risk_count = ?18
             WHERE id = ?1
                OR (
                    ?11 = 'complete'
                    AND EXISTS (
                        SELECT 1 FROM images source
                        WHERE source.id = ?1
                          AND source.image_kind IN ('raw', 'jpeg', 'heif')
                          AND images.image_kind IN ('raw', 'jpeg', 'heif')
                          AND source.file_stem = images.file_stem
                          AND source.directory_path = images.directory_path
                    )
                )",
            params![
                result.id,
                score,
                basis,
                human_score,
                animal_score,
                foreground_score,
                saliency_score,
                animal_pose_score,
                whole_image_score,
                result.algorithm_version,
                result.status,
                result.analysis_run_id,
                face_count,
                face_quality_best,
                face_quality_average,
                face_quality_min,
                face_eyes_open_count,
                face_blink_risk_count,
            ],
        ) {
            Ok(n) => {
                updated += n as u64;
                if n > 0
                    && !replace_face_observations_for_targets(
                        conn,
                        &result,
                        &target_ids,
                        is_complete,
                    )
                {
                    let _ = conn.execute_batch("ROLLBACK;");
                    return 0;
                }

                if n > 0 && is_complete {
                    let mut labels = result
                        .auto_keywords
                        .iter()
                        .map(|label| label.trim().to_string())
                        .filter(|label| !label.is_empty())
                        .collect::<Vec<_>>();
                    labels.sort_by_key(|label| label.to_lowercase());
                    labels.dedup_by(|a, b| a.eq_ignore_ascii_case(b));

                    for target_id in &target_ids {
                        for label in &labels {
                            if !insert_active_keyword_for_image(
                                conn,
                                *target_id,
                                std::slice::from_ref(label),
                            ) {
                                let _ = conn.execute_batch("ROLLBACK;");
                                return 0;
                            }
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!(
                    "update_focus_analysis_results: row {} failed: {}",
                    result.id, e
                );
            }
        }
    }

    if let Err(e) = conn.execute_batch("COMMIT;") {
        eprintln!("update_focus_analysis_results: commit {}", e);
        return 0;
    }

    updated
}

/// Count durable face observations for the requested focus/enrichment algorithm
/// version. This is intentionally scoped by version so old detection rows never
/// leak into the current recognition/crop pipeline.
pub async fn face_observation_count(algorithm_version: String) -> u64 {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    match conn.query_row(
        "SELECT COUNT(*)
         FROM face_observation
         WHERE algorithm_version = ?1",
        params![algorithm_version],
        |row| row.get::<_, i64>(0),
    ) {
        Ok(count) => count.max(0) as u64,
        Err(e) => {
            eprintln!("face_observation_count: query {}", e);
            0
        }
    }
}

/// Return durable face observations for explicit image ids and algorithm
/// version. Empty id lists intentionally return no rows.
pub async fn face_observations_for_image_ids(
    ids: Vec<i64>,
    algorithm_version: String,
) -> Vec<FaceObservationRecord> {
    let id_filter = match image_id_in_list(&ids) {
        Some(filter) => filter,
        None => return Vec::new(),
    };

    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    let sql = format!(
        "SELECT
             id,
             image_id,
             analyzed_image_id,
             face_index,
             algorithm_version,
             analysis_run_id,
             bounding_box_x,
             bounding_box_y,
             bounding_box_width,
             bounding_box_height,
             detection_confidence,
             face_capture_quality,
             face_focus_score,
             left_eye_open_score,
             right_eye_open_score,
             eyes_open_score,
             blink_risk_score,
             left_eye_x,
             left_eye_y,
             right_eye_x,
             right_eye_y,
             nose_x,
             nose_y,
             mouth_left_x,
             mouth_left_y,
             mouth_right_x,
             mouth_right_y,
             CAST(created_at AS VARCHAR)
         FROM face_observation
         WHERE algorithm_version = ?1
           AND {}
         ORDER BY image_id, face_index",
        id_filter
    );

    let mut stmt = match conn.prepare(&sql) {
        Ok(stmt) => stmt,
        Err(e) => {
            eprintln!("face_observations_for_image_ids: prepare {}", e);
            return Vec::new();
        }
    };

    let rows = match stmt.query_map(params![algorithm_version], |row| {
        let face_index = row.get::<_, i64>(3)?;
        Ok(FaceObservationRecord {
            id: row.get(0)?,
            image_id: row.get(1)?,
            analyzed_image_id: row.get(2)?,
            face_index: face_index.clamp(0, u32::MAX as i64) as u32,
            algorithm_version: row.get(4)?,
            analysis_run_id: row.get(5)?,
            bounding_box_x: row.get(6)?,
            bounding_box_y: row.get(7)?,
            bounding_box_width: row.get(8)?,
            bounding_box_height: row.get(9)?,
            detection_confidence: row.get(10)?,
            face_capture_quality: row.get(11)?,
            face_focus_score: row.get(12)?,
            left_eye_open_score: row.get(13)?,
            right_eye_open_score: row.get(14)?,
            eyes_open_score: row.get(15)?,
            blink_risk_score: row.get(16)?,
            left_eye_x: row.get(17)?,
            left_eye_y: row.get(18)?,
            right_eye_x: row.get(19)?,
            right_eye_y: row.get(20)?,
            nose_x: row.get(21)?,
            nose_y: row.get(22)?,
            mouth_left_x: row.get(23)?,
            mouth_left_y: row.get(24)?,
            mouth_right_x: row.get(25)?,
            mouth_right_y: row.get(26)?,
            created_at: row.get(27)?,
        })
    }) {
        Ok(rows) => rows,
        Err(e) => {
            eprintln!("face_observations_for_image_ids: query {}", e);
            return Vec::new();
        }
    };

    rows.filter_map(|row| row.ok()).collect()
}

pub async fn face_recognition_menu_states(
    ids: Vec<i64>,
    algorithm_version: String,
    model_version: String,
    preprocessing_version: String,
) -> Vec<FaceRecognitionMenuState> {
    let image_table_id_filter = match id_in_list(&ids) {
        Some(filter) => filter,
        None => return Vec::new(),
    };
    let face_image_id_filter = match image_id_in_list(&ids) {
        Some(filter) => filter,
        None => return Vec::new(),
    };

    let embedded_ids = FACE_EMBEDDING_RUNTIME
        .spawn(async move {
            stored_face_embeddings(&model_version, &preprocessing_version)
                .await
                .into_iter()
                .map(|record| record.face_observation_id)
                .collect::<std::collections::HashSet<_>>()
        })
        .await
        .unwrap_or_default();

    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    let mut states = std::collections::HashMap::<i64, FaceRecognitionMenuState>::new();
    let image_sql = format!(
        "SELECT id, focus_analysis_status
         FROM images
         WHERE {}",
        image_table_id_filter
    );
    let mut image_stmt = match conn.prepare(&image_sql) {
        Ok(stmt) => stmt,
        Err(e) => {
            eprintln!("face_recognition_menu_states: image prepare {}", e);
            return Vec::new();
        }
    };
    let image_rows = match image_stmt.query_map([], |row| {
        Ok(FaceRecognitionMenuState {
            image_id: row.get(0)?,
            analysis_status: row.get(1)?,
            face_observation_count: 0,
            indexed_face_count: 0,
        })
    }) {
        Ok(rows) => rows,
        Err(e) => {
            eprintln!("face_recognition_menu_states: image query {}", e);
            return Vec::new();
        }
    };
    for state in image_rows.filter_map(|row| row.ok()) {
        states.insert(state.image_id, state);
    }

    let face_sql = format!(
        "SELECT id, image_id
         FROM face_observation
         WHERE algorithm_version = ?1
           AND {}
         ORDER BY image_id, face_index",
        face_image_id_filter
    );
    let mut face_stmt = match conn.prepare(&face_sql) {
        Ok(stmt) => stmt,
        Err(e) => {
            eprintln!("face_recognition_menu_states: face prepare {}", e);
            return Vec::new();
        }
    };
    let face_rows = match face_stmt.query_map(params![algorithm_version], |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
    }) {
        Ok(rows) => rows,
        Err(e) => {
            eprintln!("face_recognition_menu_states: face query {}", e);
            return Vec::new();
        }
    };
    for (face_observation_id, image_id) in face_rows.filter_map(|row| row.ok()) {
        if let Some(state) = states.get_mut(&image_id) {
            state.face_observation_count = state.face_observation_count.saturating_add(1);
            if embedded_ids.contains(&face_observation_id) {
                state.indexed_face_count = state.indexed_face_count.saturating_add(1);
            }
        }
    }

    ids.into_iter()
        .filter_map(|id| states.get(&id).cloned())
        .collect()
}

/// Return face observations that do not yet have a LanceDB embedding for the
/// requested model/preprocessing pair. `limit == 0` means no limit.
pub async fn face_embedding_missing_observations(
    algorithm_version: String,
    model_version: String,
    preprocessing_version: String,
    limit: u32,
) -> Vec<FaceObservationRecord> {
    let embedded_ids = FACE_EMBEDDING_RUNTIME
        .spawn(async move {
            stored_face_embeddings(&model_version, &preprocessing_version)
                .await
                .into_iter()
                .map(|record| record.face_observation_id)
                .collect::<std::collections::HashSet<_>>()
        })
        .await
        .unwrap_or_default();

    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    let mut stmt = match conn.prepare(
        "SELECT
             id,
             image_id,
             analyzed_image_id,
             face_index,
             algorithm_version,
             analysis_run_id,
             bounding_box_x,
             bounding_box_y,
             bounding_box_width,
             bounding_box_height,
             detection_confidence,
             face_capture_quality,
             face_focus_score,
             left_eye_open_score,
             right_eye_open_score,
             eyes_open_score,
             blink_risk_score,
             left_eye_x,
             left_eye_y,
             right_eye_x,
             right_eye_y,
             nose_x,
             nose_y,
             mouth_left_x,
             mouth_left_y,
             mouth_right_x,
             mouth_right_y,
             CAST(created_at AS VARCHAR)
         FROM face_observation
         WHERE algorithm_version = ?1
         ORDER BY analyzed_image_id, image_id, face_index",
    ) {
        Ok(stmt) => stmt,
        Err(e) => {
            eprintln!("face_embedding_missing_observations: prepare {}", e);
            return Vec::new();
        }
    };

    let rows = match stmt.query_map(params![algorithm_version], |row| {
        let face_index = row.get::<_, i64>(3)?;
        Ok(FaceObservationRecord {
            id: row.get(0)?,
            image_id: row.get(1)?,
            analyzed_image_id: row.get(2)?,
            face_index: face_index.clamp(0, u32::MAX as i64) as u32,
            algorithm_version: row.get(4)?,
            analysis_run_id: row.get(5)?,
            bounding_box_x: row.get(6)?,
            bounding_box_y: row.get(7)?,
            bounding_box_width: row.get(8)?,
            bounding_box_height: row.get(9)?,
            detection_confidence: row.get(10)?,
            face_capture_quality: row.get(11)?,
            face_focus_score: row.get(12)?,
            left_eye_open_score: row.get(13)?,
            right_eye_open_score: row.get(14)?,
            eyes_open_score: row.get(15)?,
            blink_risk_score: row.get(16)?,
            left_eye_x: row.get(17)?,
            left_eye_y: row.get(18)?,
            right_eye_x: row.get(19)?,
            right_eye_y: row.get(20)?,
            nose_x: row.get(21)?,
            nose_y: row.get(22)?,
            mouth_left_x: row.get(23)?,
            mouth_left_y: row.get(24)?,
            mouth_right_x: row.get(25)?,
            mouth_right_y: row.get(26)?,
            created_at: row.get(27)?,
        })
    }) {
        Ok(rows) => rows,
        Err(e) => {
            eprintln!("face_embedding_missing_observations: query {}", e);
            return Vec::new();
        }
    };

    let max_count = if limit == 0 {
        usize::MAX
    } else {
        limit as usize
    };

    rows.filter_map(|row| row.ok())
        .filter(|record| !embedded_ids.contains(&record.id))
        .take(max_count)
        .collect()
}

const FACE_EMBEDDING_TABLE: &str = "face_embeddings";

static FACE_EMBEDDING_RUNTIME: once_cell::sync::Lazy<tokio::runtime::Runtime> =
    once_cell::sync::Lazy::new(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("photolibrarian-face-embeddings")
            .build()
            .expect("create face embedding Tokio runtime")
    });

fn face_embedding_schema(dimension: u32) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("face_observation_id", DataType::Int64, false),
        Field::new("image_id", DataType::Int64, false),
        Field::new("analyzed_image_id", DataType::Int64, false),
        Field::new("face_index", DataType::UInt32, false),
        Field::new("model_name", DataType::Utf8, false),
        Field::new("model_version", DataType::Utf8, false),
        Field::new("preprocessing_version", DataType::Utf8, false),
        Field::new("input_size", DataType::UInt32, false),
        Field::new("color_order", DataType::Utf8, false),
        Field::new("normalization", DataType::Utf8, false),
        Field::new("embedding_dimension", DataType::UInt32, false),
        Field::new("embedding_l2_norm", DataType::Float64, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dimension as i32,
            ),
            false,
        ),
    ]))
}

fn face_embedding_store_uri() -> Option<String> {
    let catalogue_path = CATALOGUE_PATH.lock().unwrap();
    let path = catalogue_path.as_ref()?;
    let parent = path.parent()?;
    Some(parent.join("vectors.lancedb").to_string_lossy().to_string())
}

async fn open_or_create_face_embedding_table(
    dimension: u32,
) -> Result<lancedb::Table, lancedb::Error> {
    let uri = face_embedding_store_uri().ok_or_else(|| lancedb::Error::InvalidInput {
        message: "catalogue path is not initialized".to_string(),
    })?;

    let db = lancedb::connect(&uri).execute().await?;
    match db.open_table(FACE_EMBEDDING_TABLE).execute().await {
        Ok(table) => Ok(table),
        Err(_) => {
            db.create_empty_table(FACE_EMBEDDING_TABLE, face_embedding_schema(dimension))
                .execute()
                .await
        }
    }
}

async fn open_face_embedding_table() -> Result<lancedb::Table, lancedb::Error> {
    let uri = face_embedding_store_uri().ok_or_else(|| lancedb::Error::InvalidInput {
        message: "catalogue path is not initialized".to_string(),
    })?;

    let db = lancedb::connect(&uri).execute().await?;
    db.open_table(FACE_EMBEDDING_TABLE).execute().await
}

fn face_embedding_version_filter(model_version: &str, preprocessing_version: &str) -> String {
    format!(
        "model_version = {} AND preprocessing_version = {}",
        sql_string_literal(model_version),
        sql_string_literal(preprocessing_version)
    )
}

fn face_embedding_batch(
    records: &[FaceEmbeddingVectorRecord],
    dimension: u32,
) -> Result<
    RecordBatchIterator<std::vec::IntoIter<Result<RecordBatch, arrow_schema::ArrowError>>>,
    arrow_schema::ArrowError,
> {
    let schema = face_embedding_schema(dimension);
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from_iter_values(
                records.iter().map(|record| record.face_observation_id),
            )),
            Arc::new(Int64Array::from_iter_values(
                records.iter().map(|record| record.image_id),
            )),
            Arc::new(Int64Array::from_iter_values(
                records.iter().map(|record| record.analyzed_image_id),
            )),
            Arc::new(UInt32Array::from_iter_values(
                records.iter().map(|record| record.face_index),
            )),
            Arc::new(StringArray::from_iter_values(
                records.iter().map(|record| record.model_name.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                records.iter().map(|record| record.model_version.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                records
                    .iter()
                    .map(|record| record.preprocessing_version.as_str()),
            )),
            Arc::new(UInt32Array::from_iter_values(
                records.iter().map(|record| record.input_size),
            )),
            Arc::new(StringArray::from_iter_values(
                records.iter().map(|record| record.color_order.as_str()),
            )),
            Arc::new(StringArray::from_iter_values(
                records.iter().map(|record| record.normalization.as_str()),
            )),
            Arc::new(UInt32Array::from_iter_values(
                records.iter().map(|record| record.embedding_dimension),
            )),
            Arc::new(Float64Array::from_iter_values(
                records.iter().map(|record| record.embedding_l2_norm),
            )),
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                    records.iter().map(|record| {
                        Some(record.vector.iter().copied().map(Some).collect::<Vec<_>>())
                    }),
                    dimension as i32,
                ),
            ),
        ],
    )?;

    Ok(RecordBatchIterator::new(
        vec![Ok(batch)].into_iter(),
        schema.clone(),
    ))
}

pub async fn upsert_face_embeddings(
    records: Vec<FaceEmbeddingVectorRecord>,
) -> FaceEmbeddingStoreResult {
    match FACE_EMBEDDING_RUNTIME
        .spawn(upsert_face_embeddings_impl(records))
        .await
    {
        Ok(result) => result,
        Err(e) => FaceEmbeddingStoreResult {
            requested_count: 0,
            stored_count: 0,
            total_count: 0,
            status: "runtime_failed".to_string(),
            message: format!("Face embedding runtime task failed: {}", e),
        },
    }
}

async fn upsert_face_embeddings_impl(
    records: Vec<FaceEmbeddingVectorRecord>,
) -> FaceEmbeddingStoreResult {
    let requested_count = records.len() as u64;
    if records.is_empty() {
        let total_count = face_embedding_total_count().await;
        return FaceEmbeddingStoreResult {
            requested_count,
            stored_count: 0,
            total_count,
            status: "empty".to_string(),
            message: "No face embeddings were provided.".to_string(),
        };
    }

    let model_version = records[0].model_version.clone();
    let preprocessing_version = records[0].preprocessing_version.clone();
    let dimension = records[0].embedding_dimension;
    let valid_records = records
        .into_iter()
        .filter(|record| {
            record.embedding_dimension == dimension
                && record.model_version == model_version
                && record.preprocessing_version == preprocessing_version
                && record.vector.len() == dimension as usize
                && record.vector.iter().all(|value| value.is_finite())
                && record.embedding_l2_norm.is_finite()
        })
        .collect::<Vec<_>>();

    if valid_records.is_empty() {
        let total_count = face_embedding_total_count().await;
        return FaceEmbeddingStoreResult {
            requested_count,
            stored_count: 0,
            total_count,
            status: "invalid".to_string(),
            message:
                "No valid face embeddings matched the first record's model/preprocessing contract."
                    .to_string(),
        };
    }

    let table = match open_or_create_face_embedding_table(dimension).await {
        Ok(table) => table,
        Err(e) => {
            return FaceEmbeddingStoreResult {
                requested_count,
                stored_count: 0,
                total_count: 0,
                status: "open_failed".to_string(),
                message: format!("Failed to open LanceDB face embedding table: {}", e),
            };
        }
    };

    let ids = valid_records
        .iter()
        .map(|record| record.face_observation_id.to_string())
        .collect::<Vec<_>>();
    let delete_filter = format!(
        "{} AND face_observation_id IN ({})",
        face_embedding_version_filter(&model_version, &preprocessing_version),
        ids.join(",")
    );
    if let Err(e) = table.delete(&delete_filter).await {
        return FaceEmbeddingStoreResult {
            requested_count,
            stored_count: 0,
            total_count: table.count_rows(None).await.unwrap_or(0) as u64,
            status: "replace_failed".to_string(),
            message: format!("Failed to replace existing face embeddings: {}", e),
        };
    }

    let batch = match face_embedding_batch(&valid_records, dimension) {
        Ok(batch) => batch,
        Err(e) => {
            return FaceEmbeddingStoreResult {
                requested_count,
                stored_count: 0,
                total_count: table.count_rows(None).await.unwrap_or(0) as u64,
                status: "batch_failed".to_string(),
                message: format!("Failed to build Arrow batch for face embeddings: {}", e),
            };
        }
    };

    if let Err(e) = table.add(batch).execute().await {
        return FaceEmbeddingStoreResult {
            requested_count,
            stored_count: 0,
            total_count: table.count_rows(None).await.unwrap_or(0) as u64,
            status: "store_failed".to_string(),
            message: format!("Failed to store face embeddings: {}", e),
        };
    }

    let total_count = table.count_rows(None).await.unwrap_or(0) as u64;
    FaceEmbeddingStoreResult {
        requested_count,
        stored_count: valid_records.len() as u64,
        total_count,
        status: "stored".to_string(),
        message: "Face embeddings stored in LanceDB.".to_string(),
    }
}

async fn face_embedding_total_count() -> u64 {
    match open_face_embedding_table().await {
        Ok(table) => table.count_rows(None).await.unwrap_or(0) as u64,
        Err(_) => 0,
    }
}

pub async fn face_embedding_count(model_version: String, preprocessing_version: String) -> u64 {
    FACE_EMBEDDING_RUNTIME
        .spawn(face_embedding_count_impl(
            model_version,
            preprocessing_version,
        ))
        .await
        .unwrap_or(0)
}

async fn face_embedding_count_impl(model_version: String, preprocessing_version: String) -> u64 {
    let table = match open_face_embedding_table().await {
        Ok(table) => table,
        Err(_) => return 0,
    };
    let filter = face_embedding_version_filter(&model_version, &preprocessing_version);
    table.count_rows(Some(filter)).await.unwrap_or(0) as u64
}

#[derive(Debug, Clone)]
struct StoredFaceEmbedding {
    face_observation_id: i64,
    image_id: i64,
    analyzed_image_id: i64,
    face_index: u32,
    vector: Vec<f32>,
}

async fn stored_face_embeddings(
    model_version: &str,
    preprocessing_version: &str,
) -> Vec<StoredFaceEmbedding> {
    let table = match open_face_embedding_table().await {
        Ok(table) => table,
        Err(e) => {
            eprintln!("stored_face_embeddings: open {}", e);
            return Vec::new();
        }
    };

    let filter = face_embedding_version_filter(model_version, preprocessing_version);
    let matching_count = table.count_rows(Some(filter.clone())).await.unwrap_or(0);
    if matching_count == 0 {
        return Vec::new();
    }

    let batches = match table
        .query()
        .only_if(filter)
        .limit(matching_count as usize)
        .execute()
        .await
    {
        Ok(stream) => match stream.try_collect::<Vec<_>>().await {
            Ok(batches) => batches,
            Err(e) => {
                eprintln!("stored_face_embeddings: collect {}", e);
                return Vec::new();
            }
        },
        Err(e) => {
            eprintln!("stored_face_embeddings: query {}", e);
            return Vec::new();
        }
    };

    let mut records = Vec::new();
    for batch in batches {
        let Some(face_ids) = batch
            .column_by_name("face_observation_id")
            .and_then(|array| array.as_any().downcast_ref::<Int64Array>())
        else {
            continue;
        };
        let Some(image_ids) = batch
            .column_by_name("image_id")
            .and_then(|array| array.as_any().downcast_ref::<Int64Array>())
        else {
            continue;
        };
        let Some(analyzed_ids) = batch
            .column_by_name("analyzed_image_id")
            .and_then(|array| array.as_any().downcast_ref::<Int64Array>())
        else {
            continue;
        };
        let Some(face_indices) = batch
            .column_by_name("face_index")
            .and_then(|array| array.as_any().downcast_ref::<UInt32Array>())
        else {
            continue;
        };
        let Some(vectors) = batch
            .column_by_name("vector")
            .and_then(|array| array.as_any().downcast_ref::<FixedSizeListArray>())
        else {
            continue;
        };
        let Some(values) = vectors
            .values()
            .as_any()
            .downcast_ref::<arrow_array::Float32Array>()
        else {
            continue;
        };

        let dimension = vectors.value_length().max(0) as usize;
        for row in 0..batch.num_rows() {
            let start = row * dimension;
            let end = start + dimension;
            if end > values.len() {
                continue;
            }
            records.push(StoredFaceEmbedding {
                face_observation_id: face_ids.value(row),
                image_id: image_ids.value(row),
                analyzed_image_id: analyzed_ids.value(row),
                face_index: face_indices.value(row),
                vector: (start..end).map(|index| values.value(index)).collect(),
            });
        }
    }

    records.sort_by(|left, right| {
        left.face_observation_id
            .cmp(&right.face_observation_id)
            .then(left.image_id.cmp(&right.image_id))
            .then(left.face_index.cmp(&right.face_index))
    });
    records
}

fn cosine_f32(left: &[f32], right: &[f32]) -> f64 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for (left_value, right_value) in left.iter().zip(right.iter()) {
        let l = f64::from(*left_value);
        let r = f64::from(*right_value);
        dot += l * r;
        left_norm += l * l;
        right_norm += r * r;
    }

    if left_norm <= 0.0 || right_norm <= 0.0 {
        return 0.0;
    }

    dot / (left_norm.sqrt() * right_norm.sqrt())
}

pub async fn face_embedding_nearest_neighbors(
    model_version: String,
    preprocessing_version: String,
    limit_per_face: u32,
) -> Vec<FaceEmbeddingNeighborRecord> {
    FACE_EMBEDDING_RUNTIME
        .spawn(face_embedding_nearest_neighbors_impl(
            model_version,
            preprocessing_version,
            limit_per_face,
        ))
        .await
        .unwrap_or_default()
}

async fn face_embedding_nearest_neighbors_impl(
    model_version: String,
    preprocessing_version: String,
    limit_per_face: u32,
) -> Vec<FaceEmbeddingNeighborRecord> {
    let limit_per_face = limit_per_face.max(1) as usize;
    let records = stored_face_embeddings(&model_version, &preprocessing_version).await;
    let mut neighbors = Vec::new();

    for (query_index, query) in records.iter().enumerate() {
        let mut scored = records
            .iter()
            .enumerate()
            .filter(|(neighbor_index, _)| *neighbor_index != query_index)
            .map(|(_, neighbor)| (neighbor, cosine_f32(&query.vector, &neighbor.vector)))
            .collect::<Vec<_>>();
        scored.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (neighbor, cosine) in scored.into_iter().take(limit_per_face) {
            neighbors.push(FaceEmbeddingNeighborRecord {
                query_face_observation_id: query.face_observation_id,
                neighbor_face_observation_id: neighbor.face_observation_id,
                query_image_id: query.image_id,
                neighbor_image_id: neighbor.image_id,
                query_analyzed_image_id: query.analyzed_image_id,
                neighbor_analyzed_image_id: neighbor.analyzed_image_id,
                query_face_index: query.face_index,
                neighbor_face_index: neighbor.face_index,
                cosine,
            });
        }
    }

    neighbors.sort_by(|left, right| {
        left.query_face_observation_id
            .cmp(&right.query_face_observation_id)
            .then_with(|| {
                right
                    .cosine
                    .partial_cmp(&left.cosine)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then(
                left.neighbor_face_observation_id
                    .cmp(&right.neighbor_face_observation_id),
            )
    });
    neighbors
}

pub async fn face_embedding_search(
    seed_face_observation_ids: Vec<i64>,
    candidate_image_ids: Vec<i64>,
    model_version: String,
    preprocessing_version: String,
    threshold: f64,
    limit: u32,
) -> Vec<FaceSearchMatchRecord> {
    FACE_EMBEDDING_RUNTIME
        .spawn(face_embedding_search_impl(
            seed_face_observation_ids,
            candidate_image_ids,
            model_version,
            preprocessing_version,
            threshold,
            limit,
        ))
        .await
        .unwrap_or_default()
}

pub async fn face_embedding_search_vector(
    seed_vector: Vec<f32>,
    candidate_image_ids: Vec<i64>,
    model_version: String,
    preprocessing_version: String,
    threshold: f64,
    limit: u32,
) -> Vec<FaceSearchMatchRecord> {
    FACE_EMBEDDING_RUNTIME
        .spawn(face_embedding_search_vector_impl(
            seed_vector,
            candidate_image_ids,
            model_version,
            preprocessing_version,
            threshold,
            limit,
        ))
        .await
        .unwrap_or_default()
}

async fn face_embedding_search_impl(
    seed_face_observation_ids: Vec<i64>,
    candidate_image_ids: Vec<i64>,
    model_version: String,
    preprocessing_version: String,
    threshold: f64,
    limit: u32,
) -> Vec<FaceSearchMatchRecord> {
    if seed_face_observation_ids.is_empty() {
        return Vec::new();
    }

    let records = stored_face_embeddings(&model_version, &preprocessing_version).await;
    if records.is_empty() {
        return Vec::new();
    }

    let seed_ids = seed_face_observation_ids
        .into_iter()
        .collect::<std::collections::HashSet<_>>();
    let seeds = records
        .iter()
        .filter(|record| seed_ids.contains(&record.face_observation_id))
        .collect::<Vec<_>>();
    if seeds.is_empty() {
        return Vec::new();
    }

    let candidate_ids = if candidate_image_ids.is_empty() {
        None
    } else {
        Some(
            candidate_image_ids
                .into_iter()
                .collect::<std::collections::HashSet<_>>(),
        )
    };
    let threshold = if threshold.is_finite() {
        threshold
    } else {
        0.0
    };

    let mut best_by_image = std::collections::HashMap::<i64, FaceSearchMatchRecord>::new();
    for candidate in records.iter() {
        if let Some(candidate_ids) = &candidate_ids {
            if !candidate_ids.contains(&candidate.image_id) {
                continue;
            }
        }

        for seed in seeds.iter() {
            let cosine = cosine_f32(&seed.vector, &candidate.vector);
            if cosine < threshold {
                continue;
            }

            let match_record = FaceSearchMatchRecord {
                image_id: candidate.image_id,
                face_observation_id: candidate.face_observation_id,
                seed_face_observation_id: seed.face_observation_id,
                face_index: candidate.face_index,
                cosine,
            };

            match best_by_image.get(&candidate.image_id) {
                Some(existing)
                    if existing.cosine > cosine
                        || (existing.cosine == cosine
                            && existing.face_observation_id <= candidate.face_observation_id) => {}
                _ => {
                    best_by_image.insert(candidate.image_id, match_record);
                }
            }
        }
    }

    let mut matches = best_by_image.into_values().collect::<Vec<_>>();
    matches.sort_by(|left, right| {
        right
            .cosine
            .partial_cmp(&left.cosine)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(left.image_id.cmp(&right.image_id))
            .then(left.face_observation_id.cmp(&right.face_observation_id))
    });

    if limit > 0 {
        matches.truncate(limit as usize);
    }
    matches
}

async fn face_embedding_search_vector_impl(
    seed_vector: Vec<f32>,
    candidate_image_ids: Vec<i64>,
    model_version: String,
    preprocessing_version: String,
    threshold: f64,
    limit: u32,
) -> Vec<FaceSearchMatchRecord> {
    if seed_vector.is_empty() || !seed_vector.iter().all(|value| value.is_finite()) {
        return Vec::new();
    }

    let records = stored_face_embeddings(&model_version, &preprocessing_version).await;
    if records.is_empty() {
        return Vec::new();
    }

    let candidate_ids = if candidate_image_ids.is_empty() {
        None
    } else {
        Some(
            candidate_image_ids
                .into_iter()
                .collect::<std::collections::HashSet<_>>(),
        )
    };
    let threshold = if threshold.is_finite() {
        threshold
    } else {
        0.0
    };

    let mut best_by_image = std::collections::HashMap::<i64, FaceSearchMatchRecord>::new();
    for candidate in records.iter() {
        if let Some(candidate_ids) = &candidate_ids {
            if !candidate_ids.contains(&candidate.image_id) {
                continue;
            }
        }

        let cosine = cosine_f32(&seed_vector, &candidate.vector);
        if cosine < threshold {
            continue;
        }

        let match_record = FaceSearchMatchRecord {
            image_id: candidate.image_id,
            face_observation_id: candidate.face_observation_id,
            seed_face_observation_id: 0,
            face_index: candidate.face_index,
            cosine,
        };

        match best_by_image.get(&candidate.image_id) {
            Some(existing)
                if existing.cosine > cosine
                    || (existing.cosine == cosine
                        && existing.face_observation_id <= candidate.face_observation_id) => {}
            _ => {
                best_by_image.insert(candidate.image_id, match_record);
            }
        }
    }

    let mut matches = best_by_image.into_values().collect::<Vec<_>>();
    matches.sort_by(|left, right| {
        right
            .cosine
            .partial_cmp(&left.cosine)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(left.image_id.cmp(&right.image_id))
            .then(left.face_observation_id.cmp(&right.face_observation_id))
    });

    if limit > 0 {
        matches.truncate(limit as usize);
    }
    matches
}

fn face_cluster_run_is_valid(run: &FaceClusterRunRecord) -> bool {
    !run.run_id.trim().is_empty()
        && !run.face_algorithm_version.trim().is_empty()
        && !run.model_version.trim().is_empty()
        && !run.preprocessing_version.trim().is_empty()
        && run.threshold.is_finite()
        && run.threshold > 0.0
        && run.threshold <= 1.0
}

fn face_cluster_member_is_valid(member: &FaceClusterMemberRecord, run_id: &str) -> bool {
    member.run_id == run_id
        && member.cluster_id > 0
        && member.face_observation_id > 0
        && member.image_id > 0
        && member.analyzed_image_id > 0
        && member.cluster_size > 1
        && member
            .nearest_neighbor_cosine
            .map(|value| value.is_finite())
            .unwrap_or(true)
}

fn replace_face_cluster_run_impl(
    conn: &Connection,
    run: FaceClusterRunRecord,
    members: Vec<FaceClusterMemberRecord>,
) -> u64 {
    if !face_cluster_run_is_valid(&run) {
        eprintln!("replace_face_cluster_run: invalid run metadata");
        return 0;
    }

    let run_id = run.run_id.trim().to_string();
    let valid_members = members
        .into_iter()
        .filter(|member| face_cluster_member_is_valid(member, &run_id))
        .collect::<Vec<_>>();

    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;") {
        eprintln!("replace_face_cluster_run: begin {}", e);
        return 0;
    }

    if let Err(e) = conn.execute(
        "DELETE FROM face_cluster_member WHERE run_id = ?1",
        params![run_id.as_str()],
    ) {
        eprintln!("replace_face_cluster_run: delete members {}", e);
        let _ = conn.execute_batch("ROLLBACK;");
        return 0;
    }

    if let Err(e) = conn.execute(
        "DELETE FROM face_cluster_run WHERE run_id = ?1",
        params![run_id.as_str()],
    ) {
        eprintln!("replace_face_cluster_run: delete run {}", e);
        let _ = conn.execute_batch("ROLLBACK;");
        return 0;
    }

    if let Err(e) = conn.execute(
        "INSERT INTO face_cluster_run (
             run_id, face_algorithm_version, model_version, preprocessing_version,
             threshold, cluster_count, member_count
         )
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            run_id.as_str(),
            run.face_algorithm_version,
            run.model_version,
            run.preprocessing_version,
            run.threshold,
            run.cluster_count as i64,
            valid_members.len() as i64,
        ],
    ) {
        eprintln!("replace_face_cluster_run: insert run {}", e);
        let _ = conn.execute_batch("ROLLBACK;");
        return 0;
    }

    let mut inserted = 0u64;
    for member in valid_members {
        match conn.execute(
            "INSERT INTO face_cluster_member (
                 run_id, cluster_id, face_observation_id, image_id, analyzed_image_id,
                 face_index, member_rank, cluster_size,
                 nearest_neighbor_face_observation_id, nearest_neighbor_cosine
             )
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                run_id.as_str(),
                member.cluster_id,
                member.face_observation_id,
                member.image_id,
                member.analyzed_image_id,
                member.face_index as i64,
                member.member_rank as i64,
                member.cluster_size as i64,
                member.nearest_neighbor_face_observation_id,
                member.nearest_neighbor_cosine,
            ],
        ) {
            Ok(n) => inserted += n as u64,
            Err(e) => {
                eprintln!(
                    "replace_face_cluster_run: insert face_observation_id={} failed: {}",
                    member.face_observation_id, e
                );
                let _ = conn.execute_batch("ROLLBACK;");
                return 0;
            }
        }
    }

    if let Err(e) = conn.execute_batch("COMMIT;") {
        eprintln!("replace_face_cluster_run: commit {}", e);
        return 0;
    }

    inserted
}

pub async fn replace_face_cluster_run(
    run: FaceClusterRunRecord,
    members: Vec<FaceClusterMemberRecord>,
) -> u64 {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    replace_face_cluster_run_impl(conn, run, members)
}

fn face_cluster_runs_impl(conn: &Connection, limit: u32) -> Vec<FaceClusterRunSummary> {
    let limit = i64::from(limit.clamp(1, 100));
    let mut stmt = match conn.prepare(
        "SELECT
             run_id,
             face_algorithm_version,
             model_version,
             preprocessing_version,
             threshold,
             cluster_count,
             member_count,
             CAST(created_at AS VARCHAR)
         FROM face_cluster_run
         ORDER BY created_at DESC, run_id DESC
         LIMIT ?1",
    ) {
        Ok(stmt) => stmt,
        Err(e) => {
            eprintln!("face_cluster_runs: prepare {}", e);
            return Vec::new();
        }
    };

    let rows = match stmt.query_map(params![limit], |row| {
        let cluster_count = row.get::<_, i64>(5)?;
        let member_count = row.get::<_, i64>(6)?;
        Ok(FaceClusterRunSummary {
            run_id: row.get(0)?,
            face_algorithm_version: row.get(1)?,
            model_version: row.get(2)?,
            preprocessing_version: row.get(3)?,
            threshold: row.get(4)?,
            cluster_count: cluster_count.clamp(0, u32::MAX as i64) as u32,
            member_count: member_count.clamp(0, u32::MAX as i64) as u32,
            created_at: row.get(7)?,
        })
    }) {
        Ok(rows) => rows,
        Err(e) => {
            eprintln!("face_cluster_runs: query {}", e);
            return Vec::new();
        }
    };

    rows.filter_map(|row| row.ok()).collect()
}

pub async fn face_cluster_runs(limit: u32) -> Vec<FaceClusterRunSummary> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    face_cluster_runs_impl(conn, limit)
}

fn face_cluster_members_impl(conn: &Connection, run_id: &str) -> Vec<FaceClusterMemberRecord> {
    let run_id = run_id.trim();
    if run_id.is_empty() {
        return Vec::new();
    }

    let mut stmt = match conn.prepare(
        "SELECT
             run_id,
             cluster_id,
             face_observation_id,
             image_id,
             analyzed_image_id,
             face_index,
             member_rank,
             cluster_size,
             nearest_neighbor_face_observation_id,
             nearest_neighbor_cosine
         FROM face_cluster_member
         WHERE run_id = ?1
         ORDER BY cluster_id, member_rank, face_observation_id",
    ) {
        Ok(stmt) => stmt,
        Err(e) => {
            eprintln!("face_cluster_members: prepare {}", e);
            return Vec::new();
        }
    };

    let rows = match stmt.query_map(params![run_id], |row| {
        let face_index = row.get::<_, i64>(5)?;
        let member_rank = row.get::<_, i64>(6)?;
        let cluster_size = row.get::<_, i64>(7)?;
        Ok(FaceClusterMemberRecord {
            run_id: row.get(0)?,
            cluster_id: row.get(1)?,
            face_observation_id: row.get(2)?,
            image_id: row.get(3)?,
            analyzed_image_id: row.get(4)?,
            face_index: face_index.clamp(0, u32::MAX as i64) as u32,
            member_rank: member_rank.clamp(0, u32::MAX as i64) as u32,
            cluster_size: cluster_size.clamp(0, u32::MAX as i64) as u32,
            nearest_neighbor_face_observation_id: row.get(8)?,
            nearest_neighbor_cosine: row.get(9)?,
        })
    }) {
        Ok(rows) => rows,
        Err(e) => {
            eprintln!("face_cluster_members: query {}", e);
            return Vec::new();
        }
    };

    rows.filter_map(|row| row.ok()).collect()
}

pub async fn face_cluster_members(run_id: String) -> Vec<FaceClusterMemberRecord> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    face_cluster_members_impl(conn, &run_id)
}

fn clean_person_display_name(name: &str) -> Option<String> {
    let display = name.split_whitespace().collect::<Vec<_>>().join(" ");
    if display.is_empty() || display.contains(KEYWORD_PATH_SEPARATOR) {
        None
    } else {
        Some(display)
    }
}

fn normalized_person_name(display_name: &str) -> String {
    display_name.to_lowercase()
}

fn person_accept_error(status: &str, message: &str) -> PersonClusterAcceptResult {
    PersonClusterAcceptResult {
        person_id: 0,
        person_name: String::new(),
        assigned_face_count: 0,
        keyword_image_count: 0,
        keyword_row_count: 0,
        keyword_path: String::new(),
        status: status.to_string(),
        message: message.to_string(),
    }
}

#[derive(Debug, Clone)]
struct FaceClusterRunForPersonAccept {
    model_version: String,
    preprocessing_version: String,
    threshold: f64,
}

fn face_cluster_run_for_person_accept(
    conn: &Connection,
    run_id: &str,
) -> Option<FaceClusterRunForPersonAccept> {
    conn.query_row(
        "SELECT model_version, preprocessing_version, threshold
         FROM face_cluster_run
         WHERE run_id = ?1",
        params![run_id],
        |row| {
            Ok(FaceClusterRunForPersonAccept {
                model_version: row.get(0)?,
                preprocessing_version: row.get(1)?,
                threshold: row.get(2)?,
            })
        },
    )
    .ok()
}

fn face_cluster_members_for_person_accept(
    conn: &Connection,
    run_id: &str,
    cluster_id: i64,
) -> Vec<FaceClusterMemberRecord> {
    face_cluster_members_impl(conn, run_id)
        .into_iter()
        .filter(|member| member.cluster_id == cluster_id)
        .collect()
}

fn get_or_create_person_impl(
    conn: &Connection,
    display_name: &str,
    normalized_name: &str,
) -> Result<i64, duckdb::Error> {
    if let Ok(id) = conn.query_row(
        "SELECT id FROM person WHERE normalized_name = ?1",
        params![normalized_name],
        |row| row.get(0),
    ) {
        conn.execute(
            "UPDATE person SET updated_at = CURRENT_TIMESTAMP WHERE id = ?1",
            params![id],
        )?;
        return Ok(id);
    }

    conn.query_row(
        "INSERT INTO person (display_name, normalized_name, created_at, updated_at)
         VALUES (?1, ?2, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
         RETURNING id",
        params![display_name, normalized_name],
        |row| row.get(0),
    )
}

fn accept_face_cluster_as_person_impl(
    conn: &Connection,
    run_id: &str,
    cluster_id: i64,
    person_name: &str,
) -> PersonClusterAcceptResult {
    let run_id = run_id.trim();
    if run_id.is_empty() || cluster_id <= 0 {
        return person_accept_error("invalid", "A run id and positive cluster id are required.");
    }

    let Some(display_name) = clean_person_display_name(person_name) else {
        return person_accept_error("invalid", "A non-empty person name is required.");
    };
    let normalized_name = normalized_person_name(&display_name);
    let keyword_segments = vec!["People".to_string(), display_name.clone()];
    let keyword_path = keyword_segments.join(KEYWORD_PATH_SEPARATOR);

    let Some(run) = face_cluster_run_for_person_accept(conn, run_id) else {
        return person_accept_error(
            "missing_run",
            "The selected face cluster run was not found.",
        );
    };

    let members = face_cluster_members_for_person_accept(conn, run_id, cluster_id);
    if members.is_empty() {
        return person_accept_error(
            "empty_cluster",
            "The selected cluster has no persisted members.",
        );
    }

    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;") {
        eprintln!("accept_face_cluster_as_person: begin {}", e);
        return person_accept_error(
            "failed",
            "Could not start the person assignment transaction.",
        );
    }

    let person_id = match get_or_create_person_impl(conn, &display_name, &normalized_name) {
        Ok(id) => id,
        Err(e) => {
            eprintln!("accept_face_cluster_as_person: person {}", e);
            let _ = conn.execute_batch("ROLLBACK;");
            return person_accept_error("failed", "Could not create or reuse the person row.");
        }
    };

    let mut assigned_face_count = 0u64;
    let mut image_ids = std::collections::BTreeSet::new();
    let mut keyword_row_count = 0u64;

    for member in &members {
        match conn.execute(
            "INSERT INTO person_face_assignment (
                 face_observation_id, person_id, image_id, analyzed_image_id, face_index,
                 assignment_source, face_cluster_run_id, face_cluster_id,
                 model_version, preprocessing_version, threshold, confidence_cosine,
                 created_at, updated_at
             )
             VALUES (?1, ?2, ?3, ?4, ?5, 'cluster_accept', ?6, ?7, ?8, ?9, ?10, ?11, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
             ON CONFLICT (face_observation_id) DO UPDATE SET
                 person_id = excluded.person_id,
                 image_id = excluded.image_id,
                 analyzed_image_id = excluded.analyzed_image_id,
                 face_index = excluded.face_index,
                 assignment_source = excluded.assignment_source,
                 face_cluster_run_id = excluded.face_cluster_run_id,
                 face_cluster_id = excluded.face_cluster_id,
                 model_version = excluded.model_version,
                 preprocessing_version = excluded.preprocessing_version,
                 threshold = excluded.threshold,
                 confidence_cosine = excluded.confidence_cosine,
                 updated_at = now()",
            params![
                member.face_observation_id,
                person_id,
                member.image_id,
                member.analyzed_image_id,
                member.face_index as i64,
                run_id,
                cluster_id,
                run.model_version.as_str(),
                run.preprocessing_version.as_str(),
                run.threshold,
                member.nearest_neighbor_cosine,
            ],
        ) {
            Ok(_) => {
                assigned_face_count += 1;
                image_ids.insert(member.image_id);
            }
            Err(e) => {
                eprintln!(
                    "accept_face_cluster_as_person: assign face_observation_id={} {}",
                    member.face_observation_id, e
                );
                let _ = conn.execute_batch("ROLLBACK;");
                return person_accept_error("failed", "Could not persist all face assignments.");
            }
        }
    }

    for image_id in &image_ids {
        match insert_or_merge_active_keyword_for_image(
            conn,
            *image_id,
            &keyword_segments,
            KEYWORD_ORIGIN_USER,
        ) {
            Ok(changed) => keyword_row_count += changed,
            Err(e) => {
                eprintln!(
                    "accept_face_cluster_as_person: keyword image_id={} {}",
                    image_id, e
                );
                let _ = conn.execute_batch("ROLLBACK;");
                return person_accept_error("failed", "Could not mirror the People keyword.");
            }
        }
    }

    if let Err(e) = conn.execute_batch("COMMIT;") {
        eprintln!("accept_face_cluster_as_person: commit {}", e);
        return person_accept_error("failed", "Could not commit the person assignment.");
    }

    PersonClusterAcceptResult {
        person_id,
        person_name: display_name,
        assigned_face_count,
        keyword_image_count: image_ids.len() as u64,
        keyword_row_count,
        keyword_path,
        status: "accepted".to_string(),
        message: "Face cluster accepted as a named person.".to_string(),
    }
}

pub async fn accept_face_cluster_as_person(
    run_id: String,
    cluster_id: i64,
    person_name: String,
) -> PersonClusterAcceptResult {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return person_accept_error("failed", "Catalogue not initialized.");
        }
    };

    accept_face_cluster_as_person_impl(conn, &run_id, cluster_id, &person_name)
}

fn similar_photo_candidates_impl(
    conn: &Connection,
    algorithm_version: &str,
    scoped_ids: Option<&[i64]>,
) -> Vec<SimilarPhotoCandidate> {
    let id_filter = match scoped_ids {
        Some(ids) => match id_in_list(ids) {
            Some(filter) => Some(filter),
            None => return Vec::new(),
        },
        None => None,
    };

    let mut predicates = vec![
        "is_video IS NOT TRUE".to_string(),
        "focus_analysis_status = 'complete'".to_string(),
        "focus_algorithm_version = ?1".to_string(),
    ];
    if let Some(filter) = id_filter {
        predicates.push(format!("({})", filter));
    }

    let sql = format!(
        "SELECT id, file_path, file_size, created_timestamp, capture_datetime, directory_path, camera_model
         FROM images
         WHERE {}
         ORDER BY directory_path ASC NULLS LAST,
                  camera_model ASC NULLS LAST,
                  capture_datetime ASC NULLS LAST,
                  created_timestamp ASC,
                  id ASC",
        predicates.join(" AND ")
    );

    let mut stmt = match conn.prepare(&sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("similar_photo_candidates: prepare {}", e);
            return Vec::new();
        }
    };

    let mapped = stmt.query_map(params![algorithm_version], |row| {
        Ok(SimilarPhotoCandidate {
            id: row.get(0)?,
            file_path: row.get(1)?,
            file_size: row.get::<_, i64>(2)? as u64,
            created_timestamp: row.get(3)?,
            capture_datetime: row.get(4)?,
            directory_path: row.get(5)?,
            camera_model: row.get(6)?,
        })
    });

    match mapped {
        Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
        Err(e) => {
            eprintln!("similar_photo_candidates: query {}", e);
            Vec::new()
        }
    }
}

/// Candidate rows for similar-photo grouping across the full catalogue. Rows are
/// limited to stills whose current Intelligent Culling pass completed for the
/// supplied algorithm version.
pub async fn similar_photo_candidates(algorithm_version: String) -> Vec<SimilarPhotoCandidate> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    similar_photo_candidates_impl(conn, &algorithm_version, None)
}

/// Candidate rows for similar-photo grouping intersected with an explicit image
/// selection. Empty selection means empty queue.
pub async fn similar_photo_candidates_for_ids(
    ids: Vec<i64>,
    algorithm_version: String,
) -> Vec<SimilarPhotoCandidate> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    similar_photo_candidates_impl(conn, &algorithm_version, Some(&ids))
}

fn similar_photo_featureprints_for_ids_impl(
    conn: &Connection,
    ids: &[i64],
    algorithm_version: &str,
) -> Vec<SimilarPhotoFeatureprint> {
    let Some(filter) = id_in_list(ids) else {
        return Vec::new();
    };
    let sql = format!(
        "SELECT image_id, source_stamp, featureprint_blob
         FROM similar_photo_featureprint
         WHERE algorithm_version = ?1 AND image_id IN (SELECT id FROM images WHERE {})
         ORDER BY image_id",
        filter
    );
    let mut stmt = match conn.prepare(&sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("similar_photo_featureprints_for_ids: prepare {}", e);
            return Vec::new();
        }
    };

    let mapped = stmt.query_map(params![algorithm_version], |row| {
        Ok(SimilarPhotoFeatureprint {
            image_id: row.get(0)?,
            source_stamp: row.get(1)?,
            featureprint_blob: row.get(2)?,
        })
    });

    match mapped {
        Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
        Err(e) => {
            eprintln!("similar_photo_featureprints_for_ids: query {}", e);
            Vec::new()
        }
    }
}

pub async fn similar_photo_featureprints_for_ids(
    ids: Vec<i64>,
    algorithm_version: String,
) -> Vec<SimilarPhotoFeatureprint> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    similar_photo_featureprints_for_ids_impl(conn, &ids, &algorithm_version)
}

fn upsert_similar_photo_featureprints_impl(
    conn: &Connection,
    entries: Vec<SimilarPhotoFeatureprint>,
    algorithm_version: &str,
) -> u64 {
    let algorithm_version = algorithm_version.trim();
    if entries.is_empty() || algorithm_version.is_empty() {
        return 0;
    }
    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;") {
        eprintln!("upsert_similar_photo_featureprints: begin {}", e);
        return 0;
    }

    let mut changed = 0u64;
    for entry in entries {
        if entry.image_id <= 0
            || entry.source_stamp.trim().is_empty()
            || entry.featureprint_blob.is_empty()
        {
            continue;
        }
        match conn.execute(
            "INSERT INTO similar_photo_featureprint (
                 image_id, algorithm_version, source_stamp, featureprint_blob
             )
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT (image_id, algorithm_version)
             DO UPDATE SET
                 source_stamp = excluded.source_stamp,
                 featureprint_blob = excluded.featureprint_blob,
                 updated_at = now()",
            params![
                entry.image_id,
                algorithm_version,
                entry.source_stamp,
                entry.featureprint_blob,
            ],
        ) {
            Ok(n) => changed += n as u64,
            Err(e) => {
                eprintln!(
                    "upsert_similar_photo_featureprints: image_id={} {}",
                    entry.image_id, e
                );
                let _ = conn.execute_batch("ROLLBACK;");
                return 0;
            }
        }
    }

    if let Err(e) = conn.execute_batch("COMMIT;") {
        eprintln!("upsert_similar_photo_featureprints: commit {}", e);
        return 0;
    }
    changed
}

pub async fn upsert_similar_photo_featureprints(
    entries: Vec<SimilarPhotoFeatureprint>,
    algorithm_version: String,
) -> u64 {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    upsert_similar_photo_featureprints_impl(conn, entries, &algorithm_version)
}

fn completed_similar_photo_work_units_impl(
    conn: &Connection,
    algorithm_version: &str,
    scope_key: &str,
) -> Vec<SimilarPhotoWorkUnit> {
    let mut stmt = match conn.prepare(
        "SELECT unit_index, start_image_id, end_image_id, candidate_count, member_count
         FROM similar_photo_group_work_unit
         WHERE algorithm_version = ?1 AND scope_key = ?2 AND status = 'complete'
         ORDER BY unit_index",
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("completed_similar_photo_work_units: prepare {}", e);
            return Vec::new();
        }
    };

    let mapped = stmt.query_map(params![algorithm_version, scope_key], |row| {
        Ok(SimilarPhotoWorkUnit {
            unit_index: row.get(0)?,
            start_image_id: row.get(1)?,
            end_image_id: row.get(2)?,
            candidate_count: row.get(3)?,
            member_count: row.get(4)?,
        })
    });

    match mapped {
        Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
        Err(e) => {
            eprintln!("completed_similar_photo_work_units: query {}", e);
            Vec::new()
        }
    }
}

pub async fn completed_similar_photo_work_units(
    algorithm_version: String,
    scope_key: String,
) -> Vec<SimilarPhotoWorkUnit> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    completed_similar_photo_work_units_impl(conn, &algorithm_version, &scope_key)
}

fn mark_similar_photo_work_unit_complete_impl(
    conn: &Connection,
    algorithm_version: &str,
    scope_key: &str,
    unit: SimilarPhotoWorkUnit,
) -> bool {
    if algorithm_version.trim().is_empty() || scope_key.trim().is_empty() || unit.unit_index < 0 {
        return false;
    }
    match conn.execute(
        "INSERT INTO similar_photo_group_work_unit (
             algorithm_version, scope_key, unit_index, start_image_id, end_image_id,
             candidate_count, member_count, status
         )
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'complete')
         ON CONFLICT (algorithm_version, scope_key, unit_index)
         DO UPDATE SET
             start_image_id = excluded.start_image_id,
             end_image_id = excluded.end_image_id,
             candidate_count = excluded.candidate_count,
             member_count = excluded.member_count,
             status = 'complete',
             updated_at = now()",
        params![
            algorithm_version,
            scope_key,
            unit.unit_index,
            unit.start_image_id,
            unit.end_image_id,
            unit.candidate_count,
            unit.member_count,
        ],
    ) {
        Ok(n) => n > 0,
        Err(e) => {
            eprintln!("mark_similar_photo_work_unit_complete: {}", e);
            false
        }
    }
}

pub async fn mark_similar_photo_work_unit_complete(
    algorithm_version: String,
    scope_key: String,
    unit: SimilarPhotoWorkUnit,
) -> bool {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return false;
        }
    };

    mark_similar_photo_work_unit_complete_impl(conn, &algorithm_version, &scope_key, unit)
}

fn similar_photo_member_is_valid(member: &SimilarPhotoGroupMember) -> bool {
    member.image_id > 0
        && member.group_id > 0
        && member.representative_id > 0
        && member.threshold.is_finite()
        && member.threshold >= 0.0
        && member
            .distance_to_representative
            .map(|distance| distance.is_finite() && distance >= 0.0)
            .unwrap_or(true)
}

fn replace_similar_photo_groups_impl(
    conn: &Connection,
    members: Vec<SimilarPhotoGroupMember>,
    algorithm_version: &str,
    scoped_ids: Option<&[i64]>,
) -> u64 {
    let algorithm_version = algorithm_version.trim();
    if algorithm_version.is_empty() {
        eprintln!("replace_similar_photo_groups: empty algorithm version");
        return 0;
    }

    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;") {
        eprintln!("replace_similar_photo_groups: begin {}", e);
        return 0;
    }

    let delete_result = match scoped_ids {
        Some(ids) => match id_in_list(ids) {
            Some(filter) => {
                let sql = format!("DELETE FROM similar_photo_group_member WHERE image_id IN (SELECT id FROM images WHERE {})", filter);
                conn.execute(&sql, [])
            }
            None => Ok(0),
        },
        None => conn.execute(
            "DELETE FROM similar_photo_group_member WHERE algorithm_version = ?1",
            params![algorithm_version],
        ),
    };

    if let Err(e) = delete_result {
        eprintln!("replace_similar_photo_groups: delete {}", e);
        let _ = conn.execute_batch("ROLLBACK;");
        return 0;
    }

    let mut inserted = 0u64;
    for member in members {
        if !similar_photo_member_is_valid(&member) {
            eprintln!(
                "replace_similar_photo_groups: skipped invalid member image_id={}",
                member.image_id
            );
            continue;
        }

        match conn.execute(
            "INSERT INTO similar_photo_group_member (
                 image_id, group_id, representative_id, member_rank,
                 distance_to_representative, algorithm_version, threshold
             )
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                member.image_id,
                member.group_id,
                member.representative_id,
                member.member_rank as i64,
                member.distance_to_representative,
                algorithm_version,
                member.threshold,
            ],
        ) {
            Ok(n) => inserted += n as u64,
            Err(e) => {
                eprintln!(
                    "replace_similar_photo_groups: insert image_id={} failed: {}",
                    member.image_id, e
                );
                let _ = conn.execute_batch("ROLLBACK;");
                return 0;
            }
        }
    }

    if let Err(e) = conn.execute_batch("COMMIT;") {
        eprintln!("replace_similar_photo_groups: commit {}", e);
        return 0;
    }

    inserted
}

/// Replace all similar-photo group memberships for the supplied algorithm
/// version. Used after whole-catalogue Intelligent Culling.
pub async fn replace_similar_photo_groups(
    members: Vec<SimilarPhotoGroupMember>,
    algorithm_version: String,
) -> u64 {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    replace_similar_photo_groups_impl(conn, members, &algorithm_version, None)
}

/// Replace similar-photo group memberships for an explicit image-id selection.
/// Empty selection clears nothing and inserts nothing.
pub async fn replace_similar_photo_groups_for_ids(
    ids: Vec<i64>,
    members: Vec<SimilarPhotoGroupMember>,
    algorithm_version: String,
) -> u64 {
    if ids.is_empty() {
        return 0;
    }

    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    replace_similar_photo_groups_impl(conn, members, &algorithm_version, Some(&ids))
}

fn upsert_similar_photo_groups_for_ids_impl(
    conn: &Connection,
    ids: &[i64],
    members: Vec<SimilarPhotoGroupMember>,
    algorithm_version: &str,
) -> u64 {
    let algorithm_version = algorithm_version.trim();
    if ids.is_empty() || algorithm_version.is_empty() {
        return 0;
    }
    let Some(filter) = id_in_list(ids) else {
        return 0;
    };

    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;") {
        eprintln!("upsert_similar_photo_groups_for_ids: begin {}", e);
        return 0;
    }

    let delete_sql = format!(
        "DELETE FROM similar_photo_group_member
         WHERE algorithm_version = ?1 AND image_id IN (SELECT id FROM images WHERE {})",
        filter
    );
    if let Err(e) = conn.execute(&delete_sql, params![algorithm_version]) {
        eprintln!("upsert_similar_photo_groups_for_ids: delete {}", e);
        let _ = conn.execute_batch("ROLLBACK;");
        return 0;
    }

    let mut changed = 0u64;
    for member in members {
        if !similar_photo_member_is_valid(&member) {
            eprintln!(
                "upsert_similar_photo_groups_for_ids: skipped invalid member image_id={}",
                member.image_id
            );
            continue;
        }
        match conn.execute(
            "INSERT INTO similar_photo_group_member (
                 image_id, group_id, representative_id, member_rank,
                 distance_to_representative, algorithm_version, threshold
             )
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT (image_id)
             DO UPDATE SET
                 group_id = excluded.group_id,
                 representative_id = excluded.representative_id,
                 member_rank = excluded.member_rank,
                 distance_to_representative = excluded.distance_to_representative,
                 algorithm_version = excluded.algorithm_version,
                 threshold = excluded.threshold,
                 created_at = now()",
            params![
                member.image_id,
                member.group_id,
                member.representative_id,
                member.member_rank as i64,
                member.distance_to_representative,
                algorithm_version,
                member.threshold,
            ],
        ) {
            Ok(n) => changed += n as u64,
            Err(e) => {
                eprintln!(
                    "upsert_similar_photo_groups_for_ids: insert image_id={} failed: {}",
                    member.image_id, e
                );
                let _ = conn.execute_batch("ROLLBACK;");
                return 0;
            }
        }
    }

    if let Err(e) = conn.execute_batch("COMMIT;") {
        eprintln!("upsert_similar_photo_groups_for_ids: commit {}", e);
        return 0;
    }

    changed
}

/// Progressive similar-photo write for a completed work unit. Clears only the
/// supplied unit ids, then upserts discovered memberships so partial stacks are
/// visible while the larger grouping run continues.
pub async fn upsert_similar_photo_groups_for_ids(
    ids: Vec<i64>,
    members: Vec<SimilarPhotoGroupMember>,
    algorithm_version: String,
) -> u64 {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    upsert_similar_photo_groups_for_ids_impl(conn, &ids, members, &algorithm_version)
}

/// Return compact stack counts for visible gallery representatives.
///
/// The caller passes page-visible ids. Each id is resolved to its durable
/// similar-photo group, then counted two ways: physical image rows and logical
/// photos where RAW/JPEG/HEIF siblings with the same directory/stem count once.
pub async fn similar_photo_stack_summaries_for_ids(
    ids: Vec<i64>,
    algorithm_version: String,
) -> Vec<SimilarPhotoStackSummary> {
    if ids.is_empty() {
        return Vec::new();
    }

    let algorithm_version = algorithm_version.trim().to_string();
    if algorithm_version.is_empty() {
        return Vec::new();
    }

    let id_filter = match id_in_list(&ids) {
        Some(filter) => filter,
        None => return Vec::new(),
    };

    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    let sql = format!(
        r#"
        WITH requested AS (
            SELECT id AS image_id
            FROM images
            WHERE {}
        ),
        requested_membership AS (
            SELECT r.image_id, m.group_id
            FROM requested r
            JOIN similar_photo_group_member m
              ON m.image_id = r.image_id
             AND m.algorithm_version = ?1
        )
        SELECT
            rm.image_id,
            rm.group_id,
            COUNT(DISTINCT (
                COALESCE(i.directory_path, '') || '/' ||
                COALESCE(i.file_stem, CAST(i.id AS VARCHAR))
            )) AS logical_count,
            COUNT(*) AS physical_count
        FROM requested_membership rm
        JOIN similar_photo_group_member gm
          ON gm.group_id = rm.group_id
         AND gm.algorithm_version = ?1
        JOIN images i ON i.id = gm.image_id
        GROUP BY rm.image_id, rm.group_id
        "#,
        id_filter
    );

    let mut stmt = match conn.prepare(&sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("similar_photo_stack_summaries_for_ids: prepare {}", e);
            return Vec::new();
        }
    };

    let mapped = stmt.query_map(params![algorithm_version], |row| {
        let logical_count: i64 = row.get(2)?;
        let physical_count: i64 = row.get(3)?;
        Ok(SimilarPhotoStackSummary {
            image_id: row.get(0)?,
            group_id: row.get(1)?,
            logical_count: logical_count.max(0) as u32,
            physical_count: physical_count.max(0) as u32,
        })
    });

    match mapped {
        Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
        Err(e) => {
            eprintln!("similar_photo_stack_summaries_for_ids: query {}", e);
            Vec::new()
        }
    }
}

/// Return the physical similar-photo stack members represented by visible ids.
///
/// Input ids that are not in a stack are returned as singleton rows with
/// `group_id == representative_id == image_id`. On errors, degrade to those
/// singleton rows so callers still act on the visible selection.
fn similar_photo_stack_members_for_ids_impl(
    conn: &Connection,
    ids: &[i64],
    algorithm_version: &str,
) -> Vec<SimilarPhotoStackMember> {
    if ids.is_empty() {
        return Vec::new();
    }

    let fallback = || {
        ids.iter()
            .map(|id| SimilarPhotoStackMember {
                image_id: *id,
                group_id: *id,
                representative_id: *id,
                member_rank: 0,
            })
            .collect::<Vec<_>>()
    };

    let algorithm_version = algorithm_version.trim().to_string();
    if algorithm_version.is_empty() {
        return fallback();
    }

    let id_filter = match id_in_list(&ids) {
        Some(filter) => filter,
        None => return Vec::new(),
    };

    let sql = format!(
        r#"
        WITH requested AS (
            SELECT id AS image_id
            FROM images
            WHERE {}
        ),
        requested_membership AS (
            SELECT r.image_id, m.group_id
            FROM requested r
            LEFT JOIN similar_photo_group_member m
              ON m.image_id = r.image_id
             AND m.algorithm_version = ?1
        ),
        expanded AS (
            SELECT
                gm.image_id,
                gm.group_id,
                gm.representative_id,
                gm.member_rank
            FROM requested_membership rm
            JOIN similar_photo_group_member gm
              ON gm.group_id = rm.group_id
             AND gm.algorithm_version = ?1
            UNION
            SELECT
                rm.image_id,
                rm.image_id AS group_id,
                rm.image_id AS representative_id,
                0 AS member_rank
            FROM requested_membership rm
            WHERE rm.group_id IS NULL
        )
        SELECT image_id, group_id, representative_id, member_rank
        FROM expanded
        ORDER BY group_id, member_rank, image_id
        "#,
        id_filter
    );

    let mut stmt = match conn.prepare(&sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("similar_photo_stack_members_for_ids: prepare {}", e);
            return fallback();
        }
    };

    let mapped = stmt.query_map(params![algorithm_version], |row| {
        let rank: i64 = row.get(3)?;
        Ok(SimilarPhotoStackMember {
            image_id: row.get(0)?,
            group_id: row.get(1)?,
            representative_id: row.get(2)?,
            member_rank: rank.max(0) as u32,
        })
    });

    match mapped {
        Ok(iter) => {
            let members = iter.filter_map(|r| r.ok()).collect::<Vec<_>>();
            if members.is_empty() {
                fallback()
            } else {
                members
            }
        }
        Err(e) => {
            eprintln!("similar_photo_stack_members_for_ids: query {}", e);
            fallback()
        }
    }
}

pub async fn similar_photo_stack_members_for_ids(
    ids: Vec<i64>,
    algorithm_version: String,
) -> Vec<SimilarPhotoStackMember> {
    if ids.is_empty() {
        return Vec::new();
    }

    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return ids
                .iter()
                .map(|id| SimilarPhotoStackMember {
                    image_id: *id,
                    group_id: *id,
                    representative_id: *id,
                    member_rank: 0,
                })
                .collect();
        }
    };

    similar_photo_stack_members_for_ids_impl(conn, &ids, &algorithm_version)
}

/// Distinct collection names — labels carrying `collection = TRUE` on any row,
/// read from the RAW `keyword` table (membership is independent of search
/// visibility). The gallery "Select Collection" picker's autofill/dropdown —
/// collections ONLY, the deliberate opposite of `keyword_labels` (the Add
/// dialog's flag-agnostic list). A collection whose members were all removed
/// (every `collection` switch flipped back FALSE) simply doesn't appear here;
/// its label still re-suggests in the Add dialog and re-creates by name.
pub async fn collection_labels() -> Vec<String> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    let mut stmt = match conn
        .prepare("SELECT DISTINCT label FROM keyword WHERE collection = TRUE ORDER BY label")
    {
        Ok(s) => s,
        Err(e) => {
            eprintln!("collection_labels: prepare {}", e);
            return Vec::new();
        }
    };

    let mapped = stmt.query_map([], |row| {
        let label: String = row.get(0)?;
        Ok(label)
    });

    match mapped {
        Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
        Err(e) => {
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
pub async fn add_images_to_collections(ids: Vec<i64>, labels: Vec<String>) -> u64 {
    if ids.is_empty() {
        return 0;
    }

    // A collection name is a single flat segment — trim, drop empties and any that
    // carry the path separator (mirrors keyword_materialized_rows), then de-dupe.
    let mut clean: Vec<String> = Vec::new();
    for label in &labels {
        let trimmed = label.trim();
        if trimmed.is_empty() || trimmed.contains(KEYWORD_PATH_SEPARATOR) {
            continue;
        }
        let s = trimmed.to_string();
        if !clean.contains(&s) {
            clean.push(s);
        }
    }
    if clean.is_empty() {
        return 0;
    }

    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;") {
        eprintln!("add_images_to_collections: begin failed: {}", e);
        return 0;
    }

    let mut changed: u64 = 0;
    for id in &ids {
        for label in &clean {
            // Flip ON any existing visible row(s) with this label that aren't
            // already a collection (FALSE or, on migrated catalogues, NULL).
            let flipped = match conn.execute(
                "UPDATE keyword SET collection = TRUE \
                 WHERE image_id = ? AND label = ? AND status = 1 \
                   AND (collection = FALSE OR collection IS NULL)",
                params![id, label],
            ) {
                Ok(n) => n as u64,
                Err(e) => {
                    eprintln!("add_images_to_collections: update failed: {}", e);
                    let _ = conn.execute_batch("ROLLBACK;");
                    return 0;
                }
            };
            if flipped > 0 {
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
            if already.is_ok() {
                continue;
            }

            // No visible row with this label → insert a flat collection row.
            match conn.execute(
                // Carry is_video from the image (was defaulting to FALSE) — S72 closeout.
                "INSERT INTO keyword (image_id, label, path, status, origin, created_at, collection, is_video) \
                 VALUES (?, ?, ?, 1, ?, CURRENT_TIMESTAMP, TRUE, COALESCE((SELECT is_video FROM images WHERE id = ?), FALSE))",
                params![id, label, label, KEYWORD_ORIGIN_USER, id],
            ) {
                Ok(_) => changed += 1,
                Err(e) => {
                    eprintln!("add_images_to_collections: insert failed: {}", e);
                    let _ = conn.execute_batch("ROLLBACK;");
                    return 0;
                }
            }
        }
    }

    if let Err(e) = conn.execute_batch("COMMIT;") {
        eprintln!("add_images_to_collections: commit failed: {}", e);
        return 0;
    }
    changed
}

/// The body of `assign_color_keyword_for_ids`, against an explicit connection
/// so the unit tests can drive it on an in-memory database (the saved-query
/// impl/wrapper pattern). Mirrors `add_images_to_collections` for the COLOR
/// switch: per image, flip `color = TRUE` on any existing visible row carrying
/// this label, else insert a flat row (path = label) with `color = TRUE`.
/// Idempotent; one transaction; returns rows changed (flipped + inserted).
fn assign_color_keyword_for_ids_impl(conn: &Connection, ids: &[i64], label: &str) -> u64 {
    let trimmed = label.trim();
    if ids.is_empty() || trimmed.is_empty() || trimmed.contains(KEYWORD_PATH_SEPARATOR) {
        return 0;
    }

    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;") {
        eprintln!("assign_color_keyword_for_ids: begin failed: {}", e);
        return 0;
    }

    let mut changed: u64 = 0;
    for id in ids {
        // Flip ON any existing visible row(s) with this label that aren't
        // already marked (FALSE or, on migrated catalogues, NULL).
        let flipped = match conn.execute(
            "UPDATE keyword SET color = TRUE \
             WHERE image_id = ? AND label = ? AND status = 1 \
               AND (color = FALSE OR color IS NULL)",
            params![id, trimmed],
        ) {
            Ok(n) => n as u64,
            Err(e) => {
                eprintln!("assign_color_keyword_for_ids: update failed: {}", e);
                let _ = conn.execute_batch("ROLLBACK;");
                return 0;
            }
        };
        if flipped > 0 {
            changed += flipped;
            continue;
        }

        // Already marked (a visible row with color already TRUE)? no-op.
        let already: Result<i64, _> = conn.query_row(
            "SELECT 1 FROM keyword \
             WHERE image_id = ? AND label = ? AND status = 1 AND color = TRUE LIMIT 1",
            params![id, trimmed],
            |r| r.get(0),
        );
        if already.is_ok() {
            continue;
        }

        // No visible row with this label → insert a flat color row.
        match conn.execute(
            // Carry is_video from the image (was defaulting to FALSE) — S72 closeout.
            "INSERT INTO keyword (image_id, label, path, status, origin, created_at, collection, color, is_video) \
             VALUES (?, ?, ?, 1, ?, CURRENT_TIMESTAMP, FALSE, TRUE, COALESCE((SELECT is_video FROM images WHERE id = ?), FALSE))",
            params![id, trimmed, trimmed, KEYWORD_ORIGIN_USER, id],
        ) {
            Ok(_) => changed += 1,
            Err(e) => {
                eprintln!("assign_color_keyword_for_ids: insert failed: {}", e);
                let _ = conn.execute_batch("ROLLBACK;");
                return 0;
            }
        }
    }

    if let Err(e) = conn.execute_batch("COMMIT;") {
        eprintln!("assign_color_keyword_for_ids: commit failed: {}", e);
        return 0;
    }
    changed
}

/// FFI: mark a custom color-label keyword on a batch of images (S66 — the
/// Lightroom import's color pass). A custom color label IS a `keyword` row
/// carrying `color = TRUE` — the third independent switch on the row, exactly
/// parallel to `collection`. The five STANDARD color names never come here
/// (they live in `images.color_label`); the reader's SQL filters them out.
pub async fn assign_color_keyword_for_ids(ids: Vec<i64>, label: String) -> u64 {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };
    assign_color_keyword_for_ids_impl(conn, &ids, &label)
}

/// Remove images from one or more collections — the scope-gated right-click
/// "Remove from <collection>" (S65). Per image x label, flip the `collection`
/// switch OFF on any row carrying it. The deliberate asymmetry with
/// `add_images_to_collections`: the add flips ON only VISIBLE rows (status = 1),
/// but removal ignores `status` entirely — membership counts even on a hidden
/// keyword row (`collection_is` reads the raw table), so a status filter here
/// would strand a hidden-keyword member in the collection with no way out.
/// Rows are NEVER deleted (`status`/visibility untouched — a row that also
/// serves keyword search keeps doing so); a collection whose every switch flips
/// back FALSE simply stops appearing in `collection_labels`. Idempotent (not a
/// member -> no-op). One transaction. Returns the number of rows flipped.
pub async fn remove_images_from_collections(ids: Vec<i64>, labels: Vec<String>) -> u64 {
    if ids.is_empty() {
        return 0;
    }

    // Same label hygiene as the add — trim, drop empties and any carrying the
    // path separator, de-dupe. (Scope labels arrive exact from the picker, but
    // the FFI stays defensive.)
    let mut clean: Vec<String> = Vec::new();
    for label in &labels {
        let trimmed = label.trim();
        if trimmed.is_empty() || trimmed.contains(KEYWORD_PATH_SEPARATOR) {
            continue;
        }
        let s = trimmed.to_string();
        if !clean.contains(&s) {
            clean.push(s);
        }
    }
    if clean.is_empty() {
        return 0;
    }

    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;") {
        eprintln!("remove_images_from_collections: begin failed: {}", e);
        return 0;
    }

    let mut changed: u64 = 0;
    for id in &ids {
        for label in &clean {
            // Flip OFF any row carrying membership — deliberately NO status
            // filter (see the doc comment).
            match conn.execute(
                "UPDATE keyword SET collection = FALSE \
                 WHERE image_id = ? AND label = ? AND collection = TRUE",
                params![id, label],
            ) {
                Ok(n) => changed += n as u64,
                Err(e) => {
                    eprintln!("remove_images_from_collections: update failed: {}", e);
                    let _ = conn.execute_batch("ROLLBACK;");
                    return 0;
                }
            }
        }
    }

    if let Err(e) = conn.execute_batch("COMMIT;") {
        eprintln!("remove_images_from_collections: commit failed: {}", e);
        return 0;
    }
    changed
}

/// Hidden (removed) keyword rows for one image — the recovery surface. Reads the
/// RAW table (not the view), newest-hidden first.
pub async fn hidden_keywords_for_image(image_id: i64) -> Vec<KeywordRow> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    let mut stmt = match conn.prepare(
        "SELECT label, path, status, origin, CAST(created_at AS VARCHAR), CAST(hidden_at AS VARCHAR) \
         FROM keyword WHERE image_id = ? AND status = 0 ORDER BY hidden_at DESC",
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("hidden_keywords_for_image: prepare {}", e);
            return Vec::new();
        }
    };

    let mapped = stmt.query_map(params![image_id], |row| {
        Ok(KeywordRow {
            label: row.get(0)?,
            path: row.get(1)?,
            status: row.get(2)?,
            origin: row.get(3)?,
            created_at: row.get(4)?,
            hidden_at: row.get(5)?,
        })
    });

    match mapped {
        Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
        Err(e) => {
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
pub async fn reparent_keyword(source_path: Vec<String>, new_parent: Vec<String>) -> u64 {
    if source_path.is_empty() {
        return 0;
    }
    for s in source_path.iter().chain(new_parent.iter()) {
        if s.trim().is_empty() || s.contains(KEYWORD_PATH_SEPARATOR) {
            eprintln!("reparent_keyword: invalid segment");
            return 0;
        }
    }

    let source_joined = source_path.join(KEYWORD_PATH_SEPARATOR);
    let source_prefix = format!("{}{}", source_joined, KEYWORD_PATH_SEPARATOR);
    let last_seg = source_path.last().unwrap().trim().to_string();
    let new_root = if new_parent.is_empty() {
        last_seg
    } else {
        format!(
            "{}{}{}",
            new_parent.join(KEYWORD_PATH_SEPARATOR),
            KEYWORD_PATH_SEPARATOR,
            last_seg
        )
    };
    // char count + 1 = 1-indexed position of the first char AFTER source_joined
    // ("" for the subtree root, "␟Yellow" for a descendant).
    let suffix_start = (source_joined.chars().count() + 1) as i64;
    let new_parent_rows = keyword_materialized_rows(&new_parent);

    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;") {
        eprintln!("reparent_keyword: begin failed: {}", e);
        return 0;
    }

    // (1) Ensure the new-parent ancestor chain exists for every affected image.
    for (label, path) in &new_parent_rows {
        let sql = "INSERT INTO keyword (image_id, label, path, status, origin, created_at, is_video) \
                   SELECT k.image_id, ?, ?, 1, \
                          (CASE WHEN SUM(CASE WHEN (k.origin & 1) <> 0 THEN 1 ELSE 0 END) > 0 THEN 1 ELSE 0 END) + \
                          (CASE WHEN SUM(CASE WHEN (k.origin & 2) <> 0 THEN 1 ELSE 0 END) > 0 THEN 2 ELSE 0 END), \
                          CURRENT_TIMESTAMP, k.is_video \
                   FROM keyword k \
                   WHERE k.status = 1 AND (k.path = ? OR starts_with(k.path, ?)) \
                   AND NOT EXISTS (SELECT 1 FROM keyword existing \
                                   WHERE existing.image_id = k.image_id \
                                     AND existing.status = 1 \
                                     AND existing.path = ?) \
                   GROUP BY k.image_id, k.is_video";
        if let Err(e) = conn.execute(
            sql,
            params![label, path, source_joined, source_prefix, path],
        ) {
            eprintln!("reparent_keyword: ancestor insert failed: {}", e);
            let _ = conn.execute_batch("ROLLBACK;");
            return 0;
        }
    }

    // (2) Re-root the moved subtree (label unchanged; path = new_root || suffix).
    let move_sql = "INSERT INTO keyword (image_id, label, path, status, origin, created_at, is_video) \
                    SELECT image_id, label, ? || substr(path, ?), 1, origin, CURRENT_TIMESTAMP, is_video FROM keyword \
                    WHERE status = 1 AND (path = ? OR starts_with(path, ?))";
    if let Err(e) = conn.execute(
        move_sql,
        params![new_root, suffix_start, source_joined, source_prefix],
    ) {
        eprintln!("reparent_keyword: move insert failed: {}", e);
        let _ = conn.execute_batch("ROLLBACK;");
        return 0;
    }

    // (3) Hide the old subtree.
    let hide_sql = "UPDATE keyword SET status = 0, hidden_at = CURRENT_TIMESTAMP \
                    WHERE status = 1 AND (path = ? OR starts_with(path, ?))";
    let changed = match conn.execute(hide_sql, params![source_joined, source_prefix]) {
        Ok(c) => c as u64,
        Err(e) => {
            eprintln!("reparent_keyword: hide failed: {}", e);
            let _ = conn.execute_batch("ROLLBACK;");
            return 0;
        }
    };

    if let Err(e) = conn.execute_batch("COMMIT;") {
        eprintln!("reparent_keyword: commit failed: {}", e);
        return 0;
    }
    changed
}

/// Rename a keyword node GLOBALLY (its label + the corresponding path segment),
/// cascading to descendants. Same machinery as reparent, but the subtree is
/// re-rooted under the SAME parent with the new label (so no ancestor insert is
/// needed — the parent already exists). One transaction. Returns rows hidden.
pub async fn rename_keyword(target_path: Vec<String>, new_label: String) -> u64 {
    if target_path.is_empty() {
        return 0;
    }
    let new_label = new_label.trim().to_string();
    if new_label.is_empty() || new_label.contains(KEYWORD_PATH_SEPARATOR) {
        eprintln!("rename_keyword: invalid new label");
        return 0;
    }
    for s in target_path.iter() {
        if s.trim().is_empty() || s.contains(KEYWORD_PATH_SEPARATOR) {
            eprintln!("rename_keyword: invalid segment");
            return 0;
        }
    }

    let target_joined = target_path.join(KEYWORD_PATH_SEPARATOR);
    let target_prefix = format!("{}{}", target_joined, KEYWORD_PATH_SEPARATOR);
    let suffix_start = (target_joined.chars().count() + 1) as i64;
    let parent = &target_path[..target_path.len() - 1];
    let new_root = if parent.is_empty() {
        new_label.clone()
    } else {
        format!(
            "{}{}{}",
            parent.join(KEYWORD_PATH_SEPARATOR),
            KEYWORD_PATH_SEPARATOR,
            new_label
        )
    };

    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;") {
        eprintln!("rename_keyword: begin failed: {}", e);
        return 0;
    }

    // Re-root the subtree; the root row's label becomes new_label (descendants
    // keep theirs), path re-rooted for the whole subtree.
    let move_sql = "INSERT INTO keyword (image_id, label, path, status, origin, created_at, is_video) \
                    SELECT image_id, \
                           CASE WHEN path = ? THEN ? ELSE label END, \
                           ? || substr(path, ?), 1, origin, CURRENT_TIMESTAMP, is_video FROM keyword \
                    WHERE status = 1 AND (path = ? OR starts_with(path, ?))";
    if let Err(e) = conn.execute(
        move_sql,
        params![
            target_joined,
            new_label,
            new_root,
            suffix_start,
            target_joined,
            target_prefix
        ],
    ) {
        eprintln!("rename_keyword: move insert failed: {}", e);
        let _ = conn.execute_batch("ROLLBACK;");
        return 0;
    }

    let hide_sql = "UPDATE keyword SET status = 0, hidden_at = CURRENT_TIMESTAMP \
                    WHERE status = 1 AND (path = ? OR starts_with(path, ?))";
    let changed = match conn.execute(hide_sql, params![target_joined, target_prefix]) {
        Ok(c) => c as u64,
        Err(e) => {
            eprintln!("rename_keyword: hide failed: {}", e);
            let _ = conn.execute_batch("ROLLBACK;");
            return 0;
        }
    };

    if let Err(e) = conn.execute_batch("COMMIT;") {
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
pub struct MergeChunkResult {
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
fn merge_records_into(conn: &Connection, records: &[ImageMetadata]) -> MergeChunkResult {
    let mut out = MergeChunkResult {
        inserted: 0,
        updated: 0,
        image_ids: Vec::with_capacity(records.len()),
    };
    if records.is_empty() {
        return out;
    }

    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;") {
        eprintln!("merge_records_into: begin failed: {}", e);
        return out;
    }

    for record in records {
        // Pre-check: an existing row keeps its id (UPDATE); else a fresh INSERT.
        let existing: Result<i64, _> = conn.query_row(
            "SELECT id FROM images WHERE file_path = ?1",
            params![record.file_path],
            |r| r.get(0),
        );

        // Each arm yields Result<(was_insert, id), Error>.
        let row_result: Result<(bool, i64), _> = if let Ok(id) = existing {
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
                    color_label = COALESCE(?17, color_label), \
                    rotation    = COALESCE(?18, rotation), \
                    external_source_id = COALESCE(?19, external_source_id) \
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
                    record.rotation.map(|v| v as i64),
                    record.external_source_id,
                ],
            )
            .map(|_| (false, id))
        } else {
            // INSERT — mirror ingest_metadata's column set + the canonical
            // directory_path expression; RETURNING id (a plain insert is reliable).
            let parsed = parse_filename(record.file_name.clone());
            let image_kind_str = match parsed.kind {
                ImageKind::Jpeg => "jpeg",
                ImageKind::Raw => "raw",
                ImageKind::Other => "other",
                ImageKind::Heif => "heif",
                ImageKind::Dng => "dng",
                ImageKind::Psd => "psd",
                ImageKind::Tiff => "tiff",
                ImageKind::Png => "png",
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
                    rating, flag, color_label, rotation, external_source_id, \
                    is_video, \
                    duration_seconds, frame_rate, video_kind, video_codec, video_bitrate, \
                    color_primaries, color_transfer, color_matrix, color_range, dv_profile, \
                    has_audio, audio_codec, audio_channels, audio_sample_rate, audio_bitrate, \
                    live_photo_id \
                 ) VALUES ( \
                    ?1, ?2, ?3, ?4, ?5, ?6, \
                    SUBSTRING(?1, 1, LENGTH(?1) - INSTR(REVERSE(?1), '/')), \
                    ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, \
                    ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29, ?30, ?31, \
                    ?32, \
                    ?33, ?34, ?35, ?36, ?37, \
                    ?38, ?39, ?40, ?41, ?42, \
                    ?43, ?44, ?45, ?46, ?47, \
                    ?48 \
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
                    record.rotation.unwrap_or(0),
                    record.external_source_id,
                    // Video / unified-media columns (mirror ingest_metadata) — false/None for stills.
                    record.is_video,
                    record.duration_seconds,
                    record.frame_rate,
                    record.video_kind,
                    record.video_codec,
                    record.video_bitrate,
                    record.color_primaries,
                    record.color_transfer,
                    record.color_matrix,
                    record.color_range,
                    record.dv_profile,
                    record.has_audio,
                    record.audio_codec,
                    record.audio_channels,
                    record.audio_sample_rate,
                    record.audio_bitrate,
                    record.live_photo_id,
                ],
                |r| r.get::<_, i64>(0),
            )
            .map(|new_id| (true, new_id))
        };

        match row_result {
            Ok((true, id)) => {
                out.inserted += 1;
                out.image_ids.push(id);
            }
            Ok((false, id)) => {
                out.updated += 1;
                out.image_ids.push(id);
            }
            Err(e) => {
                eprintln!(
                    "merge_records_into: row failed for {}: {}",
                    record.file_path, e
                );
                let _ = conn.execute_batch("ROLLBACK;");
                return MergeChunkResult {
                    inserted: 0,
                    updated: 0,
                    image_ids: Vec::new(),
                };
            }
        }
    }

    if let Err(e) = conn.execute_batch("COMMIT;") {
        eprintln!("merge_records_into: commit failed: {}", e);
        return MergeChunkResult {
            inserted: 0,
            updated: 0,
            image_ids: Vec::new(),
        };
    }
    out
}

/// FFI entry: merge a chunk of Lightroom-sourced image records into the
/// catalogue (matched on file_path). Reuses `ImageMetadata` as the input — it is
/// an exact superset of what LR provides (§10). Returns per-chunk stats + the
/// resulting catalogue ids (aligned to input order) for the keyword pass.
pub async fn merge_lightroom_records(records: Vec<ImageMetadata>) -> MergeChunkResult {
    if records.is_empty() {
        return MergeChunkResult {
            inserted: 0,
            updated: 0,
            image_ids: Vec::new(),
        };
    }
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("merge_lightroom_records: catalogue not initialized");
            return MergeChunkResult {
                inserted: 0,
                updated: 0,
                image_ids: Vec::new(),
            };
        }
    };
    merge_records_into(conn, &records)
}

/// One Lightroom-sourced VIDEO record (input to merge_lightroom_videos). No
/// `ImageMetadata` analogue — videos carry duration/frame_rate/has_audio/
/// video_kind and no EXIF. `directory_path` is derived Rust-side (like images).
#[derive(Debug, Clone)]
pub struct LightroomVideoRecord {
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
fn merge_videos_into(conn: &Connection, records: &[LightroomVideoRecord]) -> MergeChunkResult {
    let mut out = MergeChunkResult {
        inserted: 0,
        updated: 0,
        image_ids: Vec::with_capacity(records.len()),
    };
    if records.is_empty() {
        return out;
    }

    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;") {
        eprintln!("merge_videos_into: begin failed: {}", e);
        return out;
    }

    for record in records {
        let existing: Result<i64, _> = conn.query_row(
            "SELECT id FROM videos WHERE file_path = ?1",
            params![record.file_path],
            |r| r.get(0),
        );

        let row_result: Result<(bool, i64), _> = if let Ok(id) = existing {
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
            )
            .map(|_| (false, id))
        } else {
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
            )
            .map(|new_id| (true, new_id))
        };

        match row_result {
            Ok((true, id)) => {
                out.inserted += 1;
                out.image_ids.push(id);
            }
            Ok((false, id)) => {
                out.updated += 1;
                out.image_ids.push(id);
            }
            Err(e) => {
                eprintln!(
                    "merge_videos_into: row failed for {}: {}",
                    record.file_path, e
                );
                let _ = conn.execute_batch("ROLLBACK;");
                return MergeChunkResult {
                    inserted: 0,
                    updated: 0,
                    image_ids: Vec::new(),
                };
            }
        }
    }

    if let Err(e) = conn.execute_batch("COMMIT;") {
        eprintln!("merge_videos_into: commit failed: {}", e);
        return MergeChunkResult {
            inserted: 0,
            updated: 0,
            image_ids: Vec::new(),
        };
    }
    out
}

/// FFI entry: merge a chunk of Lightroom-sourced VIDEO records into the `videos`
/// table (matched on file_path). Returns per-chunk stats + the video-row ids.
pub async fn merge_lightroom_videos(records: Vec<LightroomVideoRecord>) -> MergeChunkResult {
    if records.is_empty() {
        return MergeChunkResult {
            inserted: 0,
            updated: 0,
            image_ids: Vec::new(),
        };
    }
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("merge_lightroom_videos: catalogue not initialized");
            return MergeChunkResult {
                inserted: 0,
                updated: 0,
                image_ids: Vec::new(),
            };
        }
    };
    merge_videos_into(conn, &records)
}

#[cfg(test)]
mod face_embedding_tests {
    use super::*;

    fn embedding_record(face_observation_id: i64, vector: Vec<f32>) -> FaceEmbeddingVectorRecord {
        FaceEmbeddingVectorRecord {
            face_observation_id,
            image_id: face_observation_id + 10,
            analyzed_image_id: face_observation_id + 20,
            face_index: 0,
            model_name: "test-model".to_string(),
            model_version: "test-version".to_string(),
            preprocessing_version: "test-preprocessing".to_string(),
            input_size: 112,
            color_order: "RGB".to_string(),
            normalization: "test-normalization".to_string(),
            embedding_dimension: vector.len() as u32,
            embedding_l2_norm: 1.0,
            vector,
        }
    }

    #[test]
    fn face_embedding_batch_accepts_fixed_size_float_vectors() {
        let records = vec![
            embedding_record(1, vec![0.1, 0.2, 0.3]),
            embedding_record(2, vec![0.4, 0.5, 0.6]),
        ];

        let batch_iterator = face_embedding_batch(&records, 3).expect("embedding batch");
        let batches = batch_iterator
            .collect::<Result<Vec<_>, _>>()
            .expect("record batches");

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 2);
        let vectors = batches[0]
            .column_by_name("vector")
            .and_then(|array| array.as_any().downcast_ref::<FixedSizeListArray>())
            .expect("vector column");
        assert_eq!(vectors.value_length(), 3);
        assert_eq!(vectors.values().len(), 6);
    }
}

#[cfg(test)]
mod face_cluster_tests {
    use super::*;
    use duckdb::Connection;

    fn setup() -> Connection {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE face_cluster_run (
                 run_id TEXT PRIMARY KEY,
                 face_algorithm_version TEXT NOT NULL,
                 model_version TEXT NOT NULL,
                 preprocessing_version TEXT NOT NULL,
                 threshold DOUBLE NOT NULL,
                 cluster_count INTEGER NOT NULL,
                 member_count INTEGER NOT NULL,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
             );

             CREATE TABLE face_cluster_member (
                 run_id TEXT NOT NULL,
                 cluster_id INTEGER NOT NULL,
                 face_observation_id INTEGER NOT NULL,
                 image_id INTEGER NOT NULL,
                 analyzed_image_id INTEGER NOT NULL,
                 face_index INTEGER NOT NULL,
                 member_rank INTEGER NOT NULL,
                 cluster_size INTEGER NOT NULL,
                 nearest_neighbor_face_observation_id INTEGER,
                 nearest_neighbor_cosine DOUBLE,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                 PRIMARY KEY (run_id, face_observation_id)
             );

             CREATE TABLE images (
                 id INTEGER PRIMARY KEY,
                 is_video BOOLEAN NOT NULL DEFAULT FALSE
             );

             CREATE SEQUENCE keyword_id_seq START 1;
             CREATE TABLE keyword (
                 id INTEGER PRIMARY KEY DEFAULT nextval('keyword_id_seq'),
                 image_id INTEGER NOT NULL,
                 label TEXT NOT NULL,
                 path TEXT NOT NULL,
                 status INTEGER NOT NULL DEFAULT 1,
                 origin INTEGER NOT NULL DEFAULT 1,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                 hidden_at TIMESTAMP,
                 collection BOOLEAN NOT NULL DEFAULT FALSE,
                 color BOOLEAN NOT NULL DEFAULT FALSE,
                 is_video BOOLEAN NOT NULL DEFAULT FALSE
             );

             CREATE SEQUENCE person_id_seq START 1;
             CREATE TABLE person (
                 id INTEGER PRIMARY KEY DEFAULT nextval('person_id_seq'),
                 display_name TEXT NOT NULL,
                 normalized_name TEXT NOT NULL,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                 updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
             );
             CREATE UNIQUE INDEX idx_person_normalized_name ON person(normalized_name);

             CREATE TABLE person_face_assignment (
                 face_observation_id INTEGER PRIMARY KEY,
                 person_id INTEGER NOT NULL,
                 image_id INTEGER NOT NULL,
                 analyzed_image_id INTEGER NOT NULL,
                 face_index INTEGER NOT NULL,
                 assignment_source TEXT NOT NULL,
                 face_cluster_run_id TEXT,
                 face_cluster_id INTEGER,
                 model_version TEXT,
                 preprocessing_version TEXT,
                 threshold DOUBLE,
                 confidence_cosine DOUBLE,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                 updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
             );",
        )
        .expect("schema");
        conn
    }

    fn run_record() -> FaceClusterRunRecord {
        FaceClusterRunRecord {
            run_id: "test-run".to_string(),
            face_algorithm_version: "vision-face-v1".to_string(),
            model_version: "auraface-coreml-v1".to_string(),
            preprocessing_version: "center-crop-v1".to_string(),
            threshold: 0.55,
            cluster_count: 1,
            member_count: 0,
        }
    }

    fn member(
        face_observation_id: i64,
        cluster_id: i64,
        member_rank: u32,
    ) -> FaceClusterMemberRecord {
        FaceClusterMemberRecord {
            run_id: "test-run".to_string(),
            cluster_id,
            face_observation_id,
            image_id: face_observation_id + 100,
            analyzed_image_id: face_observation_id + 200,
            face_index: member_rank,
            member_rank,
            cluster_size: 2,
            nearest_neighbor_face_observation_id: Some(11),
            nearest_neighbor_cosine: Some(0.76),
        }
    }

    #[test]
    fn replace_face_cluster_run_filters_invalid_members_and_reads_back_ordered_rows() {
        let conn = setup();
        let inserted = replace_face_cluster_run_impl(
            &conn,
            run_record(),
            vec![
                member(12, 1, 1),
                FaceClusterMemberRecord {
                    cluster_size: 1,
                    ..member(99, 1, 2)
                },
                member(11, 1, 0),
            ],
        );

        assert_eq!(inserted, 2);

        let stored_member_count: i64 = conn
            .query_row(
                "SELECT member_count FROM face_cluster_run WHERE run_id = 'test-run'",
                [],
                |row| row.get(0),
            )
            .expect("run member count");
        assert_eq!(stored_member_count, 2);

        let members = face_cluster_members_impl(&conn, "test-run");
        assert_eq!(members.len(), 2);
        assert_eq!(members[0].face_observation_id, 11);
        assert_eq!(members[1].face_observation_id, 12);
        assert_eq!(members[0].nearest_neighbor_face_observation_id, Some(11));
        assert_eq!(members[0].nearest_neighbor_cosine, Some(0.76));

        let runs = face_cluster_runs_impl(&conn, 10);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].run_id, "test-run");
        assert_eq!(runs[0].threshold, 0.55);
        assert_eq!(runs[0].cluster_count, 1);
        assert_eq!(runs[0].member_count, 2);
    }

    #[test]
    fn replace_face_cluster_run_clears_existing_members_for_same_run() {
        let conn = setup();
        assert_eq!(
            replace_face_cluster_run_impl(
                &conn,
                run_record(),
                vec![member(11, 1, 0), member(12, 1, 1)]
            ),
            2
        );
        assert_eq!(
            replace_face_cluster_run_impl(
                &conn,
                run_record(),
                vec![member(21, 2, 0), member(22, 2, 1)]
            ),
            2
        );

        let members = face_cluster_members_impl(&conn, "test-run");
        assert_eq!(
            members
                .iter()
                .map(|member| member.face_observation_id)
                .collect::<Vec<_>>(),
            vec![21, 22]
        );
    }

    #[test]
    fn accept_face_cluster_as_person_assigns_faces_and_mirrors_people_keyword() {
        let conn = setup();
        assert_eq!(
            replace_face_cluster_run_impl(
                &conn,
                run_record(),
                vec![member(11, 1, 0), member(12, 1, 1)]
            ),
            2
        );
        conn.execute(
            "INSERT INTO images (id, is_video) VALUES (111, FALSE), (112, FALSE)",
            [],
        )
        .expect("images");

        let result = accept_face_cluster_as_person_impl(&conn, "test-run", 1, "  James   Wagner  ");
        assert_eq!(result.status, "accepted");
        assert_eq!(result.person_name, "James Wagner");
        assert_eq!(result.assigned_face_count, 2);
        assert_eq!(result.keyword_image_count, 2);
        assert_eq!(result.keyword_row_count, 4);
        assert_eq!(
            result.keyword_path,
            ["People", "James Wagner"].join(KEYWORD_PATH_SEPARATOR)
        );

        let person_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM person WHERE normalized_name = 'james wagner'",
                [],
                |row| row.get(0),
            )
            .expect("person count");
        assert_eq!(person_count, 1);

        let assignment_count: i64 = conn
            .query_row(
                "SELECT COUNT(*)
                 FROM person_face_assignment
                 WHERE face_cluster_run_id = 'test-run'
                   AND face_cluster_id = 1
                   AND model_version = 'auraface-coreml-v1'
                   AND preprocessing_version = 'center-crop-v1'
                   AND threshold = 0.55
                   AND assignment_source = 'cluster_accept'",
                [],
                |row| row.get(0),
            )
            .expect("assignment count");
        assert_eq!(assignment_count, 2);

        let keyword_count: i64 = conn
            .query_row(
                "SELECT COUNT(*)
                 FROM keyword
                 WHERE image_id IN (111, 112)
                   AND status = 1
                   AND origin = 1
                   AND path IN ('People', ?1)",
                params![["People", "James Wagner"].join(KEYWORD_PATH_SEPARATOR)],
                |row| row.get(0),
            )
            .expect("keyword count");
        assert_eq!(keyword_count, 4);

        let repeated = accept_face_cluster_as_person_impl(&conn, "test-run", 1, "James Wagner");
        assert_eq!(repeated.status, "accepted");
        assert_eq!(repeated.person_id, result.person_id);
        assert_eq!(repeated.keyword_row_count, 0);

        let duplicate_keyword_count: i64 = conn
            .query_row(
                "SELECT COUNT(*)
                 FROM keyword
                 WHERE image_id IN (111, 112)
                   AND status = 1
                   AND path IN ('People', ?1)",
                params![["People", "James Wagner"].join(KEYWORD_PATH_SEPARATOR)],
                |row| row.get(0),
            )
            .expect("duplicate keyword count");
        assert_eq!(duplicate_keyword_count, 4);
    }
}

#[cfg(test)]
mod face_observation_tests {
    use super::*;
    use duckdb::Connection;

    fn setup() -> Connection {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE SEQUENCE face_observation_id_seq START 1;
             CREATE TABLE face_observation (
                 id INTEGER PRIMARY KEY DEFAULT nextval('face_observation_id_seq'),
                 image_id INTEGER NOT NULL,
                 analyzed_image_id INTEGER NOT NULL,
                 face_index INTEGER NOT NULL,
                 algorithm_version TEXT NOT NULL,
                 analysis_run_id TEXT NOT NULL,
                 bounding_box_x DOUBLE NOT NULL,
                 bounding_box_y DOUBLE NOT NULL,
                 bounding_box_width DOUBLE NOT NULL,
                 bounding_box_height DOUBLE NOT NULL,
                 detection_confidence DOUBLE,
                 face_capture_quality DOUBLE,
                 face_focus_score DOUBLE,
                 left_eye_open_score DOUBLE,
                 right_eye_open_score DOUBLE,
                 eyes_open_score DOUBLE,
                 blink_risk_score DOUBLE,
                 left_eye_x DOUBLE,
                 left_eye_y DOUBLE,
                 right_eye_x DOUBLE,
                 right_eye_y DOUBLE,
                 nose_x DOUBLE,
                 nose_y DOUBLE,
                 mouth_left_x DOUBLE,
                 mouth_left_y DOUBLE,
                 mouth_right_x DOUBLE,
                 mouth_right_y DOUBLE,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                 UNIQUE (image_id, algorithm_version, face_index)
             );",
        )
        .expect("face observation DDL");
        conn
    }

    fn observation(face_index: u32) -> FaceObservationResult {
        FaceObservationResult {
            face_index,
            bounding_box_x: 0.1,
            bounding_box_y: 0.2,
            bounding_box_width: 0.3,
            bounding_box_height: 0.4,
            detection_confidence: Some(0.9),
            face_capture_quality: Some(0.8),
            face_focus_score: Some(123.0),
            left_eye_open_score: Some(0.2),
            right_eye_open_score: Some(0.21),
            eyes_open_score: Some(0.205),
            blink_risk_score: Some(0.0),
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

    fn result(face_observations: Vec<FaceObservationResult>) -> FocusAnalysisResult {
        FocusAnalysisResult {
            id: 10,
            focus_score: Some(1.0),
            focus_basis: Some("human_face".to_string()),
            algorithm_version: "faces-v1".to_string(),
            analysis_run_id: "run-1".to_string(),
            status: "complete".to_string(),
            focus_human_score: Some(1.0),
            focus_animal_score: None,
            focus_foreground_score: None,
            focus_saliency_score: None,
            focus_animal_pose_score: None,
            focus_whole_image_score: Some(1.0),
            face_count: Some(face_observations.len() as i32),
            face_quality_best: Some(0.8),
            face_quality_average: Some(0.8),
            face_quality_min: Some(0.8),
            face_eyes_open_count: Some(1),
            face_blink_risk_count: Some(0),
            auto_keywords: Vec::new(),
            face_observations,
        }
    }

    #[test]
    fn face_observations_replace_current_algorithm_for_each_target() {
        let conn = setup();
        let first = result(vec![observation(0), observation(1)]);
        assert!(replace_face_observations_for_targets(
            &conn,
            &first,
            &[1, 2],
            true
        ));

        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM face_observation", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(count, 4);

        let replacement = result(vec![observation(0)]);
        assert!(replace_face_observations_for_targets(
            &conn,
            &replacement,
            &[1, 2],
            true
        ));

        let rows: Vec<(i64, i64, i64)> = conn
            .prepare(
                "SELECT image_id, analyzed_image_id, face_index
                 FROM face_observation
                 ORDER BY image_id, face_index",
            )
            .unwrap()
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .unwrap()
            .map(|row| row.unwrap())
            .collect();

        assert_eq!(rows, vec![(1, 10, 0), (2, 10, 0)]);
    }

    #[test]
    fn failed_result_clears_stale_face_observations() {
        let conn = setup();
        let complete = result(vec![observation(0)]);
        assert!(replace_face_observations_for_targets(
            &conn,
            &complete,
            &[1],
            true
        ));

        assert!(replace_face_observations_for_targets(
            &conn,
            &complete,
            &[1],
            false
        ));

        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM face_observation", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(count, 0);
    }
}

#[cfg(test)]
mod focus_analysis_queue_tests {
    use super::*;
    use duckdb::Connection;

    fn setup() -> Connection {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE images (
                 id INTEGER PRIMARY KEY,
                 file_path TEXT NOT NULL,
                 file_size BIGINT NOT NULL,
                 image_kind VARCHAR,
                 file_stem VARCHAR,
                 directory_path VARCHAR,
                 is_video BOOLEAN,
                 focus_analysis_status TEXT,
                 focus_algorithm_version TEXT,
                 focus_analysis_attempt_id TEXT
             );
             INSERT INTO images (
                 id, file_path, file_size, image_kind, file_stem, directory_path, is_video,
                 focus_analysis_status, focus_algorithm_version, focus_analysis_attempt_id
             ) VALUES
                 (1, '/a/missing.jpg', 10, 'jpeg', 'missing', '/a', FALSE, NULL, NULL, NULL),
                 (2, '/a/current-complete.jpg', 10, 'jpeg', 'current-complete', '/a', FALSE, 'complete', 'v2', 'old-run'),
                 (3, '/a/stale-complete.jpg', 10, 'jpeg', 'stale-complete', '/a', FALSE, 'complete', 'v1', 'old-run'),
                 (4, '/a/retry-offline.jpg', 10, 'jpeg', 'retry-offline', '/a', FALSE, 'online_only', 'v2', 'old-run'),
                 (5, '/a/same-run-unreadable.jpg', 10, 'jpeg', 'same-run-unreadable', '/a', FALSE, 'unreadable', 'v2', 'run-1'),
                 (6, '/a/retry-failed.jpg', 10, 'jpeg', 'retry-failed', '/a', FALSE, 'failed', 'v2', NULL),
                 (7, '/a/video.mov', 10, 'other', 'video', '/a', TRUE, NULL, NULL, NULL);",
        )
        .expect("focus queue DDL");
        conn
    }

    fn candidate_ids(rows: Vec<FocusAnalysisCandidate>) -> Vec<i64> {
        rows.into_iter().map(|row| row.id).collect()
    }

    #[test]
    fn whole_catalogue_focus_queue_retries_later_runs_without_video() {
        let conn = setup();

        assert_eq!(
            focus_analysis_candidate_count_impl(&conn, "v2", "run-1", None),
            4
        );
        assert_eq!(
            candidate_ids(focus_analysis_candidates_impl(
                &conn, 500, "v2", "run-1", None
            )),
            vec![1, 3, 4, 6]
        );
    }

    #[test]
    fn explicit_id_focus_queue_intersects_selection_with_retry_rules() {
        let conn = setup();
        let ids = vec![2, 3, 4, 5, 7];

        assert_eq!(
            focus_analysis_candidate_count_impl(&conn, "v2", "run-1", Some(&ids)),
            2
        );
        assert_eq!(
            candidate_ids(focus_analysis_candidates_impl(
                &conn,
                500,
                "v2",
                "run-1",
                Some(&ids)
            )),
            vec![3, 4]
        );
        assert_eq!(
            focus_analysis_candidate_count_impl(&conn, "v2", "run-2", Some(&ids)),
            3
        );
        assert_eq!(
            candidate_ids(focus_analysis_candidates_impl(
                &conn,
                500,
                "v2",
                "run-2",
                Some(&ids)
            )),
            vec![3, 4, 5]
        );
    }

    #[test]
    fn explicit_empty_id_focus_queue_is_empty() {
        let conn = setup();
        let ids: Vec<i64> = Vec::new();

        assert_eq!(
            focus_analysis_candidate_count_impl(&conn, "v2", "run-1", Some(&ids)),
            0
        );
        assert!(focus_analysis_candidates_impl(&conn, 500, "v2", "run-1", Some(&ids)).is_empty());
    }

    #[test]
    fn focus_queue_collapses_raw_jpeg_pairs_to_lightweight_sibling() {
        let conn = setup();
        conn.execute_batch(
            "INSERT INTO images (
                 id, file_path, file_size, image_kind, file_stem, directory_path, is_video,
                 focus_analysis_status, focus_algorithm_version, focus_analysis_attempt_id
             ) VALUES
                 (8, '/a/pair.nef', 20, 'raw', 'pair', '/a', FALSE, NULL, NULL, NULL),
                 (9, '/a/pair.jpg', 10, 'jpeg', 'pair', '/a', FALSE, NULL, NULL, NULL),
                 (10, '/a/raw-only.nef', 20, 'raw', 'raw-only', '/a', FALSE, NULL, NULL, NULL);",
        )
        .expect("insert pair rows");

        assert_eq!(
            candidate_ids(focus_analysis_candidates_impl(
                &conn, 500, "v2", "run-1", None
            )),
            vec![1, 3, 4, 6, 9, 10]
        );

        assert_eq!(
            candidate_ids(focus_analysis_candidates_impl(
                &conn,
                500,
                "v2",
                "run-1",
                Some(&[8])
            )),
            vec![9],
            "a selection containing only the RAW half should analyze the JPEG representative"
        );
    }
}

#[cfg(test)]
mod similar_photo_group_tests {
    use super::*;
    use duckdb::Connection;

    fn setup() -> Connection {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE images (id INTEGER PRIMARY KEY);
            INSERT INTO images (id) VALUES (1), (2), (3), (4), (9);
            CREATE TABLE similar_photo_group_member (
                image_id INTEGER PRIMARY KEY,
                group_id INTEGER NOT NULL,
                representative_id INTEGER NOT NULL,
                member_rank INTEGER NOT NULL,
                distance_to_representative DOUBLE,
                algorithm_version TEXT NOT NULL,
                threshold DOUBLE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE similar_photo_featureprint (
                image_id INTEGER NOT NULL,
                algorithm_version TEXT NOT NULL,
                source_stamp TEXT NOT NULL,
                featureprint_blob BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (image_id, algorithm_version)
            );
            CREATE TABLE similar_photo_group_work_unit (
                algorithm_version TEXT NOT NULL,
                scope_key TEXT NOT NULL,
                unit_index BIGINT NOT NULL,
                start_image_id INTEGER NOT NULL,
                end_image_id INTEGER NOT NULL,
                candidate_count BIGINT NOT NULL,
                member_count BIGINT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (algorithm_version, scope_key, unit_index)
            );
            INSERT INTO similar_photo_group_member
                (image_id, group_id, representative_id, member_rank, distance_to_representative, algorithm_version, threshold)
            VALUES
                (1, 1, 1, 0, 0.0, 'old-v1', 2.35),
                (2, 2, 2, 0, 0.0, 'old-v1', 2.35),
                (3, 2, 2, 1, 1.25, 'old-v1', 2.35),
                (9, 9, 9, 0, 0.0, 'other-v', 2.35);",
        )
        .expect("similar-photo DDL");
        conn
    }

    #[test]
    fn stack_members_expand_selected_groups_and_keep_singletons() {
        let conn = setup();
        let members = similar_photo_stack_members_for_ids_impl(&conn, &[2, 4], "old-v1");
        let actual: Vec<(i64, i64, i64, u32)> = members
            .into_iter()
            .map(|m| (m.image_id, m.group_id, m.representative_id, m.member_rank))
            .collect();

        assert_eq!(actual, vec![(2, 2, 2, 0), (3, 2, 2, 1), (4, 4, 4, 0)]);
    }

    #[test]
    fn replace_similar_groups_replaces_only_target_algorithm() {
        let conn = setup();
        let members = vec![
            SimilarPhotoGroupMember {
                image_id: 2,
                group_id: 2,
                representative_id: 2,
                member_rank: 0,
                distance_to_representative: Some(0.0),
                threshold: 2.35,
            },
            SimilarPhotoGroupMember {
                image_id: 3,
                group_id: 2,
                representative_id: 2,
                member_rank: 1,
                distance_to_representative: Some(1.25),
                threshold: 2.35,
            },
        ];

        assert_eq!(
            replace_similar_photo_groups_impl(&conn, members, "old-v1", None),
            2
        );

        let rows: Vec<(i64, String)> = conn
            .prepare(
                "SELECT image_id, algorithm_version
                 FROM similar_photo_group_member
                 ORDER BY image_id",
            )
            .unwrap()
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .map(|row| row.unwrap())
            .collect();

        assert_eq!(
            rows,
            vec![
                (2, "old-v1".to_string()),
                (3, "old-v1".to_string()),
                (9, "other-v".to_string()),
            ]
        );
    }

    #[test]
    fn progressive_grouping_state_upserts_featureprints_units_and_members() {
        let conn = setup();
        let featureprints = vec![
            SimilarPhotoFeatureprint {
                image_id: 1,
                source_stamp: "stamp-a".to_string(),
                featureprint_blob: vec![1, 2, 3],
            },
            SimilarPhotoFeatureprint {
                image_id: 2,
                source_stamp: "stamp-b".to_string(),
                featureprint_blob: vec![4, 5, 6],
            },
        ];
        assert_eq!(
            upsert_similar_photo_featureprints_impl(&conn, featureprints, "new-v1"),
            2
        );
        let cached = similar_photo_featureprints_for_ids_impl(&conn, &[1, 2, 9], "new-v1");
        assert_eq!(cached.len(), 2);
        assert_eq!(cached[0].featureprint_blob, vec![1, 2, 3]);

        let written = upsert_similar_photo_groups_for_ids_impl(
            &conn,
            &[1, 2],
            vec![
                SimilarPhotoGroupMember {
                    image_id: 1,
                    group_id: 1,
                    representative_id: 1,
                    member_rank: 0,
                    distance_to_representative: Some(0.0),
                    threshold: 8.5,
                },
                SimilarPhotoGroupMember {
                    image_id: 2,
                    group_id: 1,
                    representative_id: 1,
                    member_rank: 1,
                    distance_to_representative: Some(1.0),
                    threshold: 8.5,
                },
            ],
            "new-v1",
        );
        assert_eq!(written, 2);

        let unit = SimilarPhotoWorkUnit {
            unit_index: 0,
            start_image_id: 1,
            end_image_id: 2,
            candidate_count: 2,
            member_count: written as i64,
        };
        assert!(mark_similar_photo_work_unit_complete_impl(
            &conn,
            "new-v1",
            "whole:test",
            unit
        ));
        let units = completed_similar_photo_work_units_impl(&conn, "new-v1", "whole:test");
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].member_count, 2);
    }
}

#[cfg(test)]
mod analysis_job_tests {
    use super::*;
    use duckdb::Connection;

    fn setup() -> Connection {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE SEQUENCE analysis_job_id_seq START 1;
             CREATE TABLE analysis_jobs (
                 id INTEGER PRIMARY KEY DEFAULT nextval('analysis_job_id_seq'),
                 job_kind TEXT NOT NULL,
                 scope_kind TEXT NOT NULL,
                 scope_value TEXT,
                 algorithm_version TEXT NOT NULL,
                 analysis_run_id TEXT NOT NULL UNIQUE,
                 status TEXT NOT NULL,
                 total_candidate_count BIGINT NOT NULL,
                 processed_count BIGINT NOT NULL,
                 completed_count BIGINT NOT NULL,
                 skipped_count BIGINT NOT NULL,
                 failed_count BIGINT NOT NULL,
                 updated_count BIGINT NOT NULL,
                 cancel_requested BOOLEAN NOT NULL,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                 started_at TIMESTAMP,
                 updated_at TIMESTAMP,
                 finished_at TIMESTAMP,
                 last_error TEXT,
                 current_image_id BIGINT,
                 current_file_path TEXT,
                 current_started_at TIMESTAMP,
                 last_timeout_image_id BIGINT,
                 last_timeout_file_path TEXT,
                 last_timeout_at TIMESTAMP
             );",
        )
        .expect("analysis job DDL");
        conn
    }

    #[test]
    fn create_job_initializes_durable_state() {
        let conn = setup();
        let job = create_analysis_job_impl(
            &conn,
            "focus_quality",
            "whole_catalogue",
            None,
            "multi-subject-laplacian-v2",
            "run-1",
            42,
        )
        .expect("job created");

        assert_eq!(job.id, 1);
        assert_eq!(job.job_kind, "focus_quality");
        assert_eq!(job.scope_kind, "whole_catalogue");
        assert_eq!(job.algorithm_version, "multi-subject-laplacian-v2");
        assert_eq!(job.analysis_run_id, "run-1");
        assert_eq!(job.status, "queued");
        assert_eq!(job.total_candidate_count, 42);
        assert_eq!(job.processed_count, 0);
        assert_eq!(job.completed_count, 0);
        assert_eq!(job.skipped_count, 0);
        assert_eq!(job.failed_count, 0);
        assert_eq!(job.updated_count, 0);
        assert!(!job.cancel_requested);
        assert!(!job.created_at.is_empty());
        assert!(job.started_at.is_none());
        assert!(job.finished_at.is_none());
        assert!(job.current_image_id.is_none());
        assert!(job.current_file_path.is_none());
        assert!(job.current_started_at.is_none());
        assert!(job.last_timeout_image_id.is_none());
        assert!(job.last_timeout_file_path.is_none());
        assert!(job.last_timeout_at.is_none());

        let active = active_analysis_job_impl(&conn, "focus_quality").expect("active job");
        assert_eq!(active.id, job.id);
    }

    #[test]
    fn active_job_list_returns_all_non_terminal_jobs() {
        let conn = setup();
        let foreground = create_analysis_job_impl(
            &conn,
            "focus_quality",
            "selection",
            Some("1,2,3".to_string()),
            "v2",
            "active-list-foreground",
            3,
        )
        .expect("foreground job");
        let background = create_analysis_job_impl(
            &conn,
            "focus_enrichment",
            "whole_catalogue",
            None,
            "v2",
            "active-list-background",
            100,
        )
        .expect("background job");
        let completed = create_analysis_job_impl(
            &conn,
            "focus_quality",
            "whole_catalogue",
            None,
            "v2",
            "active-list-completed",
            10,
        )
        .expect("completed job");
        finish_analysis_job_impl(&conn, completed.id, "completed", None)
            .expect("completed terminal");

        let ids: Vec<i64> = active_analysis_jobs_impl(&conn)
            .into_iter()
            .map(|job| job.id)
            .collect();
        assert_eq!(ids, vec![background.id, foreground.id]);
    }

    #[test]
    fn progress_cancel_and_finish_follow_state_machine() {
        let conn = setup();
        let job = create_analysis_job_impl(
            &conn,
            "focus_quality",
            "whole_catalogue",
            Some("all".to_string()),
            "v2",
            "run-2",
            10,
        )
        .expect("job created");

        let progressed = update_analysis_job_progress_impl(&conn, job.id, 5, 3, 1, 1, 4, Some(12))
            .expect("progress updated");
        assert_eq!(progressed.status, "running");
        assert_eq!(progressed.total_candidate_count, 12);
        assert_eq!(progressed.processed_count, 5);
        assert_eq!(progressed.completed_count, 3);
        assert_eq!(progressed.skipped_count, 1);
        assert_eq!(progressed.failed_count, 1);
        assert_eq!(progressed.updated_count, 4);
        assert!(progressed.started_at.is_some());
        assert!(progressed.updated_at.is_some());

        assert!(request_cancel_analysis_job_impl(&conn, job.id));
        let cancelling = analysis_job_by_id_impl(&conn, job.id).expect("job row");
        assert_eq!(cancelling.status, "cancelling");
        assert!(cancelling.cancel_requested);

        let finished =
            finish_analysis_job_impl(&conn, job.id, "cancelled", None).expect("job finished");
        assert_eq!(finished.status, "cancelled");
        assert!(finished.cancel_requested);
        assert!(finished.finished_at.is_some());
        assert!(active_analysis_job_impl(&conn, "focus_quality").is_none());

        assert!(update_analysis_job_progress_impl(&conn, job.id, 1, 1, 0, 0, 1, None).is_none());
        assert!(!request_cancel_analysis_job_impl(&conn, job.id));
        assert!(finish_analysis_job_impl(&conn, job.id, "completed", None).is_none());
    }

    #[test]
    fn breadcrumb_tracks_current_and_last_timeout() {
        let conn = setup();
        let job = create_analysis_job_impl(
            &conn,
            "focus_quality",
            "whole_catalogue",
            None,
            "v2",
            "breadcrumb-run",
            2,
        )
        .expect("job created");

        let current = update_analysis_job_breadcrumb_impl(
            &conn,
            job.id,
            Some(42),
            Some("/photos/wedged.nef".to_string()),
            false,
        )
        .expect("breadcrumb set");
        assert_eq!(current.current_image_id, Some(42));
        assert_eq!(
            current.current_file_path.as_deref(),
            Some("/photos/wedged.nef")
        );
        assert!(current.current_started_at.is_some());
        assert!(current.last_timeout_image_id.is_none());

        let timed_out = update_analysis_job_breadcrumb_impl(
            &conn,
            job.id,
            Some(42),
            Some("/photos/wedged.nef".to_string()),
            true,
        )
        .expect("timeout recorded");
        assert_eq!(timed_out.last_timeout_image_id, Some(42));
        assert_eq!(
            timed_out.last_timeout_file_path.as_deref(),
            Some("/photos/wedged.nef")
        );
        assert!(timed_out.last_timeout_at.is_some());

        let cleared =
            update_analysis_job_breadcrumb_impl(&conn, job.id, None, None, false).expect("cleared");
        assert!(cleared.current_image_id.is_none());
        assert!(cleared.current_file_path.is_none());
        assert!(cleared.current_started_at.is_none());
        assert_eq!(cleared.last_timeout_image_id, Some(42));
    }

    #[test]
    fn invalid_job_tokens_and_non_terminal_finish_are_rejected() {
        let conn = setup();
        assert!(create_analysis_job_impl(
            &conn,
            "Focus Quality",
            "whole_catalogue",
            None,
            "v2",
            "run-3",
            0
        )
        .is_none());

        let job = create_analysis_job_impl(
            &conn,
            "focus_quality",
            "whole_catalogue",
            None,
            "v2",
            "run-4",
            0,
        )
        .expect("job created");
        assert!(finish_analysis_job_impl(&conn, job.id, "running", None).is_none());
        assert!(
            finish_analysis_job_impl(&conn, job.id, "failed", Some("boom".to_string()))
                .expect("failed terminal")
                .last_error
                .as_deref()
                == Some("boom")
        );
    }

    #[test]
    fn recovery_terminalizes_interrupted_jobs_for_kind_only() {
        let conn = setup();
        let queued = create_analysis_job_impl(
            &conn,
            "focus_quality",
            "selection",
            None,
            "v2",
            "recover-queued",
            10,
        )
        .expect("queued job");
        let running = create_analysis_job_impl(
            &conn,
            "focus_quality",
            "selection",
            None,
            "v2",
            "recover-running",
            10,
        )
        .expect("running job");
        let cancelling = create_analysis_job_impl(
            &conn,
            "focus_quality",
            "selection",
            None,
            "v2",
            "recover-cancelling",
            10,
        )
        .expect("cancelling job");
        let completed = create_analysis_job_impl(
            &conn,
            "focus_quality",
            "selection",
            None,
            "v2",
            "recover-completed",
            10,
        )
        .expect("completed job");
        let other_kind = create_analysis_job_impl(
            &conn,
            "thumbnail_quality",
            "whole_catalogue",
            None,
            "v1",
            "recover-other-kind",
            10,
        )
        .expect("other kind job");

        update_analysis_job_progress_impl(&conn, running.id, 1, 1, 0, 0, 1, None)
            .expect("running progress");
        request_cancel_analysis_job_impl(&conn, cancelling.id);
        finish_analysis_job_impl(&conn, completed.id, "completed", None).expect("completed");

        let recovered = recover_interrupted_analysis_jobs_impl(
            &conn,
            "focus_quality",
            "cancelled",
            Some("Recovered after launch".to_string()),
        );
        assert_eq!(recovered, 3);

        for id in [queued.id, running.id, cancelling.id] {
            let job = analysis_job_by_id_impl(&conn, id).expect("recovered job");
            assert_eq!(job.status, "cancelled");
            assert!(job.cancel_requested);
            assert!(job.finished_at.is_some());
            assert_eq!(job.last_error.as_deref(), Some("Recovered after launch"));
        }

        let completed = analysis_job_by_id_impl(&conn, completed.id).expect("completed job");
        assert_eq!(completed.status, "completed");
        assert!(!completed.cancel_requested);
        assert!(completed.last_error.is_none());

        let other_kind = analysis_job_by_id_impl(&conn, other_kind.id).expect("other kind job");
        assert_eq!(other_kind.status, "queued");
        assert!(!other_kind.cancel_requested);
        assert!(other_kind.finished_at.is_none());

        assert!(active_analysis_job_impl(&conn, "focus_quality").is_none());
        assert!(active_analysis_job_impl(&conn, "thumbnail_quality").is_some());
    }

    #[test]
    fn recovery_rejects_invalid_kind_and_non_terminal_status() {
        let conn = setup();
        let job = create_analysis_job_impl(
            &conn,
            "focus_quality",
            "selection",
            None,
            "v2",
            "recover-invalid",
            10,
        )
        .expect("job");

        assert_eq!(
            recover_interrupted_analysis_jobs_impl(&conn, "Focus Quality", "cancelled", None),
            0
        );
        assert_eq!(
            recover_interrupted_analysis_jobs_impl(&conn, "focus_quality", "running", None),
            0
        );

        let job = analysis_job_by_id_impl(&conn, job.id).expect("job row");
        assert_eq!(job.status, "queued");
        assert!(job.finished_at.is_none());
    }
}

#[cfg(test)]
mod keyword_tests {
    use super::*;

    #[test]
    fn materialized_rows_builds_ancestor_chain() {
        let rows = keyword_materialized_rows(&vec![
            "Animals".to_string(),
            "Dog".to_string(),
            "Lab".to_string(),
        ]);
        let sep = KEYWORD_PATH_SEPARATOR;
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], ("Animals".to_string(), "Animals".to_string()));
        assert_eq!(rows[1], ("Dog".to_string(), format!("Animals{sep}Dog")));
        assert_eq!(
            rows[2],
            ("Lab".to_string(), format!("Animals{sep}Dog{sep}Lab"))
        );
    }

    #[test]
    fn materialized_rows_trims_and_rejects_bad_segments() {
        // Blank segment -> whole path rejected (empty).
        assert!(
            keyword_materialized_rows(&vec!["Animals".to_string(), "  ".to_string()]).is_empty()
        );
        // Segment containing the separator -> rejected.
        assert!(keyword_materialized_rows(&vec![format!("a{KEYWORD_PATH_SEPARATOR}b")]).is_empty());
        // Trimming.
        let rows = keyword_materialized_rows(&vec![" Animals ".to_string()]);
        assert_eq!(rows[0], ("Animals".to_string(), "Animals".to_string()));
    }

    #[test]
    fn image_id_in_list_assembly() {
        assert_eq!(image_id_in_list(&[]), None);
        assert_eq!(
            image_id_in_list(&[5, 9, 12]),
            Some("image_id IN (5, 9, 12)".to_string())
        );
    }

    #[test]
    fn keyword_vocabulary_filters_by_origin() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE keyword (
                 image_id INTEGER NOT NULL,
                 label TEXT NOT NULL,
                 path TEXT NOT NULL,
                 status INTEGER NOT NULL DEFAULT 1,
                 origin INTEGER NOT NULL DEFAULT 1,
                 color BOOLEAN NOT NULL DEFAULT FALSE
             );
             CREATE VIEW keyword_visible AS SELECT * FROM keyword WHERE status = 1;
             INSERT INTO keyword (image_id, label, path, status, origin) VALUES
                 (1, 'manual', 'manual', 1, 1),
                 (2, 'automatic', 'automatic', 1, 2),
                 (3, 'shared', 'shared', 1, 3),
                 (4, 'hidden', 'hidden', 0, 2);",
        )
        .expect("schema + seed");

        let labels = |origin: &str| {
            keyword_vocabulary_impl(&conn, origin)
                .into_iter()
                .map(|node| node.label)
                .collect::<Vec<_>>()
        };

        assert_eq!(labels("both"), vec!["automatic", "manual", "shared"]);
        assert_eq!(labels("user"), vec!["manual", "shared"]);
        assert_eq!(labels("auto"), vec!["automatic", "shared"]);
        assert!(labels("bogus").is_empty());
    }

    #[test]
    fn keyword_management_rows_filter_and_delete_raw_keyword_rows() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        let sep = KEYWORD_PATH_SEPARATOR;
        conn.execute_batch(&format!(
            "CREATE TABLE keyword (
                 image_id INTEGER NOT NULL,
                 label TEXT NOT NULL,
                 path TEXT NOT NULL,
                 status INTEGER NOT NULL DEFAULT 1,
                 origin INTEGER NOT NULL DEFAULT 1,
                 collection BOOLEAN NOT NULL DEFAULT FALSE
             );
             INSERT INTO keyword (image_id, label, path, status, origin, collection) VALUES
                 (1, 'Dogs', 'Animals{sep}Dogs', 1, 1, FALSE),
                 (2, 'Dogs', 'Animals{sep}Dogs', 0, 1, FALSE),
                 (3, 'Auto', 'Auto', 1, 2, FALSE),
                 (4, 'Trips', 'Trips', 0, 1, TRUE),
                 (5, 'Gone', 'Gone', 0, 2, FALSE),
                 (6, 'Puppies', 'Animals{sep}Dogs{sep}Puppies', 1, 1, FALSE);"
        ))
        .expect("schema + seed");

        let all = keyword_management_rows_impl(&conn, "both", true, true);
        assert_eq!(all.len(), 5);

        let dogs = all
            .iter()
            .find(|row| row.path == format!("Animals{sep}Dogs"))
            .expect("dogs");
        assert_eq!(dogs.visible_count, 1);
        assert_eq!(dogs.hidden_count, 1);
        assert_eq!(dogs.collection_count, 0);
        assert_eq!(dogs.total_count, 2);

        let no_collections = keyword_management_rows_impl(&conn, "both", false, true)
            .into_iter()
            .map(|row| row.label)
            .collect::<Vec<_>>();
        assert!(!no_collections.contains(&"Trips".to_string()));

        let no_orphans = keyword_management_rows_impl(&conn, "both", true, false)
            .into_iter()
            .map(|row| row.label)
            .collect::<Vec<_>>();
        assert!(!no_orphans.contains(&"Gone".to_string()));
        assert!(no_orphans.contains(&"Trips".to_string()));

        let auto = keyword_management_rows_impl(&conn, "auto", true, true)
            .into_iter()
            .map(|row| row.label)
            .collect::<Vec<_>>();
        assert_eq!(auto, vec!["Auto".to_string(), "Gone".to_string()]);

        let deleted = delete_keyword_paths_impl(&conn, &[format!("Animals{sep}Dogs")]);
        assert_eq!(deleted, 3);
        let remaining = keyword_management_rows_impl(&conn, "both", true, true)
            .into_iter()
            .map(|row| row.label)
            .collect::<Vec<_>>();
        assert!(!remaining.contains(&"Dogs".to_string()));
        assert!(!remaining.contains(&"Puppies".to_string()));
    }

    #[test]
    fn keyword_predicate_sql() {
        let has = QueryPredicate {
            kind: "keyword_has".to_string(),
            day: None,
            day_end: None,
            op: None,
            stars: None,
            num: None,
            num_end: None,
            value: Some("Wagner".to_string()),
        };
        assert_eq!(
            predicate_to_sql(&has),
            "(EXISTS (SELECT 1 FROM keyword_visible k WHERE k.image_id = images.id AND k.label = 'Wagner'))"
        );
        let mut auto_has = has.clone();
        auto_has.op = Some("auto".to_string());
        assert_eq!(
            predicate_to_sql(&auto_has),
            "(EXISTS (SELECT 1 FROM keyword_visible k WHERE k.image_id = images.id AND k.label = 'Wagner' AND (k.origin & 2) <> 0))"
        );
        let mut user_has = has.clone();
        user_has.op = Some("user".to_string());
        assert_eq!(
            predicate_to_sql(&user_has),
            "(EXISTS (SELECT 1 FROM keyword_visible k WHERE k.image_id = images.id AND k.label = 'Wagner' AND (k.origin & 1) <> 0))"
        );

        let not = QueryPredicate {
            kind: "keyword_not".to_string(),
            day: None,
            day_end: None,
            op: None,
            stars: None,
            num: None,
            num_end: None,
            value: Some("snapshot".to_string()),
        };
        assert_eq!(
            predicate_to_sql(&not),
            "(NOT EXISTS (SELECT 1 FROM keyword_visible k WHERE k.image_id = images.id AND k.label = 'snapshot'))"
        );

        // Empty value -> backstop.
        let empty = QueryPredicate {
            kind: "keyword_has".to_string(),
            day: None,
            day_end: None,
            op: None,
            stars: None,
            num: None,
            num_end: None,
            value: Some(String::new()),
        };
        assert_eq!(predicate_to_sql(&empty), "(FALSE)");

        // keyword_none takes no value (S66 Gate 2): no visible NON-COLOR
        // keyword row (color-marked rows are color labels, not keywords).
        let none = QueryPredicate {
            kind: "keyword_none".to_string(),
            day: None,
            day_end: None,
            op: None,
            stars: None,
            num: None,
            num_end: None,
            value: None,
        };
        assert_eq!(
            predicate_to_sql(&none),
            "(NOT EXISTS (SELECT 1 FROM keyword_visible k WHERE k.image_id = images.id AND (k.color = FALSE OR k.color IS NULL)))"
        );
    }

    #[test]
    fn color_predicate_sql_knows_both_halves() {
        // S66: standard colors live in images.color_label; custom color-label
        // text is a keyword row with the color switch ON. any_color/no_color
        // are exact complements across BOTH halves (raw table — hiding a
        // keyword doesn't un-color the photo).
        let any = QueryPredicate {
            kind: "any_color".to_string(),
            day: None,
            day_end: None,
            op: None,
            stars: None,
            num: None,
            num_end: None,
            value: None,
        };
        assert_eq!(
            predicate_to_sql(&any),
            "(color_label IS NOT NULL OR EXISTS (SELECT 1 FROM keyword k WHERE k.image_id = images.id AND k.color = TRUE))"
        );

        let none = QueryPredicate {
            kind: "no_color".to_string(),
            day: None,
            day_end: None,
            op: None,
            stars: None,
            num: None,
            num_end: None,
            value: None,
        };
        assert_eq!(
            predicate_to_sql(&none),
            "(color_label IS NULL AND NOT EXISTS (SELECT 1 FROM keyword k WHERE k.image_id = images.id AND k.color = TRUE))"
        );
    }

    /// The sidebar multi-select prefix arms (S67): slash-terminated path
    /// prefixes and capture-datetime prefixes, quote-escaped; empty → FALSE.
    #[test]
    fn sidebar_prefix_predicate_sql() {
        // A folder-scope path prefix now also EXCLUDES Apple Photos library
        // originals (their own sidebar scope; a folder prefix can string-match a
        // nested `.photoslibrary`). The whole atom stays parenthesized so OR-joins
        // are safe.
        let path = QueryPredicate {
            kind: "path_prefix".to_string(),
            day: None,
            day_end: None,
            op: None,
            stars: None,
            num: None,
            num_end: None,
            value: Some("/Volumes/Photo's/2026/".to_string()),
        };
        assert_eq!(
            predicate_to_sql(&path),
            "(starts_with(file_path, '/Volumes/Photo''s/2026/') AND file_path NOT LIKE '%.photoslibrary/%')"
        );

        // A prefix that IS inside a `.photoslibrary` (the Apple Library node's own
        // scope) is EXEMPT — it must keep showing those originals.
        let apple = QueryPredicate {
            kind: "path_prefix".to_string(),
            day: None,
            day_end: None,
            op: None,
            stars: None,
            num: None,
            num_end: None,
            value: Some("/Users/x/Pictures/Photos Library.photoslibrary/".to_string()),
        };
        assert_eq!(
            predicate_to_sql(&apple),
            "(starts_with(file_path, '/Users/x/Pictures/Photos Library.photoslibrary/'))"
        );

        let capture = QueryPredicate {
            kind: "capture_prefix".to_string(),
            day: None,
            day_end: None,
            op: None,
            stars: None,
            num: None,
            num_end: None,
            value: Some("2026:06:".to_string()),
        };
        assert_eq!(
            predicate_to_sql(&capture),
            "(starts_with(capture_datetime, '2026:06:'))"
        );

        let empty = QueryPredicate {
            kind: "path_prefix".to_string(),
            day: None,
            day_end: None,
            op: None,
            stars: None,
            num: None,
            num_end: None,
            value: Some(String::new()),
        };
        assert_eq!(predicate_to_sql(&empty), "(FALSE)");
    }

    /// The single-node Sources/Dates path-prefix builder (S65/S86): a folder
    /// prefix excludes Apple Photos library originals; an empty prefix (All
    /// Sources), a date-only filter, and an Apple-library prefix do NOT.
    #[test]
    fn path_date_predicate_excludes_apple_library_for_folder_scope() {
        // Empty (All Sources) → no filter at all, nothing excluded.
        assert_eq!(build_path_date_predicate("", ""), "");

        // A folder prefix → its files MINUS any nested Apple library.
        assert_eq!(
            build_path_date_predicate("/Users/richardwagner/", ""),
            "file_path LIKE '/Users/richardwagner/%' AND file_path NOT LIKE '%.photoslibrary/%'"
        );

        // The Apple Library scope (prefix inside a `.photoslibrary`) is exempt.
        assert_eq!(
            build_path_date_predicate("/Users/x/Pictures/Photos Library.photoslibrary/", ""),
            "file_path LIKE '/Users/x/Pictures/Photos Library.photoslibrary/%'"
        );

        // Date-only (no folder) is unchanged — Apple is not excluded from Dates.
        assert_eq!(
            build_path_date_predicate("", "2026:06:"),
            "capture_datetime LIKE '2026:06:%'"
        );

        // Folder + date → exclusion still appended after both clauses.
        assert_eq!(
            build_path_date_predicate("/Users/richardwagner/", "2026:06:"),
            "file_path LIKE '/Users/richardwagner/%' AND capture_datetime LIKE '2026:06:%' AND file_path NOT LIKE '%.photoslibrary/%'"
        );
    }

    /// Pair keyword parity on a real in-memory engine (S67): the sidecar
    /// pass's mirror copies whole rows (status / collection / color ride
    /// along) across same-stem-same-directory RAW<->JPEG pairs, both
    /// directions, idempotently — and never touches non-pairs.
    #[test]
    fn mirror_keyword_rows_across_pairs_end_to_end() {
        use duckdb::Connection;

        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE images (id INTEGER, file_stem VARCHAR, directory_path VARCHAR, image_kind VARCHAR);
             CREATE SEQUENCE keyword_id_seq START 1;
             CREATE TABLE keyword (
                 id INTEGER PRIMARY KEY DEFAULT nextval('keyword_id_seq'),
                 image_id INTEGER NOT NULL,
                 label TEXT NOT NULL,
                 path TEXT NOT NULL,
                 status INTEGER NOT NULL DEFAULT 1,
                 origin INTEGER NOT NULL DEFAULT 1,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                 hidden_at TIMESTAMP,
                 collection BOOLEAN NOT NULL DEFAULT FALSE,
                 color BOOLEAN NOT NULL DEFAULT FALSE
             );
             -- 1+2: a RAW+JPEG pair (the synthesized-sidecar shape; keywords on the RAW only).
             -- 3:   an unpaired RAW with a keyword (must stay untouched).
             -- 4+5: a pair where the JPEG already carries one of the RAW's rows
             --      (partial overlap -> only the missing row copies).
             -- 6:   a JPEG with a keyword and NO raw twin (reverse direction no-op).
             INSERT INTO images VALUES
                 (1, 'JAW_1', '/d/a', 'raw'), (2, 'JAW_1', '/d/a', 'jpeg'),
                 (3, 'JAW_2', '/d/a', 'raw'),
                 (4, 'JAW_3', '/d/b', 'raw'), (5, 'JAW_3', '/d/b', 'jpeg'),
                 (6, 'JAW_4', '/d/b', 'jpeg');
             INSERT INTO keyword (image_id, label, path, status, collection, color) VALUES
                 (1, 'Dogs',   'Dogs',                      1, FALSE, FALSE),
                 (1, 'Family', 'Family',                    1, TRUE,  FALSE),  -- collection switch rides
                 (1, 'Old',    'Old',                       0, FALSE, FALSE),  -- hidden row rides as hidden
                 (3, 'Lonely', 'Lonely',                    1, FALSE, FALSE),
                 (4, 'Birds',  'Birds',                     1, FALSE, FALSE),
                 (4, 'Trips',  'Trips',                     1, FALSE, FALSE),
                 (5, 'Birds',  'Birds',                     1, FALSE, FALSE),
                 (6, 'Solo',   'Solo',                      1, FALSE, FALSE);",
        )
        .expect("schema + seed");

        // Pair 1+2: all three rows copy. Pair 4+5: only 'Trips' copies down,
        // 'Birds' already exists; nothing copies UP (4 has both). Image 6's
        // 'Solo' has no raw twin. Image 3 is unpaired.
        let copied = mirror_keyword_rows_across_pairs_impl(&conn);
        assert_eq!(copied, 4, "Dogs+Family+Old onto 2, Trips onto 5");

        // The JPEG twin carries the full three-switch state.
        let (status, collection): (i64, bool) = conn
            .query_row(
                "SELECT status, collection FROM keyword WHERE image_id = 2 AND path = 'Family'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(
            (status, collection),
            (1, true),
            "collection switch copied intact"
        );
        let hidden_status: i64 = conn
            .query_row(
                "SELECT status FROM keyword WHERE image_id = 2 AND path = 'Old'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(hidden_status, 0, "hidden rows copy as hidden");

        // Untouched bystanders.
        let lonely: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM keyword WHERE image_id IN (3, 6)",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(lonely, 2, "unpaired images gained nothing");

        // Idempotent: a second pass copies nothing.
        assert_eq!(mirror_keyword_rows_across_pairs_impl(&conn), 0);
    }

    /// Copy-and-Import keyword inheritance on a real in-memory engine (S67):
    /// explicit (source, destination) pairs copy whole rows — status,
    /// collection, color — idempotently; length-mismatched arrays no-op.
    #[test]
    fn copy_keyword_rows_for_image_pairs_end_to_end() {
        use duckdb::Connection;

        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE SEQUENCE keyword_id_seq START 1;
             CREATE TABLE keyword (
                 id INTEGER PRIMARY KEY DEFAULT nextval('keyword_id_seq'),
                 image_id INTEGER NOT NULL,
                 label TEXT NOT NULL,
                 path TEXT NOT NULL,
                 status INTEGER NOT NULL DEFAULT 1,
                 origin INTEGER NOT NULL DEFAULT 1,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                 hidden_at TIMESTAMP,
                 collection BOOLEAN NOT NULL DEFAULT FALSE,
                 color BOOLEAN NOT NULL DEFAULT FALSE,
                 is_video BOOLEAN NOT NULL DEFAULT FALSE
             );
             -- Source 1 (a VIDEO -> copy 11): a plain keyword + a collection-marked row.
             -- Source 2 (-> copy 12): nothing (the copy must gain nothing).
             -- Copy 11 already carries one of source 1's paths (partial overlap).
             INSERT INTO keyword (image_id, label, path, status, collection, color, is_video) VALUES
                 (1, 'Dogs',   'Dogs',   1, FALSE, FALSE, TRUE),
                 (1, 'Family', 'Family', 1, TRUE,  FALSE, TRUE),
                 (11, 'Dogs',  'Dogs',   1, FALSE, FALSE, TRUE);",
        )
        .expect("schema + seed");

        let copied = copy_keyword_rows_for_image_pairs_impl(&conn, &[1, 2], &[11, 12]);
        assert_eq!(copied, 1, "only the missing 'Family' row copies onto 11");

        let (status, collection): (i64, bool) = conn
            .query_row(
                "SELECT status, collection FROM keyword WHERE image_id = 11 AND path = 'Family'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(
            (status, collection),
            (1, true),
            "collection switch rode the copy"
        );

        // S72 closeout: is_video rides the copy (source 1 is a video).
        let copied_is_video: bool = conn
            .query_row(
                "SELECT is_video FROM keyword WHERE image_id = 11 AND path = 'Family'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(
            copied_is_video,
            "the copied keyword row carries the source's is_video"
        );

        let copy12: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM keyword WHERE image_id = 12",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(copy12, 0, "a keywordless source copies nothing");

        // Idempotent + guard rails.
        assert_eq!(
            copy_keyword_rows_for_image_pairs_impl(&conn, &[1, 2], &[11, 12]),
            0
        );
        assert_eq!(
            copy_keyword_rows_for_image_pairs_impl(&conn, &[1], &[11, 12]),
            0,
            "length mismatch no-ops"
        );
    }

    /// The color switch end-to-end on a real in-memory engine (S66):
    /// flip-or-insert marking, idempotence, and the three predicates that
    /// read it (any_color / no_color / keyword_none).
    #[test]
    fn color_keyword_switch_end_to_end() {
        use duckdb::Connection;

        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE images (id INTEGER, color_label TEXT, is_video BOOLEAN);
             CREATE SEQUENCE keyword_id_seq START 1;
             CREATE TABLE keyword (
                 id INTEGER PRIMARY KEY DEFAULT nextval('keyword_id_seq'),
                 image_id INTEGER NOT NULL,
                 label TEXT NOT NULL,
                 path TEXT NOT NULL,
                 status INTEGER NOT NULL DEFAULT 1,
                 origin INTEGER NOT NULL DEFAULT 1,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                 hidden_at TIMESTAMP,
                 collection BOOLEAN NOT NULL DEFAULT FALSE,
                 color BOOLEAN NOT NULL DEFAULT FALSE,
                 is_video BOOLEAN NOT NULL DEFAULT FALSE
             );
             CREATE OR REPLACE VIEW keyword_visible AS
                 SELECT * FROM keyword WHERE status = 1;
             -- Photo 1: standard red via color_label, no keywords.
             -- Photo 2: a VIDEO; will get a CUSTOM color label 'Approved' (keyword row).
             -- Photo 3: a real keyword 'Approved' ALREADY applied -> the mark
             --          must flip THAT row, not insert a second.
             -- Photo 4: nothing at all.
             INSERT INTO images VALUES (1, 'red', FALSE), (2, NULL, TRUE), (3, NULL, FALSE), (4, NULL, FALSE);
             INSERT INTO keyword (image_id, label, path) VALUES (3, 'Approved', 'Approved');",
        )
        .expect("schema + seed");

        // Mark 'Approved' as a color on photos 2 and 3.
        let changed = assign_color_keyword_for_ids_impl(&conn, &[2, 3], "Approved");
        assert_eq!(changed, 2, "one insert (photo 2) + one flip (photo 3)");

        // Photo 3 must still have exactly ONE 'Approved' row (flipped, not duplicated).
        let rows3: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM keyword WHERE image_id = 3 AND label = 'Approved'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(rows3, 1);

        // S72 closeout: photo 2 is a video, so its new color-keyword row carries is_video.
        let v2: bool = conn
            .query_row(
                "SELECT is_video FROM keyword WHERE image_id = 2 AND label = 'Approved'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(v2, "the video's color-keyword row carries is_video = TRUE");

        // Idempotent: a second pass changes nothing.
        assert_eq!(
            assign_color_keyword_for_ids_impl(&conn, &[2, 3], "Approved"),
            0
        );

        let count = |pred: &QueryPredicate| -> i64 {
            let sql = format!(
                "SELECT COUNT(*) FROM images WHERE {}",
                predicate_to_sql(pred)
            );
            conn.query_row(&sql, [], |r| r.get(0))
                .expect("predicate executes")
        };
        let bare = |kind: &str| QueryPredicate {
            kind: kind.to_string(),
            day: None,
            day_end: None,
            op: None,
            stars: None,
            num: None,
            num_end: None,
            value: None,
        };

        // any_color: photo 1 (standard red) + photos 2 and 3 (custom mark).
        assert_eq!(count(&bare("any_color")), 3);
        // no_color: only photo 4 — the exact complement.
        assert_eq!(count(&bare("no_color")), 1);
        // keyword_none: color-marked rows don't count as keywords → photos 1,
        // 2, 4 are keywordless. Photo 3 is the KNOWN, ACCEPTED residue (the
        // same class as the S65 collection-add residue): its real keyword and
        // its color label share one row, so the flip merges the roles and the
        // photo reads keywordless too. Rare (needs identical text in both
        // Lightroom systems on one photo); revisit only if real catalogs care.
        assert_eq!(count(&bare("keyword_none")), 4);
    }

    #[test]
    fn filename_predicate_sql() {
        let fname = |kind: &str, v: &str| QueryPredicate {
            kind: kind.to_string(),
            day: None,
            day_end: None,
            op: None,
            stars: None,
            num: None,
            num_end: None,
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
    fn collection_predicate_sql() {
        let coll = |v: &str| QueryPredicate {
            kind: "collection_is".to_string(),
            day: None,
            day_end: None,
            op: None,
            stars: None,
            num: None,
            num_end: None,
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

    #[test]
    fn metadata_predicate_sql() {
        let meta = |kind: &str, v: &str| QueryPredicate {
            kind: kind.to_string(),
            day: None,
            day_end: None,
            op: None,
            stars: None,
            num: None,
            num_end: None,
            value: Some(v.to_string()),
        };

        // The six "is" subjects: exact equality on the picked catalogue value.
        assert_eq!(
            predicate_to_sql(&meta("extension_is", "nef")),
            "(file_extension = 'nef')"
        );
        assert_eq!(
            predicate_to_sql(&meta("kind_is", "raw")),
            "(image_kind = 'raw')"
        );
        assert_eq!(
            predicate_to_sql(&meta("camera_make_is", "NIKON CORPORATION")),
            "(camera_make = 'NIKON CORPORATION')"
        );
        assert_eq!(
            predicate_to_sql(&meta("camera_model_is", "NIKON Z 8")),
            "(camera_model = 'NIKON Z 8')"
        );
        assert_eq!(
            predicate_to_sql(&meta("lens_is", "NIKKOR Z 85mm f/1.8 S")),
            "(lens_model = 'NIKKOR Z 85mm f/1.8 S')"
        );
        // The apostrophe doubles for the SQL literal.
        assert_eq!(
            predicate_to_sql(&meta("creator_is", "Richard O'Wagner")),
            "(creator = 'Richard O''Wagner')"
        );

        // Lens "contains": case-insensitive ILIKE, wildcard-escaped fragment.
        assert_eq!(
            predicate_to_sql(&meta("lens_contains", "85mm")),
            "(lens_model ILIKE '%85mm%' ESCAPE '\\')"
        );
        assert_eq!(
            predicate_to_sql(&meta("lens_contains", "100%_O'N")),
            "(lens_model ILIKE '%100\\%\\_O''N%' ESCAPE '\\')"
        );

        // Empty value -> backstop (matches nothing).
        assert_eq!(predicate_to_sql(&meta("extension_is", "")), "(FALSE)");
        assert_eq!(predicate_to_sql(&meta("lens_contains", "")), "(FALSE)");
    }

    #[test]
    fn numeric_predicate_sql_arms() {
        let num = |kind: &str, op: &str, n: f64, end: Option<f64>| QueryPredicate {
            kind: kind.to_string(),
            day: None,
            day_end: None,
            stars: None,
            value: None,
            op: Some(op.to_string()),
            num: Some(n),
            num_end: end,
        };

        // The four columns, one mode each — shortest-round-trip literals
        // (1600 prints bare, 2.8 prints "2.8", 0.0005 prints "0.0005").
        assert_eq!(
            predicate_to_sql(&num("iso_num", "eq", 1600.0, None)),
            "(iso = 1600)"
        );
        assert_eq!(
            predicate_to_sql(&num("aperture_num", "lte", 2.8, None)),
            "(aperture <= 2.8)"
        );
        assert_eq!(
            predicate_to_sql(&num("shutter_num", "gte", 0.0005, None)),
            "(shutter_speed >= 0.0005)"
        );
        assert_eq!(
            predicate_to_sql(&num("focal_num", "between", 70.0, Some(200.0))),
            "(focal_length BETWEEN 70 AND 200)"
        );
        assert_eq!(
            predicate_to_sql(&num("focus_num", "gte", 120.0, None)),
            "(focus_score >= 120)"
        );
        assert_eq!(
            predicate_to_sql(&num("face_quality_best_num", "gte", 0.7, None)),
            "(face_quality_best >= 0.7)"
        );
        assert_eq!(
            predicate_to_sql(&num("face_quality_average_num", "lte", 0.5, None)),
            "(face_quality_average <= 0.5)"
        );
        assert_eq!(
            predicate_to_sql(&num("face_quality_min_num", "between", 0.2, Some(0.9))),
            "(face_quality_min BETWEEN 0.2 AND 0.9)"
        );
        assert_eq!(
            predicate_to_sql(&num("eyes_open_count_num", "gte", 2.0, None)),
            "(face_eyes_open_count >= 2)"
        );
        assert_eq!(
            predicate_to_sql(&num("blink_risk_count_num", "lte", 0.0, None)),
            "(face_blink_risk_count <= 0)"
        );
        assert!(predicate_to_sql(&num("focus_quality", "gte", 8.0, None))
            .contains("quantile_cont(score, 0.7)"));
        assert!(
            predicate_to_sql(&num("focus_quality", "between", 4.0, Some(8.0)))
                .starts_with("((focus_human_score IS NOT NULL")
        );
        let mut per_basis_focus = num("focus_quality", "gte", 0.0, None);
        per_basis_focus.num = None;
        per_basis_focus.value = Some("human_face=7,whole_image=3".to_string());
        let per_basis_sql = predicate_to_sql(&per_basis_focus);
        assert!(per_basis_sql.contains("focus_human_score IS NOT NULL"));
        assert!(per_basis_sql.contains("quantile_cont(score, 0.6)"));
        assert!(per_basis_sql.contains("focus_whole_image_score IS NOT NULL"));
        assert!(per_basis_sql.contains("quantile_cont(score, 0.2)"));
        let mut all_basis_focus = num("focus_quality", "gte", 0.0, None);
        all_basis_focus.num = None;
        all_basis_focus.value = Some("mode=all;human_face=7,whole_image=3".to_string());
        let all_basis_sql = predicate_to_sql(&all_basis_focus);
        assert!(all_basis_sql.contains(" AND "));
        let mut one_basis_focus = num("focus_quality", "gte", 0.0, None);
        one_basis_focus.num = None;
        one_basis_focus.value = Some("mode=one;human_face=7,whole_image=3".to_string());
        assert!(predicate_to_sql(&one_basis_focus).contains("= 1"));
        let mut face_quality = num("face_quality", "gte", 0.0, None);
        face_quality.num = None;
        face_quality.value = Some("mode=all;best=70,lowest=40".to_string());
        assert_eq!(
            predicate_to_sql(&face_quality),
            "((face_quality_best IS NOT NULL AND face_quality_best >= 0.7) AND (face_quality_min IS NOT NULL AND face_quality_min >= 0.4))"
        );
        assert_eq!(
            predicate_to_sql(&num("focus_quality", "gte", 80.5, None)),
            "(FALSE)"
        );
        assert_eq!(
            predicate_to_sql(&num("focus_quality", "gte", 101.0, None)),
            "(FALSE)"
        );

        // Malformed -> backstop: between without an upper bound, unknown op,
        // non-finite bounds, missing num / missing op.
        assert_eq!(
            predicate_to_sql(&num("focal_num", "between", 70.0, None)),
            "(FALSE)"
        );
        assert_eq!(
            predicate_to_sql(&num("iso_num", "approximately", 100.0, None)),
            "(FALSE)"
        );
        assert_eq!(
            predicate_to_sql(&num("iso_num", "eq", f64::NAN, None)),
            "(FALSE)"
        );
        assert_eq!(
            predicate_to_sql(&num("focal_num", "between", 70.0, Some(f64::INFINITY))),
            "(FALSE)"
        );
        let mut missing_num = num("iso_num", "eq", 0.0, None);
        missing_num.num = None;
        assert_eq!(predicate_to_sql(&missing_num), "(FALSE)");
        let mut missing_op = num("iso_num", "eq", 100.0, None);
        missing_op.op = None;
        assert_eq!(predicate_to_sql(&missing_op), "(FALSE)");
    }

    #[test]
    fn people_count_predicate_sql() {
        let num = |op: &str, n: f64| QueryPredicate {
            kind: "people_count".to_string(),
            day: None,
            day_end: None,
            stars: None,
            value: None,
            op: Some(op.to_string()),
            num: Some(n),
            num_end: None,
        };

        assert_eq!(predicate_to_sql(&num("eq", 1.0)), "(face_count = 1)");
        assert_eq!(predicate_to_sql(&num("gt", 3.0)), "(face_count > 3)");
        assert_eq!(
            predicate_to_sql(&num("lt", 2.0)),
            "(face_count >= 1 AND face_count < 2)"
        );
        assert_eq!(predicate_to_sql(&num("lt", 1.0)), "(FALSE)");
        assert_eq!(predicate_to_sql(&num("gte", 1.0)), "(FALSE)");
    }

    #[test]
    fn focus_quality_scale_executes_on_duckdb() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE images (
                id INTEGER,
                focus_score DOUBLE,
                focus_human_score DOUBLE,
                focus_animal_score DOUBLE,
                focus_foreground_score DOUBLE,
                focus_saliency_score DOUBLE,
                focus_animal_pose_score DOUBLE,
                focus_whole_image_score DOUBLE
             );
             INSERT INTO images VALUES
                (1, 10, NULL, NULL, NULL, NULL, NULL, 10),
                (2, 20, NULL, NULL, NULL, NULL, NULL, 20),
                (3, 30, NULL, NULL, NULL, NULL, NULL, 30),
                (4, NULL, NULL, NULL, NULL, NULL, NULL, NULL);",
        )
        .expect("seed focus scores");

        let pred = QueryPredicate {
            kind: "focus_quality".to_string(),
            day: None,
            day_end: None,
            stars: None,
            value: None,
            op: Some("gte".to_string()),
            num: Some(5.0),
            num_end: None,
        };
        let sql = format!(
            "SELECT COUNT(*) FROM images WHERE {}",
            predicate_to_sql(&pred)
        );
        let count: i64 = conn.query_row(&sql, [], |r| r.get(0)).expect("count 5+");
        assert_eq!(count, 2);

        let top = QueryPredicate {
            kind: "focus_quality".to_string(),
            day: None,
            day_end: None,
            stars: None,
            value: None,
            op: Some("gte".to_string()),
            num: Some(10.0),
            num_end: None,
        };
        let top_sql = format!(
            "SELECT COUNT(*) FROM images WHERE {}",
            predicate_to_sql(&top)
        );
        let top_count: i64 = conn
            .query_row(&top_sql, [], |r| r.get(0))
            .expect("count 10");
        assert_eq!(top_count, 1);

        conn.execute("UPDATE images SET focus_animal_score = 90 WHERE id = 1", [])
            .expect("seed animal score");
        let animal_only = QueryPredicate {
            kind: "focus_quality".to_string(),
            day: None,
            day_end: None,
            stars: None,
            value: Some("animal=10".to_string()),
            op: Some("gte".to_string()),
            num: None,
            num_end: None,
        };
        let animal_sql = format!(
            "SELECT COUNT(*) FROM images WHERE {}",
            predicate_to_sql(&animal_only)
        );
        let animal_count: i64 = conn
            .query_row(&animal_sql, [], |r| r.get(0))
            .expect("count animal 10");
        assert_eq!(animal_count, 1);
    }

    #[test]
    fn date_in_last_predicate_sql() {
        let dil = |count: &str, unit: &str| QueryPredicate {
            kind: "date_in_last".to_string(),
            day: None,
            day_end: None,
            stars: None,
            num: None,
            num_end: None,
            op: Some(unit.to_string()),
            value: Some(count.to_string()),
        };

        // The four units map to DuckDB INTERVAL keywords; the cutoff is
        // computed by the DATABASE at execution time, in the stored colon form.
        assert_eq!(
            predicate_to_sql(&dil("30", "days")),
            "(SUBSTRING(capture_datetime, 1, 10) >= strftime(CURRENT_DATE - INTERVAL 30 DAY, '%Y:%m:%d'))"
        );
        assert_eq!(
            predicate_to_sql(&dil("2", "weeks")),
            "(SUBSTRING(capture_datetime, 1, 10) >= strftime(CURRENT_DATE - INTERVAL 2 WEEK, '%Y:%m:%d'))"
        );
        assert_eq!(
            predicate_to_sql(&dil("1", "months")),
            "(SUBSTRING(capture_datetime, 1, 10) >= strftime(CURRENT_DATE - INTERVAL 1 MONTH, '%Y:%m:%d'))"
        );
        assert_eq!(
            predicate_to_sql(&dil("3", "years")),
            "(SUBSTRING(capture_datetime, 1, 10) >= strftime(CURRENT_DATE - INTERVAL 3 YEAR, '%Y:%m:%d'))"
        );

        // Malformed -> backstop: zero / negative / non-numeric / oversized
        // counts, an unknown unit, missing either field.
        assert_eq!(predicate_to_sql(&dil("0", "days")), "(FALSE)");
        assert_eq!(predicate_to_sql(&dil("-3", "days")), "(FALSE)");
        assert_eq!(predicate_to_sql(&dil("soon", "days")), "(FALSE)");
        assert_eq!(predicate_to_sql(&dil("10000", "days")), "(FALSE)");
        assert_eq!(predicate_to_sql(&dil("30", "fortnights")), "(FALSE)");
        let mut no_unit = dil("30", "days");
        no_unit.op = None;
        assert_eq!(predicate_to_sql(&no_unit), "(FALSE)");
        let mut no_count = dil("30", "days");
        no_count.value = None;
        assert_eq!(predicate_to_sql(&no_count), "(FALSE)");
    }

    /// date_in_last must EXECUTE on the real engine — the strftime/INTERVAL
    /// syntax is the risk the string test can't cover (S66 Gate 3). Rows are
    /// seeded by DuckDB's OWN CURRENT_DATE arithmetic, so the test never reads
    /// the host clock and cannot go stale. (All assertions stay correct even
    /// across a midnight boundary between seed and query.)
    #[test]
    fn date_in_last_executes_on_duckdb() {
        use duckdb::Connection;

        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE images (capture_datetime TEXT);
             INSERT INTO images VALUES
                 (strftime(CURRENT_DATE - INTERVAL 5 DAY, '%Y:%m:%d') || ' 12:00:00'),
                 (strftime(CURRENT_DATE - INTERVAL 400 DAY, '%Y:%m:%d') || ' 12:00:00'),
                 ('1999:01:01 00:00:00'),
                 (NULL);",
        )
        .expect("seed table");

        let dil = |count: &str, unit: &str| QueryPredicate {
            kind: "date_in_last".to_string(),
            day: None,
            day_end: None,
            stars: None,
            num: None,
            num_end: None,
            op: Some(unit.to_string()),
            value: Some(count.to_string()),
        };
        let count_for = |p: &QueryPredicate| -> i64 {
            let sql = format!("SELECT COUNT(*) FROM images WHERE {}", predicate_to_sql(p));
            conn.query_row(&sql, [], |row| row.get(0))
                .expect("predicate executes")
        };

        // 5-day-old row only; the 400-day row, 1999 row, and NULL never match.
        assert_eq!(count_for(&dil("30", "days")), 1);
        assert_eq!(count_for(&dil("2", "weeks")), 1);
        assert_eq!(count_for(&dil("2", "months")), 1);
        // 2 years reaches the 400-day row as well.
        assert_eq!(count_for(&dil("2", "years")), 2);
    }
}

#[cfg(test)]
mod saved_query_tests {
    use super::*;
    use duckdb::Connection;

    /// The saved-query DDL, as in the main schema (fresh CREATEs, no ALTERs).
    fn setup() -> Connection {
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
                 stars INTEGER,
                 num DOUBLE,
                 num_end DOUBLE
             );",
        )
        .expect("saved-query DDL");
        conn
    }

    fn pred(
        kind: &str,
        day: Option<&str>,
        day_end: Option<&str>,
        op: Option<&str>,
        stars: Option<u8>,
        value: Option<&str>,
    ) -> QueryPredicate {
        QueryPredicate {
            kind: kind.to_string(),
            day: day.map(str::to_string),
            day_end: day_end.map(str::to_string),
            op: op.map(str::to_string),
            stars,
            value: value.map(str::to_string),
            num: None,
            num_end: None,
        }
    }

    /// A numeric-subject predicate (S65) — kind + op + the bound(s).
    fn npred(kind: &str, op: &str, num: f64, num_end: Option<f64>) -> QueryPredicate {
        QueryPredicate {
            kind: kind.to_string(),
            day: None,
            day_end: None,
            op: Some(op.to_string()),
            stars: None,
            value: None,
            num: Some(num),
            num_end,
        }
    }

    #[test]
    fn round_trip_preserves_sentence() {
        let conn = setup();

        // The design doc's "Spring picks": two date ranges EITHER, color AND,
        // color EITHER, rating AND — repeated subjects + every connector slot —
        // plus two numeric criteria (S65: num + num_end must survive the trip).
        let predicates = vec![
            pred(
                "date_between",
                Some("2024:01:01"),
                Some("2024:02:15"),
                None,
                None,
                None,
            ),
            pred(
                "date_between",
                Some("2025:06:01"),
                Some("2025:07:04"),
                None,
                None,
                None,
            ),
            pred("color", None, None, None, None, Some("blue")),
            pred("color", None, None, None, None, Some("green")),
            pred("rating", None, None, Some("gte"), Some(3), None),
            npred("iso_num", "gte", 3200.0, None),
            npred("focal_num", "between", 70.0, Some(200.0)),
        ];
        let connectors = vec![
            Connector::Or,
            Connector::And,
            Connector::Or,
            Connector::And,
            Connector::And,
            Connector::And,
        ];

        let info = save_query_impl(&conn, "Spring picks", &predicates, &connectors)
            .expect("save succeeds");
        assert_eq!(info.name, "Spring picks");

        let payload = load_saved_query_impl(&conn, info.id).expect("load succeeds");
        assert_eq!(payload.predicates, predicates);
        assert_eq!(payload.connectors, connectors);
    }

    #[test]
    fn name_collisions_gain_numeric_suffixes() {
        let conn = setup();
        let predicates = vec![pred("rating", None, None, Some("gte"), Some(1), None)];

        let a = save_query_impl(&conn, "Dogs", &predicates, &[]).expect("first save");
        let b = save_query_impl(&conn, "Dogs", &predicates, &[]).expect("second save");
        let c = save_query_impl(&conn, "Dogs", &predicates, &[]).expect("third save");
        assert_eq!(a.name, "Dogs");
        assert_eq!(b.name, "Dogs-01");
        assert_eq!(c.name, "Dogs-02");

        // The list shows all three, name-ordered.
        let names: Vec<String> = list_saved_queries_impl(&conn)
            .into_iter()
            .map(|q| q.name)
            .collect();
        assert_eq!(names, vec!["Dogs", "Dogs-01", "Dogs-02"]);
    }

    #[test]
    fn empty_name_or_empty_sentence_rejected() {
        let conn = setup();
        let predicates = vec![pred("rating", None, None, Some("gte"), Some(1), None)];
        assert!(save_query_impl(&conn, "   ", &predicates, &[]).is_none());
        assert!(save_query_impl(&conn, "Fine", &[], &[]).is_none());
    }

    #[test]
    fn delete_removes_header_and_criteria() {
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
            .query_row("SELECT COUNT(*) FROM saved_query_criterion", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(leftover, 0);
    }
}

#[cfg(test)]
mod lightroom_import_tests {
    use super::*;
    use duckdb::{params, Connection};

    // ---- classify_extension: the new ImageKind promotions (§7) ----

    #[test]
    fn classify_promotes_new_kinds() {
        // DNG out of Raw (its own kind, checked BEFORE the RAW table).
        assert_eq!(classify_extension("dng".to_string()), ImageKind::Dng);
        // PSD / TIFF / PNG out of the Other bucket.
        assert_eq!(classify_extension("psd".to_string()), ImageKind::Psd);
        assert_eq!(classify_extension("tif".to_string()), ImageKind::Tiff);
        assert_eq!(classify_extension("tiff".to_string()), ImageKind::Tiff);
        assert_eq!(classify_extension("png".to_string()), ImageKind::Png);
    }

    #[test]
    fn classify_preserves_existing_kinds() {
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
    fn classify_is_case_insensitive_for_new_kinds() {
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
    fn duckdb_on_conflict_upsert_behaves() {
        let conn = Connection::open_in_memory().expect("open in-memory duckdb");
        conn.execute_batch("CREATE TABLE t (fp TEXT UNIQUE, rating INTEGER, cam TEXT);")
            .expect("create table");

        // One upsert mirroring the merge: rating = LR-wins, cam = fill-if-missing.
        let upsert = "INSERT INTO t (fp, rating, cam) VALUES (?1, ?2, ?3) \
                      ON CONFLICT (fp) DO UPDATE SET \
                        rating = COALESCE(excluded.rating, t.rating), \
                        cam    = COALESCE(t.cam, excluded.cam)";

        // 1. First insert (no conflict): the row is created.
        conn.execute(upsert, params!["a", 3i32, "CanonX"])
            .expect("insert 1");

        // 2. Conflict with a NEW rating + a different cam:
        //    rating -> 5 (Lightroom-wins), cam stays CanonX (fact, fill-if-missing).
        conn.execute(upsert, params!["a", 5i32, "CanonY"])
            .expect("insert 2");
        let (rating, cam): (i64, String) = conn
            .query_row("SELECT rating, cam FROM t WHERE fp = 'a'", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .expect("select 2");
        assert_eq!(rating, 5, "Lightroom-wins should overwrite rating");
        assert_eq!(
            cam, "CanonX",
            "fact should be fill-if-missing (keep existing)"
        );

        // 3. Conflict with a NULL rating must NOT erase the existing 5.
        conn.execute(upsert, params!["a", Option::<i32>::None, "CanonZ"])
            .expect("insert 3");
        let rating_after: Option<i64> = conn
            .query_row("SELECT rating FROM t WHERE fp = 'a'", [], |r| r.get(0))
            .expect("select 3");
        assert_eq!(
            rating_after,
            Some(5),
            "LR-null must not erase existing curation"
        );

        // Upserts, not duplicate inserts: exactly one row.
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM t", [], |r| r.get(0))
            .expect("count");
        assert_eq!(count, 1);
    }

    // ---- videos table DDL probe (de-risks §8 before merge_lightroom_videos) ----
    //
    // Validates the new `videos` schema against the BUNDLED engine: the
    // sequence-default PK (DuckDB drops IDENTITY), and the DOUBLE / BOOLEAN /
    // BIGINT column types not otherwise exercised by the images schema.
    #[test]
    fn videos_table_ddl_and_insert() {
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
             );",
        )
        .expect("create videos table");

        // Insert omitting id (sequence default) + exercising DOUBLE/BOOLEAN/BIGINT.
        conn.execute(
            "INSERT INTO videos
                (file_path, file_size, file_name, file_extension, directory_path,
                 created_timestamp, modified_timestamp, capture_datetime,
                 pixel_width, pixel_height, duration_seconds, frame_rate, has_audio,
                 video_kind, rating, flag, color_label)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
            params![
                "/v/clip.mov",
                4_087_595_000i64,
                "clip.mov",
                "mov",
                "/v",
                1_700_000_000i64,
                1_700_000_000i64,
                "2025-01-02T03:04:05",
                1920i32,
                1080i32,
                204.4f64,
                59.94f64,
                true,
                "mov",
                4i32,
                "pick",
                "blue"
            ],
        )
        .expect("insert video");

        let (id, dur, fps, audio, kind): (i64, f64, f64, bool, String) = conn
            .query_row(
                "SELECT id, duration_seconds, frame_rate, has_audio, video_kind \
             FROM videos WHERE file_path = '/v/clip.mov'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
            )
            .expect("read video");

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
    fn duckdb_upsert_keeps_stored_id_stable() {
        let conn = Connection::open_in_memory().expect("open in-memory duckdb");
        conn.execute_batch(
            "CREATE SEQUENCE s START 1;
             CREATE TABLE m (id INTEGER PRIMARY KEY DEFAULT nextval('s'), fp TEXT UNIQUE, rating INTEGER);"
        ).expect("create");

        conn.execute("INSERT INTO m (fp, rating) VALUES ('a', 3)", [])
            .expect("insert a");
        let id_before: i64 = conn
            .query_row("SELECT id FROM m WHERE fp = 'a'", [], |r| r.get(0))
            .expect("id before");

        // Conflicting upsert (curation COALESCE direction).
        conn.execute(
            "INSERT INTO m (fp, rating) VALUES ('a', 5) \
             ON CONFLICT (fp) DO UPDATE SET rating = COALESCE(excluded.rating, m.rating)",
            [],
        )
        .expect("upsert a");

        let id_after: i64 = conn
            .query_row("SELECT id FROM m WHERE fp = 'a'", [], |r| r.get(0))
            .expect("id after");
        let rating_after: i64 = conn
            .query_row("SELECT rating FROM m WHERE fp = 'a'", [], |r| r.get(0))
            .expect("rating after");

        assert_eq!(
            id_before, id_after,
            "STORED id must be stable across ON CONFLICT update (keyword FKs depend on it)"
        );
        assert_eq!(rating_after, 5, "curation still updates (Lightroom-wins)");
    }

    // ---- merge_records_into: insert / update / policies / id-stability ----

    // A minimal `images` table covering exactly the columns the merge touches.
    fn create_images(conn: &Connection) {
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
                rating INTEGER, flag TEXT, color_label TEXT,
                rotation INTEGER DEFAULT 0,
                external_source_id TEXT,
                is_video BOOLEAN DEFAULT FALSE,
                duration_seconds DOUBLE, frame_rate DOUBLE, video_kind TEXT, video_codec TEXT, video_bitrate BIGINT,
                color_primaries TEXT, color_transfer TEXT, color_matrix TEXT, color_range TEXT, dv_profile INTEGER,
                has_audio BOOLEAN, audio_codec TEXT, audio_channels INTEGER, audio_sample_rate INTEGER, audio_bitrate BIGINT,
                live_photo_id TEXT
             );",
        )
        .expect("create images");
    }

    // A default ImageMetadata; override fields per test.
    fn img(file_path: &str, file_name: &str) -> ImageMetadata {
        ImageMetadata {
            file_path: file_path.to_string(),
            file_size: 1000,
            file_name: file_name.to_string(),
            file_extension: Some("nef".to_string()),
            created_timestamp: 1_700_000_000,
            modified_timestamp: 1_700_000_000,
            camera_make: None,
            camera_model: None,
            lens_model: None,
            focal_length: None,
            aperture: None,
            shutter_speed: None,
            iso: None,
            capture_datetime: None,
            pixel_width: None,
            pixel_height: None,
            color_space: None,
            bit_depth: None,
            gps_latitude: None,
            gps_longitude: None,
            gps_altitude: None,
            copyright: None,
            creator: None,
            description: None,
            rating: None,
            flag: None,
            color_label: None,
            rotation: None,
            is_video: false,
            duration_seconds: None,
            frame_rate: None,
            video_kind: None,
            video_codec: None,
            video_bitrate: None,
            color_primaries: None,
            color_transfer: None,
            color_matrix: None,
            color_range: None,
            dv_profile: None,
            has_audio: None,
            audio_codec: None,
            audio_channels: None,
            audio_sample_rate: None,
            audio_bitrate: None,
            live_photo_id: None,
            external_source_id: None,
        }
    }

    #[test]
    fn merge_inserts_then_updates_with_policies() {
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
        let (kind, dir): (String, String) = conn
            .query_row(
                "SELECT image_kind, directory_path FROM images WHERE file_path = '/p/a.nef'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .expect("derived cols");
        assert_eq!(kind, "raw");
        assert_eq!(dir, "/p");

        // 2. Re-merge 'a': new rating (LR-wins), a fact that was missing (fills),
        //    and a fact already set (must NOT overwrite).
        let mut a2 = img("/p/a.nef", "a.nef");
        a2.rating = Some(5); // LR-wins -> 5
        a2.capture_datetime = Some("2025-01-01T00:00:00".to_string()); // fact was NULL -> fills
        a2.camera_model = Some("WRONG".to_string()); // fact set -> keep "Nikon Z8"
        let r2 = merge_records_into(&conn, &[a2]);
        assert_eq!(r2.inserted, 0);
        assert_eq!(r2.updated, 1);
        assert_eq!(
            r2.image_ids[0], id_a,
            "id stable across re-merge (FK safety)"
        );

        let (rating, cap, cam): (i64, String, String) = conn.query_row(
            "SELECT rating, capture_datetime, camera_model FROM images WHERE file_path = '/p/a.nef'", [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        ).expect("post-update");
        assert_eq!(rating, 5, "curation: Lightroom wins");
        assert_eq!(cap, "2025-01-01T00:00:00", "fact: filled when missing");
        assert_eq!(
            cam, "Nikon Z8",
            "fact: existing value kept, not overwritten"
        );

        // 3. LR-null rating must NOT erase the existing 5.
        let a3 = img("/p/a.nef", "a.nef"); // rating None
        let r3 = merge_records_into(&conn, &[a3]);
        assert_eq!(r3.updated, 1);
        let rating2: i64 = conn
            .query_row(
                "SELECT rating FROM images WHERE file_path = '/p/a.nef'",
                [],
                |r| r.get(0),
            )
            .expect("rating after null");
        assert_eq!(rating2, 5, "LR-null must not erase existing curation");

        // 4. rotation (S67): None at INSERT takes the schema default 0; Some
        //    at INSERT lands; None at UPDATE preserves (an LR re-import can
        //    never clobber an in-app rotation).
        let rot_default: i64 = conn
            .query_row(
                "SELECT rotation FROM images WHERE file_path = '/p/b.nef'",
                [],
                |r| r.get(0),
            )
            .expect("rotation default");
        assert_eq!(
            rot_default, 0,
            "rotation: None at insert -> schema default 0"
        );

        let mut c = img("/p/c.jpg", "c.jpg");
        c.rotation = Some(90);
        let rc = merge_records_into(&conn, &[c]);
        assert_eq!(rc.inserted, 1);
        let rot_c: i64 = conn
            .query_row(
                "SELECT rotation FROM images WHERE file_path = '/p/c.jpg'",
                [],
                |r| r.get(0),
            )
            .expect("rotation inserted");
        assert_eq!(rot_c, 90, "rotation: Some at insert lands");

        let c2 = img("/p/c.jpg", "c.jpg"); // rotation None
        let rc2 = merge_records_into(&conn, &[c2]);
        assert_eq!(rc2.updated, 1);
        let rot_c2: i64 = conn
            .query_row(
                "SELECT rotation FROM images WHERE file_path = '/p/c.jpg'",
                [],
                |r| r.get(0),
            )
            .expect("rotation after null update");
        assert_eq!(
            rot_c2, 90,
            "rotation: None at update preserves the row's value"
        );
    }

    #[test]
    fn merge_inserts_video_row_into_images() {
        // The media-aware merge writes is_video + the video columns into the
        // images table (Apple Photos import + the LR video fold-in rely on this;
        // it mirrors ingest_metadata's video binding).
        let conn = Connection::open_in_memory().expect("open in-memory duckdb");
        create_images(&conn);

        let mut v = img("/v/clip.mov", "clip.mov");
        v.is_video = true;
        v.duration_seconds = Some(12.5);
        v.frame_rate = Some(29.97);
        v.video_kind = Some("mov".to_string());
        v.video_codec = Some("hevc".to_string());
        v.has_audio = Some(true);
        v.audio_codec = Some("aac".to_string());
        v.live_photo_id = Some("ABC-123".to_string());
        v.external_source_id = Some("ZUUID-1".to_string());

        let r = merge_records_into(&conn, &[v]);
        assert_eq!(r.inserted, 1);
        assert_eq!(r.updated, 0);

        let (is_video, codec, ext_id): (bool, String, String) = conn
            .query_row(
                "SELECT is_video, video_codec, external_source_id \
                 FROM images WHERE file_path = '/v/clip.mov'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .expect("video row round-trips");
        assert!(is_video, "is_video persisted true");
        assert_eq!(codec, "hevc");
        assert_eq!(ext_id, "ZUUID-1");

        let dur: f64 = conn
            .query_row(
                "SELECT duration_seconds FROM images WHERE file_path = '/v/clip.mov'",
                [],
                |r| r.get(0),
            )
            .expect("duration round-trips");
        assert_eq!(dur, 12.5);

        // A still still inserts with is_video = false (no regression).
        let rs = merge_records_into(&conn, &[img("/p/still.jpg", "still.jpg")]);
        assert_eq!(rs.inserted, 1);
        let still_is_video: bool = conn
            .query_row(
                "SELECT is_video FROM images WHERE file_path = '/p/still.jpg'",
                [],
                |r| r.get(0),
            )
            .expect("still is_video");
        assert!(!still_is_video, "still inserts is_video = false");
    }

    // ---- merge_videos_into: insert / update / policies (the videos table) ----

    fn create_videos(conn: &Connection) {
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
             );",
        )
        .expect("create videos");
    }

    fn vid(file_path: &str, file_name: &str) -> LightroomVideoRecord {
        LightroomVideoRecord {
            file_path: file_path.to_string(),
            file_size: 5000,
            file_name: file_name.to_string(),
            file_extension: Some("mov".to_string()),
            created_timestamp: 1_700_000_000,
            modified_timestamp: 1_700_000_000,
            capture_datetime: None,
            pixel_width: None,
            pixel_height: None,
            duration_seconds: None,
            frame_rate: None,
            has_audio: None,
            video_kind: Some("mov".to_string()),
            rating: None,
            flag: None,
            color_label: None,
        }
    }

    #[test]
    fn merge_videos_inserts_then_updates() {
        let conn = Connection::open_in_memory().expect("open in-memory duckdb");
        create_videos(&conn);

        // 1. Insert a new video with a fact (duration) + curation (rating).
        let mut a = vid("/v/a.mov", "a.mov");
        a.duration_seconds = Some(204.4);
        a.rating = Some(2);
        let r1 = merge_videos_into(&conn, &[a]);
        assert_eq!(r1.inserted, 1);
        let id = r1.image_ids[0];
        let dir: String = conn
            .query_row(
                "SELECT directory_path FROM videos WHERE file_path = '/v/a.mov'",
                [],
                |r| r.get(0),
            )
            .expect("dir");
        assert_eq!(dir, "/v");

        // 2. Re-merge: rating LR-wins; duration (fact, already set) must NOT change.
        let mut a2 = vid("/v/a.mov", "a.mov");
        a2.rating = Some(4);
        a2.duration_seconds = Some(999.0);
        let r2 = merge_videos_into(&conn, &[a2]);
        assert_eq!(r2.updated, 1);
        assert_eq!(r2.image_ids[0], id, "video id stable across re-merge");

        let (rating, dur): (i64, f64) = conn
            .query_row(
                "SELECT rating, duration_seconds FROM videos WHERE file_path = '/v/a.mov'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .expect("post-update");
        assert_eq!(rating, 4, "curation: Lightroom wins");
        assert!(
            (dur - 204.4).abs() < 1e-6,
            "fact: existing duration kept, not overwritten"
        );
    }
}

/// Bulk-set the pick/reject flag on many records in ONE statement (Browse
/// "Set Flag" on a selection / the whole query). Mirrors `update_image_flag`'s
/// allow-list guard: `None` clears; any value outside {pick, reject} rejects
/// the WHOLE update (→ 0). Returns the number of rows changed.
pub async fn update_flag_for_ids(ids: Vec<i64>, flag: Option<String>) -> u64 {
    let flag_value: Option<String> = match flag.as_deref() {
        None => None,
        Some(v @ ("pick" | "reject")) => Some(v.to_string()),
        Some(other) => {
            eprintln!("Rejected invalid flag value '{}' for bulk update", other);
            return 0;
        }
    };

    let where_clause = match id_in_list(&ids) {
        Some(w) => w,
        None => return 0,
    };

    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    let update_sql = format!("UPDATE images SET flag = ? WHERE {}", where_clause);
    match conn.execute(&update_sql, params![flag_value]) {
        Ok(changed) => changed as u64,
        Err(e) => {
            eprintln!("Failed to bulk-update flag: {}", e);
            0
        }
    }
}

/// Bulk-set the color label on many records in ONE statement. Mirrors
/// `update_image_color_label`'s allow-list guard. Returns rows changed.
pub async fn update_color_label_for_ids(ids: Vec<i64>, color_label: Option<String>) -> u64 {
    let label_value: Option<String> = match color_label.as_deref() {
        None => None,
        Some(v @ ("red" | "yellow" | "green" | "blue" | "purple")) => Some(v.to_string()),
        Some(other) => {
            eprintln!("Rejected invalid color label '{}' for bulk update", other);
            return 0;
        }
    };

    let where_clause = match id_in_list(&ids) {
        Some(w) => w,
        None => return 0,
    };

    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    let update_sql = format!("UPDATE images SET color_label = ? WHERE {}", where_clause);
    match conn.execute(&update_sql, params![label_value]) {
        Ok(changed) => changed as u64,
        Err(e) => {
            eprintln!("Failed to bulk-update color label: {}", e);
            0
        }
    }
}

/// Bulk-set the star rating on many records in ONE statement. Rating 0 clears
/// (NULL), mirroring `update_image_rating`. Returns rows changed.
pub async fn update_rating_for_ids(ids: Vec<i64>, rating: u32) -> u64 {
    let rating_value: Option<i64> = if rating == 0 {
        None
    } else {
        Some(rating as i64)
    };

    let where_clause = match id_in_list(&ids) {
        Some(w) => w,
        None => return 0,
    };

    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    let update_sql = format!("UPDATE images SET rating = ? WHERE {}", where_clause);
    match conn.execute(&update_sql, params![rating_value]) {
        Ok(changed) => changed as u64,
        Err(e) => {
            eprintln!("Failed to bulk-update rating: {}", e);
            0
        }
    }
}

#[cfg(test)]
mod query_builder_tests {
    use super::*;

    #[test]
    fn media_predicate_maps_each_stance() {
        // The §11 media-type seam: StillsOnly / VideosOnly gate on is_video
        // (both NULL-safe), Both applies no media filter (None → nothing pushed).
        assert_eq!(
            media_predicate(MediaType::StillsOnly),
            Some("is_video IS NOT TRUE")
        );
        assert_eq!(
            media_predicate(MediaType::VideosOnly),
            Some("is_video IS TRUE")
        );
        assert_eq!(media_predicate(MediaType::Both), None);
    }

    // S74 — get_video_details decode: column order, integer down-casts, and the
    // is_video guard. Builds a minimal images table (the video-only columns this
    // getter reads), inserts one video row + one still, and runs the production
    // SELECT through the shared `row_to_video_details` decoder.
    #[test]
    fn video_details_decodes_and_guards_on_is_video() {
        let conn = Connection::open_in_memory().expect("open in-memory duckdb");
        conn.execute_batch(
            "CREATE TABLE images (
                id INTEGER PRIMARY KEY,
                is_video BOOLEAN,
                duration_seconds DOUBLE,
                frame_rate DOUBLE,
                video_kind TEXT,
                video_codec TEXT,
                video_bitrate BIGINT,
                color_primaries TEXT,
                color_transfer TEXT,
                color_matrix TEXT,
                color_range TEXT,
                dv_profile INTEGER,
                has_audio BOOLEAN,
                audio_codec TEXT,
                audio_channels INTEGER,
                audio_sample_rate INTEGER,
                audio_bitrate BIGINT,
                live_photo_id TEXT
             );",
        )
        .expect("create images");

        // id=1 video — iPhone HDR shape: HLG transfer + Dolby Vision profile 8,
        // AAC stereo. (Plain digit literals — no Rust-style underscores in SQL.)
        conn.execute(
            "INSERT INTO images VALUES
             (1, TRUE, 12.5, 29.97, 'mov', 'hevc', 45000000,
              'bt2020', 'arib-std-b67', 'bt2020nc', 'tv', 8,
              TRUE, 'aac', 2, 48000, 160000, 'ABC-123')",
            [],
        )
        .expect("insert video");

        // id=2 still — must be invisible to the getter (the is_video guard).
        conn.execute("INSERT INTO images (id, is_video) VALUES (2, FALSE)", [])
            .expect("insert still");

        let sql = "SELECT duration_seconds, frame_rate, video_kind, video_codec, \
                   video_bitrate, color_primaries, color_transfer, color_matrix, \
                   color_range, dv_profile, has_audio, audio_codec, audio_channels, \
                   audio_sample_rate, audio_bitrate, live_photo_id \
                   FROM images WHERE id = ?1 AND is_video IS TRUE";

        let v = conn
            .query_row(sql, params![1i64], row_to_video_details)
            .expect("video row decodes");
        assert!((v.duration_seconds.unwrap() - 12.5).abs() < 1e-6);
        assert!((v.frame_rate.unwrap() - 29.97).abs() < 1e-6);
        assert_eq!(v.video_codec.as_deref(), Some("hevc"));
        assert_eq!(v.video_bitrate, Some(45_000_000)); // BIGINT → i64
        assert_eq!(v.color_transfer.as_deref(), Some("arib-std-b67"));
        assert_eq!(v.dv_profile, Some(8)); // INTEGER → i32 down-cast
        assert_eq!(v.has_audio, Some(true));
        assert_eq!(v.audio_channels, Some(2)); // INTEGER → i32 down-cast
        assert_eq!(v.audio_sample_rate, Some(48000));
        assert_eq!(v.live_photo_id.as_deref(), Some("ABC-123"));

        // The is_video guard hides the still: query_row finds no row → None.
        let still = conn
            .query_row(sql, params![2i64], row_to_video_details)
            .ok();
        assert!(
            still.is_none(),
            "is_video IS TRUE must exclude the still row"
        );
    }

    // S87 — get_external_source_id decode: a present Apple row returns its stored
    // ZUUID; a NULL column (an ordinary non-Apple row) and an absent id both read
    // as None. Mirrors the production SELECT + Option-flatten decode (the getter
    // itself locks CATALOGUE, so the SQL is exercised in-test the same way the
    // video_details test does).
    #[test]
    fn external_source_id_reads_present_null_and_absent() {
        let conn = Connection::open_in_memory().expect("open in-memory duckdb");
        conn.execute_batch(
            "CREATE TABLE images (
                id INTEGER PRIMARY KEY,
                external_source_id TEXT
             );",
        )
        .expect("create images");

        // id=1 — an Apple-managed row carrying its ZASSET.ZUUID.
        // id=2 — an ordinary row (scan / Lightroom import): external_source_id NULL.
        conn.execute(
            "INSERT INTO images VALUES (1, '937830D8-1234-5678-9ABC-9FB11190D51D'), (2, NULL)",
            [],
        )
        .expect("insert rows");

        let sql = "SELECT external_source_id FROM images WHERE id = ?1";
        let read = |id: i64| -> Option<String> {
            conn.query_row(sql, params![id], |row| row.get::<_, Option<String>>(0))
                .ok()
                .flatten()
        };

        assert_eq!(
            read(1).as_deref(),
            Some("937830D8-1234-5678-9ABC-9FB11190D51D")
        );
        assert_eq!(
            read(2),
            None,
            "a NULL external_source_id (non-Apple row) reads as None"
        );
        assert_eq!(read(99), None, "an absent id reads as None");
    }

    fn qp(kind: &str) -> QueryPredicate {
        QueryPredicate {
            kind: kind.to_string(),
            day: None,
            day_end: None,
            op: None,
            stars: None,
            value: None,
            num: None,
            num_end: None,
        }
    }

    fn rating(op: &str, stars: u8) -> QueryPredicate {
        let mut p = qp("rating");
        p.op = Some(op.to_string());
        p.stars = Some(stars);
        p
    }

    fn flag(value: &str) -> QueryPredicate {
        let mut p = qp("flag");
        p.value = Some(value.to_string());
        p
    }

    fn value_pred(kind: &str, value: &str) -> QueryPredicate {
        let mut p = qp(kind);
        p.value = Some(value.to_string());
        p
    }

    fn color(value: &str) -> QueryPredicate {
        let mut p = qp("color");
        p.value = Some(value.to_string());
        p
    }

    fn num(kind: &str, op: &str, value: f64, value_end: Option<f64>) -> QueryPredicate {
        let mut p = qp(kind);
        p.op = Some(op.to_string());
        p.num = Some(value);
        p.num_end = value_end;
        p
    }

    #[test]
    fn day_validation() {
        assert!(is_valid_day("2026:05:15"));
        assert!(!is_valid_day("2026-05-15")); // dashes, not colons
        assert!(!is_valid_day("2026:5:15")); // wrong length
        assert!(!is_valid_day("abcd:ef:gh")); // non-digits
        assert!(!is_valid_day(""));
    }

    #[test]
    fn atom_sql() {
        assert_eq!(predicate_to_sql(&rating("gte", 4)), "(rating >= 4)");
        assert_eq!(predicate_to_sql(&flag("pick")), "(flag = 'pick')");
        assert_eq!(predicate_to_sql(&color("red")), "(color_label = 'red')");
        assert_eq!(predicate_to_sql(&qp("rating_unrated")), "(rating IS NULL)");
        assert_eq!(predicate_to_sql(&qp("unflagged")), "(flag IS NULL)");
        let mut fou = qp("flag_or_unflagged");
        fou.value = Some("pick".to_string());
        assert_eq!(predicate_to_sql(&fou), "(flag = 'pick' OR flag IS NULL)");
    }

    // S75 — Query Builder video fields. Duration / frame rate ride the shared
    // numeric machinery on new columns; codec is an exact metadata match. Mirrors
    // the existing numeric + metadata arms; checks the SQL, BETWEEN, quote-escape,
    // and the (FALSE) backstop for a missing value/op.
    #[test]
    fn video_subject_atoms() {
        let mut dur = qp("duration_num");
        dur.op = Some("gte".to_string());
        dur.num = Some(60.0);
        assert_eq!(predicate_to_sql(&dur), "(duration_seconds >= 60)");

        let mut between = qp("duration_num");
        between.op = Some("between".to_string());
        between.num = Some(10.0);
        between.num_end = Some(30.0);
        assert_eq!(
            predicate_to_sql(&between),
            "(duration_seconds BETWEEN 10 AND 30)"
        );

        let mut fps = qp("framerate_num");
        fps.op = Some("eq".to_string());
        fps.num = Some(60.0);
        assert_eq!(predicate_to_sql(&fps), "(frame_rate = 60)");

        let mut codec = qp("codec_is");
        codec.value = Some("hevc".to_string());
        assert_eq!(predicate_to_sql(&codec), "(video_codec = 'hevc')");

        // Single-quote escape — defensive parity with the other metadata arms.
        let mut q = qp("codec_is");
        q.value = Some("a'b".to_string());
        assert_eq!(predicate_to_sql(&q), "(video_codec = 'a''b')");

        // Missing value/op → the (FALSE) backstop.
        assert_eq!(predicate_to_sql(&qp("codec_is")), "(FALSE)");
        assert_eq!(predicate_to_sql(&qp("duration_num")), "(FALSE)");
    }

    // S75 — the video fixed-choice subjects (dynamic range / audio / live photo).
    // Kind-only predicates: the kind alone is the SQL, no value/op. SDR and
    // "not a live photo" carry an is_video guard so they never match a still.
    #[test]
    fn video_choice_atoms() {
        assert_eq!(
            predicate_to_sql(&qp("dynrange_hlg")),
            "(color_transfer = 'arib-std-b67')"
        );
        assert_eq!(
            predicate_to_sql(&qp("dynrange_hdr10")),
            "(color_transfer = 'smpte2084')"
        );
        assert_eq!(
            predicate_to_sql(&qp("dynrange_dv")),
            "(dv_profile IS NOT NULL)"
        );
        assert_eq!(predicate_to_sql(&qp("audio_present")), "(has_audio = TRUE)");
        assert_eq!(predicate_to_sql(&qp("audio_absent")), "(has_audio = FALSE)");
        assert_eq!(
            predicate_to_sql(&qp("livephoto_yes")),
            "(is_video IS TRUE AND live_photo_id IS NOT NULL)"
        );
        assert!(predicate_to_sql(&qp("dynrange_sdr")).contains("is_video IS TRUE"));
        assert!(predicate_to_sql(&qp("livephoto_no")).contains("is_video IS TRUE"));
    }

    #[test]
    fn date_atoms() {
        let mut eq = qp("date_equals");
        eq.day = Some("2026:05:15".to_string());
        assert_eq!(
            predicate_to_sql(&eq),
            "(SUBSTRING(capture_datetime, 1, 10) = '2026:05:15')"
        );

        let mut ge = qp("date_after");
        ge.day = Some("2026:05:15".to_string());
        assert_eq!(
            predicate_to_sql(&ge),
            "(SUBSTRING(capture_datetime, 1, 10) >= '2026:05:15')"
        );

        let mut le = qp("date_before");
        le.day = Some("2026:05:15".to_string());
        assert_eq!(
            predicate_to_sql(&le),
            "(SUBSTRING(capture_datetime, 1, 10) <= '2026:05:15')"
        );

        let mut gt = qp("date_gt");
        gt.day = Some("2026:05:15".to_string());
        assert_eq!(
            predicate_to_sql(&gt),
            "(SUBSTRING(capture_datetime, 1, 10) > '2026:05:15')"
        );

        let mut lt = qp("date_lt");
        lt.day = Some("2026:05:15".to_string());
        assert_eq!(
            predicate_to_sql(&lt),
            "(SUBSTRING(capture_datetime, 1, 10) < '2026:05:15')"
        );

        let mut bt = qp("date_between");
        bt.day = Some("2026:01:01".to_string());
        bt.day_end = Some("2026:03:31".to_string());
        assert_eq!(
            predicate_to_sql(&bt),
            "(SUBSTRING(capture_datetime, 1, 10) BETWEEN '2026:01:01' AND '2026:03:31')"
        );
    }

    #[test]
    fn invalid_atoms_become_false() {
        assert_eq!(predicate_to_sql(&flag("bogus")), "(FALSE)");
        assert_eq!(predicate_to_sql(&rating("gte", 6)), "(FALSE)"); // stars out of range
        assert_eq!(predicate_to_sql(&color("teal")), "(FALSE)");
        let mut bad_date = qp("date_equals");
        bad_date.day = Some("2026-05-15".to_string()); // dashes → rejected
        assert_eq!(predicate_to_sql(&bad_date), "(FALSE)");
    }

    #[test]
    fn empty_predicates_no_where() {
        assert_eq!(build_filter_predicate(&[], &[]), "");
    }

    #[test]
    fn scoped_filter_keeps_query_and_sidebar_groups_separate() {
        let query = vec![rating("gte", 4), flag("pick")];
        let query_connectors = vec![Connector::Or];
        let scope = vec![
            value_pred("path_prefix", "/Photos/A/"),
            value_pred("path_prefix", "/Photos/B/"),
        ];
        let scope_connectors = vec![Connector::Or];

        let scoped =
            build_scoped_filter_predicate(&query, &query_connectors, &scope, &scope_connectors);

        assert_eq!(
            scoped,
            format!(
                "({}) AND ({})",
                build_filter_predicate(&query, &query_connectors),
                build_filter_predicate(&scope, &scope_connectors)
            )
        );
    }

    #[test]
    fn id_in_list_assembly() {
        assert_eq!(id_in_list(&[]), None);
        assert_eq!(id_in_list(&[5]), Some("id IN (5)".to_string()));
        assert_eq!(id_in_list(&[1, 2, 3]), Some("id IN (1, 2, 3)".to_string()));
    }

    #[test]
    fn ordered_id_projection_collapses_raw_hits_to_visible_jpeg() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE images (
                id INTEGER,
                file_path TEXT,
                file_stem TEXT,
                image_kind TEXT,
                directory_path TEXT
             );
             INSERT INTO images (id, file_path, file_stem, image_kind, directory_path) VALUES
                (1, '/photos/a.nef', 'a', 'raw', '/photos'),
                (2, '/photos/a.jpg', 'a', 'jpeg', '/photos'),
                (3, '/photos/b.nef', 'b', 'raw', '/photos'),
                (4, '/photos/b.jpg', 'b', 'jpeg', '/photos'),
                (5, '/photos/c.nef', 'c', 'raw', '/photos');",
        )
        .expect("seed images");

        let input = vec![3, 1, 2, 4, 5];
        assert_eq!(
            project_raw_jpeg_visible_ids_impl(&conn, &input, false),
            input
        );
        assert_eq!(
            project_raw_jpeg_visible_ids_impl(&conn, &input, true),
            vec![4, 2, 5]
        );
    }

    #[test]
    fn order_by_follows_first_subject() {
        // Rating-first → stars best-to-worst.
        assert_eq!(
            order_by_for_filter(&[rating("gte", 4)]),
            RATING_FILTER_ORDER_BY
        );
        assert_eq!(
            order_by_for_filter(&[qp("rating_unrated")]),
            RATING_FILTER_ORDER_BY
        );
        assert_eq!(
            order_by_for_filter(&[num("focus_quality", "gte", 80.0, None)]),
            FOCUS_FILTER_ORDER_BY
        );
        // The FIRST subject wins even when rating appears later.
        assert_eq!(
            order_by_for_filter(&[flag("pick"), rating("gte", 4)]),
            DEFAULT_FILTER_ORDER_BY
        );
        // Date-, flag-, color-first, and empty → the default newest-first.
        let mut d = qp("date_after");
        d.day = Some("2026:05:15".to_string());
        assert_eq!(order_by_for_filter(&[d]), DEFAULT_FILTER_ORDER_BY);
        assert_eq!(
            order_by_for_filter(&[flag("pick")]),
            DEFAULT_FILTER_ORDER_BY
        );
        assert_eq!(
            order_by_for_filter(&[color("red")]),
            DEFAULT_FILTER_ORDER_BY
        );
        assert_eq!(order_by_for_filter(&[]), DEFAULT_FILTER_ORDER_BY);
    }

    #[test]
    fn focus_order_survives_duplicate_filter_projection() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE images (
                id INTEGER,
                indexed_timestamp TIMESTAMP,
                file_path TEXT,
                file_size BIGINT,
                file_name TEXT,
                file_extension TEXT,
                created_timestamp BIGINT,
                modified_timestamp BIGINT,
                camera_make TEXT,
                camera_model TEXT,
                lens_model TEXT,
                focal_length DOUBLE,
                aperture DOUBLE,
                shutter_speed DOUBLE,
                iso INTEGER,
                capture_datetime TEXT,
                pixel_width INTEGER,
                pixel_height INTEGER,
                color_space TEXT,
                bit_depth INTEGER,
                gps_latitude DOUBLE,
                gps_longitude DOUBLE,
                gps_altitude DOUBLE,
                copyright TEXT,
                creator TEXT,
                description TEXT,
                rating INTEGER,
                flag TEXT,
                color_label TEXT,
                rotation INTEGER,
                file_stem TEXT,
                image_kind TEXT,
                is_video BOOLEAN,
                focus_score DOUBLE
             );",
        )
        .expect("create images");

        conn.execute(
            "INSERT INTO images (
                id, indexed_timestamp, file_path, file_size, file_name,
                file_extension, created_timestamp, modified_timestamp,
                camera_model, capture_datetime, pixel_width, pixel_height,
                rotation, file_stem, image_kind, is_video, focus_score
             ) VALUES (
                1, TIMESTAMP '2026-06-15 10:00:00', '/tmp/a.jpg', 1000, 'a.jpg',
                'jpg', 1700000000, 1700000001,
                'Z 8', '2026:06:15 10:00:00', 100, 100,
                0, 'a', 'jpeg', FALSE, 3.5
             )",
            [],
        )
        .expect("insert image");

        let records = execute_image_record_query(
            &conn,
            "(focus_score >= 3.5)",
            FOCUS_FILTER_ORDER_BY,
            10,
            0,
            true,
            false,
            false,
            "",
            MediaType::StillsOnly,
        );

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].id, 1);
    }

    #[test]
    fn similar_collapse_picks_visible_jpeg_even_when_durable_rep_is_raw() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE images (
                id INTEGER,
                indexed_timestamp TIMESTAMP,
                file_path TEXT,
                file_size BIGINT,
                file_name TEXT,
                file_extension TEXT,
                created_timestamp BIGINT,
                modified_timestamp BIGINT,
                camera_make TEXT,
                camera_model TEXT,
                lens_model TEXT,
                focal_length DOUBLE,
                aperture DOUBLE,
                shutter_speed DOUBLE,
                iso INTEGER,
                capture_datetime TEXT,
                pixel_width INTEGER,
                pixel_height INTEGER,
                color_space TEXT,
                bit_depth INTEGER,
                gps_latitude DOUBLE,
                gps_longitude DOUBLE,
                gps_altitude DOUBLE,
                copyright TEXT,
                creator TEXT,
                description TEXT,
                rating INTEGER,
                flag TEXT,
                color_label TEXT,
                rotation INTEGER,
                file_stem TEXT,
                image_kind TEXT,
                directory_path TEXT,
                is_video BOOLEAN,
                focus_score DOUBLE
             );
             CREATE TABLE similar_photo_group_member (
                image_id BIGINT NOT NULL,
                group_id BIGINT NOT NULL,
                representative_id BIGINT NOT NULL,
                member_rank INTEGER NOT NULL,
                distance_to_representative DOUBLE,
                algorithm_version TEXT NOT NULL,
                threshold DOUBLE NOT NULL
             );",
        )
        .expect("create tables");

        conn.execute_batch(
            "INSERT INTO images (
                id, indexed_timestamp, file_path, file_size, file_name,
                file_extension, created_timestamp, modified_timestamp,
                camera_model, capture_datetime, pixel_width, pixel_height,
                rotation, file_stem, image_kind, directory_path, is_video, focus_score
             ) VALUES
                (1, TIMESTAMP '2026-06-15 10:00:00', '/tmp/a.nef', 1000, 'a.nef',
                 'nef', 1700000000, 1700000001, 'Z 8', '2026:06:15 10:00:00',
                 100, 100, 0, 'a', 'raw', '/tmp', FALSE, 3.5),
                (2, TIMESTAMP '2026-06-15 10:00:01', '/tmp/a.jpg', 900, 'a.jpg',
                 'jpg', 1700000000, 1700000001, 'Z 8', '2026:06:15 10:00:01',
                 100, 100, 0, 'a', 'jpeg', '/tmp', FALSE, 3.6),
                (3, TIMESTAMP '2026-06-15 10:00:02', '/tmp/b.nef', 1000, 'b.nef',
                 'nef', 1700000002, 1700000003, 'Z 8', '2026:06:15 10:00:02',
                 100, 100, 0, 'b', 'raw', '/tmp', FALSE, 3.4),
                (4, TIMESTAMP '2026-06-15 10:00:03', '/tmp/b.jpg', 900, 'b.jpg',
                 'jpg', 1700000002, 1700000003, 'Z 8', '2026:06:15 10:00:03',
                 100, 100, 0, 'b', 'jpeg', '/tmp', FALSE, 3.7);

             INSERT INTO similar_photo_group_member VALUES
                (1, 10, 1, 0, 0.0, 'v-test', 2.35),
                (2, 10, 1, 1, 0.1, 'v-test', 2.35),
                (3, 10, 1, 2, 0.2, 'v-test', 2.35),
                (4, 10, 1, 3, 0.3, 'v-test', 2.35);",
        )
        .expect("insert rows");

        let raw_visible_records = execute_image_record_query(
            &conn,
            "",
            "id ASC",
            10,
            0,
            false,
            false,
            true,
            "v-test",
            MediaType::StillsOnly,
        );
        assert_eq!(raw_visible_records.len(), 1);
        assert_eq!(
            raw_visible_records[0].id, 2,
            "similar collapse should prefer the JPEG representative even when RAW rows are otherwise visible"
        );

        let raw_collapsed_records = execute_image_record_query(
            &conn,
            "",
            "id ASC",
            10,
            0,
            false,
            true,
            true,
            "v-test",
            MediaType::StillsOnly,
        );
        assert_eq!(raw_collapsed_records.len(), 1);
        assert_eq!(
            raw_collapsed_records[0].id, 2,
            "RAW/JPEG collapse must not make a RAW durable representative erase the stack"
        );

        let count = execute_image_count_query(
            &conn,
            "",
            false,
            true,
            true,
            "v-test",
            MediaType::StillsOnly,
        );
        assert_eq!(count, 1);
    }

    #[test]
    fn single_predicate_wrapped() {
        assert_eq!(
            build_filter_predicate(&[rating("gte", 4)], &[]),
            "((rating >= 4))"
        );
    }

    #[test]
    fn left_to_right_accumulation() {
        // "A and B or C" → ((A AND B) OR C), left-to-right, NO precedence.
        let preds = vec![rating("gte", 4), flag("pick"), color("red")];
        let conns = vec![Connector::And, Connector::Or];
        assert_eq!(
            build_filter_predicate(&preds, &conns),
            "((((rating >= 4)) AND ((flag = 'pick'))) OR ((color_label = 'red')))"
        );
    }

    #[test]
    fn xor_is_boolean_inequality() {
        let preds = vec![flag("pick"), color("red")];
        let conns = vec![Connector::Xor];
        assert_eq!(
            build_filter_predicate(&preds, &conns),
            "(((flag = 'pick')) <> ((color_label = 'red')))"
        );
    }
}

#[cfg(test)]
mod folder_sync_tests {
    use super::*;
    use duckdb::Connection;

    fn setup() -> Connection {
        let conn = Connection::open_in_memory().expect("in-memory db");
        conn.execute_batch(
            "CREATE TABLE directory_sync_state (
                 directory_path  TEXT PRIMARY KEY,
                 last_sync_mtime BIGINT NOT NULL
             );
             CREATE TABLE images (id INTEGER, file_path TEXT, directory_path VARCHAR);
             CREATE SEQUENCE keyword_id_seq START 1;
             CREATE TABLE keyword (
                 id INTEGER PRIMARY KEY DEFAULT nextval('keyword_id_seq'),
                 image_id INTEGER NOT NULL,
                 label TEXT NOT NULL,
                 path TEXT NOT NULL,
                 status INTEGER NOT NULL DEFAULT 1,
                 origin INTEGER NOT NULL DEFAULT 1,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                 hidden_at TIMESTAMP,
                 collection BOOLEAN NOT NULL DEFAULT FALSE,
                 color BOOLEAN NOT NULL DEFAULT FALSE
             );",
        )
        .expect("schema");
        conn
    }

    #[test]
    fn sync_state_upsert_read_and_prune() {
        let conn = setup();
        // Both directories exist in the catalogue (prune must keep them).
        conn.execute_batch(
            "INSERT INTO images VALUES
                 (1, '/A/x.jpg', '/A'),
                 (2, '/B/y.jpg', '/B');",
        )
        .expect("seed images");

        // Empty input is a refused no-op.
        assert_eq!(update_directory_sync_states_impl(&conn, &[]), 0);

        // First write: two inserts.
        let first = vec![
            DirectorySyncState {
                directory_path: "/A".into(),
                last_sync_mtime: 1_000,
            },
            DirectorySyncState {
                directory_path: "/B".into(),
                last_sync_mtime: 2_000,
            },
        ];
        assert_eq!(update_directory_sync_states_impl(&conn, &first), 2);

        // Read back, indexed by path.
        let rows = directory_sync_states_impl(&conn);
        assert_eq!(rows.len(), 2);
        let mtime_of = |p: &str| {
            rows.iter()
                .find(|r| r.directory_path == p)
                .map(|r| r.last_sync_mtime)
        };
        assert_eq!(mtime_of("/A"), Some(1_000));
        assert_eq!(mtime_of("/B"), Some(2_000));

        // Second write: /A moves (an UPDATE, not a duplicate row) — including
        // BACKWARD (the restore-from-backup case the != compare exists for).
        let second = vec![DirectorySyncState {
            directory_path: "/A".into(),
            last_sync_mtime: 500,
        }];
        assert_eq!(update_directory_sync_states_impl(&conn, &second), 1);
        let rows = directory_sync_states_impl(&conn);
        assert_eq!(rows.len(), 2, "update in place — no duplicate /A row");
        let mtime_of = |p: &str| {
            rows.iter()
                .find(|r| r.directory_path == p)
                .map(|r| r.last_sync_mtime)
        };
        assert_eq!(mtime_of("/A"), Some(500));

        // Prune: /B's last record leaves the catalogue → the next upsert call
        // drops /B's bookkeeping row (and the upsert itself still counts 1).
        conn.execute_batch("DELETE FROM images WHERE id = 2;")
            .expect("remove /B record");
        let third = vec![DirectorySyncState {
            directory_path: "/A".into(),
            last_sync_mtime: 600,
        }];
        assert_eq!(update_directory_sync_states_impl(&conn, &third), 1);
        let rows = directory_sync_states_impl(&conn);
        assert_eq!(rows.len(), 1, "departed directory pruned");
        assert_eq!(rows[0].directory_path, "/A");
        assert_eq!(rows[0].last_sync_mtime, 600);
    }

    #[test]
    fn remove_images_by_ids_explicit_rows_keyword_rows_survive() {
        let conn = setup();
        conn.execute_batch(
            "INSERT INTO images VALUES
                 (1, '/A/x.jpg', '/A'),
                 (2, '/A/y.jpg', '/A'),
                 (3, '/A/z.jpg', '/A');
             INSERT INTO keyword (image_id, label, path) VALUES
                 (1, 'Dogs', 'Dogs'),
                 (2, 'Dogs', 'Dogs'),
                 (3, 'Cats', 'Cats');",
        )
        .expect("seed");

        // Empty refusal: no ids must never mean \"all ids\".
        assert_eq!(remove_images_by_ids_impl(&conn, &[]), 0);
        let total: i64 = conn
            .query_row("SELECT COUNT(*) FROM images", [], |r| r.get(0))
            .unwrap();
        assert_eq!(total, 3, "refusal deleted nothing");

        // Remove two explicit rows.
        assert_eq!(remove_images_by_ids_impl(&conn, &[1, 3]), 2);
        let survivor: String = conn
            .query_row("SELECT file_path FROM images", [], |r| r.get(0))
            .unwrap();
        assert_eq!(survivor, "/A/y.jpg");

        // Keyword rows are NEVER deleted (the S31/S65 doctrine): all three
        // remain, the two orphans now invisible to every consumer (each joins
        // through images.id).
        let keyword_rows: i64 = conn
            .query_row("SELECT COUNT(*) FROM keyword", [], |r| r.get(0))
            .unwrap();
        assert_eq!(keyword_rows, 3);

        // Idempotent re-call on already-gone ids: zero rows, no error.
        assert_eq!(remove_images_by_ids_impl(&conn, &[1, 3]), 0);
    }

    #[test]
    fn remove_images_by_ids_chunks_past_500() {
        let conn = setup();
        if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;") {
            panic!("begin failed: {}", e);
        }
        for id in 1..=1_205i64 {
            conn.execute(
                "INSERT INTO images VALUES (?, ?, '/A')",
                params![id, format!("/A/f{}.jpg", id)],
            )
            .expect("seed row");
        }
        conn.execute_batch("COMMIT;").expect("commit seed");

        let ids: Vec<i64> = (1..=1_205).collect();
        assert_eq!(remove_images_by_ids_impl(&conn, &ids), 1_205);
        let total: i64 = conn
            .query_row("SELECT COUNT(*) FROM images", [], |r| r.get(0))
            .unwrap();
        assert_eq!(total, 0, "all three chunks executed");
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
pub async fn update_image_rotation(file_path: String, degrees: i32) -> bool {
    // Acquire lock and validate connection
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return false;
        }
    };

    // Update the rotation for the specified file path
    let update_sql = "UPDATE images SET rotation = ? WHERE file_path = ?";

    match conn.execute(update_sql, params![degrees, file_path]) {
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

    let rows = match stmt.query_map([], |row| row.get::<_, String>(0)) {
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
pub async fn get_distinct_directory_paths() -> Vec<String> {
    // Acquire lock and validate connection
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
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

    let mut stmt = match conn.prepare(query_sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to prepare query: {}", e);
            return Vec::new();
        }
    };

    let rows = match stmt.query_map([], |row| row.get::<_, String>(0)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to execute query: {}", e);
            return Vec::new();
        }
    };

    // Collect results, logging errors but continuing for other rows
    let mut paths = Vec::new();
    for row_result in rows {
        match row_result {
            Ok(path) => paths.push(path),
            Err(e) => eprintln!("Failed to parse row: {}", e),
        }
    }

    paths
}

/// One directory's own image tally for the Sources sidebar (S65) — a row of
/// the aggregated GROUP BY. `directory_path` is the image's immediate parent
/// directory in the SAME canonical derived form `get_distinct_directory_paths`
/// returns (the protected S5 expression over `file_path` — deliberately NOT
/// the stored `directory_path` column), so the key set is byte-identical to
/// the tree's and one call can feed both structure and counts.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectoryImageCount {
    pub directory_path: String,
    pub image_count: i64,
}

/// One capture day's image tally for the Dates sidebar (S65). `day` is the
/// "YYYY:MM:DD" prefix — the same 10-character SUBSTRING
/// `get_distinct_date_strings` returns, so the key set is byte-identical.
#[derive(Debug, Clone, PartialEq)]
pub struct CaptureDayImageCount {
    pub day: String,
    pub image_count: i64,
}

/// Per-directory image counts in ONE pass — the cure (flagged S64) for the
/// Sources sidebar's per-node COUNT flood (one query per tree node: thousands
/// of sequential scans at 65k records). Same derived directory expression,
/// WHERE, and ORDER BY as `get_distinct_directory_paths`, so the keys double
/// as the distinct-paths list and this call replaces it for the sidebar.
/// Counts are RAW catalogue (no duplicate filter / no RAW+JPEG collapse — the
/// sidebar-count convention, design doc §11). Each image tallies under its
/// immediate parent ONLY; Swift sums ancestors while walking the trie (an
/// ancestor's old prefix-count equals the sum over its descendant leaves,
/// since every image has exactly one parent directory).
pub async fn directory_image_counts(media_type: MediaType) -> Vec<DirectoryImageCount> {
    // Acquire lock and validate connection
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    // The S5 directory expression (do NOT rewrite) — see
    // get_distinct_directory_paths for the derivation notes.
    // Media-type stance (DESIGN §11): the sidebar counts follow the Photos view's
    // media type so Dates/Sources counts equal what the gallery shows. None (Both)
    // → no media filter; the same seam as every query helper, via media_predicate.
    let media_clause = match media_predicate(media_type) {
        Some(pred) => format!("AND {}", pred),
        None => String::new(),
    };
    let query_sql = format!(
        r#"
        SELECT
            SUBSTRING(file_path, 1, LENGTH(file_path) - INSTR(REVERSE(file_path), '/')) as dir_path,
            COUNT(*) as image_count
        FROM images
        WHERE file_path IS NOT NULL AND file_path LIKE '%/%' {}
        GROUP BY dir_path
        ORDER BY dir_path ASC
    "#,
        media_clause
    );

    let mut stmt = match conn.prepare(&query_sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("directory_image_counts: prepare {}", e);
            return Vec::new();
        }
    };

    let rows = match stmt.query_map([], |row| {
        Ok(DirectoryImageCount {
            directory_path: row.get::<_, String>(0)?,
            image_count: row.get::<_, i64>(1)?,
        })
    }) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("directory_image_counts: query {}", e);
            return Vec::new();
        }
    };

    let mut counts = Vec::new();
    for row_result in rows {
        match row_result {
            Ok(c) => counts.push(c),
            Err(e) => eprintln!("directory_image_counts: row {}", e),
        }
    }

    counts
}

/// Per-capture-day image counts in ONE pass — the Dates-sidebar twin of
/// `directory_image_counts` (S65; one COUNT per year + month + distinct DAY
/// before this). Same day expression, WHERE, and ORDER BY as
/// `get_distinct_date_strings`, so the keys double as the distinct-days list
/// and this call replaces it for the sidebar. Counts are RAW catalogue (the
/// §11 sidebar convention); Swift sums months and years from the days. NULL /
/// empty capture datetimes (Undated files) are excluded here exactly as they
/// are from the tree — the "All Photos" total is a separate whole-catalogue
/// count and still includes them.
pub async fn capture_day_image_counts(media_type: MediaType) -> Vec<CaptureDayImageCount> {
    // Acquire lock and validate connection
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    // Media-type stance (DESIGN §11): follows the Photos view's media type (see
    // directory_image_counts), via media_predicate — None (Both) = no media filter.
    let media_clause = match media_predicate(media_type) {
        Some(pred) => format!("AND {}", pred),
        None => String::new(),
    };
    let query_sql = format!(
        r#"
        SELECT SUBSTRING(capture_datetime, 1, 10) as date_str,
               COUNT(*) as image_count
        FROM images
        WHERE capture_datetime IS NOT NULL AND capture_datetime != '' {}
        GROUP BY date_str
        ORDER BY date_str ASC
    "#,
        media_clause
    );

    let mut stmt = match conn.prepare(&query_sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("capture_day_image_counts: prepare {}", e);
            return Vec::new();
        }
    };

    let rows = match stmt.query_map([], |row| {
        Ok(CaptureDayImageCount {
            day: row.get::<_, String>(0)?,
            image_count: row.get::<_, i64>(1)?,
        })
    }) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("capture_day_image_counts: query {}", e);
            return Vec::new();
        }
    };

    let mut counts = Vec::new();
    for row_result in rows {
        match row_result {
            Ok(c) => counts.push(c),
            Err(e) => eprintln!("capture_day_image_counts: row {}", e),
        }
    }

    counts
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
        false,
        "",
        MediaType::StillsOnly,
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
    media_type: MediaType,
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
        false,
        "",
        media_type,
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
        false,
        "",
        MediaType::StillsOnly,
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
    media_type: MediaType,
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
        false,
        "",
        media_type,
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
    media_type: MediaType,
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

    // Build predicate via shared helper (single source of truth for the
    // four-arm path/date composition — see `build_path_date_predicate`).
    let predicate = build_path_date_predicate(&path_prefix, &date_prefix);

    execute_image_count_query(
        conn,
        &predicate,
        apply_duplicate_filter,
        apply_raw_jpeg_collapse,
        false,
        "",
        media_type,
    )
}

pub async fn get_images_for_path_prefix_gallery(
    limit: i64,
    offset: i64,
    path_prefix: String,
    date_prefix: String,
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
    apply_similar_photo_collapse: bool,
    similar_algorithm_version: String,
    media_type: MediaType,
) -> Vec<ImageRecord> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    let predicate = build_path_date_predicate(&path_prefix, &date_prefix);

    execute_image_record_query(
        conn,
        &predicate,
        "capture_datetime DESC NULLS LAST, created_timestamp DESC",
        limit,
        offset,
        apply_duplicate_filter,
        apply_raw_jpeg_collapse,
        apply_similar_photo_collapse,
        &similar_algorithm_version,
        media_type,
    )
}

pub async fn get_image_count_for_filters_gallery(
    path_prefix: String,
    date_prefix: String,
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
    apply_similar_photo_collapse: bool,
    similar_algorithm_version: String,
    media_type: MediaType,
) -> i64 {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    let predicate = build_path_date_predicate(&path_prefix, &date_prefix);

    execute_image_count_query(
        conn,
        &predicate,
        apply_duplicate_filter,
        apply_raw_jpeg_collapse,
        apply_similar_photo_collapse,
        &similar_algorithm_version,
        media_type,
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
/// - `media_type`: the media stance (Stills / Videos / Both). Folder-sync's
///   `computeDiff` passes `Both` (its disk side lists video too, and an
///   asymmetric gate would flag every catalogued video as a phantom arrival);
///   stills-only surfaces pass `StillsOnly`.
pub async fn get_file_paths_for_filters(
    path_prefix: String,
    date_prefix: String,
    apply_duplicate_filter: bool,
    apply_raw_jpeg_collapse: bool,
    media_type: MediaType,
) -> FilePathsResult {
    // Acquire lock and validate connection. Catalogue not initialized
    // is a hard failure — caller must distinguish from "zero matches".
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
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
        media_type,
    ) {
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
    media_type: MediaType,
) -> Vec<ImageRecord> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
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
        media_type,
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
pub async fn remove_images_for_filters(path_prefix: String, date_prefix: String) -> i64 {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };

    let predicate = build_path_date_predicate(&path_prefix, &date_prefix);
    if predicate.is_empty() {
        eprintln!(
            "remove_images_for_filters refused: empty predicate \
             (path_prefix and date_prefix both empty)"
        );
        return 0;
    }

    let delete_sql = format!("DELETE FROM images WHERE {}", predicate);
    match conn.execute(&delete_sql, []) {
        Ok(rows_changed) => rows_changed as i64,
        Err(e) => {
            eprintln!("remove_images_for_filters DELETE failed: {}", e);
            0
        }
    }
}

// ============================================================================
// Folder Sync (Session 68; Docs/DESIGN-Folder-Sync.md)
//   - directory_sync_states:        the bookkeeping table, whole (the sweep's
//                                   baseline: one row per catalogued directory)
//   - update_directory_sync_states: bulk upsert after a scan/sync walk (+ prune
//                                   of rows whose directory left the catalogue)
//   - remove_images_by_ids:         the full-sync removal half — explicit rows,
//                                   confirmed missing by the walk, named in the
//                                   Swift confirm before this is ever called
// ============================================================================

/// One directory's folder-sync bookkeeping row (S68): the directory's on-disk
/// mtime (epoch seconds, as stat reports it) recorded when the directory was
/// last scanned or synced. The monitor sweep flags a directory when a fresh
/// stat DIFFERS (`!=`, never `>` — a restore-from-backup moves mtime backward
/// and the directory is still changed). Mirrors the UDL `DirectorySyncState`.
pub struct DirectorySyncState {
    pub directory_path: String,
    pub last_sync_mtime: i64,
}

/// Body of `directory_sync_states` against an explicit connection (the
/// impl/wrapper pattern, for in-memory unit tests). Returns the whole table,
/// unordered; the Swift sweep indexes it by path. Empty on any error
/// (failure-as-empty, the record-returning convention).
fn directory_sync_states_impl(conn: &Connection) -> Vec<DirectorySyncState> {
    let mut stmt =
        match conn.prepare("SELECT directory_path, last_sync_mtime FROM directory_sync_state") {
            Ok(s) => s,
            Err(e) => {
                eprintln!("directory_sync_states: prepare {}", e);
                return Vec::new();
            }
        };

    let mapped = stmt.query_map([], |row| {
        Ok(DirectorySyncState {
            directory_path: row.get(0)?,
            last_sync_mtime: row.get(1)?,
        })
    });

    match mapped {
        Ok(iter) => iter.filter_map(|r| r.ok()).collect(),
        Err(e) => {
            eprintln!("directory_sync_states: query {}", e);
            Vec::new()
        }
    }
}

/// The folder-sync sweep baseline: every `directory_sync_state` row. Called by
/// `SourceLocationMonitor` once per sweep cycle; the worklist of directories
/// to stat comes from the catalogue's distinct-directory derivation, and this
/// table supplies the mtime each is compared against. A directory with no row
/// here (never scanned since the feature landed) compares as changed, which is
/// correct — its first sync records the baseline.
pub async fn directory_sync_states() -> Vec<DirectorySyncState> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };
    directory_sync_states_impl(conn)
}

/// Body of `update_directory_sync_states` against an explicit connection (the
/// impl/wrapper pattern). Per row: check-then-UPDATE-or-INSERT keyed on
/// `directory_path` (the S46 house pattern — never ON CONFLICT), all in one
/// transaction. After the upserts, one bounded housekeeping DELETE prunes rows
/// whose directory no longer exists in the catalogue (post-relocate prefixes,
/// fully-removed roots) — harmless if skipped, tidy if kept; the table stays
/// a few thousand rows. Returns rows changed (updated + inserted; the prune
/// does not count). 0 on any failure (failure-as-zero, transaction rolled
/// back).
fn update_directory_sync_states_impl(conn: &Connection, states: &[DirectorySyncState]) -> u64 {
    if states.is_empty() {
        return 0;
    }

    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;") {
        eprintln!("update_directory_sync_states: begin failed: {}", e);
        return 0;
    }

    let mut changed: u64 = 0;
    for state in states {
        // UPDATE first; an existing row is the common case after the first sync.
        let updated = match conn.execute(
            "UPDATE directory_sync_state SET last_sync_mtime = ? \
             WHERE directory_path = ?",
            params![state.last_sync_mtime, state.directory_path],
        ) {
            Ok(n) => n as u64,
            Err(e) => {
                eprintln!("update_directory_sync_states: update failed: {}", e);
                let _ = conn.execute_batch("ROLLBACK;");
                return 0;
            }
        };
        if updated > 0 {
            changed += updated;
            continue;
        }

        match conn.execute(
            "INSERT INTO directory_sync_state (directory_path, last_sync_mtime) \
             VALUES (?, ?)",
            params![state.directory_path, state.last_sync_mtime],
        ) {
            Ok(_) => changed += 1,
            Err(e) => {
                eprintln!("update_directory_sync_states: insert failed: {}", e);
                let _ = conn.execute_batch("ROLLBACK;");
                return 0;
            }
        }
    }

    // Housekeeping: drop bookkeeping rows for directories that have left the
    // catalogue entirely (relocated prefixes, removed roots). The sweep never
    // consults them (its worklist is the catalogue's distinct directories) —
    // this just keeps the table from accreting dead paths across relocates.
    if let Err(e) = conn.execute(
        "DELETE FROM directory_sync_state \
         WHERE directory_path NOT IN (\
             SELECT DISTINCT directory_path FROM images \
             WHERE directory_path IS NOT NULL)",
        [],
    ) {
        // Non-fatal: the upserts above are the contract; stale rows are inert.
        eprintln!(
            "update_directory_sync_states: prune failed (non-fatal): {}",
            e
        );
    }

    if let Err(e) = conn.execute_batch("COMMIT;") {
        eprintln!("update_directory_sync_states: commit failed: {}", e);
        return 0;
    }
    changed
}

/// Record the post-walk truth for a batch of directories: their on-disk
/// mtimes as of the scan/sync that just looked inside them. Called at the end
/// of every directory scan and every sync (and, on first run, seeded for
/// already-catalogued directories so the first sweep has a baseline).
pub async fn update_directory_sync_states(states: Vec<DirectorySyncState>) -> u64 {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };
    update_directory_sync_states_impl(conn, &states)
}

/// Body of `remove_images_by_ids` against an explicit connection (the
/// impl/wrapper pattern). Chunked `DELETE ... WHERE id IN (...)` inside one
/// transaction; integer interpolation via `id_in_list` (no injection surface).
/// Keyword rows are NEVER touched — the S31/S65 doctrine: rows are the
/// recovery surface, and orphans are invisible because every keyword consumer
/// joins through `images.id`. Returns rows deleted; 0 on failure (rolled
/// back) or empty input (refusal — no ids must never mean "all ids").
fn remove_images_by_ids_impl(conn: &Connection, ids: &[i64]) -> u64 {
    if ids.is_empty() {
        return 0;
    }

    if let Err(e) = conn.execute_batch("BEGIN TRANSACTION;") {
        eprintln!("remove_images_by_ids: begin failed: {}", e);
        return 0;
    }

    let mut deleted: u64 = 0;
    for chunk in ids.chunks(500) {
        let predicate = match id_in_list(chunk) {
            Some(p) => p,
            None => continue,
        };
        let delete_sql = format!("DELETE FROM images WHERE {}", predicate);
        match conn.execute(&delete_sql, []) {
            Ok(n) => deleted += n as u64,
            Err(e) => {
                eprintln!("remove_images_by_ids: DELETE failed: {}", e);
                let _ = conn.execute_batch("ROLLBACK;");
                return 0;
            }
        }
    }

    if let Err(e) = conn.execute_batch("COMMIT;") {
        eprintln!("remove_images_by_ids: commit failed: {}", e);
        return 0;
    }
    deleted
}

/// Session 68: catalogue-only DELETE of EXPLICIT image rows — the full-sync
/// removal half. The Swift sync walk has already (1) confirmed each file
/// missing by stat on a MOUNTED, ACCESSIBLE volume (the healthy-roots
/// eligibility gate — an unplugged drive never reaches this call), and
/// (2) shown the user a confirm naming the vanished files. Touches NO files
/// and NO thumbnails (Swift owns cache hygiene for the removed paths).
/// Refuses an empty list. Returns rows deleted, 0 on any failure.
pub async fn remove_images_by_ids(ids: Vec<i64>) -> u64 {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return 0;
        }
    };
    remove_images_by_ids_impl(conn, &ids)
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
) -> Vec<ImageRecord> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return Vec::new();
        }
    };

    let predicate = build_destination_family_predicate(&sample_file_path, &canonical_file_name);

    execute_image_record_projection_query(conn, &predicate, false, false, MediaType::StillsOnly)
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
pub async fn find_counterpart_image(file_path: String) -> Option<ImageRecord> {
    // 1. Parse parent directory and basename from the input path.
    //    Pure string operation — no filesystem I/O, no canonicalization.
    //    Trusts the input. Risk 3 in DESIGN-image-classification.md.
    let last_slash = match file_path.rfind('/') {
        Some(pos) => pos,
        None => {
            // No slash → no parent directory. Cannot be a real file path.
            return None;
        }
    };

    let parent_dir = &file_path[..last_slash];
    let basename = &file_path[(last_slash + 1)..];

    // 2. Parse stem and extension from the basename.
    //    rfind('.') finds the LAST dot, which handles multi-dot filenames
    //    such as IMG.2024-05-13.NEF correctly (stem = "IMG.2024-05-13").
    let last_dot = match basename.rfind('.') {
        Some(pos) => pos,
        None => {
            // No extension → cannot be classified as JPEG or RAW.
            return None;
        }
    };

    let stem = &basename[..last_dot];
    let ext = &basename[(last_dot + 1)..];

    // 3. Classify the input extension; Other-kind inputs have no counterpart.
    let input_kind = classify_extension(ext.to_string());
    let target_kind = match input_kind {
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
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
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

    let mut stmt = match conn.prepare(query_sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to prepare counterpart query: {}", e);
            return None;
        }
    };

    let rows = match stmt.query_map(params![parent_dir, stem, file_path], |row| {
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
    }) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to execute counterpart query: {}", e);
            return None;
        }
    };

    // 6. Iterate candidates in alphabetical-by-extension order; return the
    //    first record whose extension classifies as the opposite kind of
    //    the input. The SQL ORDER BY guarantees determinism across runs;
    //    the Rust-side filter ensures we don't accidentally return a same-
    //    kind sibling (multi-RAW edge case).
    for row_result in rows {
        match row_result {
            Ok(record) => {
                // file_extension is filtered NOT NULL/non-empty in SQL,
                // but match defensively in case future schema changes
                // weaken that guard.
                let candidate_ext = match &record.file_extension {
                    Some(e) => e.clone(),
                    None => continue,
                };

                if classify_extension(candidate_ext) == target_kind {
                    return Some(record);
                }
            }
            Err(e) => eprintln!("Failed to parse counterpart row: {}", e),
        }
    }

    // No opposite-kind candidate found.
    None
}

/// S74 — the video-exclusive columns for the detail panel's Video group,
/// fetched on demand for the one video being viewed (see `get_video_details`).
///
/// Deliberately NOT folded into `ImageRecord`: that struct is lifted in bulk on
/// every gallery page, ⌘A-whole-query, and the 167k builder load — all through
/// the @MainActor UniFFI lift (S57's launch-beachball root cause). Weighting it
/// with 16 fields that only the single-record detail panel consumes would tax
/// the hottest read path for nothing. The shared fields a video also carries
/// (capture date, dimensions, rotation, bit depth, color space, GPS) already
/// ride on ImageRecord; this carries only the video-only set.
#[derive(Debug, Clone)]
pub struct VideoDetails {
    pub duration_seconds: Option<f64>,   // container duration, seconds
    pub frame_rate: Option<f64>,         // nominal fps (e.g. 29.97)
    pub video_kind: Option<String>,      // container: "mov" / "mp4" / "mxf"
    pub video_codec: Option<String>,     // "hevc" / "prores" / "h264"
    pub video_bitrate: Option<i64>,      // bits/sec
    pub color_primaries: Option<String>, // CICP canonical (bt2020 / smpte432 / bt709)
    pub color_transfer: Option<String>,  // arib-std-b67 (HLG) / smpte2084 (PQ) / bt709
    pub color_matrix: Option<String>,    // bt2020nc / smpte170m / bt709
    pub color_range: Option<String>,     // "tv" / "pc"
    pub dv_profile: Option<i32>,         // Dolby Vision profile (8 on iPhone); None = none
    pub has_audio: Option<bool>,
    pub audio_codec: Option<String>, // aac / pcm_s16le / pcm_s24le
    pub audio_channels: Option<i32>,
    pub audio_sample_rate: Option<i32>,
    pub audio_bitrate: Option<i64>,    // bits/sec
    pub live_photo_id: Option<String>, // QuickTime content.identifier
}

/// Decode the 16 video-only columns (in the SELECT order used by
/// `get_video_details`) into a `VideoDetails`. Shared by the FFI fn and its
/// unit test so the column positions and integer down-casts are verified in one
/// place. DuckDB hands integer columns back as i64 (mirrors the ImageRecord
/// decode), so the INTEGER fields cast to i32 explicitly; BIGINT stays i64.
fn row_to_video_details(row: &duckdb::Row) -> Result<VideoDetails, duckdb::Error> {
    Ok(VideoDetails {
        duration_seconds: row.get(0)?,
        frame_rate: row.get(1)?,
        video_kind: row.get(2)?,
        video_codec: row.get(3)?,
        video_bitrate: row.get(4)?,
        color_primaries: row.get(5)?,
        color_transfer: row.get(6)?,
        color_matrix: row.get(7)?,
        color_range: row.get(8)?,
        dv_profile: row.get::<_, Option<i64>>(9)?.map(|v| v as i32),
        has_audio: row.get(10)?,
        audio_codec: row.get(11)?,
        audio_channels: row.get::<_, Option<i64>>(12)?.map(|v| v as i32),
        audio_sample_rate: row.get::<_, Option<i64>>(13)?.map(|v| v as i32),
        audio_bitrate: row.get(14)?,
        live_photo_id: row.get(15)?,
    })
}

/// S74 — fetch the video-only columns for one image by id, for the detail
/// panel's Video group. Single-row, by primary key, guarded `is_video IS TRUE`
/// (the caller only asks for videos). Returns None when the catalogue is
/// uninitialized, the id is absent, or the row is not a video — all benign
/// "no Video group" outcomes for the panel, never an error worth surfacing.
pub async fn get_video_details(image_id: i64) -> Option<VideoDetails> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return None;
        }
    };

    let query_sql = r#"
        SELECT
            duration_seconds, frame_rate, video_kind, video_codec, video_bitrate,
            color_primaries, color_transfer, color_matrix, color_range, dv_profile,
            has_audio, audio_codec, audio_channels, audio_sample_rate, audio_bitrate,
            live_photo_id
        FROM images
        WHERE id = ?1 AND is_video IS TRUE
    "#;

    conn.query_row(query_sql, params![image_id], row_to_video_details)
        .ok()
}

/// S87 (Apple Photos Phase 2 prerequisite) — fetch the Apple asset UUID
/// (`external_source_id`, the ZASSET.ZUUID stored at import) for one image by id.
/// Single-row, by primary key. Deliberately NOT folded onto `ImageRecord`: that
/// struct is lifted in bulk on every gallery page, ⌘A-whole-query, and the 167k
/// builder load — all through the @MainActor UniFFI lift (S57's launch-beachball
/// root cause) — and this provenance/PhotoKit handle is read only when a single
/// cloud-only Apple row needs materializing. Mirrors `get_video_details`. Returns
/// None when the catalogue is uninitialized, the id is absent, or the column is
/// NULL (an ordinary non-Apple row) — all benign "no Apple handle" outcomes, never
/// an error worth surfacing.
pub async fn get_external_source_id(image_id: i64) -> Option<String> {
    let catalogue = CATALOGUE.lock().unwrap();
    let conn = match catalogue.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Catalogue not initialized");
            return None;
        }
    };

    conn.query_row(
        "SELECT external_source_id FROM images WHERE id = ?1",
        params![image_id],
        |row| row.get::<_, Option<String>>(0),
    )
    .ok()
    .flatten()
}
