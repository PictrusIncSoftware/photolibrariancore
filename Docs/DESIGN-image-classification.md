# Design: Image Classification Primitives

**Status:** Approved, pre-implementation
**Author:** Richard Wagner (with Claude)
**Date:** May 13, 2026 (Session 12)
**Scope:** `photolibrariancore` Rust core, with one Swift consumer refactor

---

## Summary

Introduce a small classification subsystem to the Rust core that answers two questions about images by file extension:

1. **What kind of image is this?** (`Jpeg`, `Raw`, or `Other`)
2. **Where is its counterpart?** (Given a JPEG, find the matching RAW in the catalogue, or vice versa.)

These primitives are the foundation for the upcoming JPEG+RAW pair-aware right-click menu (Session 13) and replace an inline string-based classification check currently embedded in `PhotosView.displayedImages` on the Swift side.

---

## Motivation

The catalogue already collapses JPEG+RAW pairs for display: when both files share a parent directory and filename stem, only the JPEG is shown in the gallery. This logic currently lives in Swift, in `PhotosView.displayedImages`, and hardcodes the JPEG check as a literal string comparison against `"jpg"` and `"jpeg"`.

Session 13 will introduce a right-click context menu with "Open With" entries. For a collapsed pair, opening the image should default to the RAW (the negative), not the JPEG (the preview). To support this, the gallery — which is showing the JPEG record — must be able to find the corresponding RAW record in the catalogue.

This requires answering both questions above, and answering them consistently across the app. Today, "is this a JPEG?" is answered in one place (Swift). After this change, "what kind of image is this?" is answered in one place (Rust), and Swift consults the Rust answer.

Centralizing classification in the Rust core gives us:

- A single source of truth for image-kind classification, owned by the catalogue layer.
- Trivial future extensibility: adding HEIC, AVIF, or new RAW formats becomes a one-line edit to a `const` array.
- A natural home for `find_counterpart_image`, which is fundamentally a catalogue lookup and belongs alongside the other catalogue query functions.

---

## Design

### Constants (Rust, `src/lib.rs`)

Immutable, compile-time `const` slices defining the recognized extension sets. Lowercase, no leading dot.

```rust
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

const JPEG_EXTENSIONS: &[&str] = &["jpg", "jpeg"];
```

Rationale for `const` (not `static`, not mutable, not user-configurable):

- Compile-time known, zero runtime overhead, thread-safe by construction.
- The lists are facts about the world (these are the formats we recognize), not user state. Future user-configurable extensions are an explicit non-goal; that conversation can happen if and when a real requirement emerges.
- The compiler enforces the single source of truth: no runtime mutation possible.

### Enum (Rust, `src/lib.rs`, exposed via UniFFI)

```rust
enum ImageKind {
    Jpeg,
    Raw,
    Other,
}
```

Three variants chosen deliberately over a boolean `is_raw`:

- `Other` is a real category. A `.png` screenshot, a `.tiff` scan, a `.heic` iPhone capture — none of these are RAW, none participate in JPEG+RAW pairing, and we should be able to say so explicitly rather than forcing them into a misleading "not RAW" bucket.
- Pattern-matching on three variants in Swift is clearer than chained booleans, and future format categories (e.g. `Heic` if we ever distinguish it) extend the enum without changing the call sites' shape.

This is the first enum exposed across the UniFFI boundary in this project. UniFFI supports enums natively; no special handling required.

### Classification function (Rust, **synchronous**)

```rust
fn classify_extension(ext: String) -> ImageKind {
    let lower = ext.to_lowercase();
    if JPEG_EXTENSIONS.contains(&lower.as_str()) {
        ImageKind::Jpeg
    } else if RAW_EXTENSIONS.contains(&lower.as_str()) {
        ImageKind::Raw
    } else {
        ImageKind::Other
    }
}
```

**Synchronous** is the correct semantic match: this is a pure string-to-enum lookup with no I/O. Forcing it async to "match convention" would push the Swift-side consumer (`displayedImages`, a SwiftUI computed property that cannot `await`) into awkward shapes.

This will be the first synchronous function in the project's UniFFI surface. Integration risk noted below.

The function accepts the extension as a separate `String` parameter (without leading dot, case-insensitive) rather than parsing a path. The catalogue already stores `file_extension` as a separate column, lowercase; callers can pass it directly without further preprocessing.

### Counterpart lookup function (Rust, **async**)

```rust
async fn find_counterpart_image(file_path: String) -> Option<ImageRecord>
```

Semantics:

1. Read the input file's extension.
2. Classify it via `classify_extension`.
3. If `Other`, return `None` (a non-pairing format has no counterpart by definition).
4. Otherwise, compute the parent directory and filename stem.
5. Query the catalogue for an image record with:
   - The same parent directory.
   - The same filename stem.
   - An extension whose kind is the *opposite* of the input (`Jpeg` → look for `Raw`; `Raw` → look for `Jpeg`).
6. Return the first matching record, or `None` if no counterpart exists.

Implementation notes:

- The SQL query uses **parameterized binding** for any `LIKE` clauses from day one. No `format!` injection. This matches the fix planned for the existing Chunk 6 carryover; we will not introduce new code in the broken pattern.
- "First matching record" rather than "all matching records" because the pairing model is one-to-one by stem. If a directory somehow contains `IMG_0001.jpg`, `IMG_0001.NEF`, and `IMG_0001.ARW`, returning the first match is a defensible default for v1. A future refinement can address multi-RAW edge cases if real users hit them.
- Return type `Option<ImageRecord>` because "no counterpart" is a legitimate, non-error result. A solo JPEG with no RAW sibling returns `None`, not an error.

### UDL changes (`src/photolibrariancore.udl`)

Add the enum declaration, the synchronous classifier, and the async counterpart lookup:

```idl
enum ImageKind {
    "Jpeg",
    "Raw",
    "Other",
};

namespace photolibrariancore {
    // ... existing entries ...

    // Classify a file extension into JPEG, RAW, or Other.
    // Synchronous — pure string-to-enum lookup, no I/O.
    // ext: file extension without leading dot, case-insensitive
    ImageKind classify_extension(string ext);

    // Find the JPEG+RAW counterpart of an image in the catalogue.
    // Given a JPEG, returns the matching RAW (same parent directory, same stem).
    // Given a RAW, returns the matching JPEG.
    // Returns None if the input is Other-kind, or if no counterpart exists.
    [Async]
    ImageRecord? find_counterpart_image(string file_path);
};
```

### Swift consumer refactor (`PhotosView.swift`)

The existing `displayedImages` computed property hardcodes the JPEG check:

```swift
if ext == "jpg" || ext == "jpeg" {
    // treat as JPEG
}
```

After this change, it consults the Rust classifier:

```swift
if classifyExtension(ext: ext) == .jpeg {
    // treat as JPEG
}
```

Because `classify_extension` is synchronous, this fits cleanly inside the computed property without restructuring.

---

## Non-goals

- **No right-click menu work in this session.** That is Session 13. This session lays the foundation.
- **No single-select infrastructure.** Also Session 13.
- **No user-configurable extension lists.** If a real requirement emerges later, that's a separate design conversation involving persistence, re-classification of existing rows, and UI.
- **No changes to the existing pair-collapsing display behavior.** The Swift refactor is functionally a no-op: the same images are shown, the same images are hidden, only the classification *mechanism* changes.
- **No fix to the Chunk 6 SQL parameterization issue in `get_images_filtered` / `get_filtered_image_count`.** Tracked separately; this session writes new code that follows the correct pattern but does not retrofit existing code.

---

## Risks and mitigations

**Risk 1: First synchronous UniFFI export.** All existing exports are `[Async]`. The build script `build_for_xcode.sh` has only been exercised against async functions. UniFFI fully supports sync, but the build pipeline may need a small adjustment.

*Mitigation:* Verify at integration time by building immediately after adding `classify_extension`. If the build fails on the sync export, adjustments are likely small (UniFFI macro attribute, possibly a build script tweak). If it fails in a way we can't quickly resolve, the fallback is to make `classify_extension` async and accept the awkwardness in `displayedImages` (e.g., precompute kinds when records load rather than calling per-render).

**Risk 2: `displayedImages` refactor is the riskiest integration point.** The current implementation is in production and correct. Replacing the inline string check must preserve behavior exactly.

*Mitigation:* Manual verification after refactor — load a library with known JPEG+RAW pairs, confirm pair-collapse is identical to pre-refactor behavior. If the refactor turns out more disruptive than expected (e.g., performance issues from per-render FFI calls), the fallback is to **ship the Rust primitives this session and defer the Swift refactor to a follow-up commit**. The Rust primitives are independently valuable as the foundation for Session 13's counterpart-lookup work.

**Risk 3: Edge cases in stem matching.** Filenames with multiple dots (`IMG.2024-05-13.NEF`), unusual capitalization, or whitespace need to be handled consistently between the existing display-time filter and the new `find_counterpart_image` query.

*Mitigation:* The Rust function uses the same parent-directory + stem-without-extension matching logic that the Swift filter already uses. The catalogue already stores filenames in a canonical form. Manual spot-check with a few edge-case filenames during integration.

---

## What this enables

After this session lands, Session 13 can implement right-click on a collapsed pair as:

```swift
// Pseudo-code for the right-click handler
let clickedRecord: ImageRecord = ...  // the JPEG the user sees
let counterpart = await findCounterpartImage(filePath: clickedRecord.filePath)

// counterpart is the RAW, or nil if it's a solo JPEG
// Menu can now show "Open RAW" and "Open JPEG" submenu entries
//   when counterpart != nil
```

The classification primitive also unlocks any future feature that needs to reason about image kinds (export pipelines, metadata-aware UI, format-specific tooling, etc.).

---

## Implementation order

1. Rust: add `RAW_EXTENSIONS`, `JPEG_EXTENSIONS`, `ImageKind` enum, `classify_extension` function in `src/lib.rs`.
2. UDL: expose `ImageKind` enum and `classify_extension` function. Build via `./build_for_xcode.sh`. **Verify the sync export builds cleanly — this is the first one.** Fix if not.
3. Rust: add `find_counterpart_image` function in `src/lib.rs`, using parameterized SQL bindings.
4. UDL: expose `find_counterpart_image`. Build again.
5. Swift: refactor `PhotosView.displayedImages` to call `classifyExtension` via the regenerated bindings.
6. Xcode build, exercise the gallery, confirm pair-collapse behavior is unchanged on a library with known pairs.
7. Cross-machine verification on MacBook Pro M1 Max.
8. Manual commit (Richard) at the milestone.

If step 5 proves disruptive at integration time, ship steps 1–4 and 6–8 as the session deliverable and defer step 5 to a follow-up commit.

---

## Open questions deferred

- **Multi-RAW edge case.** What if a directory contains both a `.NEF` and a `.DNG` with the same stem as a `.jpg`? Current design returns the first match. Real-world frequency unknown. Revisit if users report it.
- **HEIC and other modern formats.** Should `.heic` be a fourth `ImageKind` variant, or stay in `Other`? Current design says `Other`. Revisit when HEIC handling becomes a real product question.
- **Case sensitivity edge cases.** The catalogue stores extensions lowercase. If any code path bypasses that normalization, the classifier could miss a match. Spot-check during integration; add a normalization assertion if needed.

---

*This design note is the contract for Session 12's implementation work. Claude Code should read this document before doing any work in `photolibrariancore` for this session.*