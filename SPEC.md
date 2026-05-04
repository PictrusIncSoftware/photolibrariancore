# PhotoLibrarian — Version 1.0 High-Level Specification

**Document Type:** Product Specification — Version 1  
**Status:** Final Draft  
**Last Updated:** April 2026  
**Scope:** What this product is, what it includes in v1, and what it explicitly is not.

---

## 1. Problem Statement

Professional photographers and enthusiasts are effectively locked into Adobe's ecosystem — not because competing tools lack editing capability, but because no standalone, one-time-purchase alternative replicates Lightroom's photograph organization and cataloguing functionality. Tools such as Photomator and Pixelmator Pro address post-production editing competently. The gap is in the library: the ability to ingest, organize, tag, rate, filter, and manage large volumes of photographs across formats and shoots.

This product exists to close that gap — giving photographers a permanent, locally-owned, subscription-free home for their image library, from which any external editing tool of their choice can be launched.

---

## 2. Product Vision

A lightweight, fast, native macOS application that serves as the definitive home for a photographer's image library. It does not attempt to be an editor. Instead, it is the organisational backbone of a photographer's workflow — the hub from which any external editing tool can be launched. The user pays once and owns the software outright. There is no subscription. There is no cloud dependency.

---

## 3. What This Product Is

- A **photo library manager and organiser** for RAW and JPEG image formats
- A **catalogue system** that allows photographers to import, browse, rate, tag, filter, and search their image collections
- A **launch point** for external editing applications, via right-click context menu integration
- A **native macOS application**, built exclusively for Apple Silicon, delivering performance and platform cohesion
- A **one-time purchase** product, installed and run locally on the user's device
- A **storage utility** capable of consolidating and exporting image archives to local drives or network-attached storage

---

## 4. What This Product Is Not (v1 Scope Boundary)

- It is **not a photo editor**. No pixel manipulation, masking, tone curve adjustment, exposure control, or retouching tools are included
- It is **not a subscription service**. No recurring fees, no cloud lock-in, no remote storage requirement
- It is **not cross-platform**. Windows and Linux are permanently out of scope. Intel-based Macs are not supported
- It is **not a DAM (Digital Asset Management) platform** for enterprise or team use. v1 targets individual photographers
- It is **not a replacement for editing tools** such as Photomator, Pixelmator Pro, or Capture One. It is designed to complement them

---

## 5. Core Functional Requirements (v1)

### 5.1 Image Ingestion
- Import photographs from connected cameras, memory cards, and local or external drives
- Support for RAW formats including Canon CR2/CR3, Nikon NEF, Sony ARW, Fuji RAF, and other major manufacturer formats, as well as JPEG
- Non-destructive import: original files are never altered by the application

### 5.2 Library & Catalogue Management
- Maintain a persistent catalogue of all imported images, including metadata and file references
- Support for multiple catalogues or libraries
- Folder-based and catalogue-based views of the image collection

### 5.3 Organisation Tools
- Star ratings (1–5) and flag/reject markers
- Colour labels
- User-defined tags and keywords
- Collections and smart collections (rule-based dynamic groupings)
- Shoot/session grouping by date, import batch, or folder

### 5.4 Search & Filtering
- Filter by rating, label, tag, date range, camera model, lens, ISO, and other EXIF metadata
- Full-text search across tags and keywords
- Smart collection rules as persistent saved filters

### 5.5 Metadata
- Read and display embedded EXIF, IPTC, and XMP metadata
- Allow users to write and edit IPTC fields including caption, copyright, and creator
- Sidecar XMP support for RAW files to preserve metadata non-destructively in an open, interoperable format

### 5.6 External Editor Integration
- Right-click context menu on any image to open in a registered external application
- Support for multiple registered editors, including Photomator, Pixelmator Pro, and Affinity Photo
- Pass the original file or a user-configured copy to the external editor
- Detect and display edited versions returned from external editors

### 5.7 AI-Powered Organisation
- Automatic tagging and keyword suggestion using on-device machine learning via Apple's Core ML framework
- Face recognition to identify and group photographs by person, processed entirely on-device
- AI-generated tags are non-destructive, clearly distinguished from user-created tags, and fully editable

### 5.8 Semantic & Similarity Search
- Search the library using natural language descriptions (e.g., "sunset on the beach", "portrait with shallow depth of field")
- Surface visually similar images to any selected photograph
- Powered by LanceDB's embedded vector index; all inference and search is local and on-device, with no data sent externally

### 5.9 Cloud Backup & Sync
- Optional backup of the catalogue and/or image files to a user-configured cloud destination
- Sync catalogue state — ratings, tags, collections — across multiple devices belonging to the same user
- Cloud connectivity is entirely optional; the application is fully functional in a completely offline configuration

### 5.10 iOS & iPadOS Companion App
- Browse, rate, tag, and filter the synced library from an iPhone or iPad
- Full-screen image preview and loupe view on mobile
- External editor hand-off where supported on iPadOS
- Catalogue changes made on iOS/iPadOS sync back to the macOS application

### 5.11 Print Module & Export to Web
- Print layouts including single image, contact sheet, and multi-image templates with configurable margins and captions
- Export to web: generate a lightweight static HTML gallery from a selection or collection
- Export presets for common output formats covering JPEG at various resolutions, sRGB colour space conversion, and optional watermarking

### 5.12 Collation & Export to Storage
- Consolidate a selection, collection, or entire library into a single destination folder on a local drive or a network-attached storage (NAS) device
- Configurable folder naming conventions by date, shoot name, camera, or custom user-defined template
- Designed to function as a standalone archiving utility — a photographer can use this feature to bring order to an existing image archive without adopting a new workflow
- Preserves original file names and sidecar XMP files in the destination
- Supports both copy and move operations, with a configurable conflict-resolution policy: skip, rename, or overwrite

### 5.13 Image Preview & Browsing
- Fast thumbnail generation and caching for large libraries
- Loupe and full-screen single image view
- Basic zoom and pan in preview (non-editing)
- Side-by-side comparison of two images

---

## 6. Technical Stack

| Layer | Technology | Rationale |
|---|---|---|
| UI | Swift / SwiftUI | Native macOS performance, platform-consistent UX, full Apple Silicon optimisation |
| Backend / Core Logic | Rust | Performance, memory safety, suitability for high-throughput file I/O and metadata processing |
| Rust–Swift Binding | UniFFI | Mozilla's open-source binding generator; produces idiomatic Swift from Rust APIs, supports async/await, eliminates hand-written FFI |
| Catalogue Database | DuckDB | Fast analytical queries over large image catalogues; fully embedded, no server process required |
| Vector / Semantic Index | LanceDB | Embedded vector database for semantic search and visual similarity; runs entirely on-device alongside DuckDB |
| ML / AI | Apple Core ML | On-device inference for auto-tagging and face recognition; optimised for Apple Silicon Neural Engine |

### 6.1 Rust–Swift Architecture Note

The Rust backend and Swift frontend communicate exclusively through a UniFFI-defined interface layer. The API surface between the two is declared in a `.udl` interface definition file, from which UniFFI generates both the Rust scaffolding and the Swift bindings. This contract enforces a clean separation between backend logic and UI, and ensures the binding layer remains in sync with the underlying Rust API throughout development. All long-running backend operations — file I/O, database queries, thumbnail generation, ML inference — are exposed as async Rust functions and consumed as Swift async/await calls, keeping the UI thread unblocked at all times.

### 6.2 Sandboxing Note

The application must comply with Apple's Mac App Store sandboxing requirements. The Rust backend, DuckDB, and LanceDB must all be bundled within the application container. No shared library installations or external processes are permitted. This constraint is a known architectural consideration and must be validated early in the development cycle.

---

## 7. Data & Storage Model

- All catalogue data is stored locally on the user's device
- The catalogue database files are portable and self-contained — a user can relocate them to another machine or external drive
- Image files are referenced by path; the application never duplicates or repackages original files
- XMP sidecar files are written alongside RAW files to store ratings, tags, and metadata in an open format readable by other applications
- No telemetry, usage data, or image content is transmitted externally under any circumstances

---

## 8. Platform Requirements

- **Architecture:** Apple Silicon exclusively (M1 and later)
- **Minimum OS:** macOS Monterey 12
- **Intel Mac support:** None. Not supported in v1 or any future version
- **iOS/iPadOS companion:** Requires iOS/iPadOS 16 or later (companion app, v1 feature)

---

## 9. Licensing & Distribution

- **Distribution:** Mac App Store exclusively
- **Pricing:** $10.00 USD, one-time purchase, no subscription
- **Revenue share:** Apple Small Business Program (15% commission) to be applied for upon first submission
- **License scope:** Single device, single user. The license is tied to the purchaser's Apple ID. The application is intended for use on one machine. Re-purchase is required for a new or replacement machine. This is stated clearly in the App Store description and in the End User License Agreement (EULA)
- **Hardware locking:** Not implemented at the application level. Apple ID authentication via the Mac App Store serves as the access control mechanism
- **Copy protection:** No custom license screens, license keys, or third-party copy protection mechanisms are used, in compliance with Mac App Store guidelines
- **Updates:** Distributed exclusively through the Mac App Store. No third-party update mechanisms are used
- **Developer Program enrollment:** Apple Developer Program ($99/year) to be activated when the application approaches submission readiness, not during active development
- **Small Business Program:** To be applied for at the point of first submission to qualify for the 15% commission rate

---

## 10. Development Strategy

### 10.1 Code Generation
The majority of application code — Swift UI, Rust backend, UniFFI interface definitions, database schema and queries, Core ML integration, and external editor handoff logic — will be produced using Claude (Anthropic) as the primary AI-assisted development tool. Final compilation, packaging, and App Store submission will be performed in Xcode, which is the required toolchain for Mac App Store distribution.

### 10.2 Development Priorities
The following architectural concerns are identified as requiring early validation before broader feature development:

- **UniFFI bridge setup** — establishing the Rust-Swift FFI layer and confirming async interop before any feature code is written
- **Sandbox compliance** — validating that DuckDB and LanceDB operate correctly within Apple's Mac App Store sandbox entitlements
- **RAW format support** — confirming decode coverage across the major manufacturer RAW formats using macOS-native frameworks or a suitable Rust crate

### 10.3 Testing Agents
Automated testing agents will be developed as a formal part of the project, running alongside feature development rather than after it. The following agents are planned for v1:

- **Functional Test Agent** — drives the application through defined end-to-end user journeys: importing a shoot, rating and tagging images, filtering, launching an external editor, and verifying catalogue persistence across restarts
- **Library Stress Test Agent** — generates large synthetic image libraries to validate query performance and UI responsiveness at scale against DuckDB
- **Metadata Validation Agent** — confirms that EXIF, IPTC, and XMP data is read and written correctly and completely across all supported RAW and JPEG formats
- **Regression Agent** — executes a defined suite of functional checks after every significant code change to detect breakage early

---

## 11. Out of Scope (Future Consideration)

The following are explicitly deferred beyond v1:

- Windows port — permanently out of scope
- Intel Mac support — permanently out of scope
- Plugin or extension API for third-party developers
- Video file support

---

## 12. Success Criteria for v1

A v1 release is considered complete when a photographer can:

1. Import a full shoot of RAW and JPEG files from a memory card or drive
2. Browse, rate, tag, and filter those images within the application
3. Search their library using natural language and receive meaningful results
4. Launch any registered external editor from a right-click on any image
5. Consolidate and export their library to a local drive or NAS with a configurable folder structure
6. Close and reopen the application and find all their work — ratings, tags, collections, AI tags — fully intact
7. Do all of the above without an internet connection and without an ongoing subscription

---

*End of v1 Specification — PhotoLibrarian*
