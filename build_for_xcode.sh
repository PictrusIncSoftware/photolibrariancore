#!/bin/bash

# ============================================================================
# PhotoLibrarian Core - Xcode Build Helper
# ============================================================================
#
# This script performs a complete build pipeline:
# 1. Builds the Rust static library (aarch64-apple-darwin)
# 2. Regenerates UniFFI Swift bindings from the .udl file
# 3. Copies all artifacts to the Xcode-active location
#
# The UniFFI bindings are regenerated on every build to ensure they stay
# in sync with the .udl interface definition.
#
# ============================================================================
#
# DIRECTORY LAYOUT (CRITICAL - READ THIS):
#
# This project has THREE "Libraries" locations that historically caused confusion:
#
# 1. ~/DeveloperProjects/PhotoLibrarian/photolibrariancore/xcode-libs/
#    - Local build output directory (kept for reference/backup)
#
# 2. ~/DeveloperProjects/PhotoLibrarian/Libraries/
#    - Project root Libraries directory (legacy, may be used by other tools)
#
# 3. ~/DeveloperProjects/PhotoLibrarian/PhotoLibrarian/Libraries/  ⭐ PRIMARY
#    - THIS is where Xcode actually links from
#    - The .xcodeproj references this location in its Library Search Paths
#    - Files here MUST be kept up-to-date or Xcode uses stale binaries
#
# 4. ~/DeveloperProjects/PhotoLibrarian/PhotoLibrarian/PhotoLibrarian/
#    - Swift bindings source file location (photolibrariancore.swift only)
#    - Xcode compiles the Swift bindings directly as source code
#    - Only the .swift file goes here; binary artifacts (.a, .h, .modulemap)
#      stay in location #3 (Libraries/)
#
# This script ensures location #3 (the Xcode-active location) always gets
# fresh builds. We also copy to #1 and #2 for compatibility, and the Swift
# bindings to #4 for Xcode to compile.
#
# WHY THIS MATTERS:
# If you run `cargo build` but don't run this script, Xcode will continue
# using the old library file. You'll see "successful builds" that don't
# reflect your code changes. This caused a full day of debugging.
#
# ============================================================================

set -e

echo "🦀 Building photolibrariancore for Apple Silicon..."
echo ""

# Ensure the target is installed
rustup target add aarch64-apple-darwin

# Build release version
echo "Building release build..."
cargo build --release --target aarch64-apple-darwin

echo ""
echo "🔄 Generating UniFFI Swift bindings..."
cargo run --bin uniffi-bindgen generate src/photolibrariancore.udl \
    --language swift --out-dir generated-swift

# Define paths
RUST_LIB="target/aarch64-apple-darwin/release/libphotolibrariancore.a"
GENERATED_SWIFT_DIR="generated-swift"

# Output directories
LOCAL_OUTPUT="xcode-libs"
PROJECT_ROOT_LIBS="../Libraries"
XCODE_LIBS="../PhotoLibrarian/Libraries"  # ⭐ PRIMARY - Xcode links here
XCODE_SWIFT="../PhotoLibrarian/PhotoLibrarian/photolibrariancore.swift"  # Swift source location

# Create output directories if they don't exist
mkdir -p "$LOCAL_OUTPUT"
mkdir -p "$PROJECT_ROOT_LIBS"
mkdir -p "$XCODE_LIBS"

echo ""
echo "📦 Copying build artifacts..."
echo ""

# Copy static library to all three locations
echo "→ Copying libphotolibrariancore.a..."
cp "$RUST_LIB" "$LOCAL_OUTPUT/"
cp "$RUST_LIB" "$PROJECT_ROOT_LIBS/"
cp "$RUST_LIB" "$XCODE_LIBS/"  # ⭐ PRIMARY

# Copy header and modulemap to all three locations
echo "→ Copying photolibrariancoreFFI.h..."
cp "$GENERATED_SWIFT_DIR/photolibrariancoreFFI.h" "$LOCAL_OUTPUT/"
cp "$GENERATED_SWIFT_DIR/photolibrariancoreFFI.h" "$PROJECT_ROOT_LIBS/"
cp "$GENERATED_SWIFT_DIR/photolibrariancoreFFI.h" "$XCODE_LIBS/"  # ⭐ PRIMARY

echo "→ Copying photolibrariancoreFFI.modulemap..."
cp "$GENERATED_SWIFT_DIR/photolibrariancoreFFI.modulemap" "$LOCAL_OUTPUT/"
cp "$GENERATED_SWIFT_DIR/photolibrariancoreFFI.modulemap" "$PROJECT_ROOT_LIBS/"
cp "$GENERATED_SWIFT_DIR/photolibrariancoreFFI.modulemap" "$XCODE_LIBS/"  # ⭐ PRIMARY

echo "→ Copying photolibrariancore.swift..."
cp "$GENERATED_SWIFT_DIR/photolibrariancore.swift" "$LOCAL_OUTPUT/"
cp "$GENERATED_SWIFT_DIR/photolibrariancore.swift" "$PROJECT_ROOT_LIBS/"
cp "$GENERATED_SWIFT_DIR/photolibrariancore.swift" "$XCODE_LIBS/"  # ⭐ PRIMARY
cp "$GENERATED_SWIFT_DIR/photolibrariancore.swift" "$XCODE_SWIFT"  # Swift source for Xcode

echo ""
echo "✅ Build complete!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📍 PRIMARY (Xcode-active) location updated:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ls -lh "$XCODE_LIBS"
echo ""
echo "Static library size: $(du -h "$XCODE_LIBS/libphotolibrariancore.a" | cut -f1)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔄 Also copied to (for compatibility):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  • $(cd "$LOCAL_OUTPUT" && pwd)"
echo "  • $(cd "$PROJECT_ROOT_LIBS" && pwd)"
echo ""
echo "Swift bindings source (photolibrariancore.swift) also copied to:"
echo "  • $(cd "$(dirname "$XCODE_SWIFT")" && pwd)/$(basename "$XCODE_SWIFT")"
echo ""
echo "⏰ All files updated at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
