#!/bin/bash

# ============================================================================
# PhotoLibrarian Core - Xcode Build Helper
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
# This script ensures location #3 (the Xcode-active location) always gets
# fresh builds. We also copy to #1 and #2 for compatibility.
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

# Define paths
RUST_LIB="target/aarch64-apple-darwin/release/libphotolibrariancore.a"
GENERATED_SWIFT_DIR="generated-swift"

# Output directories
LOCAL_OUTPUT="xcode-libs"
PROJECT_ROOT_LIBS="../Libraries"
XCODE_LIBS="../PhotoLibrarian/Libraries"  # ⭐ PRIMARY - Xcode links here

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
echo "⏰ All files updated at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
