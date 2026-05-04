#!/bin/bash

# PhotoLibrarian Core - Xcode Build Helper
# Builds the Rust library for Apple Silicon and prepares it for Xcode integration

set -e

echo "🦀 Building photolibrariancore for Apple Silicon..."
echo ""

# Ensure the target is installed
rustup target add aarch64-apple-darwin

# Build release version
echo "Building release build..."
cargo build --release --target aarch64-apple-darwin

# Create output directory for Xcode
mkdir -p xcode-libs

# Copy the static library
echo "Copying static library to xcode-libs/..."
cp target/aarch64-apple-darwin/release/libphotolibrariancore.a xcode-libs/

# Show the output
echo ""
echo "✅ Build complete!"
echo ""
echo "Static library: $(pwd)/xcode-libs/libphotolibrariancore.a"
echo "Size: $(du -h xcode-libs/libphotolibrariancore.a | cut -f1)"
echo ""
echo "Next steps:"
echo "1. Add libphotolibrariancore.a to your Xcode project"
echo "2. See UNIFFI_INTEGRATION.md for complete integration instructions"
echo ""
