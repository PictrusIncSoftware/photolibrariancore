// UniFFI binding generator entry point
//
// This binary is used during the build process to generate Swift bindings from the
// Rust code and .udl interface definition. It's not part of the final application;
// it's a build-time tool.
//
// Build process flow:
// 1. This binary is compiled during `cargo build`
// 2. Xcode build script runs: `cargo run --bin uniffi-bindgen generate ...`
// 3. uniffi-bindgen reads photolibrariancore.udl
// 4. Generates Swift code (photolibrariancore.swift) and C headers
// 5. Generated code is compiled into the Xcode project
//
// Why a separate binary? UniFFI's bindgen is a CLI tool that needs to run as a
// standalone process during builds. This is the standard UniFFI pattern for
// integrating with Xcode.
//
// The actual logic lives in the `uniffi` crate's main function.
fn main() {
    uniffi::uniffi_bindgen_main()
}
