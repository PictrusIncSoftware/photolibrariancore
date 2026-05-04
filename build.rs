fn main() {
    uniffi::generate_scaffolding("src/photolibrariancore.udl")
        .expect("Failed to generate UniFFI scaffolding");
}
