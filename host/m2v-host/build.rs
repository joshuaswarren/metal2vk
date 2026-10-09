// Generates include/metal2vk.h from src/capi.rs. The header is checked in;
// CI regenerates it and expects no diff.
fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let config = cbindgen::Config::from_file(format!("{crate_dir}/cbindgen.toml"))
        .expect("cbindgen.toml next to Cargo.toml");
    let bindings = cbindgen::generate_with_config(&crate_dir, config).expect("cbindgen run");
    bindings.write_to_file(format!("{crate_dir}/include/metal2vk.h"));
    println!("cargo:rerun-if-changed=src/capi.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}
