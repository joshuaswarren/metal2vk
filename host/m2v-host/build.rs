// Generates include/metal2vk.h from src/capi.rs. The header is checked in;
// CI regenerates it and expects no diff.
//
// cbindgen 0.29 emits C23 fixed-underlying-type enums guarded by
// `#if __STDC_VERSION__ >= 202311L`, with a plain-int typedef in the #else.
// Under C23 the enum tag already names the type, so that #else typedef would
// collide; Zig 0.17's translate-c (whose clang runs in C23 mode) rejects the
// pattern outright. We normalize to the plain C99 form
// `typedef enum m2v_status m2v_status;` - valid C99, C23 and C++ - before
// writing the file.
fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let config = cbindgen::Config::from_file(format!("{crate_dir}/cbindgen.toml"))
        .expect("cbindgen.toml next to Cargo.toml");
    let bindings = cbindgen::generate_with_config(&crate_dir, config).expect("cbindgen run");
    let raw_path = format!("{crate_dir}/include/metal2vk.h.raw");
    if !bindings.write_to_file(&raw_path) {
        panic!("cbindgen could not write {raw_path}");
    }
    let raw = std::fs::read_to_string(&raw_path).expect("read raw cbindgen output");

    let mut out = String::with_capacity(raw.len());
    let mut lines = raw.lines().peekable();
    while let Some(line) = lines.next() {
        let trimmed = line.trim();
        if trimmed.starts_with("#if __STDC_VERSION__ >= 202311L") {
            let rest: Vec<&str> = lines.clone().collect();
            // arm 1: fixed underlying type ("  : int32_t" then "#endif")
            if rest.len() >= 2
                && rest[0].trim().starts_with(": int")
                && rest[1].trim().starts_with("#endif")
            {
                for _ in 0..2 {
                    lines.next();
                }
                continue;
            }
            // arm 2: guarded typedef pair -> keep only the enum typedef
            if rest.len() >= 4
                && rest[0].trim().starts_with("typedef enum ")
                && rest[1].trim() == "#else"
                && rest[2].trim().starts_with("typedef int32_t")
                && rest[3].trim().starts_with("#endif")
            {
                out.push_str(rest[0]);
                out.push('\n');
                for _ in 0..4 {
                    lines.next();
                }
                continue;
            }
        }
        out.push_str(line);
        out.push('\n');
    }
    std::fs::write(format!("{crate_dir}/include/metal2vk.h"), out).expect("write header");
    let _ = std::fs::remove_file(&raw_path);
    println!("cargo:rerun-if-changed=src/capi.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}
