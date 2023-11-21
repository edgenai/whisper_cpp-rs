use std::{env, fs};
use std::path::PathBuf;
use std::process::exit;

const SUBMODULE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/thirdparty/whisper.cpp");

fn main() {
    let submodule_dir = &PathBuf::from(SUBMODULE_DIR);
    let header_path = submodule_dir.join("whisper.h");

    if fs::read_dir(submodule_dir).is_err() {
        eprintln!("Could not find {SUBMODULE_DIR}. Did you forget to initialize submodules?");

        exit(1);
    }

    let dst = cmake::Config::new(submodule_dir)
        .configure_arg("-DBUILD_SHARED_LIBS=Off")
        .configure_arg("-DWHISPER_BUILD_EXAMPLES=Off")
        .configure_arg("-DWHISPER_BUILD_TESTS=Off")
        .build();

    println!("cargo:rustc-link-search=native={}/lib/static", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64/static", dst.display());
    println!("cargo:rustc-link-lib=static=whisper");

    let bindings = bindgen::Builder::default()
        .header(header_path.to_string_lossy())
        .parse_callbacks(Box::new(
            bindgen::CargoCallbacks::new().rerun_on_header_files(false),
        ))
        .generate_comments(false)
        .allowlist_function("whisper_.*")
        .allowlist_type("whisper_.*")
        .allowlist_function("ggml_.*")
        .allowlist_type("ggml_.*")
        .clang_arg("-xc++")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
