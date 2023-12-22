use std::{env, fs};
use std::path::PathBuf;

const SUBMODULE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/thirdparty/whisper.cpp");

fn main() {
    let submodule_dir = &PathBuf::from(SUBMODULE_DIR);
    let header_path = submodule_dir.join("whisper.h");

    if fs::read_dir(submodule_dir).is_err() {
        panic!("Could not find {SUBMODULE_DIR}. Did you forget to initialize submodules?");
    }

    let mut config = cmake::Config::new(submodule_dir);

    config.define("BUILD_SHARED_LIBS", "OFF")
        .define("WHISPER_BUILD_EXAMPLES", "OFF")
        .define("WHISPER_BUILD_TESTS", "OFF");

    #[cfg(not(feature = "avx"))]
    {
        config.define("WHISPER_NO_AVX", "ON")
    }

    #[cfg(not(feature = "avx2"))]
    {
        config.define("WHISPER_NO_AVX2", "ON")
    }

    #[cfg(not(feature = "fma"))]
    {
        config.define("WHISPER_NO_FMA", "ON")
    }

    #[cfg(not(feature = "f16c"))]
    {
        config.define("WHISPER_NO_F16C", "ON")
    }

    let dst = config.build();

    println!("cargo:rustc-link-search=native={}/lib/static", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64/static", dst.display());
    println!("cargo:rustc-link-lib=static=whisper");

    let bindings = bindgen::Builder::default()
        .header(submodule_dir.join("ggml.h").to_string_lossy())
        .header(submodule_dir.join("whisper.h").to_string_lossy())
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
