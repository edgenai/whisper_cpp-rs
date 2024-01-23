use std::path::PathBuf;
use std::process::Command;
use std::{env, fs};

// TODO add feature compatibility checks

const SUBMODULE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/thirdparty/whisper.cpp");

fn main() {
    let submodule_dir = &PathBuf::from(SUBMODULE_DIR);

    if fs::read_dir(submodule_dir).is_err() {
        panic!("Could not find {SUBMODULE_DIR}. Did you forget to initialize submodules?");
    }

    let mut config = cmake::Config::new(submodule_dir);

    config
        .define("BUILD_SHARED_LIBS", "OFF")
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

    println!(
        "cargo:rustc-link-search=native={}/lib/static",
        dst.display()
    );
    println!(
        "cargo:rustc-link-search=native={}/lib64/static",
        dst.display()
    );
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
        .allowlist_type("ggml_.*")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    #[cfg(feature = "compat")]
    {
        // TODO this whole section is a bit hacky, could probably clean it up a bit, particularly the retrieval of symbols from the library files

        let (whisper_lib_name, nm_name, objcopy_name) =
            if cfg!(target_os = "linux") || cfg!(target_os = "macos") {
                ("libwhisper.a", "nm", "objcopy")
            } else {
                ("whisper.lib", "llvm-nm", "llvm-objcopy")
            };

        let lib_path = out_path.join("lib").join("static");

        // Modifying symbols exposed by the ggml library

        let output = Command::new(nm_name)
            .current_dir(&lib_path)
            .arg(whisper_lib_name)
            .output()
            .expect("Failed to acquire symbols from the compiled library.");
        if !output.status.success() {
            panic!(
                "An error has occurred while acquiring symbols from the compiled library ({})",
                output.status
            );
        }
        let out_str = String::from_utf8_lossy(output.stdout.as_slice());
        let symbols = out_str.split('\n');

        let mut cmd = Command::new(objcopy_name);
        cmd.current_dir(&lib_path);
        for symbol in symbols {
            if !(symbol.contains("T ggml")
                || symbol.contains("B ggml")
                || symbol.contains("T gguf")
                || symbol.contains("T quantize")
                || symbol.contains("T dequantize"))
            {
                continue;
            }

            let formatted = &symbol[11..];
            cmd.arg(format!("--redefine-sym={formatted}=whisper_{formatted}"));
        }
        let status = cmd
            .arg(whisper_lib_name)
            .status()
            .expect("Failed to modify global symbols from the ggml library.");
        if !status.success() {
            panic!(
                "An error as occurred while modifying global symbols from library file ({})",
                status
            );
        }
    }
}
