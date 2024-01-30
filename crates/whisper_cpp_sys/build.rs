use std::{env, fs};
use std::path::PathBuf;
#[cfg(feature = "compat")]
use std::process::Command;

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
            .args(["-p", "-P"])
            .output()
            .expect("Failed to acquire symbols from the compiled library.");
        if !output.status.success() {
            panic!(
                "An error has occurred while acquiring symbols from the compiled library ({})",
                output.status
            );
        }
        let out_str = String::from_utf8_lossy(output.stdout.as_slice());
        let symbols = get_symbols(
            &out_str,
            [
                Filter {
                    prefix: "ggml",
                    sym_type: 'T',
                },
                Filter {
                    prefix: "ggml",
                    sym_type: 'B',
                },
                Filter {
                    prefix: "gguf",
                    sym_type: 'T',
                },
                Filter {
                    prefix: "quantize",
                    sym_type: 'T',
                },
                Filter {
                    prefix: "dequantize",
                    sym_type: 'T',
                },
            ],
        );

        let mut cmd = Command::new(objcopy_name);
        cmd.current_dir(&lib_path);
        for symbol in symbols {
            cmd.arg(format!("--redefine-sym={symbol}=whisper_{symbol}"));
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

#[cfg(feature = "compat")]
struct Filter<'a> {
    prefix: &'a str,
    sym_type: char,
}

/// Helper function to turn **`nm`**'s output into an iterator of [`str`] symbols.
///
/// This function expects **`nm`** to be called using the **`-p`** and **`-P`** flags.
#[cfg(feature = "compat")]
fn get_symbols<'a, const N: usize>(
    nm_output: &'a str,
    filters: [Filter<'a>; N],
) -> impl Iterator<Item=&'a str> + 'a {
    nm_output
        .lines()
        .map(|symbol| {
            // Strip irrelevant information

            let mut stripped = symbol;
            while stripped.split(' ').count() > 2 {
                let idx = unsafe { stripped.rfind(' ').unwrap_unchecked() };
                stripped = &stripped[..idx]
            }
            stripped
        })
        .filter(move |symbol| {
            // Filter matching symbols

            if symbol.split(' ').count() == 2 {
                for filter in &filters {
                    if symbol.ends_with(filter.sym_type) && symbol.starts_with(filter.prefix) {
                        return true;
                    }
                }
            }
            false
        })
        .map(|symbol| &symbol[..symbol.len() - 2]) // Strip the type, so only the symbol remains
}
