use std::path::PathBuf;
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
        .define("WHISPER_BUILD_TESTS", "OFF")
        .define("WHISPER_METAL", "OFF"); // TODO this is on by default on Apple devices, and is causing issues, see why

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
        compat::redefine_symbols(out_path);
    }
}

#[cfg(feature = "compat")]
mod compat {
    use std::path::Path;
    use std::process::Command;

    pub fn redefine_symbols(out_path: impl AsRef<Path>) {
        // TODO this whole section is a bit hacky, could probably clean it up a bit, particularly the retrieval of symbols from the library files
        // TODO do this for cuda if necessary

        let whisper_lib_name = lib_name();
        let (nm_name, objcopy_name) = tool_names();
        println!("Modifying {whisper_lib_name}, symbols acquired via \"{nm_name}\" and modified via \"{objcopy_name}\"");

        let lib_path = out_path.as_ref().join("lib").join("static");

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

    /// Returns *Whisper.cpp*'s compiled library name, based on the operating system.
    fn lib_name() -> &'static str {
        if cfg!(target_family = "windows") {
            "whisper.lib"
        } else if cfg!(target_family = "unix") {
            "libwhisper.a"
        } else {
            println!("cargo:warning=Unknown target family, defaulting to Unix lib names");
            "libwhisper.a"
        }
    }

    /// Returns the names of tools equivalent to [nm][nm] and [objcopy][objcopy].
    ///
    /// [nm]: https://www.man7.org/linux/man-pages/man1/nm.1.html
    /// [objcopy]: https://www.man7.org/linux/man-pages/man1/objcopy.1.html
    fn tool_names() -> (&'static str, &'static str) {
        let nm_names;
        let objcopy_names;
        if cfg!(target_family = "unix") {
            nm_names = vec!["nm", "llvm-nm"];
            objcopy_names = vec!["objcopy", "llvm-objcopy"];
        } else {
            nm_names = vec!["llvm-nm"];
            objcopy_names = vec!["llvm-objcopy"];
        }

        let nm_name;

        if let Some(path) = option_env!("NM_PATH") {
            nm_name = path;
        } else {
            println!("Looking for \"nm\" or an equivalent tool");
            nm_name = find_tool(&nm_names).expect(
                "No suitable tool equivalent to \"nm\" has been found in \
            PATH, if one is already installed, either add it to PATH or set NM_PATH to its full path",
            );
        }

        let objcopy_name;
        if let Some(path) = option_env!("OBJCOPY_PATH") {
            objcopy_name = path;
        } else {
            println!("Looking for \"objcopy\" or an equivalent tool");
            objcopy_name = find_tool(&objcopy_names).expect("No suitable tool equivalent to \"objcopy\" has \
            been found in PATH, if one is already installed, either add it to PATH or set OBJCOPY_PATH to its full path");
        }

        (nm_name, objcopy_name)
    }

    /// Returns the first tool found in the system, given a list of tool names, returning the first one found and
    /// printing its version.
    ///
    /// Returns [`Option::None`] if no tool is found.
    fn find_tool<'a>(names: &[&'a str]) -> Option<&'a str> {
        for name in names {
            if let Ok(output) = Command::new(name).arg("--version").output() {
                if output.status.success() {
                    let out_str = String::from_utf8_lossy(&output.stdout);
                    println!("Valid \"tool\" found:\n{out_str}");
                    return Some(name);
                }
            }
        }

        None
    }

    /// A filter for a symbol in a library.
    struct Filter<'a> {
        prefix: &'a str,
        sym_type: char,
    }

    /// Turns **`nm`**'s output into an iterator of [`str`] symbols.
    ///
    /// This function expects **`nm`** to be called using the **`-p`** and **`-P`** flags.
    fn get_symbols<'a, const N: usize>(
        nm_output: &'a str,
        filters: [Filter<'a>; N],
    ) -> impl Iterator<Item = &'a str> + 'a {
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
}
