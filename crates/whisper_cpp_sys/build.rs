use std::{env, fs};
use std::path::PathBuf;

// TODO add feature compatibility checks

const SUBMODULE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/thirdparty/whisper.cpp");

fn main() {
    let submodule_dir = &PathBuf::from(SUBMODULE_DIR);

    if fs::read_dir(submodule_dir).is_err() {
        panic!("Could not find {SUBMODULE_DIR}. Did you forget to initialize submodules?");
    }

    let mut config = cmake::Config::new(submodule_dir);

    config
        .define("BUILD_SHARED_LIBS", "ON")
        .define("WHISPER_BUILD_EXAMPLES", "OFF")
        .define("WHISPER_BUILD_TESTS", "OFF")
        .define("WHISPER_NO_ACCELERATE", "ON") // TODO accelerate is used by default, but is causing issues atm, check why
        .define("WHISPER_METAL", "OFF"); // TODO this is on by default on Apple devices, and is causing issues, see why

    #[cfg(not(feature = "avx"))]
    {
        config.define("WHISPER_NO_AVX", "ON");
    }

    #[cfg(not(feature = "avx2"))]
    {
        config.define("WHISPER_NO_AVX2", "ON");
    }

    #[cfg(not(feature = "fma"))]
    {
        config.define("WHISPER_NO_FMA", "ON");
    }

    #[cfg(not(feature = "f16c"))]
    {
        config.define("WHISPER_NO_F16C", "ON");
    }

    #[cfg(feature = "cuda")]
    {
        config.define("WHISPER_CUBLAS", "ON");
    }

    let dst = config.build();

    if cfg!(target_family = "windows") {
        println!("cargo:rustc-link-search=native={}/bin", dst.display());
        println!(
            "cargo:rustc-link-search=native={}/lib/static",
            dst.display()
        );
        println!("cargo:rustc-link-lib=dylib=whisper");
    } else {
        println!("cargo:rustc-link-search=native={}/lib", dst.display());
        println!("cargo:rustc-link-lib=dylib=whisper");
    }

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
    use std::collections::HashSet;
    use std::env;
    use std::fmt::{Display, Formatter};
    use std::path::{Path, PathBuf};
    use std::process::Command;

    pub fn redefine_symbols(out_path: impl AsRef<Path>) {
        let whisper_lib_name = lib_name();
        let (nm, objcopy) = tools();
        println!("Modifying {whisper_lib_name}, symbols acquired via \"{nm}\" and modified via \"{objcopy}\"");

        let lib_path = if cfg!(target_family = "windows") {
            out_path.as_ref().join("bin")
        } else {
            out_path.as_ref().join("lib")
        };

        // Modifying symbols exposed by the ggml library

        let out_str = nm_symbols(&nm, whisper_lib_name, &lib_path);
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
        objcopy_redefine(&objcopy, whisper_lib_name, "whisp_", symbols, &lib_path);
    }

    /// Returns *Whisper.cpp*'s compiled library name, based on the operating system.
    fn lib_name() -> &'static str {
        if cfg!(target_family = "windows") {
            "whisper.dll"
        } else if cfg!(target_os = "linux") {
            "libwhisper.so"
        } else if cfg!(any(
            target_os = "macos",
            target_os = "ios",
            target_os = "dragonfly"
        )) {
            "libwhisper.dylib"
        } else {
            println!("cargo:warning=Unknown target family, defaulting to Unix lib names");
            "libwhisper.so"
        }
    }

    /// Returns [`Tool`]s equivalent to [nm][nm] and [objcopy][objcopy].
    ///
    /// [nm]: https://www.man7.org/linux/man-pages/man1/nm.1.html
    /// [objcopy]: https://www.man7.org/linux/man-pages/man1/objcopy.1.html
    fn tools() -> (Tool, Tool) {
        let nm_names;
        let objcopy_names;
        let nm_help;
        let objcopy_help;
        if cfg!(target_os = "linux") {
            nm_names = vec!["nm", "llvm-nm"];
            objcopy_names = vec!["objcopy", "llvm-objcopy"];
            nm_help = vec!["\"nm\" from GNU Binutils", "\"llvm-nm\" from LLVM"];
            objcopy_help = vec![
                "\"objcopy\" from GNU Binutils",
                "\"llvm-objcopy\" from LLVM",
            ];
        } else if cfg!(any(
            target_os = "macos",
            target_os = "ios",
            target_os = "dragonfly"
        )) {
            nm_names = vec!["nm", "llvm-nm"];
            objcopy_names = vec!["llvm-objcopy"];
            nm_help = vec!["\"llvm-nm\" from LLVM 17"];
            objcopy_help = vec!["\"llvm-objcopy\" from LLVM 17"];
        } else {
            nm_names = vec!["llvm-nm"];
            objcopy_names = vec!["llvm-objcopy"];
            nm_help = vec!["\"llvm-nm\" from LLVM 17"];
            objcopy_help = vec!["\"llvm-objcopy\" from LLVM 17"];
        }

        let nm_env = "NM_PATH";
        println!("cargo:rerun-if-env-changed={nm_env}");
        println!("Looking for \"nm\" or an equivalent tool");
        let nm_name = find_tool(&nm_names, nm_env).unwrap_or_else(move || {
            panic_tool_help("nm", nm_env, &nm_help);
            unreachable!("The function above should have panicked")
        });

        let objcopy_env = "OBJCOPY_PATH";
        println!("cargo:rerun-if-env-changed={objcopy_env}");
        println!("Looking for \"objcopy\" or an equivalent tool..");
        let objcopy_name = find_tool(&objcopy_names, objcopy_env).unwrap_or_else(move || {
            panic_tool_help("objcopy", objcopy_env, &objcopy_help);
            unreachable!("The function above should have panicked")
        });

        (nm_name, objcopy_name)
    }

    /// A command line tool name present in `PATH` or its full [`Path`].
    enum Tool {
        /// The name of a tool present in `PATH`.
        Name(&'static str),

        /// The full [`Path`] to a tool.
        FullPath(PathBuf),
    }

    impl Display for Tool {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            match self {
                Tool::Name(name) => write!(f, "{}", name),
                Tool::FullPath(path) => write!(f, "{}", path.display()),
            }
        }
    }

    /// Returns the first [`Tool`] found in the system `PATH`, given a list of tool names, returning
    /// the first one found and printing its version.
    ///
    /// If a value is present in the provided environment variable name, it will get checked
    /// instead.
    ///
    /// ## Panic
    /// Returns [`Option::None`] if no [`Tool`] is found.
    fn find_tool(names: &[&'static str], env: &str) -> Option<Tool> {
        if let Ok(path_str) = env::var(env) {
            let path_str = path_str.trim_matches([' ', '"', '\''].as_slice());
            println!("{env} is set, checking if \"{path_str}\" is a valid tool");
            let path = PathBuf::from(&path_str);

            if !path.is_file() {
                panic!("\"{path_str}\" is not a file path.")
            }

            let output = Command::new(path_str)
                .arg("--version")
                .output()
                .unwrap_or_else(|e| panic!("Failed to run \"{path_str} --version\". ({e})"));

            if output.status.success() {
                let out_str = String::from_utf8_lossy(&output.stdout);
                println!("Valid tool found:\n{out_str}");
            } else {
                println!("cargo:warning=Tool \"{path_str}\" found, but could not execute \"{path_str} --version\"")
            }

            return Some(Tool::FullPath(path));
        }

        println!("{env} not set, looking for {names:?} in PATH");
        for name in names {
            if let Ok(output) = Command::new(name).arg("--version").output() {
                if output.status.success() {
                    let out_str = String::from_utf8_lossy(&output.stdout);
                    println!("Valid tool found:\n{out_str}");
                    return Some(Tool::Name(name));
                }
            }
        }

        None
    }

    /// Always panics, printing suggestions for finding the specified tool.
    fn panic_tool_help(name: &str, env: &str, suggestions: &[&str]) {
        let suggestions_str = if suggestions.is_empty() {
            String::new()
        } else {
            let mut suggestions_str = "For your Operating System we recommend:\n".to_string();
            for suggestion in &suggestions[..suggestions.len() - 1] {
                suggestions_str.push_str(&format!("{suggestion}\nOR\n"));
            }
            suggestions_str.push_str(suggestions[suggestions.len() - 1]);
            suggestions_str
        };

        panic!("No suitable tool equivalent to \"{name}\" has been found in PATH, if one is already installed, either add its directory to PATH or set {env} to its full path. {suggestions_str}")
    }

    /// Executes [nm][nm] or an equivalent tool in portable mode and returns the output.
    ///
    /// ## Panic
    /// Will panic on any errors.
    ///
    /// [nm]: https://www.man7.org/linux/man-pages/man1/nm.1.html
    fn nm_symbols(tool: &Tool, target_lib: &str, out_path: impl AsRef<Path>) -> String {
        let output = Command::new(tool.to_string())
            .current_dir(&out_path)
            .arg(target_lib)
            .args(["-p", "-P"])
            .output()
            .unwrap_or_else(move |e| panic!("Failed to run \"{tool}\". ({e})"));

        if !output.status.success() {
            panic!(
                "An error has occurred while acquiring symbols from the compiled library \"{target_lib}\" ({}):\n{}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            );
        }

        String::from_utf8_lossy(&output.stdout).to_string()
    }

    /// Executes [objcopy][objcopy], adding a prefix to the specified symbols of the target library.
    ///
    /// ## Panic
    /// Will panic on any errors.
    ///
    /// [objcopy]: https://www.man7.org/linux/man-pages/man1/objcopy.1.html
    fn objcopy_redefine(
        tool: &Tool,
        target_lib: &str,
        prefix: &str,
        symbols: HashSet<&str>,
        out_path: impl AsRef<Path>,
    ) {
        let mut cmd = Command::new(tool.to_string());
        cmd.current_dir(&out_path);
        for symbol in symbols {
            cmd.arg(format!("--redefine-sym={symbol}={prefix}{symbol}"));
        }

        let output = cmd
            .arg(target_lib)
            .output()
            .unwrap_or_else(move |e| panic!("Failed to run \"{tool}\". ({e})"));

        if !output.status.success() {
            panic!(
                "An error has occurred while redefining symbols from library file \"{target_lib}\" ({}):\n{}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            );
        }
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
    ) -> HashSet<&'a str> {
        let iter = nm_output
            .lines()
            .map(|symbol| {
                // Strip irrelevant information

                let mut stripped = symbol;
                while stripped.split(' ').count() > 2 {
                    // SAFETY: We just made sure ' ' is present above
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
            .map(|symbol| &symbol[..symbol.len() - 2]); // Strip the type, so only the symbol remains

        // Filter duplicates
        HashSet::from_iter(iter)
    }
}
