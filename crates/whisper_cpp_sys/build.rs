use std::path::{Path, PathBuf};
use std::process::exit;
use std::{env, fs, io};

const SUBMODULE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/thirdparty/whisper.cpp");

fn copy_recursively(src: &Path, dst: &Path) -> io::Result<()> {
    if !dst.exists() {
        fs::create_dir_all(dst)?;
    }

    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;

        if file_type.is_dir() {
            copy_recursively(&entry.path(), &dst.join(entry.file_name()))?;
        } else {
            fs::copy(entry.path(), dst.join(entry.file_name()))?;
        }
    }

    Ok(())
}

fn main() {
    let submodule_dir = &PathBuf::from(SUBMODULE_DIR);

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());

    let build_dir = out_dir.join("build");
    let header_path = out_dir.join("build/whisper.h");

    if fs::read_dir(submodule_dir).is_err() {
        eprintln!("Could not find {SUBMODULE_DIR}. Did you forget to initialize submodules?");

        exit(1);
    }

    if let Err(err) = fs::create_dir_all(&build_dir) {
        eprintln!("Could not create {build_dir:#?}: {err}");

        exit(1);
    }

    if let Err(err) = copy_recursively(submodule_dir, &build_dir) {
        eprintln!("Could not copy {submodule_dir:#?} into {build_dir:#?}: {err}");

        exit(1);
    }

    let dst = cmake::Config::new(&build_dir)
        .configure_arg("-DBUILD_SHARED_LIBS=Off")
        .configure_arg("-DWHISPER_BUILD_EXAMPLES=Off")
        .configure_arg("-DWHISPER_BUILD_TESTS=Off")
        .build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());
    println!("cargo:rustc-link-lib=static=llama");

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
