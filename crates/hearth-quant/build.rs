fn main() {
    println!("cargo::rustc-check-cfg=cfg(msvc_kernel)");
    let feature = std::env::var("CARGO_FEATURE_MSVC_KERNEL");
    if feature.is_err() {
        return;
    }

    let vs_base = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools";
    let msvc_ver = "14.51.36231";
    let winsdk_ver = r"C:\Program Files (x86)\Windows Kits\10";

    // Find Windows SDK version
    let sdk_include = std::fs::read_dir(format!("{}\\Include", winsdk_ver))
        .ok()
        .and_then(|mut d| d.next().and_then(|e| e.ok()).map(|e| e.path()))
        .unwrap_or_else(|| format!("{}\\Include\\10.0.26100.0", winsdk_ver).into());
    let _sdk_lib = std::fs::read_dir(format!("{}\\Lib", winsdk_ver))
        .ok()
        .and_then(|mut d| d.next().and_then(|e| e.ok()).map(|e| e.path()))
        .unwrap_or_else(|| format!("{}\\Lib\\10.0.26100.0", winsdk_ver).into());

    let inc = format!("{}\\VC\\Tools\\MSVC\\{}\\include", vs_base, msvc_ver);
    let _lib = format!("{}\\VC\\Tools\\MSVC\\{}\\lib\\x64", vs_base, msvc_ver);

    let mut build = cc::Build::new();
    build
        .file("src/msvc_kernel.c")
        .opt_level(2)
        .flag("/arch:SSE2")
        .flag("/Oi")
        .flag("/Ot")
        .include(&inc)
        .include(format!("{}\\ucrt", sdk_include.display()))
        .include(format!("{}\\um", sdk_include.display()))
        .include(format!("{}\\shared", sdk_include.display()));

    // Use explicit toolchain if env var points to one
    if let Ok(tc) = std::env::var("MSVC_TOOLCHAIN") {
        let cl = std::path::PathBuf::from(&tc)
            .join("bin")
            .join("Hostx64")
            .join("x64")
            .join("cl.exe");
        if cl.exists() {
            build.compiler(cl);
        }
    } else if let Ok(cc) = std::env::var("CC") {
        build.compiler(&cc);
    }

    build.compile("msvc_kernel");
    println!("cargo:rerun-if-changed=src/msvc_kernel.c");
    println!("cargo:rustc-cfg=msvc_kernel");
}
