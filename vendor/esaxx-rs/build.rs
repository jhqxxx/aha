#[cfg(feature = "cpp")]
#[cfg(not(target_os = "macos"))]
fn main() {
    let mut build = cc::Build::new();
    build.cpp(true).flag("-std=c++11");

    if std::env::var("CARGO_CFG_TARGET_FEATURE")
        .unwrap_or_default()
        .contains("crt-static")
    {
        build.static_crt(true);
    }

    build.file("src/esaxx.cpp").include("src").compile("esaxx");
}

#[cfg(feature = "cpp")]
#[cfg(target_os = "macos")]
fn main() {
    let mut build = cc::Build::new();
    build.cpp(true).flag("-std=c++11").flag("-stdlib=libc++");

    if std::env::var("CARGO_CFG_TARGET_FEATURE")
        .unwrap_or_default()
        .contains("crt-static")
    {
        build.static_crt(true);
    }

    build.file("src/esaxx.cpp").include("src").compile("esaxx");
}

#[cfg(not(feature = "cpp"))]
fn main() {}
