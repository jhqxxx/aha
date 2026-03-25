use anyhow::{Result, anyhow};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default)]
pub struct OnnxRuntimeConfig {
    pub intra_threads: Option<usize>,
    pub inter_threads: Option<usize>,
}

#[cfg(feature = "onnx-runtime")]
pub struct OnnxSessionBundle {
    pub session: ort::session::Session,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
}

#[cfg(not(feature = "onnx-runtime"))]
pub struct OnnxSessionBundle;

#[cfg(feature = "onnx-runtime")]
fn default_ort_dylib_filename() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "onnxruntime.dll"
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        "libonnxruntime.so"
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        "libonnxruntime.dylib"
    }
}

#[cfg(feature = "onnx-runtime")]
fn candidate_ort_dylib_paths() -> Vec<PathBuf> {
    let file = default_ort_dylib_filename();
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut candidates = vec![
        manifest_dir.join("lib").join(file),
        std::env::current_dir()
            .unwrap_or_else(|_| manifest_dir.clone())
            .join("lib")
            .join(file),
    ];

    if let Ok(current_exe) = std::env::current_exe()
        && let Some(exe_dir) = current_exe.parent()
    {
        candidates.push(exe_dir.join(file));
        candidates.push(exe_dir.join("lib").join(file));
        if let Some(parent) = exe_dir.parent() {
            candidates.push(parent.join(file));
            candidates.push(parent.join("lib").join(file));
        }
    }

    candidates
}

#[cfg(feature = "onnx-runtime")]
pub fn ensure_ort_dylib_path() -> Result<PathBuf> {
    if let Ok(path) = std::env::var("ORT_DYLIB_PATH")
        && !path.is_empty()
    {
        let path = PathBuf::from(path);
        if path.exists() {
            return Ok(path);
        }
        return Err(anyhow!(
            "ORT_DYLIB_PATH is set but the file does not exist: {}",
            path.display()
        ));
    }

    if let Some(path) = candidate_ort_dylib_paths()
        .into_iter()
        .find(|candidate| candidate.exists())
    {
        unsafe { std::env::set_var("ORT_DYLIB_PATH", &path) };
        return Ok(path);
    }

    Err(anyhow!(
        "unable to locate {}; searched repo/current working directory/current executable folders. \
set ORT_DYLIB_PATH explicitly or place the runtime under lib/",
        default_ort_dylib_filename()
    ))
}

#[cfg(not(feature = "onnx-runtime"))]
pub fn ensure_ort_dylib_path() -> Result<PathBuf> {
    Err(anyhow!(
        "onnx runtime support is not enabled; rebuild with --features onnx-runtime"
    ))
}

fn find_first_onnx_file(dir: &Path) -> Result<PathBuf> {
    let mut stack = vec![dir.to_path_buf()];
    let mut matches = Vec::new();
    while let Some(current) = stack.pop() {
        for entry in std::fs::read_dir(&current)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("onnx"))
            {
                matches.push(path);
            }
        }
    }
    matches.sort();
    matches
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("no .onnx file found under {}", dir.display()))
}

pub fn resolve_onnx_file(path: &str) -> Result<PathBuf> {
    let model_path = Path::new(path);
    if !model_path.exists() {
        return Err(anyhow!("onnx model path not found: {}", path));
    }
    if model_path.is_dir() {
        return find_first_onnx_file(model_path);
    }
    if model_path
        .extension()
        .is_none_or(|ext| !ext.eq_ignore_ascii_case("onnx"))
    {
        return Err(anyhow!(
            "onnx model path does not point to a .onnx file: {}",
            path
        ));
    }
    Ok(model_path.to_path_buf())
}

pub fn validate_onnx_file(path: &str) -> Result<()> {
    let _ = resolve_onnx_file(path)?;
    Ok(())
}

pub fn resolve_tokenizer_dir(
    onnx_path: &str,
    tokenizer_dir: Option<&str>,
    required_files: &[&str],
) -> Result<PathBuf> {
    fn has_required_files(path: &Path, required_files: &[&str]) -> bool {
        required_files.iter().all(|file| path.join(file).exists())
    }

    fn without_onnx_suffix(path: &Path) -> Option<PathBuf> {
        let name = path.file_name()?.to_string_lossy();
        for suffix in ["-ONNX", "-onnx"] {
            if name.ends_with(suffix) {
                let base = &name[..name.len() - suffix.len()];
                if !base.is_empty() {
                    return Some(path.with_file_name(base));
                }
            }
        }
        None
    }

    let mut candidates = Vec::new();
    if let Some(dir) = tokenizer_dir {
        candidates.push(PathBuf::from(dir));
    }

    let onnx_file = resolve_onnx_file(onnx_path)?;
    if let Some(parent) = onnx_file.parent() {
        candidates.push(parent.to_path_buf());
        if let Some(grand) = parent.parent() {
            candidates.push(grand.to_path_buf());
        }
    }

    let mut expanded = candidates.clone();
    for candidate in candidates {
        if let Some(sibling) = without_onnx_suffix(&candidate) {
            expanded.push(sibling);
        }
    }

    let mut unique = Vec::new();
    for candidate in expanded {
        if !unique
            .iter()
            .any(|existing: &PathBuf| existing == &candidate)
        {
            unique.push(candidate);
        }
    }

    for candidate in unique {
        if has_required_files(&candidate, required_files) {
            return Ok(candidate);
        }
    }

    Err(anyhow!(
        "unable to infer tokenizer directory for onnx artifact {}; provide --tokenizer-dir",
        onnx_file.display()
    ))
}

#[cfg(feature = "onnx-runtime")]
pub fn create_session(path: &str, _cfg: Option<&OnnxRuntimeConfig>) -> Result<OnnxSessionBundle> {
    let _ = ensure_ort_dylib_path()?;
    let model_file = resolve_onnx_file(path)?;
    let session = ort::session::Session::builder()?.commit_from_file(model_file)?;
    let input_names = session
        .inputs()
        .iter()
        .map(|input| input.name().to_string())
        .collect::<Vec<_>>();
    let output_names = session
        .outputs()
        .iter()
        .map(|output| output.name().to_string())
        .collect::<Vec<_>>();
    Ok(OnnxSessionBundle {
        session,
        input_names,
        output_names,
    })
}

#[cfg(not(feature = "onnx-runtime"))]
pub fn create_session(_path: &str, _cfg: Option<&OnnxRuntimeConfig>) -> Result<OnnxSessionBundle> {
    Err(anyhow!(
        "onnx runtime support is not enabled; rebuild with --features onnx-runtime"
    ))
}
