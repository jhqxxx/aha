use std::io::BufRead;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use tauri::{Emitter, State};

// ── State ────────────────────────────────────────────

struct AppState {
    server_process: Arc<Mutex<Option<Child>>>,
    server_logs: Arc<Mutex<Vec<String>>>,
}

// ── Data Types ───────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
struct ModelInfo {
    model_id: String,
    owner: String,
    model_type: String,
    downloaded: bool,
    size: Option<u64>,
    size_human: Option<String>,
    path: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ModelDetail {
    model_id: String,
    owner: String,
    model_type: String,
    downloaded: bool,
    size: Option<u64>,
    size_human: Option<String>,
    path: Option<String>,
    is_gguf: bool,
    is_onnx: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct LaunchConfig {
    model_id: String,
    address: Option<String>,
    port: Option<u16>,
    weight_path: Option<String>,
    gguf_path: Option<String>,
    mmproj_path: Option<String>,
    save_dir: Option<String>,
    download_retries: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
struct ServerStatusResponse {
    running: bool,
    pid: Option<u32>,
    logs: Vec<String>,
}

// ── Helpers ──────────────────────────────────────────

#[tauri::command]
fn get_default_save_dir() -> Result<String, String> {
    aha::utils::get_default_save_dir().ok_or_else(|| "无法获取 home 目录".to_string())
}

fn get_save_dir() -> Result<String, String> {
    aha::utils::get_default_save_dir().ok_or_else(|| "无法获取 home 目录".to_string())
}

fn find_aha_binary() -> Result<PathBuf, String> {
    // 1. 检查 PATH 环境变量
    if let Some(paths) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&paths) {
            let candidate = dir.join("aha").with_extension("exe");
            if candidate.exists() || dir.join("aha").exists() {
                let found = dir.join("aha");
                // check with extension on Windows
                if cfg!(windows) {
                    let with_ext = found.with_extension("exe");
                    if with_ext.exists() {
                        return Ok(with_ext);
                    }
                }
                if found.exists() {
                    return Ok(found);
                }
            }
        }
    }

    // 2. 在 Tauri 可执行文件附近查找（开发模式）
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            // aha-ui/target/debug/ 同级应该有 aha/target/debug/aha
            // 或者 workspace 根 target 目录
            let candidates = vec![
                exe_dir.join("aha").with_extension("exe"),
                exe_dir.join("../aha").join("aha").with_extension("exe"),
            ];
            for c in &candidates {
                if c.exists() {
                    return Ok(c.clone());
                }
            }

            // 尝试 workspace target 目录
            // exe_dir 是 aha-ui/target/debug/ 或 aha-ui/src-tauri/target/debug/
            if let Some(target_dir) = exe_dir.parent() {
                let ws = target_dir.join("aha").with_extension("exe");
                if ws.exists() {
                    return Ok(ws);
                }
            }
        }
    }

    // 3. 根据 CARGO_MANIFEST_DIR 推断（编译时）
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // manifest = aha-ui/src-tauri
    let workspace_target = manifest
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("target").join("debug").join("aha").with_extension("exe"));
    if let Some(p) = workspace_target {
        if p.exists() {
            return Ok(p);
        }
        // try release
        let release = manifest
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("target").join("release").join("aha").with_extension("exe"));
        if let Some(r) = release {
            if r.exists() {
                return Ok(r);
            }
        }
    }

    Err("未找到 aha 可执行文件，请先编译: cd aha && cargo build".to_string())
}

// ── Model Commands ───────────────────────────────────

#[tauri::command]
fn list_models() -> Vec<ModelInfo> {
    use aha::models::common::model_mapping::WhichModel;
    use aha::utils::{bytes_to_human, dir_size, get_default_weight_path, is_model_downloaded};

    let models = WhichModel::model_list();

    models
        .iter()
        .map(|m| {
            let model_id = m.as_string();
            let downloaded = is_model_downloaded(*m);
            let (size, size_human, path) = if downloaded {
                let p = get_default_weight_path(*m);
                let sz = dir_size(std::path::Path::new(&p)).ok();
                (sz, sz.map(bytes_to_human), Some(p))
            } else {
                (None, None, None)
            };

            ModelInfo {
                model_id,
                owner: m.model_owner(),
                model_type: m.model_type().to_string(),
                downloaded,
                size,
                size_human,
                path,
            }
        })
        .collect()
}

#[tauri::command]
fn get_model_detail(model_id: String) -> Result<ModelDetail, String> {
    use aha::models::common::model_mapping::WhichModel;
    use aha::utils::{bytes_to_human, dir_size, get_default_weight_path, is_model_downloaded};

    // 根据 model_id 找到对应的 WhichModel
    let model = WhichModel::model_list()
        .into_iter()
        .find(|m| m.as_string() == model_id)
        .ok_or_else(|| format!("未知模型: {}", model_id))?;

    let downloaded = is_model_downloaded(model);
    let (size, size_human, path) = if downloaded {
        let p = get_default_weight_path(model);
        let sz = dir_size(std::path::Path::new(&p)).ok();
        (sz, sz.map(bytes_to_human), Some(p))
    } else {
        (None, None, None)
    };

    Ok(ModelDetail {
        model_id: model.as_string(),
        owner: model.model_owner(),
        model_type: model.model_type().to_string(),
        downloaded,
        size,
        size_human,
        path,
        is_gguf: model.is_gguf(),
        is_onnx: model.is_onnx(),
    })
}

#[tauri::command]
async fn download_model(model_id: String, save_dir: Option<String>) -> Result<(), String> {
    use aha::utils::download_model;

    let save_dir = match save_dir {
        Some(dir) => dir,
        None => get_save_dir()?,
    };
    download_model(&model_id, &save_dir, 3)
        .await
        .map_err(|e| format!("下载失败: {}", e))
}

#[tauri::command]
fn delete_model(model_id: String) -> Result<(), String> {
    use aha::utils::get_default_weight_path;
    use aha::models::common::model_mapping::WhichModel;

    let model = WhichModel::model_list()
        .into_iter()
        .find(|m| m.as_string() == model_id)
        .ok_or_else(|| format!("未知模型: {}", model_id))?;

    let path = get_default_weight_path(model);
    let p = std::path::Path::new(&path);
    if p.exists() {
        std::fs::remove_dir_all(p).map_err(|e| format!("删除失败: {}", e))?;
    }
    Ok(())
}

// ── Server Commands ──────────────────────────────────

#[tauri::command]
fn start_server(
    config: LaunchConfig,
    state: State<'_, AppState>,
    app: tauri::AppHandle,
) -> Result<(), String> {
    let mut guard = state.server_process.lock().map_err(|e| e.to_string())?;

    // 检查是否已经在运行
    if let Some(ref mut child) = *guard {
        match child.try_wait() {
            Ok(Some(_)) => {} // 已退出
            Ok(None) => return Err("服务已经在运行中".to_string()),
            Err(e) => return Err(format!("检查服务状态失败: {}", e)),
        }
    }
    // 进程已退出，清理
    *guard = None;

    let binary = find_aha_binary()?;
    let address = config.address.unwrap_or_else(|| "127.0.0.1".to_string());
    let port = config.port.unwrap_or(10100);

    let mut cmd = Command::new(&binary);
    cmd.arg("cli")
        .arg("-m")
        .arg(&config.model_id)
        .arg("--address")
        .arg(&address)
        .arg("--port")
        .arg(port.to_string());

    if let Some(wp) = &config.weight_path {
        cmd.arg("--weight-path").arg(wp);
    }
    if let Some(gp) = &config.gguf_path {
        cmd.arg("--gguf-path").arg(gp);
    }
    if let Some(mp) = &config.mmproj_path {
        cmd.arg("--mmproj-path").arg(mp);
    }
    if let Some(sd) = &config.save_dir {
        cmd.arg("--save-dir").arg(sd);
    }
    if let Some(dr) = config.download_retries {
        cmd.arg("--download-retries").arg(dr.to_string());
    }

    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("启动服务失败: {}", e))?;

    let pid = child.id();
    let logs = state.server_logs.clone();
    let app_clone = app.clone();

    // 启动前追加一条日志
    {
        let mut l = logs.lock().map_err(|e| e.to_string())?;
        l.push(format!(
            "[aha] 启动服务: {}:{}  模型: {}  PID: {}",
            address, port, config.model_id, pid
        ));
    }

    // 后台读取 stdout
    if let Some(stdout) = child.stdout.take() {
        let logs = logs.clone();
        let app = app_clone.clone();
        std::thread::spawn(move || {
            let reader = std::io::BufReader::new(stdout);
            for line in reader.lines() {
                if let Ok(line) = line {
                    let mut l = logs.lock().unwrap();
                    l.push(line.clone());
                    if l.len() > 2000 {
                        let excess = l.len() - 2000;
                        l.drain(0..excess);
                    }
                    drop(l);
                    let _ = app.emit("server-log", &line);
                }
            }
        });
    }

    // 后台读取 stderr
    if let Some(stderr) = child.stderr.take() {
        let logs = logs.clone();
        let app = app_clone.clone();
        std::thread::spawn(move || {
            let reader = std::io::BufReader::new(stderr);
            for line in reader.lines() {
                if let Ok(line) = line {
                    let mut l = logs.lock().unwrap();
                    l.push(line.clone());
                    if l.len() > 2000 {
                        let excess = l.len() - 2000;
                        l.drain(0..excess);
                    }
                    drop(l);
                    let _ = app.emit("server-log", &line);
                }
            }
        });
    }

    *guard = Some(child);

    let _ = app.emit("server-started", pid);

    Ok(())
}

#[tauri::command]
fn stop_server(state: State<'_, AppState>, app: tauri::AppHandle) -> Result<(), String> {
    let mut guard = state.server_process.lock().map_err(|e| e.to_string())?;

    if let Some(mut child) = guard.take() {
        let _ = child.kill();
        let _ = child.wait();
        let mut l = state.server_logs.lock().map_err(|e| e.to_string())?;
        l.push("[aha] 服务已停止".to_string());
        let _ = app.emit("server-stopped", ());
    }

    Ok(())
}

#[tauri::command]
fn get_server_status(state: State<'_, AppState>) -> ServerStatusResponse {
    let mut running = false;
    let pid = {
        let mut guard = state.server_process.lock().unwrap();
        if let Some(ref mut child) = *guard {
            match child.try_wait() {
                Ok(Some(_)) => false,
                Ok(None) => {
                    running = true;
                    true
                }
                Err(_) => false,
            }
        } else {
            false
        }
        .then(|| guard.as_ref().map(|c| c.id()))
        .flatten()
    };

    let logs = state.server_logs.lock().unwrap().clone();
    ServerStatusResponse { running, pid, logs }
}

#[tauri::command]
fn clear_logs(state: State<'_, AppState>) -> Result<(), String> {
    let mut l = state.server_logs.lock().map_err(|e| e.to_string())?;
    l.clear();
    Ok(())
}

// ── Entry Point ──────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(AppState {
            server_process: Arc::new(Mutex::new(None)),
            server_logs: Arc::new(Mutex::new(Vec::new())),
        })
        .invoke_handler(tauri::generate_handler![
            list_models,
            get_model_detail,
            download_model,
            delete_model,
            start_server,
            stop_server,
            get_server_status,
            clear_logs,
            get_default_save_dir,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
