//! Process management module for AHA services
//!
//! This module provides functionality for:
//! - Managing PID files for service tracking
//! - Discovering running AHA services
//! - Service information display

use std::fs;
use std::path::PathBuf;

use anyhow::{Result, anyhow};
use sysinfo::{Pid, ProcessesToUpdate, System};

/// Service information structure
#[derive(Debug, Clone)]
pub struct ServiceInfo {
    /// Service unique identifier (format: pid@port)
    pub service_id: String,
    /// Process ID
    pub pid: u32,
    /// Model name (if available)
    pub model: Option<String>,
    /// Listen port
    pub port: u16,
    /// Listen address
    pub address: String,
    /// Service status
    pub status: ServiceStatus,
}

/// Service status
#[derive(Debug, Clone, PartialEq)]
pub enum ServiceStatus {
    Running,
    Stopping,
    Unknown,
}

/// Get the PID file directory
///
/// Returns the appropriate directory for storing PID files:
/// - Linux/macOS: $XDG_RUNTIME_DIR/aha or ~/.aha/run
/// - Windows: %LOCALAPPDATA%\aha\run
pub fn get_pid_dir() -> Result<PathBuf> {
    #[cfg(unix)]
    {
        // Try XDG_RUNTIME_DIR first
        if let Ok(runtime_dir) = std::env::var("XDG_RUNTIME_DIR") {
            let pid_dir = PathBuf::from(runtime_dir).join("aha");
            fs::create_dir_all(&pid_dir)?;
            return Ok(pid_dir);
        }

        // Fallback to ~/.aha/run
        let home = dirs::home_dir().ok_or_else(|| anyhow!("Cannot determine home directory"))?;
        let pid_dir = home.join(".aha").join("run");
        fs::create_dir_all(&pid_dir)?;
        Ok(pid_dir)
    }

    #[cfg(windows)]
    {
        let local_app_data = std::env::var("LOCALAPPDATA")
            .map_err(|_| anyhow!("Cannot determine LOCALAPPDATA directory"))?;
        let pid_dir = PathBuf::from(local_app_data).join("aha").join("run");
        fs::create_dir_all(&pid_dir)?;
        Ok(pid_dir)
    }
}

/// Create a PID file for the current service
///
/// # Arguments
/// * `pid` - Process ID
/// * `port` - Listen port
pub fn create_pid_file(pid: u32, port: u16) -> Result<()> {
    let pid_dir = get_pid_dir()?;
    let pid_file = pid_dir.join(format!("{}.pid", port));

    let content = format!("{}\n", pid);
    fs::write(&pid_file, content)?;

    Ok(())
}

/// Clean up a PID file
///
/// # Arguments
/// * `port` - Listen port
pub fn cleanup_pid_file(port: u16) -> Result<()> {
    let pid_dir = get_pid_dir()?;
    let pid_file = pid_dir.join(format!("{}.pid", port));

    if pid_file.exists() {
        fs::remove_file(&pid_file)?;
    }

    Ok(())
}

/// Get the PID from a PID file
///
/// # Arguments
/// * `port` - Listen port
pub fn get_pid_from_file(port: u16) -> Option<u32> {
    let pid_dir = get_pid_dir().ok()?;
    let pid_file = pid_dir.join(format!("{}.pid", port));

    if !pid_file.exists() {
        return None;
    }

    let content = fs::read_to_string(&pid_file).ok()?;
    content.trim().parse::<u32>().ok()
}

/// Check if a process is an AHA service
///
/// Verifies that the process command line contains "aha serv" or "aha cli"
fn is_aha_process(sys: &System, pid: Pid) -> bool {
    if let Some(process) = sys.process(pid) {
        let cmd = process.cmd();
        let cmd_str: String = cmd.iter()
            .filter_map(|s| s.to_str())
            .collect::<Vec<&str>>()
            .join(" ");
        return cmd_str.contains("aha serv") || cmd_str.contains("aha cli");
    }
    false
}

/// Find all running AHA services
///
/// Returns a list of ServiceInfo for all running AHA services
pub fn find_aha_services() -> Result<Vec<ServiceInfo>> {
    let mut services = Vec::new();
    let mut sys = System::new_all();
    sys.refresh_processes(ProcessesToUpdate::All, true);

    // First, try to discover services from PID files
    let pid_dir = get_pid_dir()?;
    if let Ok(entries) = fs::read_dir(&pid_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("pid") {
                continue;
            }

            // Extract port from filename
            let port_str = path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            let port: u16 = port_str.parse().unwrap_or(0);

            if port == 0 {
                continue;
            }

            // Read PID from file
            if let Ok(content) = fs::read_to_string(&path) {
                if let Ok(pid) = content.trim().parse::<u32>() {
                    let sys_pid = Pid::from_u32(pid);
                    if is_aha_process(&sys, sys_pid) {
                        services.push(ServiceInfo {
                            service_id: format!("{}@{}", pid, port),
                            pid,
                            model: None, // TODO: Extract from command line
                            port,
                            address: "127.0.0.1".to_string(),
                            status: ServiceStatus::Running,
                        });
                    } else {
                        // Stale PID file, remove it
                        let _ = fs::remove_file(&path);
                    }
                }
            }
        }
    }

    // Fallback: scan processes for AHA services
    for (pid, process) in sys.processes() {
        if services.iter().any(|s| s.pid == pid.as_u32()) {
            continue; // Already found via PID file
        }

        let cmd = process.cmd();
        let cmd_str: String = cmd.iter()
            .filter_map(|s| s.to_str())
            .collect::<Vec<&str>>()
            .join(" ");

        if cmd_str.contains("aha serv") || cmd_str.contains("aha cli") {
            // Try to extract port from command line
            let port_str = cmd.iter()
                .position(|s| s.to_str() == Some("--port"))
                .and_then(|i| cmd.get(i + 1))
                .and_then(|s| s.to_str());
            let port = port_str
                .and_then(|s| s.parse::<u16>().ok())
                .unwrap_or(10100);

            services.push(ServiceInfo {
                service_id: format!("{}@{}", pid.as_u32(), port),
                pid: pid.as_u32(),
                model: None,
                port,
                address: "127.0.0.1".to_string(),
                status: ServiceStatus::Running,
            });
        }
    }

    Ok(services)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_pid_dir() {
        let pid_dir = get_pid_dir();
        assert!(pid_dir.is_ok());
        let dir = pid_dir.unwrap();
        assert!(dir.exists());
    }

    #[test]
    fn test_create_and_cleanup_pid_file() {
        let port = 19999;
        create_pid_file(12345, port).unwrap();
        let pid = get_pid_from_file(port);
        assert_eq!(pid, Some(12345));
        cleanup_pid_file(port).unwrap();
        let pid = get_pid_from_file(port);
        assert_eq!(pid, None);
    }

    #[test]
    fn test_get_pid_from_file_nonexistent() {
        let port = 19998; // Use a port that likely doesn't have a PID file
        let pid = get_pid_from_file(port);
        assert_eq!(pid, None);
    }

    #[test]
    fn test_service_status_debug() {
        // Test ServiceStatus Debug implementation
        assert_eq!(format!("{:?}", ServiceStatus::Running), "Running");
        assert_eq!(format!("{:?}", ServiceStatus::Stopping), "Stopping");
        assert_eq!(format!("{:?}", ServiceStatus::Unknown), "Unknown");
    }

    #[test]
    fn test_service_info_clone() {
        let service = ServiceInfo {
            service_id: "12345@10100".to_string(),
            pid: 12345,
            model: Some("qwen3-0.6b".to_string()),
            port: 10100,
            address: "127.0.0.1".to_string(),
            status: ServiceStatus::Running,
        };
        let service_clone = service.clone();
        assert_eq!(service_clone.service_id, "12345@10100");
        assert_eq!(service_clone.pid, 12345);
        assert_eq!(service_clone.model, Some("qwen3-0.6b".to_string()));
        assert_eq!(service_clone.port, 10100);
    }

    #[test]
    fn test_find_aha_services() {
        // This test will find actual running AHA services or return empty
        let services = find_aha_services();
        assert!(services.is_ok());
        let services_list = services.unwrap();
        // We can't assert specific services here since it depends on what's running
        // but we can verify the structure is correct
        for service in services_list {
            assert!(!service.service_id.is_empty());
            assert!(service.pid > 0);
            assert!(service.port > 0);
            assert!(!service.address.is_empty());
        }
    }
}
