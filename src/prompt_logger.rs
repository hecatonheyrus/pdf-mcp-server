//! Optional logging of user prompts received by the MCP server.
//!
//! Logging is opt-in (`--log-prompts` CLI flag or `PDF_MCP_LOG_PROMPTS=true`)
//! and only supported when the server runs on Windows. Prompts are appended as
//! one JSON object per line (JSON Lines) to `./log/prompts.log`, relative to the
//! server's working directory.
//!
//! Logging failures never break a tool call: they are reported on stderr only.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value, json};

const LOG_DIR: &str = "log";
const LOG_FILE_NAME: &str = "prompts.log";

/// Serializes writes so concurrent tool calls never interleave lines.
static WRITE_LOCK: Mutex<()> = Mutex::new(());
/// Set once at startup; `None` means logging is disabled.
static LOG_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Configures prompt logging. Call once at startup.
///
/// Returns the log file path when logging was enabled.
pub fn init(requested: bool) -> Option<&'static PathBuf> {
    let path = if !requested {
        None
    } else if cfg!(target_os = "windows") {
        Some(PathBuf::from(".").join(LOG_DIR).join(LOG_FILE_NAME))
    } else {
        eprintln!("Prompt logging is only supported on Windows; ignoring --log-prompts.");
        None
    };
    LOG_PATH.get_or_init(|| path).as_ref()
}

/// Appends one prompt record to the log file, if logging is enabled.
///
/// * `tool`    – name of the MCP tool that received the prompt
/// * `prompt`  – the user's prompt text
/// * `context` – extra request arguments worth recording (file, pages, ...)
pub fn log_prompt(tool: &str, prompt: &str, context: &[(&str, &str)]) {
    let Some(path) = LOG_PATH.get().and_then(Option::as_ref) else {
        return;
    };

    if let Err(e) = append_line(path, &format_record(tool, prompt, context)) {
        eprintln!("Failed to write prompt log to {:?}: {}", path, e);
    }
}

fn format_record(tool: &str, prompt: &str, context: &[(&str, &str)]) -> String {
    let mut record = Map::new();
    record.insert("timestamp".into(), json!(utc_timestamp()));
    record.insert("tool".into(), json!(tool));
    record.insert("prompt".into(), json!(prompt));
    for (key, value) in context {
        record.insert((*key).into(), json!(value));
    }
    Value::Object(record).to_string()
}

/// Current UTC time as RFC 3339, e.g. `2026-11-25T14:03:07Z`.
fn utc_timestamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));

    // Civil-from-days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + if month <= 2 { 1 } else { 0 };

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, rem / 3_600, rem % 3_600 / 60, rem % 60
    )
}

fn append_line(path: &Path, line: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let _guard = WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{}", line)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_json_line() {
        let dir = std::env::temp_dir().join(format!("pdf-mcp-prompt-log-{}", std::process::id()));
        let path = dir.join(LOG_DIR).join(LOG_FILE_NAME);

        let line = format_record("test_tool", "summarize \"chapter\" 1", &[("file", "a.pdf"), ("page_list", "1-3")]);
        append_line(&path, &line).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        let line: Value = serde_json::from_str(content.lines().last().unwrap()).unwrap();
        assert_eq!(line["tool"], "test_tool");
        assert_eq!(line["prompt"], "summarize \"chapter\" 1");
        assert_eq!(line["file"], "a.pdf");
        assert_eq!(line["page_list"], "1-3");
        assert!(line["timestamp"].is_string());

        let _ = fs::remove_dir_all(dir);
    }
}
