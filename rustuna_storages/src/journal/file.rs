use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
#[cfg(target_os = "macos")]
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rustuna_core::{Error, ErrorKind, Result};
use serde::Serialize;

use super::{JournalBackend, JournalLog};

const LOCK_FILE_SUFFIX: &str = ".lock";
const RENAME_FILE_SUFFIX: &str = ".rename";

fn fsync_file(file: &File) -> std::io::Result<()> {
    #[cfg(target_os = "macos")]
    {
        // NOTE:
        // Rust's File::sync_all() uses F_FULLFSYNC on macOS, which can be very slow.
        // Using fsync(2) is faster but may provide weaker durability guarantees
        // (e.g., device internal cache might not be flushed on power loss).
        loop {
            let rc = unsafe { libc::fsync(file.as_raw_fd()) };
            if rc == 0 {
                return Ok(());
            }
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue; // retry on EINTR
            }
            return Err(err);
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        file.sync_all()
    }
}

pub trait JournalFileLock: Send + Sync {
    fn acquire(&self) -> Result<()>;
    fn release(&self) -> Result<()>;
}

pub struct JournalFileBackend {
    file_path: PathBuf,
    lock: Box<dyn JournalFileLock>,
    log_number_offset: HashMap<usize, u64>,
}

impl JournalFileBackend {
    pub fn new(
        file_path: impl AsRef<Path>,
        lock: Option<Box<dyn JournalFileLock>>,
    ) -> Result<Self> {
        let file_path = file_path.as_ref().to_path_buf();
        if !file_path.exists() {
            File::create(&file_path)
                .map_err(|e| Error::with_reason(ErrorKind::StorageError, e.to_string()))?;
        }
        let lock = lock.unwrap_or_else(|| Box::new(JournalFileSymlinkLock::new(&file_path)));
        let mut log_number_offset = HashMap::new();
        log_number_offset.insert(0, 0);
        Ok(JournalFileBackend {
            file_path,
            lock,
            log_number_offset,
        })
    }
}

impl JournalBackend for JournalFileBackend {
    fn read_logs(
        &mut self,
        log_number_from: usize,
        handler: &mut dyn FnMut(JournalLog) -> Result<()>,
    ) -> Result<()> {
        let mut file = File::open(&self.file_path)
            .map_err(|e| Error::with_reason(ErrorKind::StorageError, e.to_string()))?;
        let file_size = file
            .metadata()
            .map_err(|e| Error::with_reason(ErrorKind::StorageError, e.to_string()))?
            .len();
        let mut remaining_log_size = file_size as i64;

        let mut log_number_start = 0usize;
        if let Some(offset) = self.log_number_offset.get(&log_number_from).copied() {
            file.seek(SeekFrom::Start(offset))
                .map_err(|e| Error::with_reason(ErrorKind::StorageError, e.to_string()))?;
            log_number_start = log_number_from;
            remaining_log_size -= offset as i64;
        }

        let mut reader = BufReader::new(file);
        let mut last_decode_error: Option<Error> = None;
        let mut log_number = log_number_start;
        loop {
            let mut line = Vec::new();
            let bytes = reader.read_until(b'\n', &mut line).map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Failed to read log file: {e}"),
                )
            })?;
            if bytes == 0 {
                break;
            }

            let byte_len = line.len() as u64;
            remaining_log_size -= byte_len as i64;
            if remaining_log_size < 0 {
                break;
            }
            if let Some(err) = last_decode_error.take() {
                return Err(err);
            }

            if !self.log_number_offset.contains_key(&(log_number + 1)) {
                let next_offset = self
                    .log_number_offset
                    .get(&log_number)
                    .copied()
                    .unwrap_or(0)
                    .saturating_add(byte_len);
                self.log_number_offset.insert(log_number + 1, next_offset);
            }

            if log_number < log_number_from {
                log_number += 1;
                continue;
            }

            if !line.ends_with(b"\n") {
                last_decode_error = Some(Error::with_reason(
                    ErrorKind::StorageError,
                    "Invalid log format.".to_string(),
                ));
                self.log_number_offset.remove(&(log_number + 1));
                log_number += 1;
                continue;
            }

            match decode_log_line(&line) {
                Ok(log) => handler(log)?,
                Err(err) => {
                    last_decode_error = Some(err);
                    self.log_number_offset.remove(&(log_number + 1));
                }
            }
            log_number += 1;
        }

        Ok(())
    }
    fn append_logs(&mut self, logs: &[JournalLog]) -> Result<()> {
        let _guard = JournalFileLockGuard::new(self.lock.as_ref())?;
        let what_to_write = encode_logs(logs)?;
        let mut file = OpenOptions::new()
            .append(true)
            .open(&self.file_path)
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Failed to open a file: {e}"),
                )
            })?;
        file.write_all(&what_to_write).map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to write logs: {e}"),
            )
        })?;
        file.flush().map_err(|e| {
            Error::with_reason(ErrorKind::StorageError, format!("Failed to flush: {e}"))
        })?;
        fsync_file(&file).map_err(|e| {
            Error::with_reason(ErrorKind::StorageError, format!("Failed to fsync: {e}"))
        })?;
        Ok(())
    }
}

pub struct JournalFileSymlinkLock {
    lock_target_file: PathBuf,
    lock_file: PathBuf,
}

impl JournalFileSymlinkLock {
    pub fn new(filepath: impl AsRef<Path>) -> Self {
        let filepath = filepath.as_ref().to_path_buf();
        JournalFileSymlinkLock {
            lock_target_file: filepath.clone(),
            lock_file: PathBuf::from(format!("{}{}", filepath.display(), LOCK_FILE_SUFFIX)),
        }
    }
}

impl JournalFileLock for JournalFileSymlinkLock {
    fn acquire(&self) -> Result<()> {
        let mut sleep_secs = 0.001f64;
        loop {
            match std::os::unix::fs::symlink(&self.lock_target_file, &self.lock_file) {
                Ok(_) => return Ok(()),
                Err(err) => {
                    if err.kind() == std::io::ErrorKind::AlreadyExists {
                        thread::sleep(Duration::from_secs_f64(sleep_secs));
                        sleep_secs = (sleep_secs * 2.0).min(1.0);
                        continue;
                    }
                    return Err(Error::with_reason(
                        ErrorKind::StorageError,
                        "Failed to create a symlink filej",
                    ));
                }
            }
        }
    }

    fn release(&self) -> Result<()> {
        let rename_file = PathBuf::from(format!(
            "{}{}{}",
            self.lock_file.display(),
            unique_suffix(),
            RENAME_FILE_SUFFIX
        ));
        match fs::rename(&self.lock_file, &rename_file) {
            Ok(_) => {}
            Err(e) => {
                let _ = fs::remove_file(&rename_file);
                return Err(Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Failed to rename lock file: {e}"),
                ));
            }
        }
        match fs::remove_file(&rename_file) {
            Ok(_) => Ok(()),
            Err(e) => {
                let _ = fs::remove_file(&rename_file);
                Err(Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Failed to remove rename file: {e}"),
                ))
            }
        }
    }
}

pub struct JournalFileOpenLock {
    lock_file: PathBuf,
}

impl JournalFileOpenLock {
    pub fn new(filepath: impl AsRef<Path>) -> Self {
        let filepath = filepath.as_ref().to_path_buf();
        JournalFileOpenLock {
            lock_file: PathBuf::from(format!("{}{}", filepath.display(), LOCK_FILE_SUFFIX)),
        }
    }
}

impl JournalFileLock for JournalFileOpenLock {
    fn acquire(&self) -> Result<()> {
        let mut sleep_secs = 0.001f64;
        loop {
            let res = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&self.lock_file);
            match res {
                Ok(_) => return Ok(()),
                Err(err) => {
                    if err.kind() == std::io::ErrorKind::AlreadyExists {
                        thread::sleep(Duration::from_secs_f64(sleep_secs));
                        sleep_secs = (sleep_secs * 2.0).min(1.0);
                        continue;
                    }
                    return Err(Error::with_reason(ErrorKind::StorageError, err.to_string()));
                }
            }
        }
    }

    fn release(&self) -> Result<()> {
        let rename_file = PathBuf::from(format!(
            "{}{}{}",
            self.lock_file.display(),
            unique_suffix(),
            RENAME_FILE_SUFFIX
        ));
        match fs::rename(&self.lock_file, &rename_file) {
            Ok(_) => {}
            Err(e) => {
                let _ = fs::remove_file(&rename_file);
                return Err(Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Failed to rename lock file: {e}"),
                ));
            }
        }
        match fs::remove_file(&rename_file) {
            Ok(_) => Ok(()),
            Err(e) => {
                let _ = fs::remove_file(&rename_file);
                Err(Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Failed to remove rename file: {e}"),
                ))
            }
        }
    }
}

struct JournalFileLockGuard<'a> {
    lock: &'a dyn JournalFileLock,
}

impl<'a> JournalFileLockGuard<'a> {
    fn new(lock: &'a dyn JournalFileLock) -> Result<Self> {
        lock.acquire()?;
        Ok(JournalFileLockGuard { lock })
    }
}

impl<'a> Drop for JournalFileLockGuard<'a> {
    fn drop(&mut self) {
        let _ = self.lock.release();
    }
}

fn unique_suffix() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!(".{}-{:?}", nanos, thread::current().id())
}

pub fn encode_logs(logs: &[JournalLog]) -> Result<Vec<u8>> {
    let mut result = Vec::new();
    for (i, log) in logs.iter().enumerate() {
        if i > 0 {
            result.push(b'\n');
        }
        let mut buf = Vec::new();
        let mut serializer =
            serde_json::Serializer::with_formatter(&mut buf, serde_json::ser::CompactFormatter);
        serde_json::to_value(log)
            .map_err(|e| Error::with_reason(ErrorKind::StorageError, e.to_string()))?
            .serialize(&mut serializer)
            .map_err(|e| Error::with_reason(ErrorKind::StorageError, e.to_string()))?;
        result.extend_from_slice(&buf);
    }
    result.push(b'\n');
    Ok(result)
}

pub fn decode_log_line(line: &[u8]) -> Result<JournalLog> {
    let mut slice = line;
    if slice.ends_with(b"\n") {
        slice = &slice[..slice.len() - 1];
    }
    if slice.ends_with(b"\r") {
        slice = &slice[..slice.len() - 1];
    }
    serde_json::from_slice(slice)
        .map_err(|e| Error::with_reason(ErrorKind::StorageError, e.to_string()))
}
