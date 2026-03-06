use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use rustuna_core::trial_queue::TrialQueue;
use rustuna_core::{Error, ErrorKind, Result};

static SEQUENCE_COUNTER: AtomicU64 = AtomicU64::new(0);

pub struct DirectoryTrialQueue {
    pending_dir: PathBuf,
    processing_dir: PathBuf,
}

impl DirectoryTrialQueue {
    /// Creates a new DirectoryTrialQueue with the specified base directory.
    ///
    /// The base directory will contain two subdirectories:
    /// - `pending/`: Contains files representing queued trial IDs
    /// - `processing/`: Contains files that have been popped but may need recovery
    ///
    /// # Arguments
    ///
    /// * `base_dir` - Base directory path. Caller should pass a study-specific path
    ///   like `{storage_dir}/queue/{study_id}/` to ensure isolation between studies.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use rustuna_storages::directory_queue::DirectoryTrialQueue;
    /// let queue = DirectoryTrialQueue::new("/path/to/storage/queue/1").unwrap();
    /// ```
    pub fn new(base_dir: impl AsRef<Path>) -> Result<Self> {
        let base = base_dir.as_ref().to_path_buf();
        let pending_dir = base.join("pending");
        let processing_dir = base.join("processing");

        fs::create_dir_all(&pending_dir).map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to create pending directory: {e}"),
            )
        })?;
        fs::create_dir_all(&processing_dir).map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to create processing directory: {e}"),
            )
        })?;

        Ok(Self {
            pending_dir,
            processing_dir,
        })
    }

    fn trial_id_to_filename(trial_id: u32) -> String {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let sequence = SEQUENCE_COUNTER.fetch_add(1, Ordering::SeqCst);
        format!("{timestamp:032x}_{sequence:016x}_{trial_id:010}")
    }

    fn filename_to_trial_id(filename: &str) -> Option<u32> {
        filename.split('_').nth(2)?.parse().ok()
    }
}

impl TrialQueue for DirectoryTrialQueue {
    fn push(&mut self, trial_id: u32) -> Result<()> {
        let filename = Self::trial_id_to_filename(trial_id);
        let target_path = self.pending_dir.join(&filename);

        // Create a temporary file first, then atomically rename it to the target path.
        // This ensures that incomplete writes are never visible.
        let temp_file = tempfile::NamedTempFile::new_in(&self.pending_dir).map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to create temporary file: {e}"),
            )
        })?;

        temp_file.persist(&target_path).map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to persist trial file: {e}"),
            )
        })?;

        Ok(())
    }

    fn pop(&mut self) -> Result<u32> {
        // Read all entries from pending directory and sort by filename
        let mut entries: Vec<_> = fs::read_dir(&self.pending_dir)
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Failed to read pending directory: {e}"),
                )
            })?
            .filter_map(|entry| entry.ok())
            .collect();

        if entries.is_empty() {
            return Err(Error::new(ErrorKind::TrialQueueEmpty));
        }

        entries.sort_by_key(|entry| entry.file_name());

        // Try to move the first available file from pending to processing.
        // Only one process will succeed in moving a specific file due to
        // the atomicity of rename on POSIX systems.
        for entry in entries {
            let filename = entry.file_name();
            let pending_path = entry.path();
            let processing_path = self.processing_dir.join(&filename);

            match fs::rename(&pending_path, &processing_path) {
                Ok(()) => {
                    // Successfully moved the file - parse and return the trial_id
                    if let Some(filename_str) = filename.to_str() {
                        if let Some(trial_id) = Self::filename_to_trial_id(filename_str) {
                            return Ok(trial_id);
                        }
                    }
                    // If we can't parse the filename, treat it as an error
                    return Err(Error::with_reason(
                        ErrorKind::Unexpected,
                        format!("Invalid trial queue filename: {filename:?}"),
                    ));
                }
                Err(e) if e.kind() == io::ErrorKind::NotFound => {
                    // Another process took this file, try the next one
                    continue;
                }
                Err(e) => {
                    return Err(Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Failed to move trial file: {e}"),
                    ));
                }
            }
        }

        // All files were taken by other processes
        Err(Error::new(ErrorKind::TrialQueueEmpty))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustuna_core::trial_queue::TrialQueue;

    #[test]
    fn test_push_and_pop() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut queue = DirectoryTrialQueue::new(temp_dir.path()).unwrap();

        queue.push(1).unwrap();
        queue.push(2).unwrap();
        queue.push(3).unwrap();

        assert_eq!(queue.pop().unwrap(), 1);
        assert_eq!(queue.pop().unwrap(), 2);
        assert_eq!(queue.pop().unwrap(), 3);
    }

    #[test]
    fn test_pop_empty_queue() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut queue = DirectoryTrialQueue::new(temp_dir.path()).unwrap();

        assert!(queue.pop().is_err());
    }

    #[test]
    fn test_fifo_order() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut queue = DirectoryTrialQueue::new(temp_dir.path()).unwrap();

        // Push in non-sequential order
        queue.push(5).unwrap();
        queue.push(2).unwrap();
        queue.push(8).unwrap();
        queue.push(1).unwrap();

        // Should pop in the order they were pushed (FIFO), not sorted by trial_id
        assert_eq!(queue.pop().unwrap(), 5);
        assert_eq!(queue.pop().unwrap(), 2);
        assert_eq!(queue.pop().unwrap(), 8);
        assert_eq!(queue.pop().unwrap(), 1);
    }

    #[test]
    fn test_multiple_queues_same_directory() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut queue1 = DirectoryTrialQueue::new(temp_dir.path()).unwrap();
        let mut queue2 = DirectoryTrialQueue::new(temp_dir.path()).unwrap();

        queue1.push(1).unwrap();
        queue1.push(2).unwrap();

        // Both queues share the same directory, so queue2 should see the items
        assert_eq!(queue2.pop().unwrap(), 1);
        assert_eq!(queue1.pop().unwrap(), 2);
    }
}
