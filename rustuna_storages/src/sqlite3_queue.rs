use rusqlite::{params, Connection};
use rustuna_core::trial_queue::TrialQueue;
use rustuna_core::{Error, ErrorKind, Result};
use std::path::Path;
use std::sync::Mutex;

pub struct SQLite3TrialQueue {
    conn: Mutex<Connection>,
    study_id: u32,
}

impl SQLite3TrialQueue {
    /// Creates a new SQLite3TrialQueue with the specified database path and study ID.
    ///
    /// Multiple studies can share the same database file, with study_id used for isolation.
    /// The `trial_queue` table will be created if it doesn't exist.
    ///
    /// # Arguments
    ///
    /// * `db_path` - Path to the SQLite database file
    /// * `study_id` - Study ID to isolate trials for this queue
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use rustuna_storages::sqlite3_queue::SQLite3TrialQueue;
    /// let queue = SQLite3TrialQueue::new("/path/to/queue.db", 1).unwrap();
    /// ```
    pub fn new(db_path: impl AsRef<Path>, study_id: u32) -> Result<Self> {
        let conn = Connection::open(db_path).map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to open SQLite database: {e}"),
            )
        })?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS trial_queue (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                study_id INTEGER NOT NULL,
                trial_id INTEGER NOT NULL,
                enqueued_at INTEGER NOT NULL,
                sequence INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_trial_queue_study_order 
                ON trial_queue(study_id, enqueued_at, sequence);",
        )
        .map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to create trial_queue table: {e}"),
            )
        })?;

        Ok(Self {
            conn: Mutex::new(conn),
            study_id,
        })
    }
}

impl TrialQueue for SQLite3TrialQueue {
    fn push(&mut self, trial_id: u32) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| Error::new(ErrorKind::StorageError))?;

        // Use current timestamp in microseconds for ordering
        let enqueued_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64;

        // Get next sequence number for this timestamp to handle concurrent inserts
        let sequence: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(sequence), -1) + 1 
                 FROM trial_queue 
                 WHERE study_id = ? AND enqueued_at = ?",
                params![self.study_id, enqueued_at],
                |row| row.get(0),
            )
            .unwrap_or(0);

        conn.execute(
            "INSERT INTO trial_queue (study_id, trial_id, enqueued_at, sequence) 
             VALUES (?, ?, ?, ?)",
            params![self.study_id, trial_id, enqueued_at, sequence],
        )
        .map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to enqueue trial: {e}"),
            )
        })?;

        Ok(())
    }

    fn pop(&mut self) -> Result<u32> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| Error::new(ErrorKind::StorageError))?;

        // Use a transaction with IMMEDIATE to ensure exclusive access
        conn.execute("BEGIN IMMEDIATE", []).map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to begin transaction: {e}"),
            )
        })?;

        let result: std::result::Result<u32, rusqlite::Error> = (|| {
            // Find and delete the first entry in FIFO order
            let trial_id: u32 = conn.query_row(
                "DELETE FROM trial_queue
                 WHERE rowid = (
                     SELECT rowid FROM trial_queue 
                     WHERE study_id = ? 
                     ORDER BY enqueued_at, sequence 
                     LIMIT 1
                 )
                 RETURNING trial_id",
                params![self.study_id],
                |row| row.get(0),
            )?;
            Ok(trial_id)
        })();

        match result {
            Ok(trial_id) => {
                conn.execute("COMMIT", []).map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Failed to commit transaction: {e}"),
                    )
                })?;
                Ok(trial_id)
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                let _ = conn.execute("ROLLBACK", []);
                Err(Error::new(ErrorKind::TrialQueueEmpty))
            }
            Err(e) => {
                let _ = conn.execute("ROLLBACK", []);
                Err(Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Failed to dequeue trial: {e}"),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_push_and_pop() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut queue = SQLite3TrialQueue::new(temp_file.path(), 1).unwrap();

        queue.push(10).unwrap();
        queue.push(20).unwrap();
        queue.push(30).unwrap();

        assert_eq!(queue.pop().unwrap(), 10);
        assert_eq!(queue.pop().unwrap(), 20);
        assert_eq!(queue.pop().unwrap(), 30);
    }

    #[test]
    fn test_pop_empty_queue() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut queue = SQLite3TrialQueue::new(temp_file.path(), 1).unwrap();

        assert!(queue.pop().is_err());
    }

    #[test]
    fn test_fifo_ordering() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut queue = SQLite3TrialQueue::new(temp_file.path(), 1).unwrap();

        for i in 1..=100 {
            queue.push(i).unwrap();
        }

        for i in 1..=100 {
            assert_eq!(queue.pop().unwrap(), i);
        }
    }

    #[test]
    fn test_study_isolation() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut queue1 = SQLite3TrialQueue::new(temp_file.path(), 1).unwrap();
        let mut queue2 = SQLite3TrialQueue::new(temp_file.path(), 2).unwrap();

        queue1.push(10).unwrap();
        queue1.push(20).unwrap();
        queue2.push(30).unwrap();
        queue2.push(40).unwrap();

        assert_eq!(queue1.pop().unwrap(), 10);
        assert_eq!(queue2.pop().unwrap(), 30);
        assert_eq!(queue1.pop().unwrap(), 20);
        assert_eq!(queue2.pop().unwrap(), 40);

        assert!(queue1.pop().is_err());
        assert!(queue2.pop().is_err());
    }

    #[test]
    fn test_push_pop_interleaved() {
        let temp_file = NamedTempFile::new().unwrap();
        let mut queue = SQLite3TrialQueue::new(temp_file.path(), 1).unwrap();

        queue.push(1).unwrap();
        queue.push(2).unwrap();
        assert_eq!(queue.pop().unwrap(), 1);

        queue.push(3).unwrap();
        assert_eq!(queue.pop().unwrap(), 2);
        assert_eq!(queue.pop().unwrap(), 3);

        assert!(queue.pop().is_err());
    }

    #[test]
    fn test_reopen_queue() {
        let temp_file = NamedTempFile::new().unwrap();
        let db_path = temp_file.path().to_path_buf();

        {
            let mut queue = SQLite3TrialQueue::new(&db_path, 1).unwrap();
            queue.push(10).unwrap();
            queue.push(20).unwrap();
        }

        {
            let mut queue = SQLite3TrialQueue::new(&db_path, 1).unwrap();
            assert_eq!(queue.pop().unwrap(), 10);
            assert_eq!(queue.pop().unwrap(), 20);
            assert!(queue.pop().is_err());
        }
    }
}
