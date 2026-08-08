//! Conversions between the in-memory and the persisted representation of trial datetimes.
//!
//! Rustuna follows the same layering as Optuna (5.0.0rc1 and later):
//!
//! - `PersistedTrial::datetime_start` / `datetime_complete` are **timezone-naive local time**, so
//!   that `FrozenTrial.datetime_start` keeps returning local time to users.
//! - [`crate::sqlite3::SQLite3Storage`] persists **timezone-naive UTC**. Timezone-aware column
//!   types would need a schema migration where they are supported at all, so the offset is dropped
//!   and the value is normalized to UTC instead.
//! - [`crate::journal::JournalStorage`] persists **timezone-aware UTC**, because a journal log is
//!   JSON and can carry the offset without any schema concerns.
//!
//! Every conversion happens at a persistence boundary, so nothing outside this module needs to
//! know which encoding a backend uses.

use chrono::{DateTime, Local, NaiveDateTime, SecondsFormat, TimeZone, Utc};
use rustuna_core::{Error, ErrorKind, Result};

/// Format used for naive datetimes, matching SQLite's `strftime('%Y-%m-%d %H:%M:%f', ...)` and
/// Python's `str(datetime)`.
const NAIVE_FORMAT: &str = "%Y-%m-%d %H:%M:%S%.f";

fn invalid(value: &str) -> Error {
    Error::with_reason(
        ErrorKind::StorageError,
        format!("Failed to parse datetime: {value}"),
    )
}

fn parse_naive(value: &str) -> Result<NaiveDateTime> {
    NaiveDateTime::parse_from_str(value, NAIVE_FORMAT)
        .or_else(|_| NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S"))
        .or_else(|_| NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S%.f"))
        .map_err(|_| invalid(value))
}

fn format_naive(value: NaiveDateTime) -> String {
    value.format(NAIVE_FORMAT).to_string()
}

/// Interprets a naive local datetime, resolving the ambiguity of a DST fold towards the earlier
/// instant the way `datetime.astimezone()` does in Python.
fn local_to_utc(value: NaiveDateTime) -> Result<DateTime<Utc>> {
    let local = Local
        .from_local_datetime(&value)
        .earliest()
        // A local time skipped by a DST jump forward has no instant at all; map it through the
        // offset in effect just before the gap rather than failing the whole read.
        .or_else(|| Local.from_local_datetime(&value).latest())
        .ok_or_else(|| invalid(&format_naive(value)))?;
    Ok(local.with_timezone(&Utc))
}

/// Converts naive local time (as held by `PersistedTrial`) into the naive UTC stored by SQLite.
pub fn naive_local_to_naive_utc(value: &str) -> Result<String> {
    Ok(format_naive(local_to_utc(parse_naive(value)?)?.naive_utc()))
}

/// Converts the naive UTC stored by SQLite back into the naive local time held by
/// `PersistedTrial`.
pub fn naive_utc_to_naive_local(value: &str) -> Result<String> {
    let utc = Utc.from_utc_datetime(&parse_naive(value)?);
    Ok(format_naive(utc.with_timezone(&Local).naive_local()))
}

/// Converts naive local time into the timezone-aware UTC written to a journal log.
///
/// The output matches Python's `datetime.isoformat(timespec="microseconds")` on an aware UTC
/// datetime, so Optuna can read logs written by Rustuna.
pub fn naive_local_to_aware_utc(value: &str) -> Result<String> {
    Ok(local_to_utc(parse_naive(value)?)?.to_rfc3339_opts(SecondsFormat::Micros, false))
}

/// Converts a datetime read from a journal log into the naive local time held by
/// `PersistedTrial`.
///
/// Logs written before Rustuna and Optuna moved to aware UTC carry a naive local datetime with no
/// offset. Those are read back as local time, which is what they were, so old journals keep
/// reporting the same wall-clock values.
pub fn journal_datetime_to_naive_local(value: &str) -> Result<String> {
    match DateTime::parse_from_rfc3339(value) {
        Ok(aware) => Ok(format_naive(aware.with_timezone(&Local).naive_local())),
        Err(_) => Ok(format_naive(parse_naive(value)?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn naive_local_and_naive_utc_round_trip() -> Result<()> {
        let local = "2024-01-02 03:04:05.678";
        let utc = naive_local_to_naive_utc(local)?;
        assert_eq!(naive_utc_to_naive_local(&utc)?, local);
        Ok(())
    }

    #[test]
    fn naive_utc_conversion_applies_the_local_offset() -> Result<()> {
        let local = "2024-01-02 03:04:05.678";
        let utc = naive_local_to_naive_utc(local)?;
        // The stored value differs from the local one by exactly the local offset.
        let offset = Local
            .from_local_datetime(&parse_naive(local)?)
            .earliest()
            .expect("unambiguous local time")
            .offset()
            .local_minus_utc();
        assert_eq!(
            parse_naive(&utc)?,
            parse_naive(local)? - chrono::Duration::seconds(offset as i64)
        );
        Ok(())
    }

    #[test]
    fn aware_utc_round_trips_and_carries_an_offset() -> Result<()> {
        let local = "2024-06-02 03:04:05.678";
        let aware = naive_local_to_aware_utc(local)?;
        assert!(
            aware.ends_with("+00:00"),
            "journal datetimes must be aware UTC: {aware}"
        );
        assert_eq!(journal_datetime_to_naive_local(&aware)?, local);
        Ok(())
    }

    #[test]
    fn journal_datetimes_written_by_optuna_are_accepted() -> Result<()> {
        // datetime.now(tz=timezone.utc).isoformat(timespec="microseconds")
        let value = "2024-01-02T03:04:05.678000+00:00";
        let local = journal_datetime_to_naive_local(value)?;
        assert_eq!(
            parse_naive(&local)?,
            DateTime::parse_from_rfc3339(value)
                .expect("valid rfc3339")
                .with_timezone(&Local)
                .naive_local()
        );
        Ok(())
    }

    #[test]
    fn journal_datetimes_from_older_logs_are_read_as_local_time() -> Result<()> {
        // Logs written before the move to aware UTC carry naive local time.
        assert_eq!(
            journal_datetime_to_naive_local("2024-01-02 03:04:05.678")?,
            "2024-01-02 03:04:05.678"
        );
        Ok(())
    }

    #[test]
    fn seconds_only_datetimes_are_accepted() -> Result<()> {
        assert_eq!(
            naive_utc_to_naive_local("2024-01-02 03:04:05")?,
            naive_utc_to_naive_local("2024-01-02 03:04:05.000")?
        );
        Ok(())
    }

    #[test]
    fn invalid_datetimes_are_rejected() {
        assert!(naive_utc_to_naive_local("not-a-datetime").is_err());
        assert!(naive_local_to_aware_utc("").is_err());
    }
}
