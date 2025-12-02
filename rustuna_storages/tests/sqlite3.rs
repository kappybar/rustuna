use std::path::PathBuf;
use std::process::Command;

use rustuna_core::storage_cache::CachedStorageBackend;
use rustuna_core::trial::TrialStateValues;
use rustuna_storages::sqlite3::SQLite3Storage;

fn run_optuna_script(python: &str, db_path: &PathBuf, script: &str) -> bool {
    Command::new(python)
        .args(["-c", script, db_path.to_string_lossy().as_ref()])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[test]
fn load_studies_from_optuna_sqlite() {
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("optuna.sqlite3");
    let script = r#"
import optuna, sys

storage = "sqlite:///" + sys.argv[1]
optuna.create_study(storage=storage, study_name="test-0")
optuna.create_study(storage=storage, study_name="test-1", directions=["maximize", "minimize"])
"#;
    assert!(run_optuna_script(&python, &db_path, script));

    let mut storage = SQLite3Storage::new(db_path.to_string_lossy().as_ref()).unwrap();
    let studies = storage.get_studies().unwrap();
    assert_eq!(studies.len(), 2);
    assert_eq!(studies[0].name, "test-0");
    assert_eq!(studies[1].name, "test-1");

    assert_eq!(studies[0].directions.len(), 1);
    assert_eq!(
        studies[0].directions[0],
        rustuna_core::study::Direction::Minimize
    );

    assert_eq!(studies[1].directions.len(), 2);
    assert_eq!(
        studies[1].directions[0],
        rustuna_core::study::Direction::Maximize
    );
    assert_eq!(
        studies[1].directions[1],
        rustuna_core::study::Direction::Minimize
    );
}
