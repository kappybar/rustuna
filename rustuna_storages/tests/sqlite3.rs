use std::path::PathBuf;
use std::process::Command;

use rustuna_core::distribution::Distribution;
use rustuna_core::storage::Storage;
use rustuna_core::storage_cache::{CachedStorage, CachedStorageBackend};
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

#[test]
fn load_trial() {
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("optuna.sqlite3");
    let script = r#"
import optuna, sys

def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", 1, 10, log=True)
    y = trial.suggest_int("y", -10, 10)
    trial.suggest_categorical("z", [True, False, "foo", 10])
    return x ** 2 + y

storage = "sqlite:///" + sys.argv[1]
study = optuna.create_study(storage=storage)
study.optimize(objective, n_trials=10)
"#;

    assert!(run_optuna_script(&python, &db_path, script));

    let mut storage = SQLite3Storage::new(db_path.to_string_lossy().as_ref()).unwrap();
    let studies = storage.get_studies().unwrap();
    assert_eq!(studies.len(), 1);

    let trial0 = storage.get_trial(studies[0].id, 0).unwrap();
    assert_eq!(trial0.number, 0);

    // Distributions
    assert_eq!(trial0.distributions.len(), 3);
    assert_eq!(
        trial0.distributions["x"],
        rustuna_core::distribution::Distribution::Float {
            low: 1.0,
            high: 10.0,
            log: true,
            step: None
        }
    );
    assert_eq!(
        trial0.distributions["y"],
        rustuna_core::distribution::Distribution::Int {
            low: -10,
            high: 10,
            log: false,
            step: Some(1)
        }
    );
    assert_eq!(
        trial0.distributions["z"],
        rustuna_core::distribution::Distribution::Categorical { cardinality: 4 }
    );

    // Objective value
    assert!(matches!(trial0.state_values, TrialStateValues::Complete(_)));
}

#[test]
fn get_trials() {
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("optuna.sqlite3");
    let script = r#"
import optuna, sys

def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", 1, 10, log=True)
    y = trial.suggest_int("y", -10, 10)
    trial.suggest_categorical("z", [True, False, "foo", 10])
    trial.set_user_attr("key", "value")
    return x ** 2 + y

storage = "sqlite:///" + sys.argv[1]
study = optuna.create_study(storage=storage, study_name="foo", load_if_exists=True)
study.optimize(objective, n_trials=10)
"#;
    // Evaluate 10 trials
    assert!(run_optuna_script(&python, &db_path, script));
    let mut storage = CachedStorage::new(Box::new(
        SQLite3Storage::new(db_path.to_string_lossy().as_ref()).unwrap(),
    ));
    let study_id = {
        let studies = storage.get_studies().unwrap();
        assert_eq!(studies.len(), 1);
        studies[0].id
    };
    let trials = storage.get_trials(study_id).unwrap();
    assert_eq!(trials.len(), 10);

    // Evaluate more 10 trials
    assert!(run_optuna_script(&python, &db_path, script));
    let trials = storage.get_trials(study_id).unwrap();
    assert_eq!(trials.len(), 20);
    assert_eq!(trials[0].distributions.len(), 3);
    assert_eq!(
        trials[0].distributions["x"],
        Distribution::Float {
            low: 1.0,
            high: 10.0,
            step: None,
            log: true
        }
    );
    assert_eq!(
        trials[0].distributions["y"],
        Distribution::Int {
            low: -10,
            high: 10,
            step: Some(1),
            log: false
        }
    );
    assert_eq!(
        trials[0].distributions["z"],
        Distribution::Categorical { cardinality: 4 }
    );
    assert_eq!(trials[0].internal_params.len(), 3);
    assert_eq!(trials[0].attrs.len(), 1);
    assert!(matches!(
        trials[0].state_values,
        TrialStateValues::Complete(_)
    ));
}
