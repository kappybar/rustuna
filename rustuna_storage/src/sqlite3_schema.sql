CREATE TABLE IF NOT EXISTS studies (
	study_id INTEGER NOT NULL,
	study_name VARCHAR(512) NOT NULL,
	PRIMARY KEY (study_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_studies_study_name ON studies (study_name);
CREATE TABLE IF NOT EXISTS version_info (
	version_info_id INTEGER NOT NULL,
	schema_version INTEGER,
	library_version VARCHAR(256),
	PRIMARY KEY (version_info_id),
	CHECK (version_info_id=1)
);
CREATE TABLE IF NOT EXISTS study_directions (
	study_direction_id INTEGER NOT NULL,
	direction VARCHAR(8) NOT NULL,
	study_id INTEGER NOT NULL,
	objective INTEGER NOT NULL,
	PRIMARY KEY (study_direction_id),
	UNIQUE (study_id, objective),
	FOREIGN KEY(study_id) REFERENCES studies (study_id)
);
CREATE TABLE IF NOT EXISTS trials (
	trial_id INTEGER NOT NULL,
	number INTEGER,
	study_id INTEGER,
	state VARCHAR(8) NOT NULL,
	datetime_start DATETIME,
	datetime_complete DATETIME,
	-- Rustuna-specific column; it is not part of Optuna's SQLite schema.
	-- NULL means the trial is live. It hides the trial from reads and doubles as the
	-- synchronization cursor other processes use to pick up newly discarded trials.
	-- Stored as naive UTC, unlike datetime_start and datetime_complete above.
	discarded_at DATETIME,
	PRIMARY KEY (trial_id),
	FOREIGN KEY(study_id) REFERENCES studies (study_id)
);
CREATE TABLE IF NOT EXISTS alembic_version (
	version_num VARCHAR(32) NOT NULL,
	CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);
CREATE TABLE IF NOT EXISTS trial_heartbeats (
	trial_heartbeat_id INTEGER NOT NULL,
	trial_id INTEGER NOT NULL,
	heartbeat DATETIME NOT NULL,
	PRIMARY KEY (trial_heartbeat_id),
	UNIQUE (trial_id),
	FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
);
CREATE TABLE IF NOT EXISTS "study_user_attributes" (
	study_user_attribute_id INTEGER NOT NULL,
	study_id INTEGER,
	"key" VARCHAR(512),
	value_json TEXT,
	PRIMARY KEY (study_user_attribute_id),
	FOREIGN KEY(study_id) REFERENCES studies (study_id),
	UNIQUE (study_id, "key")
);
CREATE TABLE IF NOT EXISTS "study_system_attributes" (
	study_system_attribute_id INTEGER NOT NULL,
	study_id INTEGER,
	"key" VARCHAR(512),
	value_json TEXT,
	PRIMARY KEY (study_system_attribute_id),
	UNIQUE (study_id, "key"),
	FOREIGN KEY(study_id) REFERENCES studies (study_id)
);
CREATE TABLE IF NOT EXISTS "trial_user_attributes" (
	trial_user_attribute_id INTEGER NOT NULL,
	trial_id INTEGER,
	"key" VARCHAR(512),
	value_json TEXT,
	PRIMARY KEY (trial_user_attribute_id),
	UNIQUE (trial_id, "key"),
	FOREIGN KEY(trial_id) REFERENCES trials (trial_id)
);
CREATE TABLE IF NOT EXISTS "trial_system_attributes" (
	trial_system_attribute_id INTEGER NOT NULL,
	trial_id INTEGER,
	"key" VARCHAR(512),
	value_json TEXT,
	PRIMARY KEY (trial_system_attribute_id),
	FOREIGN KEY(trial_id) REFERENCES trials (trial_id),
	UNIQUE (trial_id, "key")
);
CREATE TABLE IF NOT EXISTS "trial_params" (
	param_id INTEGER NOT NULL,
	trial_id INTEGER,
	param_name VARCHAR(512),
	param_value FLOAT,
	distribution_json TEXT,
	PRIMARY KEY (param_id),
	FOREIGN KEY(trial_id) REFERENCES trials (trial_id),
	UNIQUE (trial_id, param_name)
);
CREATE TABLE IF NOT EXISTS "trial_intermediate_values" (
	trial_intermediate_value_id INTEGER NOT NULL,
	trial_id INTEGER NOT NULL,
	step INTEGER NOT NULL,
	intermediate_value FLOAT,
	intermediate_value_type VARCHAR(7) NOT NULL,
	PRIMARY KEY (trial_intermediate_value_id),
	FOREIGN KEY(trial_id) REFERENCES trials (trial_id),
	UNIQUE (trial_id, step)
);
CREATE TABLE IF NOT EXISTS "trial_values" (
	trial_value_id INTEGER NOT NULL,
	trial_id INTEGER NOT NULL,
	objective INTEGER NOT NULL,
	value FLOAT,
	value_type VARCHAR(7) NOT NULL,
	PRIMARY KEY (trial_value_id),
	FOREIGN KEY(trial_id) REFERENCES trials (trial_id),
	UNIQUE (trial_id, objective)
);
CREATE INDEX IF NOT EXISTS trials_study_id_key ON trials (study_id);
-- This composite index is not included in Optuna's SQLite schema. Rustuna adds it
-- to efficiently refresh trials by study and trial number.
CREATE INDEX IF NOT EXISTS trials_study_id_number_key ON trials (study_id, number);
-- This index is not included in Optuna's SQLite schema either. Rustuna adds it so that
-- picking up trials discarded by other processes costs one range scan over the new entries
-- instead of a scan over every discarded trial.
CREATE INDEX IF NOT EXISTS trials_study_id_discarded_at_key ON trials (study_id, discarded_at);
INSERT OR IGNORE INTO version_info (version_info_id, schema_version, library_version) VALUES (1, 12, '4.6.0.dev')
