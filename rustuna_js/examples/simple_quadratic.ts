import * as rustuna from "../pkg/node/rustuna.js";

const study = rustuna.create_study("test");

const objective = (trial: rustuna.Trial) => {
	const x = trial.suggest_float("x", -10.0, 10.0);
	const y = trial.suggest_int("y", -10, 10);
	const z = trial.suggest_categorical("z", ["foo", "bar"]);

	const value = (x - 5) ** 2 + (y + 2) ** 2;
	console.log(`x: ${x}, y: ${y}, z: ${z}, value: ${value}`);
	return value;
};
study.optimize(objective, 10);

console.log(study.best_trial);
