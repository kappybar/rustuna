use rustuna_core::trial::PersistedTrial;
use rustuna_core::distribution::Distribution;
pub struct PedAnovaImportanceEvaluator {
    target_quantile: f64,
    region_quantile: f64,
    evaluate_on_local: bool,
    n_steps: usize,
    prior_weight: f64,
    min_n_top_trials: usize,
}



fn count_numerical_param_in_grid(
    param_name: &str,
    dist: &Distribution,
    trials: &[PersistedTrial],
    n_steps: usize,
) -> Vec<u32> {
    let (low, high, log, n_steps) = match dist {
        Distribution::Int {
            low,
            high,
            step,
            log,
        } => {
            let n_steps = if *log {
                let log2_domain_size = ((high - low + 1) as f64).log2().ceil() as usize + 1;
                n_steps.min(log2_domain_size)
            } else {
                n_steps.min(((high - low) / step + 1) as usize)
            };
            (*low as f64, *high as f64, *log, n_steps)
        },
        Distribution::Float {
            low,
            high,
            step,
            log,
        } => {
            let n_steps = if let Some(step) = step {
                n_steps.min(((high - low) / step).round() as usize + 1)
            } else {
                n_steps
            };
            (*low, *high, *log, n_steps)
        },
        _ => unreachable!("Invalid distribution type for numerical calculation"),
    };
    let (low, high) = if log {
        (low.ln(), high.ln())
    } else {
        (low, high)
    };
    let param_values = trials.iter().map(|t| {
        let v = t.internal_params[param_name];
        if log { v.ln() } else { v }
    });
    let step_size = (high - low) / (n_steps as f64);
    let mut counts = vec![0u32; n_steps];
    for v in param_values {
        let idx = ((v - low) / step_size).floor().max(0.0).min((n_steps - 1) as f64) as usize;
        counts[idx] += 1;
    }
    counts
}




fn count_categorical_param_in_grid(
    param_name: &str,
    dist: &Distribution,
    trials: &[PersistedTrial],
) -> Vec<u32> {
    let cardinality = match dist {
        Distribution::Categorical { cardinality } => *cardinality,
        _ => unreachable!("Invalid distribution type for categorical calculation"),
    };
    let mut counts = vec![0u32; cardinality];
    for t in trials {
        let v = t.internal_params[param_name];
        counts[v as usize] += 1;
    }
    counts
}