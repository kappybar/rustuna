use rustuna_core::trial::PersistedTrial;
use rustuna_core::distribution::Distribution;


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