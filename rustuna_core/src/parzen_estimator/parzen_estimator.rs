use rand::rngs::StdRng;
use std::{collections::HashMap, vec};

use super::probability_distributions::CategoricalDistributions;
use super::probability_distributions::{
    DiscreteTruncLogNormDistributions, DiscreteTruncNormDistributions, Distributions,
    MixtureOfProductDistribution, TruncLogNormDistributions, TruncNormDistributions,
};
use crate::distribution::Distribution;

pub struct ParzenEstimator {
    mixuture_distribution: MixtureOfProductDistribution,
}

impl ParzenEstimator {
    pub fn new(
        observations: HashMap<String, Vec<f64>>,
        search_space: HashMap<String, Distribution>,
        weights: Vec<f64>,
        prior_weight: f64,
    ) -> Self {
        let n_observations = match observations.values().next() {
            None => 0,
            Some(first) => {
                let first_len = first.len();
                assert!(
                    observations.values().all(|v| v.len() == first_len),
                    "Parameter observations have inconsistent lengths"
                );
                first_len
            }
        };
        assert!(
            n_observations == weights.len(),
            "Number of observations and length of weights must be equal"
        );

        let mut distributions = HashMap::<String, Distributions>::new();
        for param_name in search_space.keys() {
            distributions.insert(
                param_name.clone(),
                Self::calculate_distribution(&observations[param_name], &search_space[param_name]),
            );
        }
        let weights_with_prior_weight = {
            let mut w = weights.clone();
            w.push(prior_weight);
            w.iter().map(|&x| x / w.iter().sum::<f64>()).collect()
        };

        ParzenEstimator {
            mixuture_distribution: MixtureOfProductDistribution::new(
                distributions,
                weights_with_prior_weight,
            ),
        }
    }

    fn calculate_distribution(
        observations: &Vec<f64>,
        search_space: &Distribution,
    ) -> Distributions {
        match search_space {
            Distribution::Float { .. } | Distribution::Int { .. } => {
                Self::calculate_numerical_distribution(observations, search_space)
            }
            Distribution::Categorical { .. } => {
                Self::calculate_categorical_distribution(observations, search_space)
            }
        }
    }

    fn calculate_numerical_distribution(
        observations: &Vec<f64>,
        search_space: &Distribution,
    ) -> Distributions {
        // Currently, we assume consider_prior=True, consider_endpoints=True, and consider_magic_clip=True.
        let (low, high, step, log) = match search_space {
            Distribution::Float {
                low,
                high,
                step,
                log,
            } => (*low, *high, *step, *log),
            Distribution::Int {
                low,
                high,
                step,
                log,
            } => (*low as f64, *high as f64, Some(*step as f64), *log),
            _ => unreachable!("Invalid distribution type for numerical calculation"),
        };

        // Handle step
        let (adjusted_low, adjusted_high) = match step {
            Some(s) => (low - s / 2.0, high + s / 2.0),
            None => (low, high),
        };

        // Handle log scale
        let (adjusted_low, adjusted_high) = if log {
            (adjusted_low.ln(), adjusted_high.ln())
        } else {
            (adjusted_low, adjusted_high)
        };
        let mus: Vec<f64> = observations
            .iter()
            .map(|&m| if log { m.ln() } else { m })
            .chain(std::iter::once((adjusted_low + adjusted_high) / 2.0)) // Add prior
            .collect();

        let sigmas = {
            let mus = &mus[0..mus.len() - 1]; // Exclude prior for sigma calculation
            let sorted_indices = {
                let mut indices: Vec<usize> = (0..mus.len()).collect();
                indices.sort_by(|&i, &j| mus[i].partial_cmp(&mus[j]).unwrap());
                indices
            };
            let sorted_mus = sorted_indices.iter().map(|&i| mus[i]).collect::<Vec<f64>>();
            let sorted_mus_with_endpoints: Vec<f64> = {
                let mut v = vec![adjusted_low];
                v.extend(sorted_mus.iter());
                v.push(adjusted_high);
                v
            };
            let mut sorted_sigmas = Vec::<f64>::new();
            for i in 1..(sorted_mus_with_endpoints.len() - 1) {
                let left_diff = sorted_mus_with_endpoints[i] - sorted_mus_with_endpoints[i - 1];
                let right_diff = sorted_mus_with_endpoints[i + 1] - sorted_mus_with_endpoints[i];
                sorted_sigmas.push(left_diff.max(right_diff));
            }
            // Reorder sigmas to match original mus order
            let mut sigmas = vec![0.0; sorted_sigmas.len()];
            for (i, &sorted_idx) in sorted_indices.iter().enumerate() {
                sigmas[sorted_idx] = sorted_sigmas[i];
            }
            sigmas.push(adjusted_high - adjusted_low); // Sigma for prior

            let maxsigma = adjusted_high - adjusted_low;
            let minsigma =
                (adjusted_high - adjusted_low) / (100.0f64.min(1.0 + sigmas.len() as f64));
            sigmas
                .iter()
                .map(|&s| s.clamp(minsigma, maxsigma))
                .collect()
        };

        match (step, log) {
            (None, false) => Distributions::TruncNorm(TruncNormDistributions {
                mus,
                sigmas,
                low,
                high,
            }),
            (None, true) => Distributions::TruncLogNorm(TruncLogNormDistributions {
                mus,
                sigmas,
                low,
                high,
            }),
            (Some(value), false) => {
                Distributions::DiscreteTruncNorm(DiscreteTruncNormDistributions {
                    mus,
                    sigmas,
                    low,
                    high,
                    step: value,
                })
            }
            (Some(value), true) => {
                Distributions::DiscreteTruncLogNorm(DiscreteTruncLogNormDistributions {
                    mus,
                    sigmas,
                    low,
                    high,
                    step: value,
                })
            }
        }
    }

    fn calculate_categorical_distribution(
        observations: &Vec<f64>,
        search_space: &Distribution,
    ) -> Distributions {
        let cardinality = match search_space {
            Distribution::Categorical { cardinality } => *cardinality,
            _ => unreachable!("Invalid distribution type for categorical calculation"),
        };

        if observations.is_empty() {
            // Return uniform distribution if there is no observation
            let weights: Vec<Vec<f64>> = vec![vec![1.0 / cardinality as f64; cardinality]];
            return Distributions::Categorical(CategoricalDistributions { weights });
        }

        let n_kernels = observations.len() + 1; // +1 for prior
        let prior_mass_per_kernel = 1.0 / (n_kernels as f64);
        let mut weights: Vec<Vec<f64>> = vec![vec![prior_mass_per_kernel; cardinality]; n_kernels];
        let observed_indices: Vec<usize> = observations.iter().map(|&v| v as usize).collect();
        for (i, &col) in observed_indices.iter().enumerate() {
            assert!(
                col < cardinality,
                "Observed index {} out of range (cardinality = {})",
                col,
                cardinality
            );
            weights[i][col] += 1.0;
        }
        let row_sums: Vec<f64> = weights.iter().map(|row| row.iter().sum::<f64>()).collect();
        for i in 0..weights.len() {
            let denom = if row_sums[i] == 0.0 { 1.0 } else { row_sums[i] };
            for j in 0..weights[i].len() {
                weights[i][j] /= denom;
            }
        }
        Distributions::Categorical(CategoricalDistributions { weights })
    }

    pub fn sample(&self, rng: &mut StdRng, size: usize) -> Vec<HashMap<String, f64>> {
        self.mixuture_distribution.sample(rng, size)
    }

    pub fn log_pdf(&self, x: &HashMap<String, f64>) -> f64 {
        self.mixuture_distribution.log_pdf(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn build_parzen_estimator() {
        let mut observations = HashMap::<String, Vec<f64>>::new();
        observations.insert("a".to_string(), vec![0.1, 0.4, 0.35]);
        observations.insert("c".to_string(), vec![0.05, 0.3, 0.7]);
        observations.insert("b".to_string(), vec![1.0, 3.0, 4.0]);
        observations.insert("d".to_string(), vec![2.0, 4.0, 5.0]);
        observations.insert("e".to_string(), vec![0.0, 1.0, 2.0]);

        let mut search_space = HashMap::<String, Distribution>::new();
        search_space.insert(
            "a".to_string(),
            Distribution::Float {
                low: 0.01,
                high: 1.0,
                step: None,
                log: false,
            },
        );
        search_space.insert(
            "b".to_string(),
            Distribution::Float {
                low: 0.01,
                high: 1.0,
                step: None,
                log: true,
            },
        );
        search_space.insert(
            "c".to_string(),
            Distribution::Int {
                low: 1,
                high: 5,
                step: 1,
                log: false,
            },
        );
        search_space.insert(
            "d".to_string(),
            Distribution::Int {
                low: 1,
                high: 5,
                step: 1,
                log: true,
            },
        );
        search_space.insert(
            "e".to_string(),
            Distribution::Categorical { cardinality: 3 },
        );

        let parzen_estimator =
            ParzenEstimator::new(observations, search_space, vec![0.2, 0.5, 0.3], 1.0);
        let mut rng = StdRng::seed_from_u64(42);
        let samples = parzen_estimator.sample(&mut rng, 10);
        assert_eq!(samples.len(), 10);
        for sample in samples.iter() {
            let a = sample.get("a").unwrap();
            assert!(*a >= 0.01 && *a <= 1.0);
            let b = sample.get("b").unwrap();
            assert!(*b >= 0.01 && *b <= 1.0);
            let c = sample.get("c").unwrap();
            assert!(*c == 1.0 || *c == 2.0 || *c == 3.0 || *c == 4.0 || *c == 5.0);
            let d = sample.get("d").unwrap();
            assert!(*d == 1.0 || *d == 2.0 || *d == 3.0 || *d == 4.0 || *d == 5.0);
            let e = sample.get("e").unwrap();
            assert!(*e == 0.0 || *e == 1.0 || *e == 2.0);
        }
    }
}
