use rand::rngs::StdRng;
use std::cmp::Ordering;
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
        observations: &HashMap<String, Vec<f64>>,
        search_space: &HashMap<String, Distribution>,
        weights: &[f64],
        prior_weight: f64,
    ) -> Self {
        let n_observations = observations.values().next().map_or(0, |v| {
            assert!(
                observations.values().all(|w| w.len() == v.len()),
                "Observations have inconsistent lengths"
            );
            v.len()
        });
        assert_eq!(
            n_observations, weights.len(),
            "Number of observations and length of weights must be equal"
        );

        let mut keys: Vec<_> = search_space.keys().collect();
        keys.sort();

        let mut distributions = HashMap::with_capacity(keys.len());
        for key in keys.iter() {
            let obs_vec = observations.get(*key).map(Vec::as_slice).unwrap_or(&[]);
            let dist = Self::calculate_distribution(obs_vec, &search_space[*key]);
            distributions.insert((*key).clone(), dist);
        }

        let weights_sum = {
            let s = weights.iter().sum::<f64>() + prior_weight;
            if s == 0.0 { (weights.len() + 1) as f64 } else { s }
        };

        let weights_with_prior_weight= weights
            .iter()
            .chain(std::iter::once(&prior_weight))
            .map(|w| w / weights_sum)
            .collect();

        Self {
            mixuture_distribution: MixtureOfProductDistribution::new(
                distributions,
                weights_with_prior_weight,
            ),
        }
    }

    fn calculate_distribution(observations: &[f64], search_space: &Distribution) -> Distributions {
        match search_space {
            Distribution::Float { .. } | Distribution::Int { .. } => {
                Self::calculate_numerical_distribution(observations, search_space)
            }
            Distribution::Categorical { .. } => {
                Self::calculate_categorical_distribution(observations, search_space)
            }
        }
    }

    pub fn sample(&self, rng: &mut StdRng, size: usize) -> Vec<HashMap<String, f64>> {
        self.mixuture_distribution.sample(rng, size)
    }

    pub fn log_pdf(&self, x: &HashMap<String, f64>) -> f64 {
        self.mixuture_distribution.log_pdf(x)
    }
}

pub trait NumericalDistributionBuilder {
    fn calculate_numerical_distribution(
        &self,
        observations: &[f64],
        search_space: &Distribution,
    ) -> Distributions;
}

pub trait CategoricalDistributionBuilder {
    fn calculate_categorical_distribution(
        &self,
        observations: &[f64],
        search_space: &Distribution,
    ) -> Distributions;
}

pub struct DefaultParazenDistributionBuilder;

impl ParzenDistributionBuilder for DefaultParazenDistributionBuilder {
    fn calculate_numerical_distribution(
        &self,
        observations: &[f64],
        search_space: &Distribution,
    ) -> Distributions {
        // Currently, we assume consider_prior=True, consider_endpoints=True, and consider_magic_clip=True.
        let (low, high, step_opt, log) = match search_space {
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

        let (mut adj_low, mut adj_high) = match step_opt {
            Some(s) => (low - s / 2.0, high + s / 2.0),
            None => (low, high),
        };

        if log {
            adj_low = adj_low.ln();
            adj_high = adj_high.ln();
        }

        let mus = observations
            .iter()
            .map(|&m| if log { m.ln() } else { m })
            .chain(std::iter::once((adj_low + adj_high) / 2.0)) // Add prior
            .collect::<Vec<_>>();

        let mut sigmas = Vec::with_capacity(mus.len());
        if mus.len() == 1 {
            // Case: prior only
            sigmas.push(adj_high - adj_low);
        } else {
            let m = mus.len() - 1; // exclude prior
            let mut idx_vals: Vec<(usize, f64)> = (0..m).map(|i| (i, mus[i])).collect();
            idx_vals.sort_by(|a, b| a.1.total_cmp(&b.1));
            let sorted_mus = idx_vals.iter().map(|&(_, v)| v);
            
            let extended = std::iter::once(adj_low)
                .chain(sorted_mus)
                .chain(std::iter::once(adj_high))
                .collect::<Vec<_>>();

            let sorted_sigmas = (1..(extended.len() - 1))
                .map(|i| {
                    let left_diff = extended[i] - extended[i - 1];
                    let right_diff = extended[i + 1] - extended[i];
                    left_diff.max(right_diff)
                })
                .collect::<Vec<_>>();

            sigmas.resize(m, 0.0);
            for (&(orig_idx, _), &sigma) in idx_vals.iter().zip(sorted_sigmas.iter()) {
                sigmas[orig_idx] = sigma;
            }
            // Sigma for prior
            sigmas.push(adj_high - adj_low);

            // Clamp (minsigma, maxsigma)
            let maxsigma = adj_high - adj_low;
            let minsigma = (adj_high - adj_low) / (100.0_f64.min(1.0 + sigmas.len() as f64));
            for s in sigmas.iter_mut() {
                *s = s.clamp(minsigma, maxsigma);
            }
        }

        match (step_opt, log) {
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
            (Some(step), false) => {
                Distributions::DiscreteTruncNorm(DiscreteTruncNormDistributions {
                    mus,
                    sigmas,
                    low,
                    high,
                    step,
                })
            }
            (Some(step), true) => {
                Distributions::DiscreteTruncLogNorm(DiscreteTruncLogNormDistributions {
                    mus,
                    sigmas,
                    low,
                    high,
                    step,
                })
            }
        }
    }

    fn calculate_categorical_distribution(
        &self,
        observations: &[f64],
        search_space: &Distribution,
    ) -> Distributions {
        let cardinality = match search_space {
            Distribution::Categorical { cardinality } => *cardinality,
            _ => unreachable!("Invalid distribution type for categorical calculation"),
        };

        if observations.is_empty() {
            // Case: prior only
            let weights_row = vec![1.0 / cardinality as f64; cardinality];
            return Distributions::Categorical(CategoricalDistributions {
                weights: vec![weights_row],
            });
        }

        let n_kernels = observations.len() + 1; // +1 for prior
        let prior_mass_per_kernel = 1.0 / (n_kernels as f64);
        let mut weights: Vec<Vec<f64>> = vec![vec![prior_mass_per_kernel; cardinality]; n_kernels];
        for (i, &v) in observations.iter().enumerate() {
            let col = v as usize;
            assert!(
                col < cardinality,
                "Observed index {} out of range (cardinality = {})",
                col,
                cardinality
            );
            weights[i][col] += 1.0;
        }
        for row in weights.iter_mut() {
            let s = row.iter().sum::<f64>();
            let denom = if s == 0.0 { 1.0 } else { s };
            for x in row.iter_mut() {
                *x /= denom;
            }
        }

        Distributions::Categorical(CategoricalDistributions { weights })
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
            ParzenEstimator::new(&observations, &search_space, &vec![0.2, 0.5, 0.3], 1.0);
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
