use super::truncnorm;
use rand::rngs::StdRng;
use rand::Rng;
use rand_distr::{Distribution as RandDistribution, WeightedAliasIndex};
use std::collections::HashMap;

#[derive(Debug)]
pub(crate) struct TruncNormDistributions {
    pub mus: Vec<f64>,
    pub sigmas: Vec<f64>,
    pub low: f64,
    pub high: f64,
}

#[derive(Debug)]
pub(crate) struct TruncLogNormDistributions {
    pub mus: Vec<f64>,
    pub sigmas: Vec<f64>,
    pub low: f64,
    pub high: f64,
}

#[derive(Debug)]
pub(crate) struct DiscreteTruncNormDistributions {
    pub mus: Vec<f64>,
    pub sigmas: Vec<f64>,
    pub low: f64,
    pub high: f64,
    pub step: f64,
}

#[derive(Debug)]
pub(crate) struct DiscreteTruncLogNormDistributions {
    pub mus: Vec<f64>,
    pub sigmas: Vec<f64>,
    pub low: f64,
    pub high: f64,
    pub step: f64,
}

#[derive(Debug)]
pub(crate) struct CategoricalDistributions {
    pub weights: Vec<Vec<f64>>,
}

#[derive(Debug)]
pub(crate) enum Distributions {
    TruncNorm(TruncNormDistributions),
    TruncLogNorm(TruncLogNormDistributions),
    DiscreteTruncNorm(DiscreteTruncNormDistributions),
    DiscreteTruncLogNorm(DiscreteTruncLogNormDistributions),
    Categorical(CategoricalDistributions),
}
impl Clone for Distributions {
    fn clone(&self) -> Self {
        match self {
            Distributions::TruncNorm(d) => Distributions::TruncNorm(TruncNormDistributions {
                mus: d.mus.clone(),
                sigmas: d.sigmas.clone(),
                low: d.low,
                high: d.high,
            }),
            Distributions::TruncLogNorm(d) => {
                Distributions::TruncLogNorm(TruncLogNormDistributions {
                    mus: d.mus.clone(),
                    sigmas: d.sigmas.clone(),
                    low: d.low,
                    high: d.high,
                })
            }
            Distributions::DiscreteTruncNorm(d) => {
                Distributions::DiscreteTruncNorm(DiscreteTruncNormDistributions {
                    mus: d.mus.clone(),
                    sigmas: d.sigmas.clone(),
                    low: d.low,
                    high: d.high,
                    step: d.step,
                })
            }
            Distributions::DiscreteTruncLogNorm(d) => {
                Distributions::DiscreteTruncLogNorm(DiscreteTruncLogNormDistributions {
                    mus: d.mus.clone(),
                    sigmas: d.sigmas.clone(),
                    low: d.low,
                    high: d.high,
                    step: d.step,
                })
            }
            Distributions::Categorical(d) => Distributions::Categorical(CategoricalDistributions {
                weights: d.weights.clone(),
            }),
        }
    }
}

pub(crate) struct MixtureOfProductDistribution {
    pub distributions: HashMap<String, Distributions>,
    pub weights: Vec<f64>,
    log_sum_weights: f64,
}
impl MixtureOfProductDistribution {
    pub fn new(distributions: HashMap<String, Distributions>, weights: Vec<f64>) -> Self {
        let log_sum_weights = weights.iter().sum::<f64>().ln();
        MixtureOfProductDistribution {
            distributions,
            weights,
            log_sum_weights: log_sum_weights,
        }
    }

    pub fn sample(&self, rng: &mut StdRng, size: usize) -> Vec<HashMap<String, f64>> {
        let indices_distribution = WeightedAliasIndex::new(self.weights.clone()).unwrap();
        let active_indices: Vec<usize> = (0..size)
            .map(|_| indices_distribution.sample(rng))
            .collect();

        let mut samples: Vec<HashMap<String, f64>> = vec![];
        let mut sorted_params: Vec<_> = self.distributions.keys().collect();
        sorted_params.sort();
        for i in active_indices.iter() {
            let mut sample: HashMap<String, f64> = HashMap::new();
            for param in sorted_params.iter() {
                let distribution = &self.distributions[*param];
                match distribution {
                    Distributions::TruncNorm(d) => {
                        let value = truncnorm::rvs(
                            rng,
                            (d.low - d.mus[*i]) / d.sigmas[*i],
                            (d.high - d.mus[*i]) / d.sigmas[*i],
                            d.mus[*i],
                            d.sigmas[*i],
                        )
                        .unwrap();
                        sample.insert((*param).clone(), value);
                    }
                    Distributions::TruncLogNorm(d) => {
                        let log_scale_value = truncnorm::rvs(
                            rng,
                            (d.low.ln() - d.mus[*i]) / d.sigmas[*i],
                            (d.high.ln() - d.mus[*i]) / d.sigmas[*i],
                            d.mus[*i],
                            d.sigmas[*i],
                        )
                        .unwrap();
                        sample.insert((*param).clone(), log_scale_value.exp());
                    }
                    Distributions::DiscreteTruncNorm(d) => {
                        let value = truncnorm::rvs(
                            rng,
                            (d.low - d.step / 2.0 - d.mus[*i]) / d.sigmas[*i],
                            (d.high + d.step / 2.0 - d.mus[*i]) / d.sigmas[*i],
                            d.mus[*i],
                            d.sigmas[*i],
                        )
                        .unwrap();
                        let discrete_value = (d.low + ((value - d.low) / d.step).round() * d.step)
                            .max(d.low)
                            .min(d.high);
                        sample.insert((*param).clone(), discrete_value);
                    }
                    Distributions::DiscreteTruncLogNorm(d) => {
                        let log_scale_value = truncnorm::rvs(
                            rng,
                            ((d.low - d.step / 2.0).max(f64::MIN_POSITIVE).ln() - d.mus[*i])
                                / d.sigmas[*i],
                            ((d.high + d.step / 2.0).max(f64::MIN_POSITIVE).ln() - d.mus[*i])
                                / d.sigmas[*i],
                            d.mus[*i],
                            d.sigmas[*i],
                        )
                        .unwrap();
                        let original_scale_value = log_scale_value.exp();
                        let discrete_value = (d.low
                            + ((original_scale_value - d.low) / d.step).round() * d.step)
                            .max(d.low)
                            .min(d.high);
                        sample.insert((*param).clone(), discrete_value);
                    }
                    Distributions::Categorical(d) => {
                        let probs = &d.weights[*i];

                        let sum: f64 = probs.iter().sum();
                        assert!(sum >= 0.0, "Categorical distribution has negative total probability for parameter {}", param);

                        let mut cum = 0.0;
                        let u: f64 = rng.gen::<f64>() * sum; // No need to normalize, multiply by sum
                        let mut chosen = None;
                        for (category, &p) in probs.iter().enumerate() {
                            cum += p;
                            if u <= cum {
                                chosen = Some(category as f64);
                                break;
                            }
                        }
                        // Fallback: choose the last category if none was chosen due to ordering
                        let cat = chosen.unwrap_or((probs.len() - 1) as f64);
                        sample.insert((*param).clone(), cat);
                    }
                }
            }
            samples.push(sample);
        }

        samples
    }

    pub fn log_pdf(&self, x: &HashMap<String, f64>) -> f64 {
        let n_kernels = self.weights.len();
        let mut weighted_log_pdf = vec![0.0_f64; n_kernels];

        let mut sorted_params: Vec<_> = self.distributions.keys().collect();
        sorted_params.sort();
        for param in sorted_params.iter() {
            let distribution = &self.distributions[*param];
            match distribution {
                Distributions::TruncNorm(d) => {
                    for k in 0..n_kernels {
                        if weighted_log_pdf[k] == f64::NEG_INFINITY {
                            continue;
                        }

                        let mu_k = d.mus[k];
                        let sigma_k = d.sigmas[k];
                        let val = truncnorm::log_pdf(
                            x[*param],
                            (d.low - mu_k) / sigma_k,
                            (d.high - mu_k) / sigma_k,
                            mu_k,
                            sigma_k,
                        )
                        .unwrap_or(f64::NEG_INFINITY);

                        weighted_log_pdf[k] += val;
                    }
                }
                Distributions::TruncLogNorm(d) => {
                    let x_val = x[*param];
                    if x_val <= 0.0 {
                        for k in 0..n_kernels {
                            weighted_log_pdf[k] = f64::NEG_INFINITY;
                        }
                        continue;
                    }
                    let ln_x = x_val.ln();
                    for k in 0..n_kernels {
                        if weighted_log_pdf[k] == f64::NEG_INFINITY {
                            continue;
                        }

                        let mu_k = d.mus[k];
                        let sigma_k = d.sigmas[k];
                        let val = truncnorm::log_pdf(
                            ln_x,
                            (d.low.ln() - mu_k) / sigma_k,
                            (d.high.ln() - mu_k) / sigma_k,
                            mu_k,
                            sigma_k,
                        )
                        .unwrap_or(f64::NEG_INFINITY);

                        weighted_log_pdf[k] += val - ln_x;
                    }
                }
                Distributions::DiscreteTruncNorm(d) => {
                    let x_val = x[*param];
                    for k in 0..n_kernels {
                        if weighted_log_pdf[k] == f64::NEG_INFINITY {
                            continue;
                        }

                        let mu_k = d.mus[k];
                        let sigma_k = d.sigmas[k];
                        let a = if x_val <= d.low {
                            f64::NEG_INFINITY
                        } else {
                            (x_val - d.step / 2.0 - mu_k) / sigma_k
                        };
                        let b = if x_val >= d.high {
                            f64::INFINITY
                        } else {
                            (x_val + d.step / 2.0 - mu_k) / sigma_k
                        };
                        let a_trunc = (d.low - mu_k) / sigma_k;
                        let b_trunc = (d.high - mu_k) / sigma_k;

                        match truncnorm::log_mass_interval(a, b, a_trunc, b_trunc) {
                            Ok(val) => weighted_log_pdf[k] += val,
                            Err(_) => weighted_log_pdf[k] = f64::NEG_INFINITY,
                        }
                    }
                }
                Distributions::DiscreteTruncLogNorm(d) => {
                    let x_val = x[*param];
                    if x_val <= 0.0 {
                        for k in 0..n_kernels {
                            weighted_log_pdf[k] = f64::NEG_INFINITY;
                        }
                        continue;
                    }
                    for k in 0..n_kernels {
                        if weighted_log_pdf[k] == f64::NEG_INFINITY {
                            continue;
                        }

                        let mu_k = d.mus[k];
                        let sigma_k = d.sigmas[k];
                        let low_bound = (x_val - d.step / 2.0).max(f64::MIN_POSITIVE);
                        let high_bound = x_val + d.step / 2.0;

                        let a = if x_val <= d.low {
                            f64::NEG_INFINITY
                        } else {
                            (low_bound.ln() - mu_k) / sigma_k
                        };
                        let b = if x_val >= d.high {
                            f64::INFINITY
                        } else {
                            (high_bound.ln() - mu_k) / sigma_k
                        };
                        let a_trunc = (d.low.ln() - mu_k) / sigma_k;
                        let b_trunc = (d.high.ln() - mu_k) / sigma_k;

                        match truncnorm::log_mass_interval(a, b, a_trunc, b_trunc) {
                            Ok(val) => weighted_log_pdf[k] += val,
                            Err(_) => weighted_log_pdf[k] = f64::NEG_INFINITY,
                        }
                    }
                }
                Distributions::Categorical(d) => {
                    let xi = x[*param] as usize;
                    for k in 0..n_kernels {
                        if weighted_log_pdf[k] == f64::NEG_INFINITY {
                            continue;
                        }

                        assert!(xi < d.weights[k].len(), "Categorical index out of bounds");
                        let p = d.weights[k][xi];
                        if p <= 0.0 {
                            weighted_log_pdf[k] = f64::NEG_INFINITY;
                        } else {
                            weighted_log_pdf[k] += p.ln();
                        }
                    }
                }
            }
        }

        // Add log mixture weights
        for k in 0..n_kernels {
            let w = self.weights[k];
            if w <= 0.0 {
                weighted_log_pdf[k] = f64::NEG_INFINITY;
            } else {
                weighted_log_pdf[k] += w.ln();
            }
        }

        // Log-sum-exp across kernels
        let max = weighted_log_pdf
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, |a, b| if a > b { a } else { b });
        if max.is_infinite() && max.is_sign_negative() {
            // All -inf -> return -inf
            return f64::NEG_INFINITY;
        }
        let sum_exp: f64 = weighted_log_pdf.iter().map(|v| (v - max).exp()).sum();

        // Weights are basically normalized, but sum to 1 may not hold due to float rounding error
        let log_total_weight = self.log_sum_weights;
        (max + sum_exp.ln()) - log_total_weight // Normalize by total weight to avoid float rounding error
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_mixture_of_product_distribution() {
        let truncnorm_dist = TruncNormDistributions {
            mus: vec![-0.5, 0.0],   // mus[-1] is prior
            sigmas: vec![2.0, 1.0], // sigma[-1] is prior
            low: -1.0,
            high: 1.0,
        };
        let trunclognorm_dist = TruncLogNormDistributions {
            mus: vec![2.0, 3.0],    // ditto
            sigmas: vec![2.0, 1.0], // ditto
            low: 1.0,
            high: 5.0,
        };
        let discrete_truncnorm_dist = DiscreteTruncNormDistributions {
            mus: vec![-0.5, 0.0],   // ditto
            sigmas: vec![1.0, 1.0], // ditto
            low: -1.0,
            high: 1.0,
            step: 1.0,
        };
        let discrete_trunclognorm_dist = DiscreteTruncLogNormDistributions {
            mus: vec![2.0, 3.0],    // ditto
            sigmas: vec![1.0, 1.0], // ditto
            low: 1.0,
            high: 5.0,
            step: 1.0,
        };
        let categorical_dist = CategoricalDistributions {
            weights: vec![
                vec![0.9, 0.1], // vec.len() == cardinality
                vec![0.5, 0.5], // uniform prior
            ], // weigths.len() == mus.len()
        };
        let distributions = vec![
            (
                "param_truncnorm".to_string(),
                Distributions::TruncNorm(truncnorm_dist),
            ),
            (
                "param_trunclognorm".to_string(),
                Distributions::TruncLogNorm(trunclognorm_dist),
            ),
            (
                "param_discretetruncnorm".to_string(),
                Distributions::DiscreteTruncNorm(discrete_truncnorm_dist),
            ),
            (
                "param_discretetrunclognorm".to_string(),
                Distributions::DiscreteTruncLogNorm(discrete_trunclognorm_dist),
            ),
            (
                "param_categorical".to_string(),
                Distributions::Categorical(categorical_dist),
            ),
        ];
        let distributions_map: std::collections::HashMap<String, Distributions> =
            distributions.into_iter().collect();
        let mixture = MixtureOfProductDistribution::new(
            distributions_map,
            vec![0.5, 0.5], // weights.len() == mus.len()
        );
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let samples = mixture.sample(&mut rng, 10);
        for sample in samples.iter() {
            let val_truncnorm = sample.get("param_truncnorm").unwrap();
            assert!(*val_truncnorm >= -1.0 && *val_truncnorm <= 1.0);
            let val_trunclognorm = sample.get("param_trunclognorm").unwrap();
            assert!(*val_trunclognorm >= 1.0 && *val_trunclognorm <= 5.0);
            let val_discretetruncnorm = sample.get("param_discretetruncnorm").unwrap();
            assert!(*val_discretetruncnorm >= -1.0 && *val_discretetruncnorm <= 1.0);
            let val_discretetrunclognorm = sample.get("param_discretetrunclognorm").unwrap();
            assert!(*val_discretetrunclognorm >= 1.0 && *val_discretetrunclognorm <= 5.0);
            let val_categorical = sample.get("param_categorical").unwrap();
            assert!(*val_categorical == 0.0 || *val_categorical == 1.0);
        }
    }
}
