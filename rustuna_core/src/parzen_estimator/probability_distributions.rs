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

#[derive(Debug)]
pub(crate) struct MixtureOfProductDistribution {
    pub param_names: Vec<String>,          // Sorted param names
    pub distributions: Vec<Distributions>, // Sorted distributions
    pub log_weights: Vec<f64>,             // ln(w_i)
    pub log_sum_weights: f64,              // ln(sum weights)
    pub alias: WeightedAliasIndex<f64>,
    pub n_kernels: usize,
}

impl MixtureOfProductDistribution {
    pub fn new(distributions_map: HashMap<String, Distributions>, weights: Vec<f64>) -> Self {
        let sum_w = weights.iter().sum::<f64>();
        let log_sum_weights = if sum_w > 0.0 {
            sum_w.ln()
        } else {
            0.0_f64.ln()
        };
        let log_weights = weights
            .iter()
            .map(|&w| if w > 0.0 { w.ln() } else { f64::NEG_INFINITY })
            .collect::<Vec<_>>();

        let alias = WeightedAliasIndex::new(weights.clone())
            .expect("weights must be non-empty and non-negative");

        let mut param_names: Vec<String> = distributions_map.keys().cloned().collect();
        param_names.sort();

        let mut distributions: Vec<Distributions> = Vec::with_capacity(param_names.len());
        for name in param_names.iter() {
            distributions.push(distributions_map.get(name).unwrap().clone());
        }

        let n_kernels = weights.len();

        MixtureOfProductDistribution {
            param_names,
            distributions,
            log_weights,
            log_sum_weights,
            alias,
            n_kernels,
        }
    }

    pub fn sample(&self, rng: &mut StdRng, size: usize) -> Vec<HashMap<String, f64>> {
        let mut samples: Vec<HashMap<String, f64>> = Vec::with_capacity(size);

        for _ in 0..size {
            let k = self.alias.sample(rng); // Active kernel index
            let mut sample = HashMap::with_capacity(self.param_names.len());
            for (param, dist) in self.param_names.iter().zip(self.distributions.iter()) {
                match dist {
                    Distributions::TruncNorm(d) => {
                        let mu = d.mus[k];
                        let sigma = d.sigmas[k];
                        let value = truncnorm::rvs(
                            rng,
                            (d.low - mu) / sigma,
                            (d.high - mu) / sigma,
                            mu,
                            sigma,
                        )
                        .unwrap();
                        sample.insert(param.clone(), value);
                    }
                    Distributions::TruncLogNorm(d) => {
                        let mu = d.mus[k];
                        let sigma = d.sigmas[k];
                        let log_value = truncnorm::rvs(
                            rng,
                            (d.low.ln() - mu) / sigma,
                            (d.high.ln() - mu) / sigma,
                            mu,
                            sigma,
                        )
                        .unwrap();
                        sample.insert(param.clone(), log_value.exp());
                    }
                    Distributions::DiscreteTruncNorm(d) => {
                        let mu = d.mus[k];
                        let sigma = d.sigmas[k];
                        let value = truncnorm::rvs(
                            rng,
                            (d.low - d.step / 2.0 - mu) / sigma,
                            (d.high + d.step / 2.0 - mu) / sigma,
                            mu,
                            sigma,
                        )
                        .unwrap();
                        let discrete_value = (d.low + ((value - d.low) / d.step).round() * d.step)
                            .max(d.low)
                            .min(d.high);
                        sample.insert(param.clone(), discrete_value);
                    }
                    Distributions::DiscreteTruncLogNorm(d) => {
                        let mu = d.mus[k];
                        let sigma = d.sigmas[k];
                        let log_value = truncnorm::rvs(
                            rng,
                            ((d.low - d.step / 2.0).max(f64::MIN_POSITIVE).ln() - mu) / sigma,
                            ((d.high + d.step / 2.0).max(f64::MIN_POSITIVE).ln() - mu) / sigma,
                            mu,
                            sigma,
                        )
                        .unwrap();
                        let original = log_value.exp();
                        let discrete_value = (d.low
                            + ((original - d.low) / d.step).round() * d.step)
                            .max(d.low)
                            .min(d.high);
                        sample.insert(param.clone(), discrete_value);
                    }
                    Distributions::Categorical(d) => {
                        let probs = &d.weights[k];
                        let sum: f64 = probs.iter().sum();
                        assert!(sum > 0.0, "Categorical distribution has non-positive total probability for param {param}");

                        let u = rng.gen::<f64>() * sum;
                        let mut cum = 0.0;
                        let mut chosen = (probs.len() - 1) as f64; // fallback
                        for (category, &p) in probs.iter().enumerate() {
                            cum += p;
                            if u <= cum {
                                chosen = category as f64;
                                break;
                            }
                        }
                        sample.insert(param.clone(), chosen);
                    }
                }
            }
            samples.push(sample);
        }

        samples
    }

    pub fn log_pdf(&self, x: &HashMap<String, f64>) -> f64 {
        let n = self.n_kernels;
        let mut weighted_log_pdf = vec![0.0_f64; n];

        for (param, dist) in self.param_names.iter().zip(self.distributions.iter()) {
            let x_val = match x.get(param) {
                Some(v) => *v,
                None => {
                    return f64::NEG_INFINITY;
                }
            };

            match dist {
                Distributions::TruncNorm(d) => {
                    for (k, weight) in weighted_log_pdf.iter_mut().enumerate().take(n) {
                        if *weight == f64::NEG_INFINITY {
                            continue;
                        }
                        let mu_k = d.mus[k];
                        let sigma_k = d.sigmas[k];
                        let val = truncnorm::log_pdf(
                            x_val,
                            (d.low - mu_k) / sigma_k,
                            (d.high - mu_k) / sigma_k,
                            mu_k,
                            sigma_k,
                        )
                        .unwrap_or(f64::NEG_INFINITY);
                        *weight += val;
                    }
                }
                Distributions::TruncLogNorm(d) => {
                    if x_val <= 0.0 {
                        return f64::NEG_INFINITY;
                    }
                    let ln_x = x_val.ln();
                    for (k, weight) in weighted_log_pdf.iter_mut().enumerate().take(n) {
                        if *weight == f64::NEG_INFINITY {
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
                        *weight += val - ln_x;
                    }
                }
                Distributions::DiscreteTruncNorm(d) => {
                    for (k, weight) in weighted_log_pdf.iter_mut().enumerate().take(n) {
                        if *weight == f64::NEG_INFINITY {
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
                            Ok(v) => *weight += v,
                            Err(_) => *weight = f64::NEG_INFINITY,
                        }
                    }
                }
                Distributions::DiscreteTruncLogNorm(d) => {
                    if x_val <= 0.0 {
                        return f64::NEG_INFINITY;
                    }
                    for (k, weight) in weighted_log_pdf.iter_mut().enumerate().take(n) {
                        if *weight == f64::NEG_INFINITY {
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
                            Ok(v) => *weight += v,
                            Err(_) => *weight = f64::NEG_INFINITY,
                        }
                    }
                }
                Distributions::Categorical(d) => {
                    let xi = x_val as usize;
                    for (k, weight) in weighted_log_pdf.iter_mut().enumerate().take(n) {
                        if *weight == f64::NEG_INFINITY {
                            continue;
                        }
                        if xi >= d.weights[k].len() {
                            return f64::NEG_INFINITY;
                        }
                        let p = d.weights[k][xi];
                        if p <= 0.0 {
                            *weight = f64::NEG_INFINITY;
                        } else {
                            *weight += p.ln();
                        }
                    }
                }
            }
        }

        // Add log weights
        for (k, weight) in weighted_log_pdf.iter_mut().enumerate().take(n) {
            let lw = self.log_weights[k];
            if lw.is_infinite() && lw.is_sign_negative() {
                *weight = f64::NEG_INFINITY;
            } else {
                *weight += lw;
            }
        }

        // Log-sum-exp across kernels
        let max = weighted_log_pdf
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        if max.is_infinite() && max.is_sign_negative() {
            // All -inf -> return -inf

            return f64::NEG_INFINITY;
        }
        let sum_exp: f64 = weighted_log_pdf.iter().map(|v| (v - max).exp()).sum();
        // Weights are basically normalized, but sum to 1 may not hold due to float rounding error
        (max + sum_exp.ln()) - self.log_sum_weights // Normalize by total weight to avoid float rounding error
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
