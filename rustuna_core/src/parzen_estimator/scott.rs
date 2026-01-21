use crate::distribution::Distribution;
use crate::parzen_estimator::model::NumericalDistributionBuilder;
use crate::parzen_estimator::probability_distributions::{
    DiscreteTruncLogNormDistributions, DiscreteTruncNormDistributions, Distributions,
    TruncLogNormDistributions, TruncNormDistributions,
};

pub struct ScottNumericalDistributionBuilder<'a> {
    weights: &'a [f64],
}

impl<'a> NumericalDistributionBuilder for ScottNumericalDistributionBuilder<'a> {
    fn calculate_numerical_distribution(
        &self,
        observations: &[f64],
        search_space: &Distribution,
    ) -> Distributions {
        // NOTE: Since the Optuna TPE bandwidth selection is too wide for this analysis, we use the Scott's rule:
        // David W. Scott. 1992. Multivariate Density Estimation: Theory, Practice, and Visualization. John Wiley & Sons.
        let (low, high, step, log) = match search_space {
            Distribution::Int {
                low,
                high,
                step,
                log,
            } => (*low as f64, *high as f64, Some(*step as f64), *log),
            Distribution::Float {
                low,
                high,
                step,
                log,
            } => (*low, *high, *step, *log),
            _ => panic!("Unsupported distribution type for ScottNumericalDistributionBuilder"),
        };
        let weights_sum = self.weights.iter().sum::<f64>();
        let n_observations = observations.len();

        let mean_est = observations
            .iter()
            .zip(self.weights.iter())
            .map(|(v, w)| v * w)
            .sum::<f64>()
            / weights_sum;
        let sigma_est = {
            let var = observations
                .iter()
                .zip(self.weights.iter())
                .map(|(v, w)| w * (v - mean_est).powi(2))
                .sum::<f64>()
                / (weights_sum - 1.0).max(1.0);
            var.sqrt()
        };

        let inter_quantile_range = if observations.is_empty() {
            0.0
        } else {
            let mut sorted_obs = observations.to_vec();
            sorted_obs.sort_by(|a, b| a.total_cmp(b));
            let q1_idx =
                ((0.25 * (n_observations as f64 - 1.0)).floor() as usize).min(n_observations - 1);
            let q3_idx =
                ((0.75 * (n_observations as f64 - 1.0)).floor() as usize).min(n_observations - 1);
            sorted_obs[q3_idx] - sorted_obs[q1_idx]
        };
        let sigma_est =
            (1.059 * sigma_est.min(inter_quantile_range / 1.34) * weights_sum).powf(-0.2);
        let sigmas = std::iter::repeat_n(sigma_est, n_observations);

        let mus_with_prior = observations
            .iter()
            .copied()
            .chain(std::iter::once((low + high) / 2.0));
        let sigmas_with_prior = sigmas.chain(std::iter::once(high - low + 1.0));

        match (step, log) {
            (None, false) => Distributions::TruncNorm(TruncNormDistributions {
                mus: mus_with_prior.collect(),
                sigmas: sigmas_with_prior.collect(),
                low,
                high,
            }),
            (None, true) => Distributions::TruncLogNorm(TruncLogNormDistributions {
                mus: mus_with_prior.collect(),
                sigmas: sigmas_with_prior.collect(),
                low,
                high,
            }),
            (Some(step), false) => {
                Distributions::DiscreteTruncNorm(DiscreteTruncNormDistributions {
                    mus: mus_with_prior.collect(),
                    sigmas: sigmas_with_prior.collect(),
                    low,
                    high,
                    step,
                })
            }
            (Some(step), true) => {
                Distributions::DiscreteTruncLogNorm(DiscreteTruncLogNormDistributions {
                    mus: mus_with_prior.collect(),
                    sigmas: sigmas_with_prior.collect(),
                    low,
                    high,
                    step,
                })
            }
        }
    }
}
