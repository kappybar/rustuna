use crate::parzen_estimator::probability_distributions::{Distributions, DiscreteTruncNormDistributions};
use crate::parzen_estimator::model::NumericalDistributionBuilder;
use crate::distribution::Distribution;


pub struct ScottNumericalDistributionBuilder<'a> {
    weights: &'a [f64],
}

impl<'a> NumericalDistributionBuilder for ScottNumericalDistributionBuilder<'a> {
    fn calculate_numerical_distribution(
        &self,
        observations: &[f64],
        search_space: &Distribution::Int,
    ) -> Distributions {
        // NOTE: Since the Optuna TPE bandwidth selection is too wide for this analysis, we use the Scott's rule:
        // David W. Scott. 1992. Multivariate Density Estimation: Theory, Practice, and Visualization. John Wiley & Sons. 
        let (low, high, step, log) = (
            search_space.low as f64,
            search_space.high as f64,
            search_space.step as f64,
            search_space.log,
        );
        let weights_sum = self.weights.iter().sum();
        let n_observations = observations.len();
        
        let mean_est = observations.iter().zip(self.weights.iter()).map(|(v, w)| v * w / weights_sum);
        let sigma_est = {
            let var = observations.iter().zip(mean_est).zip(self.weights.iter())
            .map(|((v, m), w)| w * (v - m).powi(2)).sum() / (weights_sum - 1.0).max(1.0);
            var.sqrt()
        };

        let inter_quantile_range = if observations.is_empty() { 0.0 } else {
            let mut sorted_obs = observations.to_vec();
            sorted_obs.sort_by(|a, b| a.total_cmp(b));
            let q1_idx = ((0.25 * (n_observations as f64 - 1.0)).floor() as usize).min(n_observations - 1);
            let q3_idx = ((0.75 * (n_observations as f64 - 1.0)).floor() as usize).min(n_observations - 1);
            sorted_obs[q3_idx] - sorted_obs[q1_idx]
        };
        let sigma_est = (1.059 * sigma_est.min(inter_quantile_range / 1.34) * weights_sum).powf(-0.2);
        let sigmas = std::iter::repeat_n(sigma_est, n_observations);

        let mus_with_prior = observations.iter().chain(std::iter::once((low + high) / 2.0));
        let sigmas_with_prior = sigmas.chain(std::iter::once(high - low + 1.0));


        Distributions::DiscreteTruncNorm(DiscreteTruncNormDistributions {
            mus: mus_with_prior.cloned().collect(),
            sigmas: sigmas_with_prior.collect(),
            low,
            high,
            step,
        })
    }
}
