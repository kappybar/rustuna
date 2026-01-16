use crate::parzen_estimator::probability_distributions::Distributions;
use crate::parzen_estimator::model::NumericalDistributionBuilder;
use crate::distribution::Distribution;


pub struct ScottNumericalDistributionBuilder {
    weights: &[f64],
}

impl NumericalDistributionBuilder for ScottNumericalDistributionBuilder {
    fn calculate_numerical_distribution(
        &self,
        observations: &[f64],
        search_space: &Distribution::Int,
    ) -> Distributions {
        let (low, high, step, log) = (
            search_space.low as f64,
            search_space.high as f64,
            search_space.step as f64,
            search_space.log,
        );
        let weights_sum = self.weights.iter().sum();
        
        let mean_est = observations.iter().zip(self.weights.iter()).map(|(v, w)| v * w / weights_sum);
        let sigma_est = {
            let var = observations.iter().zip(mean_est).zip(self.weights.iter())
            .map(|((v, m), w)| w * (v - m).powi(2)).sum() / (weights_sum - 1.0).max(1.0);
            var.sqrt()
        };
        
    }
}
