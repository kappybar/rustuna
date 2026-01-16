use crate::parzen_estimator::probability_distributions::Distributions;
use crate::parzen_estimator::model::NumericalDistributionBuilder;
use crate::distribution::Distribution;



pub struct ScottNumericalDistributionBuilder {
    weights: &[f64],
}
