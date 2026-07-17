use crate::{Error, ErrorKind, Result};

/// Parameter distribution used by samplers and storages.
///
/// This type is the Rustuna counterpart of Optuna distributions. Trial parameters are stored
/// internally as `f64`, and categorical parameters are represented by zero-based indices.
#[derive(Debug, Clone, PartialEq)]
pub enum Distribution {
    /// Continuous range over floating-point values.
    Float {
        low: f64,
        high: f64,
        step: Option<f64>,
        log: bool,
    },
    /// Integer range with optional logarithmic sampling.
    Int {
        low: i64,
        high: i64,
        step: i64,
        log: bool,
    },
    /// Categorical distribution represented only by the number of choices.
    ///
    /// Unlike Optuna's `CategoricalDistribution`, Rustuna does not store the choice values in
    /// each distribution object. The actual labels are stored separately in study system
    /// attributes.
    Categorical { cardinality: usize },
}

impl Distribution {
    /// Checks whether two distributions are compatible for the same parameter name.
    ///
    /// Rustuna follows the same basic rule as Optuna here: the distribution kind must stay the
    /// same, and categorical cardinality or the `log` flag must not change across trials.
    pub fn check_compatibility(&self, other: &Distribution) -> Result<()> {
        match (self, other) {
            (
                Distribution::Float {
                    low: _,
                    high: _,
                    step: _,
                    log: log1,
                },
                Distribution::Float {
                    low: _,
                    high: _,
                    step: _,
                    log: log2,
                },
            ) if log1 == log2 => Ok(()),
            (
                Distribution::Int {
                    low: _,
                    high: _,
                    step: _,
                    log: log1,
                },
                Distribution::Int {
                    low: _,
                    high: _,
                    step: _,
                    log: log2,
                },
            ) if log1 == log2 => Ok(()),
            (
                Distribution::Categorical { cardinality: c1 },
                Distribution::Categorical { cardinality: c2 },
            ) if c1 == c2 => Ok(()),
            _ => Err(Error::new(ErrorKind::IncompatibleDistribution)),
        }
    }

    /// Returns whether the distribution can produce only a single value.
    pub fn is_single(&self) -> bool {
        match self {
            Distribution::Float {
                low, high, step, ..
            } => {
                if let Some(step) = step {
                    low == high || high - low - step < 1e-12
                } else {
                    low == high
                }
            }
            Distribution::Int {
                low,
                high,
                step,
                log,
            } => {
                if *log {
                    low == high
                } else {
                    low == high || high - low < *step
                }
            }
            Distribution::Categorical { cardinality } => *cardinality == 1,
        }
    }

    /// Returns whether the internal representation is contained in this distribution.
    pub fn contains(&self, internal_value: f64) -> bool {
        match self {
            Distribution::Float { low, high, .. } => {
                // TODO(c-bata): Consider checking the step attribute.
                *low <= internal_value && internal_value <= *high
            }
            Distribution::Int { low, high, .. } => {
                // TODO(c-bata): Consider checking the step attribute.
                *low as f64 <= internal_value && internal_value <= *high as f64
            }
            Distribution::Categorical { cardinality } => {
                internal_value >= 0.0 && (internal_value as usize) < *cardinality
            }
        }
    }
}


fn adjust_float_high(low: f64, high: f64, step: f64) -> f64 {
    if low > high || step <= 0.0 {
        return high;
    }

    let range = high - low;
    if (range % step).abs() > f64::EPSILON {
        (range / step).floor() * step + low
    } else {
        high
    }
}

fn adjust_int_high(low: i64, high: i64, step: i64) -> i64 {
    if low > high || step <= 0 {
        return high;
    }

    let range = high as i128 - low as i128;
    let step = step as i128;
    if range % step == 0 {
        high
    } else {
        (range / step * step + low as i128) as i64
    }
}

#[cfg(test)]
mod tests {
    use crate::distribution::Distribution;

    #[test]
    fn test_check_compatibility_different() {
        let f = Distribution::Float {
            low: 0.0,
            high: 1.0,
            step: None,
            log: false,
        };
        let i = Distribution::Int {
            low: 0,
            high: 1,
            step: 1,
            log: false,
        };
        let c = Distribution::Categorical { cardinality: 3 };
        assert!(f.check_compatibility(&i).is_err());
        assert!(f.check_compatibility(&c).is_err());
        assert!(i.check_compatibility(&f).is_err());
        assert!(i.check_compatibility(&c).is_err());
        assert!(c.check_compatibility(&f).is_err());
        assert!(c.check_compatibility(&i).is_err());
    }

    #[test]
    fn test_check_compatibility_float() {
        let f1 = Distribution::Float {
            low: 0.0,
            high: 1.0,
            step: None,
            log: false,
        };
        let f2 = Distribution::Float {
            low: -1.0,
            high: 2.0,
            step: None,
            log: false,
        };
        assert!(f1.check_compatibility(&f1).is_ok());
        assert!(f1.check_compatibility(&f2).is_ok());

        let f1_log = Distribution::Float {
            low: 0.0,
            high: 1.0,
            step: None,
            log: true,
        };
        assert!(f1.check_compatibility(&f1_log).is_err());
    }

    #[test]
    fn test_check_compatibility_int() {
        let i1 = Distribution::Int {
            low: 0,
            high: 1,
            step: 1,
            log: false,
        };
        let i2 = Distribution::Int {
            low: 0,
            high: 2,
            step: 1,
            log: false,
        };
        assert!(i1.check_compatibility(&i1).is_ok());
        assert!(i1.check_compatibility(&i2).is_ok());

        let i1_log = Distribution::Int {
            low: 0,
            high: 1,
            step: 1,
            log: true,
        };
        assert!(i1.check_compatibility(&i1_log).is_err());
    }

    #[test]
    fn test_check_compatibility_categorical() {
        let c1 = Distribution::Categorical { cardinality: 3 };
        let c2 = Distribution::Categorical { cardinality: 4 };
        assert!(c1.check_compatibility(&c1).is_ok());
        assert!(c1.check_compatibility(&c2).is_err());
    }
}
