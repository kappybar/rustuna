use rand::Rng;
use statrs::distribution::{ContinuousCDF, Normal};

/// -0.5 * ln(2π)
const NEG_HALF_LOG_2PI: f64 = -0.9189385332046727;

#[derive(Debug)]
pub enum TruncNormError {
    #[allow(dead_code)]
    InvalidScale(f64),
    #[allow(dead_code)]
    InvalidBounds(f64, f64),
    #[allow(dead_code)]
    TinyProbabilityMass(f64, f64),
    #[allow(dead_code)]
    NanBounds(f64, f64),
}

fn log_diff_cdf(a: f64, b: f64) -> Result<f64, TruncNormError> {
    if a > b {
        return Err(TruncNormError::InvalidBounds(a, b));
    }

    let std_normal = Normal::new(0.0, 1.0).unwrap();
    let fa = std_normal.cdf(a);
    let fb = std_normal.cdf(b);

    // Require positive mass
    if !(fb > fa) {
        return Err(TruncNormError::TinyProbabilityMass(a, b));
    }

    // Difference relatively large -> safe to take ln directly
    let diff = fb - fa;
    if diff > 1e-12 * fb {
        return Ok(diff.ln());
    }

    // Handle underflowed fa == 0 (rounded to zero)
    if fa == 0.0 {
        return Ok(fb.ln());
    }

    // General stable log-diff: ln Φ(b) + ln(1 - Φ(a)/Φ(b))
    // Compute in log-space to avoid cancellation
    let lfa = fa.ln(); // Finite because fa > 0
    let lfb = fb.ln();
    let r = (lfa - lfb).exp(); // fa / fb in (0,1)
    let ln_one_minus_r = (-r).ln_1p(); // ln(1 - r) computed stably
    Ok(lfb + ln_one_minus_r)
}

/// x    : Generated sample in original scale
/// a, b : Standardized bounds (a = (low - loc) / scale, b = (high - loc) / scale)
/// loc  : Mean (location parameter)
/// scale: Standard deviation (scale parameter)
pub fn rvs<R: Rng + ?Sized>(
    rng: &mut R,
    a: f64,
    b: f64,
    loc: f64,
    scale: f64,
) -> Result<f64, TruncNormError> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(TruncNormError::InvalidScale(scale));
    }
    if a.is_nan() || b.is_nan() {
        return Err(TruncNormError::NanBounds(a, b));
    }
    if a >= b {
        return Err(TruncNormError::InvalidBounds(a, b));
    }

    let std_normal = Normal::new(0.0, 1.0).unwrap();
    let fa = std_normal.cdf(a); // Φ(a)
    let fb = std_normal.cdf(b); // Φ(b)

    let mass = fb - fa;
    if !(mass > 0.0) {
        return Err(TruncNormError::TinyProbabilityMass(fa, fb));
    }

    // Sample p directly in [fa, fb) in a numerically safe way.
    // gen_range returns value in [fa, fb), avoiding exact 0 or 1 when fa==0 or fb==1.
    let p = rng.gen_range(fa..fb); // p ~ U(Φ(a), Φ(b))
    let z = std_normal.inverse_cdf(p); // z ~ N(0,1) truncated to [a, b]
    Ok(loc + scale * z) // X = μ + σ * Z (converted to original scale)
}

/// x    : Observation in original scale
/// a, b : Standardized bounds (a = (low - loc) / scale, b = (high - loc) / scale)
/// loc  : Mean (location parameter)
/// scale: Standard deviation (scale parameter)
pub fn log_pdf(x: f64, a: f64, b: f64, loc: f64, scale: f64) -> Result<f64, TruncNormError> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(TruncNormError::InvalidScale(scale));
    }
    if a.is_nan() || b.is_nan() {
        return Err(TruncNormError::NanBounds(a, b));
    }
    if a >= b {
        return Err(TruncNormError::InvalidBounds(a, b));
    }

    let z = (x - loc) / scale;
    if z < a || z > b {
        return Ok(f64::NEG_INFINITY);
    }

    let ln_phi: f64 = -0.5 * z * z + NEG_HALF_LOG_2PI;
    let ln_mass = log_diff_cdf(a, b)?;

    Ok(ln_phi - scale.ln() - ln_mass)
}

/// a      : Uower bound of the observed interval (standardized)
/// b      : Upper bound of the observed interval (standardized)
/// a_trunc: Lower truncation bound (standardized)
/// b_trunc: Upper truncation bound (standardized)
pub fn log_mass_interval(
    a: f64,
    b: f64,
    a_trunc: f64,
    b_trunc: f64,
) -> Result<f64, TruncNormError> {
    if a.is_nan() || b.is_nan() || a_trunc.is_nan() || b_trunc.is_nan() {
        return Err(TruncNormError::NanBounds(a, b));
    }
    if a_trunc >= b_trunc {
        return Err(TruncNormError::InvalidBounds(a_trunc, b_trunc));
    }
    if a >= b {
        return Ok(f64::NEG_INFINITY);
    }
    if b <= a_trunc || a >= b_trunc {
        return Ok(f64::NEG_INFINITY);
    }

    // Intersection of observed interval and truncation interval
    let a_adjusted = a.max(a_trunc);
    let b_adjusted = b.min(b_trunc);
    if !(a_adjusted < b_adjusted) {
        return Ok(f64::NEG_INFINITY);
    }

    // compute ln(numer) = ln(Φ(hi) - Φ(lo)) robustly
    let ln_numer = log_diff_cdf(a_adjusted, b_adjusted)?;
    // compute ln(denom) = ln(Φ(b_trunc) - Φ(a_trunc)) robustly
    let ln_denom = log_diff_cdf(a_trunc, b_trunc)?;

    Ok(ln_numer - ln_denom)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_rvs() {
        let mut rng = StdRng::seed_from_u64(42);
        let loc = 10.0;
        let scale = 2.0;
        let a = -1.0;
        let b = 2.0;

        for _ in 0..100 {
            let s = rvs(&mut rng, a, b, loc, scale).unwrap();
            assert!(s >= (loc + a * scale) && s <= (loc + b * scale));
        }
    }

    #[test]
    fn test_rvs_errors_for_nan_and_invalid_scale() {
        let mut rng = StdRng::seed_from_u64(42);

        // NaN bounds -> NanBounds
        let res_nan = rvs(&mut rng, f64::NAN, 1.0, 0.0, 1.0);
        assert!(matches!(res_nan, Err(TruncNormError::NanBounds(_, _))));

        // Invalid scale -> InvalidScale
        let res_scale = rvs(&mut rng, -1.0, 1.0, 0.0, 0.0);
        assert!(matches!(res_scale, Err(TruncNormError::InvalidScale(_))));
    }

    #[test]
    fn test_truncnorm_log_pdf_basic() {
        let loc = 0.0;
        let scale = 1.0;
        let a = -1.0; // Standardized lower bound
        let b = 1.0; // Standardized upper bound

        // Case: Within bounds
        let res = log_pdf(0.0, a, b, loc, scale);
        assert!(res.is_ok());
        let val = res.unwrap();
        assert!(val.is_finite());

        // Case: Outside bounds must return -infinity
        let out_low = log_pdf(-2.0, a, b, loc, scale).unwrap();
        let out_high = log_pdf(2.0, a, b, loc, scale).unwrap();
        assert_eq!(out_low, f64::NEG_INFINITY);
        assert_eq!(out_high, f64::NEG_INFINITY);
    }

    #[test]
    fn test_log_mass_interval() {
        let a = -1.0;
        let b = 0.5;
        let a_tr = -2.0;
        let b_tr = 2.0;

        // Basic: ln((Φ(b)-Φ(a)) / (Φ(b_tr)-Φ(a_tr)))
        let std = Normal::new(0.0, 1.0).unwrap();
        let numer = std.cdf(b) - std.cdf(a);
        let denom = std.cdf(b_tr) - std.cdf(a_tr);
        assert!(numer > 0.0 && denom > 0.0);
        let expected = numer.ln() - denom.ln();

        let got = log_mass_interval(a, b, a_tr, b_tr).unwrap();
        assert!((got - expected).abs() <= 1e-12);

        // Empty observed interval -> -inf
        let got_empty = log_mass_interval(1.0, 0.0, a_tr, b_tr).unwrap();
        assert_eq!(got_empty, f64::NEG_INFINITY);
    }

    #[test]
    fn test_invalid_parameters() {
        // Error: scale <= 0
        let res = log_pdf(0.0, -1.0, 1.0, 0.0, 0.0);
        assert!(matches!(res, Err(TruncNormError::InvalidScale(_))));

        // Error: a >= b
        let mut rng = StdRng::seed_from_u64(0);
        let res = rvs(&mut rng, 1.0, -1.0, 0.0, 1.0);
        assert!(matches!(res, Err(TruncNormError::InvalidBounds(_, _))));

        // Error: NaN input
        let res = log_pdf(0.0, f64::NAN, 1.0, 0.0, 1.0);
        assert!(matches!(res, Err(TruncNormError::NanBounds(_, _))));
    }
}
