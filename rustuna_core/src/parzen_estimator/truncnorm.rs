use rand::Rng;
use statrs::distribution::{ContinuousCDF, Normal};

/// -0.5 * ln(2π)
pub(crate) const NEG_HALF_LOG_2PI: f64 = -0.9189385332046727;

/// Fast erf approximation using piecewise strategy:
/// - |x| ≤ 0.5: degree-11 Taylor polynomial (no exp), error < 2e-8
/// - 0.5 < |x| < 6.0: Abramowitz&Stegun 7.1.26 erfc (one exp call), error < 1.5e-7
/// - |x| ≥ 6.0: ±1.0 (erf saturates)
#[inline]
fn erf_fast(x: f64) -> f64 {
    let ax = x.abs();
    if ax <= 0.5 {
        // erf(x) = x * P(x²), P is degree-5 polynomial in x²
        // Coefficients: (2/√π) * (-1)^k / (k! * (2k+1)) for k=0..5
        let x2 = x * x;
        x * (std::f64::consts::FRAC_2_SQRT_PI
            + x2 * (-0.37612638903183753
                + x2 * (0.11283791670955126
                    + x2 * (-0.026866170645131252
                        + x2 * (0.005223052625442188 + x2 * (-0.0008548327023450095))))))
    } else if ax < 6.0 {
        // A&S 7.1.26: erfc(x) ≈ poly(t) * exp(-x²), t = 1/(1 + 0.3275911*x)
        let t = 1.0 / (1.0 + 0.3275911 * ax);
        let poly = t
            * (0.254829592
                + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        let erfc = poly * (-ax * ax).exp();
        if x >= 0.0 {
            1.0 - erfc
        } else {
            erfc - 1.0
        }
    } else if x >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Standard normal CDF Φ(x) = 0.5 * (1 + erf(x/√2)).
/// Maximum absolute error: ~1.5e-7.
#[inline]
fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_fast(x * std::f64::consts::FRAC_1_SQRT_2))
}

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

#[inline]
pub(crate) fn log_diff_cdf(a: f64, b: f64) -> Result<f64, TruncNormError> {
    if a > b {
        return Err(TruncNormError::InvalidBounds(a, b));
    }

    let fa = norm_cdf(a);
    let fb = norm_cdf(b);

    // Require positive mass
    if fb <= fa {
        return Err(TruncNormError::TinyProbabilityMass(a, b));
    }

    let diff = fb - fa;

    // Fast path: when [a, b] covers almost all mass, diff ≈ 1.
    // Use log(1-t) ≈ -t - t²/2 to avoid the log() call entirely.
    // Error ≤ t³/3 < 4e-11 for t < 5e-4, well within our 1.5e-7 target.
    let tail_mass = 1.0 - diff; // = Φ(a) + (1-Φ(b)), total outside mass
    if tail_mass < 5e-4 {
        return Ok(-tail_mass - 0.5 * tail_mass * tail_mass);
    }

    // Common path: difference is large enough for direct log
    if diff > 1e-12 * fb {
        return Ok(diff.ln());
    }

    // Handle underflowed fa == 0 (rounded to zero)
    if fa == 0.0 {
        return Ok(fb.ln());
    }

    // Stable log-diff for near-equal CDFs (catastrophic cancellation case)
    // ln Φ(b) + ln(1 - Φ(a)/Φ(b))
    let lfa = fa.ln();
    let lfb = fb.ln();
    let r = (lfa - lfb).exp(); // fa / fb in (0,1)
    Ok(lfb + (-r).ln_1p())
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

    let fa = norm_cdf(a); // Φ(a)
    let fb = norm_cdf(b); // Φ(b)

    let mass = fb - fa;
    if mass <= 0.0 {
        return Err(TruncNormError::TinyProbabilityMass(fa, fb));
    }

    // Sample p directly in [fa, fb) in a numerically safe way.
    // gen_range returns value in [fa, fb), avoiding exact 0 or 1 when fa==0 or fb==1.
    let p = rng.gen_range(fa..fb); // p ~ U(Φ(a), Φ(b))
    let z = Normal::new(0.0, 1.0).unwrap().inverse_cdf(p); // z ~ N(0,1) truncated to [a, b]
    Ok(loc + scale * z) // X = μ + σ * Z (converted to original scale)
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
    fn test_norm_cdf_accuracy() {
        let std_normal = Normal::new(0.0, 1.0).unwrap();
        let xs = [
            -6.0, -5.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0,
            6.0,
        ];
        for &x in &xs {
            let approx = norm_cdf(x);
            let exact = std_normal.cdf(x);
            let abs_err = (approx - exact).abs();
            assert!(
                abs_err < 2e-7,
                "norm_cdf({x}) = {approx}, exact = {exact}, abs_err = {abs_err}"
            );
        }
        // Monotonicity
        let mut prev = norm_cdf(-8.0);
        for i in -79..=80 {
            let x = i as f64 * 0.1;
            let cur = norm_cdf(x);
            assert!(cur >= prev, "norm_cdf not monotone at x={x}");
            prev = cur;
        }
        // Symmetry: Φ(x) + Φ(-x) = 1
        for &x in &xs {
            let sum = norm_cdf(x) + norm_cdf(-x);
            assert!(
                (sum - 1.0).abs() < 2e-7,
                "symmetry failed at x={x}: sum={sum}"
            );
        }
    }

    #[test]
    fn test_erf_fast_accuracy() {
        let std_normal = Normal::new(0.0, 1.0).unwrap();
        // erf(x) = 2*Φ(x*√2) - 1
        let xs = [
            -5.5, -4.0, -3.0, -2.0, -1.0, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 1.0,
            2.0, 3.0, 4.0, 5.5,
        ];
        for &x in &xs {
            let approx = erf_fast(x);
            let exact = 2.0 * std_normal.cdf(x * std::f64::consts::SQRT_2) - 1.0;
            let abs_err = (approx - exact).abs();
            assert!(
                abs_err < 2e-7,
                "erf_fast({x}) = {approx}, exact = {exact}, abs_err = {abs_err}"
            );
        }
        // Small-x path: verify Taylor branch gives better accuracy
        for &x in &[-0.5_f64, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5] {
            let approx = erf_fast(x);
            let exact = 2.0 * std_normal.cdf(x * std::f64::consts::SQRT_2) - 1.0;
            assert!(
                (approx - exact).abs() < 2e-8,
                "Taylor branch error too large at x={x}"
            );
        }
        // Odd symmetry: erf(-x) == -erf(x)
        for &x in &xs {
            assert!(
                (erf_fast(x) + erf_fast(-x)).abs() < 1e-15,
                "odd symmetry failed at x={x}"
            );
        }
    }

    #[test]
    fn test_invalid_parameters() {
        // Error: a >= b
        let mut rng = StdRng::seed_from_u64(0);
        let res = rvs(&mut rng, 1.0, -1.0, 0.0, 1.0);
        assert!(matches!(res, Err(TruncNormError::InvalidBounds(_, _))));
    }
}
