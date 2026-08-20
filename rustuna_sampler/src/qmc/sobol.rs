//! Sobol' low-discrepancy sequence generator.

use rustuna_core::{Error, ErrorKind, Result};

use super::direction_numbers::{self, MAX_DIM};

/// Number of fixed-point bits behind each coordinate, matching `scipy.stats.qmc.Sobol`.
const BITS: usize = 30;

/// Number of points the sequence holds before it would start repeating.
const CAPACITY: u64 = 1 << BITS;

/// Generator for the Sobol' quasirandom sequence.
///
/// The engine reproduces `scipy.stats.qmc.Sobol(d, scramble=False)` exactly: it uses the same
/// Joe-Kuo direction numbers, the same Antonov-Saleev Gray code construction, and the same
/// convention of placing the origin at index 0.
///
/// A coordinate is the exclusive-or of the direction numbers selected by the set bits of the
/// Gray code of the point index. Since every operation is a bitwise exclusive-or, the direction
/// numbers are kept as fixed-point integers and scaled into `[0, 1)` only when a point is handed
/// out.
///
/// # Balance properties
///
/// Sobol' points are a quadrature rule, not a stream of independent samples. They form a
/// `(t, m, d)`-net only when the `2^m` points at indices `0..2^m` are taken together, so using a
/// non-power-of-two number of points, skipping index 0, or thinning the sequence all degrade
/// their uniformity.
///
/// # Examples
///
/// ```
/// use rustuna_core::Result;
/// use rustuna_sampler::qmc::SobolEngine;
///
/// fn main() -> Result<()> {
///     let engine = SobolEngine::new(2)?;
///     assert_eq!(engine.nth_point(0)?, vec![0.0, 0.0]);
///     assert_eq!(engine.nth_point(1)?, vec![0.5, 0.5]);
///     assert_eq!(engine.nth_point(2)?, vec![0.75, 0.25]);
///     assert_eq!(engine.nth_point(3)?, vec![0.25, 0.75]);
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub struct SobolEngine {
    dim: usize,
    /// Direction numbers per dimension: `sv[d][j]` is the `j`-th direction number of dimension
    /// `d`, left-aligned as a `BITS`-wide fixed-point fraction.
    sv: Vec<[u32; BITS]>,
}

impl SobolEngine {
    /// Creates an engine over `dim` dimensions.
    pub fn new(dim: usize) -> Result<Self> {
        if dim == 0 || dim > MAX_DIM {
            return Err(Error::with_reason(
                ErrorKind::SamplerError,
                format!("Sobol' dimension must be in 1..={MAX_DIM}, got {dim}"),
            ));
        }

        Ok(Self {
            dim,
            sv: initialize_direction_numbers(dim),
        })
    }

    /// Returns the number of dimensions the engine covers.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the point at index `n`, where index 0 is the origin.
    ///
    /// This evaluates the Gray code definition of the sequence directly rather than iterating the
    /// recurrence, so the cost does not grow with `n`.
    pub fn nth_point(&self, n: u64) -> Result<Vec<f64>> {
        if n >= CAPACITY {
            return Err(Error::with_reason(
                ErrorKind::SamplerError,
                format!("Sobol' point index must be below 2**{BITS}={CAPACITY}, got {n}"),
            ));
        }

        let mut quasi = vec![0u32; self.dim];
        let mut gray = n ^ (n >> 1);
        let mut j = 0;
        while gray != 0 {
            if gray & 1 == 1 {
                for (value, row) in quasi.iter_mut().zip(&self.sv) {
                    *value ^= row[j];
                }
            }
            gray >>= 1;
            j += 1;
        }

        let scale = 1.0 / CAPACITY as f64;
        Ok(quasi
            .iter()
            .map(|&value| f64::from(value) * scale)
            .collect())
    }
}

/// Builds the direction numbers for `dim` dimensions as `BITS`-wide fixed-point fractions.
///
/// This mirrors SciPy's `_initialize_v`. For each dimension the initial values `m_1..m_s` come
/// from the Joe-Kuo table and the remaining ones follow the recurrence induced by that
/// dimension's primitive polynomial `x^s + a_1 x^(s-1) + ... + a_(s-1) x + 1`:
///
/// ```text
/// m_i = 2 a_1 m_(i-1) XOR 4 a_2 m_(i-2) XOR ... XOR 2^(s-1) a_(s-1) m_(i-s+1)
///       XOR 2^s m_(i-s) XOR m_(i-s)
/// ```
fn initialize_direction_numbers(dim: usize) -> Vec<[u32; BITS]> {
    let mut sv = vec![[0u32; BITS]; dim];

    // Dimension 0 is the van der Corput sequence, which uses m_i = 1 for every i and needs no
    // primitive polynomial.
    sv[0].fill(1);

    for (index, entry) in direction_numbers::decode(dim).enumerate() {
        let row = &mut sv[index + 1];
        let degree = entry.degree;

        for (j, slot) in row.iter_mut().enumerate().take(degree.min(BITS)) {
            *slot = entry.m[j];
        }
        for j in degree..BITS {
            let mut value = row[j - degree];
            let mut pow2 = 1;
            for k in 0..degree {
                pow2 <<= 1;
                if (entry.poly >> (degree - 1 - k)) & 1 == 1 {
                    value ^= pow2 * row[j - k - 1];
                }
            }
            row[j] = value;
        }
    }

    // Left-align every m_j so that the stored integer is the direction number times 2^BITS.
    for row in &mut sv {
        for (j, value) in row.iter_mut().enumerate() {
            *value <<= BITS - 1 - j;
        }
    }
    sv
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rescales a point back to the fixed-point integers the engine stores, so that expected
    /// values taken from SciPy can be written exactly.
    fn as_fixed_point(point: &[f64]) -> Vec<u64> {
        point
            .iter()
            .map(|&value| (value * CAPACITY as f64).round() as u64)
            .collect()
    }

    /// Returns the points at indices `0..n`.
    fn first_points(dim: usize, n: u64) -> Result<Vec<Vec<f64>>> {
        let engine = SobolEngine::new(dim)?;
        (0..n).map(|index| engine.nth_point(index)).collect()
    }

    #[test]
    fn matches_scipy_in_one_dimension() -> Result<()> {
        // scipy.stats.qmc.Sobol(d=1, scramble=False).random(16)
        let expected = [
            0.0, 0.5, 0.75, 0.25, 0.375, 0.875, 0.625, 0.125, 0.1875, 0.6875, 0.9375, 0.4375,
            0.3125, 0.8125, 0.5625, 0.0625,
        ];
        let points = first_points(1, expected.len() as u64)?;
        for (point, expected) in points.iter().zip(expected) {
            assert_eq!(point, &[expected]);
        }
        Ok(())
    }

    #[test]
    fn matches_scipy_in_three_dimensions() -> Result<()> {
        // scipy.stats.qmc.Sobol(d=3, scramble=False).random(8)
        let expected = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.75, 0.25, 0.25],
            [0.25, 0.75, 0.75],
            [0.375, 0.375, 0.625],
            [0.875, 0.875, 0.125],
            [0.625, 0.125, 0.875],
            [0.125, 0.625, 0.375],
        ];
        let points = first_points(3, expected.len() as u64)?;
        assert_eq!(points, expected.map(|point| point.to_vec()));
        Ok(())
    }

    #[test]
    fn matches_scipy_in_forty_dimensions() -> Result<()> {
        // scipy.stats.qmc.Sobol(d=40, scramble=False).fast_forward(100).random(1), scaled by 2^30.
        let point = as_fixed_point(&SobolEngine::new(40)?.nth_point(100)?);
        for (index, expected) in [
            (0, 444596224),
            (1, 276824064),
            (2, 830472192),
            (37, 511705088),
            (38, 964689920),
            (39, 142606336),
        ] {
            assert_eq!(point[index], expected, "dimension {index}");
        }
        Ok(())
    }

    #[test]
    fn matches_scipy_in_the_largest_supported_dimension() -> Result<()> {
        // The tail dimensions exercise the widest polynomials in the table, so this also checks
        // that the packed direction numbers decode correctly all the way to the end.
        // scipy.stats.qmc.Sobol(d=1024, scramble=False).fast_forward(12345).random(1) * 2^30.
        let point = as_fixed_point(&SobolEngine::new(MAX_DIM)?.nth_point(12345)?);
        for (index, expected) in [
            (0, 688193536),
            (1, 873398272),
            (100, 579272704),
            (500, 324861952),
            (1021, 954925056),
            (1022, 370868224),
            (1023, 619642880),
        ] {
            assert_eq!(point[index], expected, "dimension {index}");
        }
        Ok(())
    }

    #[test]
    fn forms_a_net_in_base_two() -> Result<()> {
        // A (0, m, 2)-net puts exactly one of the 2^m points into every elementary interval of
        // volume 2^-m. Splitting the unit square 2^k1 by 2^k2 for every k1 + k2 = m enumerates
        // those intervals.
        const M: usize = 6;
        let points = first_points(2, 1 << M)?;

        for k1 in 0..=M {
            let (rows, columns) = (1usize << k1, 1usize << (M - k1));
            let mut counts = vec![0; rows * columns];
            for point in &points {
                let row = (point[0] * rows as f64) as usize;
                let column = (point[1] * columns as f64) as usize;
                counts[row * columns + column] += 1;
            }
            assert!(
                counts.iter().all(|&count| count == 1),
                "the {rows}x{columns} split is not balanced"
            );
        }
        Ok(())
    }

    #[test]
    fn rejects_unsupported_dimensions() {
        assert!(SobolEngine::new(0).is_err());
        assert!(SobolEngine::new(MAX_DIM + 1).is_err());
        assert!(SobolEngine::new(MAX_DIM).is_ok());
    }

    #[test]
    fn refuses_indices_past_the_end_of_the_sequence() -> Result<()> {
        let engine = SobolEngine::new(2)?;
        assert_eq!(engine.nth_point(CAPACITY - 1)?.len(), 2);
        assert!(engine.nth_point(CAPACITY).is_err());
        Ok(())
    }
}
