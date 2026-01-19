use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

const EPS: f64 = 1e-12;

pub fn fast_non_dominated_sort<R>(loss_values: &[R]) -> Vec<usize>
where
    R: AsRef<[f64]>,
{
    let n = loss_values.len();
    let mut ranks: Vec<usize> = vec![usize::MAX; n];
    if n == 0 {
        return ranks;
    }

    let dominates = |a: &[f64], b: &[f64]| -> bool {
        if a.len() != b.len() {
            return false;
        }
        let mut strictly_better = false;
        for (x, y) in a.iter().zip(b.iter()) {
            if x.is_nan() || y.is_nan() {
                return false;
            }
            if *x > *y {
                return false;
            }
            if *x < *y {
                strictly_better = true;
            }
        }
        strictly_better
    };

    let mut s: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut n_dom: Vec<usize> = vec![0; n];
    for p in 0..(n - 1) {
        for q in (p + 1)..n {
            let a = loss_values[p].as_ref();
            let b = loss_values[q].as_ref();

            let p_dom_q = dominates(a, b);
            let q_dom_p = dominates(b, a);

            if p_dom_q {
                s[p].push(q);
                n_dom[q] += 1;
            } else if q_dom_p {
                s[q].push(p);
                n_dom[p] += 1;
            }
        }
    }

    let mut current_front: Vec<usize> = Vec::new();
    for i in 0..n {
        if n_dom[i] == 0 {
            ranks[i] = 0;
            current_front.push(i);
        }
    }

    let mut front_idx = 0usize;
    while !current_front.is_empty() {
        let mut next_front: Vec<usize> = Vec::new();
        for &p in current_front.iter() {
            for &q in s[p].iter() {
                n_dom[q] = n_dom[q].saturating_sub(1);
                if n_dom[q] == 0 {
                    ranks[q] = front_idx + 1;
                    next_front.push(q);
                }
            }
        }
        front_idx += 1;
        current_front = next_front;
    }

    debug_assert!(
        ranks.iter().all(|&r| r != usize::MAX),
        "Some ranks were not assigned"
    );

    ranks
}

pub fn filter_pareto_front<R>(loss_values: &[R], indices: &mut Vec<usize>)
where
    R: AsRef<[f64]>,
{
    if indices.is_empty() {
        return;
    }

    let orig_indices = indices.clone();
    let subset_rows: Vec<&[f64]> = orig_indices
        .iter()
        .map(|&i| loss_values[i].as_ref())
        .collect();
    let subset_ranks = fast_non_dominated_sort(&subset_rows);
    let mut write = 0usize;
    for (subset_idx, &orig_idx) in orig_indices.iter().enumerate() {
        if subset_ranks[subset_idx] == 0 {
            indices[write] = orig_idx;
            write += 1;
        }
    }
    indices.truncate(write);
}

pub fn compute_hypervolume<R>(loss_vals: &[R], reference_point: &[f64]) -> f64
where
    R: AsRef<[f64]>,
{
    if loss_vals.is_empty() {
        return 0.0;
    }
    let n = loss_vals.len();
    let m = reference_point.len();

    assert!(m > 0, "reference_point must be non-empty");
    for (k, &v) in reference_point.iter().enumerate() {
        assert!(!v.is_nan(), "reference_point[{}] is NaN", k);
    }

    for (i, r) in loss_vals.iter().enumerate() {
        let row = r.as_ref();
        assert!(
            row.len() == m,
            "dim mismatch: loss_vals[{}].len() == {}, expected {}",
            i,
            row.len(),
            m
        );
    }

    let mut indices: Vec<usize> = (0..n).collect();
    filter_pareto_front(loss_vals, &mut indices);

    // Sort pareto indices by the first objective (ascending) to often improve recursion behavior.
    // Use total_cmp to get a total order (handles NaN in a deterministic way).
    indices.sort_unstable_by(|&a, &b| {
        let ra = loss_vals[a].as_ref()[0];
        let rb = loss_vals[b].as_ref()[0];
        ra.total_cmp(&rb)
    });

    let inclusive_hvs: Vec<f64> = indices
        .iter()
        .map(|&idx| {
            let row = loss_vals[idx].as_ref();
            let mut prod = 1.0;
            for k in 0..m {
                // Clamp the difference to be >= 0.0 to avoid negative/invalid volumes.
                // Negative differences can happen when the reference point dominates some loss_vals.
                let diff = reference_point[k] - row[k];
                let diff = if diff.is_nan() { 0.0 } else { diff };
                prod *= diff.max(0.0);
            }
            prod
        })
        .collect();

    match indices.len() {
        0 => 0.0,
        1 => inclusive_hvs.iter().sum(),
        2 => {
            let i = indices[0];
            let j = indices[1];
            let ri = loss_vals[i].as_ref();
            let rj = loss_vals[j].as_ref();
            let mut inter = 1.0;
            for k in 0..m {
                let maxval = ri[k].max(rj[k]);
                let diff = reference_point[k] - maxval;
                let diff = if diff.is_nan() { 0.0 } else { diff };
                inter *= diff.max(0.0);
            }
            inclusive_hvs.iter().sum::<f64>() - inter
        }
        _ => {
            let mut total = 0.0;
            let len = indices.len();
            for (i_pos, &i_idx) in indices.iter().enumerate() {
                let inclusive_hv = inclusive_hvs[i_pos];
                let rows = len - (i_pos + 1);
                if rows == 0 {
                    total += inclusive_hv;
                    continue;
                }
                let mut limited_rows: Vec<Vec<f64>> = Vec::with_capacity(rows);
                for j_rel in 0..rows {
                    let j_idx = indices[i_pos + 1 + j_rel];
                    let ri = loss_vals[i_idx].as_ref();
                    let rj = loss_vals[j_idx].as_ref();
                    let mut row: Vec<f64> = Vec::with_capacity(m);
                    for k in 0..m {
                        row.push(ri[k].max(rj[k]));
                    }
                    limited_rows.push(row);
                }
                let limited_refs: Vec<&[f64]> = limited_rows.iter().map(|r| r.as_slice()).collect();
                total += compute_exclusive_hypervolume(&limited_refs, inclusive_hv, reference_point);
            }
            total
        }
    }
}

fn compute_exclusive_hypervolume(
    limited_sols: &[&[f64]],
    inclusive_hv: f64,
    reference_point: &[f64],
) -> f64 {
    if limited_sols.is_empty() {
        inclusive_hv
    } else {
        inclusive_hv - compute_hypervolume(limited_sols, reference_point)
    }
}

/// Binary-heap entry for max-heap by contribution.
/// We implement Ord such that largest contribution is on top.
#[derive(Debug)]
struct HeapEntry {
    contrib: f64,
    idx: usize,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // 1. Compare by `contrib` as the primary key (treat NaN as the smallest value)
        match self.contrib.partial_cmp(&other.contrib) {
            Some(ord) => {
                // If `contrib` values are comparable
                // Use `idx` as a tie-breaker when they are equal
                ord.then_with(|| self.idx.cmp(&other.idx))
            }
            None => {
                // Handle cases where NaN is involved
                let a_nan = self.contrib.is_nan();
                let b_nan = other.contrib.is_nan();
                match (a_nan, b_nan) {
                    // Both are NaN: decide ordering by `idx`
                    (true, true) => self.idx.cmp(&other.idx),
                    // Treat NaN as the smallest value
                    (true, false) => Ordering::Less,
                    (false, true) => Ordering::Greater,
                    // This case should not occur in theory, but fall back to `idx` defensively
                    (false, false) => self.idx.cmp(&other.idx),
                }
            }
        }
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for HeapEntry {}

/// Lazy (delayed) update hypervolume subset selection.
/// Since this problem is known to be submodular maximization, 1-1/e approximation can be achieved via greedy selection.
/// rank_i_loss_vals: rows (any R: AsRef<[f64]>) aligned with rank_i_indices
/// rank_i_indices: original indices aligned with rank_i_loss_vals
pub fn hypervolume_subset_selection<R>(
    rank_i_loss_vals: &[R],
    rank_i_indices: &[usize],
    reference_point: &[f64],
    subset_size: usize,
) -> Vec<usize>
where
    R: AsRef<[f64]>,
{
    assert_eq!(rank_i_loss_vals.len(), rank_i_indices.len());

    let mut idx_to_row: HashMap<usize, &[f64]> = HashMap::with_capacity(rank_i_loss_vals.len());
    for (&idx, row) in rank_i_indices.iter().zip(rank_i_loss_vals.iter()) {
        idx_to_row.insert(idx, row.as_ref());
    }

    // Initial contributions: HV of each single solution
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(rank_i_loss_vals.len());
    for &idx in rank_i_indices.iter() {
        let single = [*idx_to_row.get(&idx).unwrap()];
        let hv = compute_hypervolume(&single, reference_point);
        heap.push(HeapEntry { contrib: hv, idx });
    }

    let mut selected_rows: Vec<&[f64]> = Vec::with_capacity(subset_size);
    let mut selected_indices: Vec<usize> = Vec::with_capacity(subset_size);
    let mut selected_set: HashSet<usize> = HashSet::with_capacity(subset_size);

    while selected_indices.len() < subset_size {
        let top = match heap.pop() {
            Some(e) => e,
            None => break,
        };

        let cand_idx = top.idx;

        // Skip if already selected (could be leftover stale entries)
        if selected_set.contains(&cand_idx) {
            continue;
        }

        // current hv of selected set
        let hv_selected = if selected_rows.is_empty() {
            0.0
        } else {
            compute_hypervolume(&selected_rows, reference_point)
        };

        // recompute true marginal contribution for candidate
        let cand_row = *idx_to_row.get(&cand_idx).unwrap();

        // create tmp = selected_rows + candidate
        let mut tmp: Vec<&[f64]> = Vec::with_capacity(selected_rows.len() + 1);
        tmp.extend(selected_rows.iter().cloned());
        tmp.push(cand_row);
        let hv_with_cand = compute_hypervolume(&tmp, reference_point);
        let true_contrib = hv_with_cand - hv_selected;

        // If the stored (popped) contrib is sufficiently close to true_contrib, accept it.
        // Otherwise push back updated value and continue (lazy update).
        if (top.contrib - true_contrib).abs() <= EPS {
            // select cand
            selected_rows.push(cand_row);
            selected_indices.push(cand_idx);
            selected_set.insert(cand_idx);
        } else {
            // push updated contribution back into heap and continue
            heap.push(HeapEntry {
                contrib: true_contrib,
                idx: cand_idx,
            });
        }
    }

    selected_indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_non_dominated_sort_basic() {
        // simple 2D example
        let pts = [
            vec![0.0, 0.0], // 0 -> front 0
            vec![1.0, 0.0], // 1 dominated by 0 -> front 1
            vec![0.0, 1.0], // 2 dominated by 0 -> front 1
            vec![2.0, 2.0], // dominated by both -> front 2
            vec![0.5, 0.5], // non-dominated w.r.t 1 & 2 -> front 1
        ];
        let refs: Vec<&[f64]> = pts.iter().map(|r| r.as_slice()).collect();
        let ranks = fast_non_dominated_sort(&refs);
        // ranks[0] must be 0; others >=1
        assert_eq!(ranks[0], 0);
        assert!(ranks[1] > 0);
        assert!(ranks[2] > 0);
        assert!(ranks[3] > ranks[1] && ranks[3] > ranks[2]);
    }

    #[test]
    fn test_compute_hypervolume_float_simple() {
        let loss_vals = [vec![1.0, 1.0], vec![2.0, 0.5]];
        let refs: Vec<&[f64]> = loss_vals.iter().map(|v| v.as_slice()).collect();
        let reference_point = vec![3.0, 3.0];

        let hv = compute_hypervolume(&refs, &reference_point);
        assert!(hv > 0.0);
    }

    #[test]
    fn test_compute_hypervolume() {
        let loss_vals = [vec![1.0, 2.0],
            vec![2.0, 1.5],
            vec![1.5, 1.0]];
        let refs: Vec<&[f64]> = loss_vals.iter().map(|v| v.as_slice()).collect();
        let reference_point = vec![3.0, 3.0];

        let hv = compute_hypervolume(&refs, &reference_point);
        // Manually computed expected hypervolume
        let expected_hv = 3.5;
        assert!((hv - expected_hv).abs() < 1e-6, "hv: {}, expected: {}", hv, expected_hv);
    }

    #[test]
    fn greedy_hv_approx_minus_1_over_e_no_itertools() {
        fn combinations_recursive(
            n: usize,
            k: usize,
            start: usize,
            current: &mut Vec<usize>,
            out: &mut Vec<Vec<usize>>,
        ) {
            if current.len() == k {
                out.push(current.clone());
                return;
            }
            for i in start..n {
                current.push(i);
                combinations_recursive(n, k, i + 1, current, out);
                current.pop();
            }
        }

        // A simple problem instance
        let pts: Vec<[f64; 2]> = vec![
            [1.0, 4.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [4.0, 1.0],
            [2.5, 2.5],
            [3.5, 3.5],
        ];
        let n = pts.len();
        let refp = vec![6.0, 6.0];

        let rows_ref: Vec<Vec<f64>> = pts.iter().map(|p| vec![p[0], p[1]]).collect();
        let row_slices: Vec<&[f64]> = rows_ref.iter().map(|v| v.as_slice()).collect();

        let k = 2usize;

        // Exhaustive search
        let mut all_combs: Vec<Vec<usize>> = Vec::new();
        combinations_recursive(n, k, 0, &mut Vec::new(), &mut all_combs);

        let mut best_hv = 0.0;
        for comb in all_combs.iter() {
            let chosen: Vec<&[f64]> = comb.iter().map(|&i| row_slices[i]).collect();
            let hv = compute_hypervolume(&chosen, &refp);
            if hv > best_hv {
                best_hv = hv;
            }
        }

        // Greedy selection
        let rank_i_loss_vals: Vec<&[f64]> = row_slices.clone();
        let rank_i_indices: Vec<usize> = (0..n).collect();

        let greedy_sel =
            hypervolume_subset_selection(&rank_i_loss_vals, &rank_i_indices, &refp, k);
        let greedy_rows: Vec<&[f64]> =
            greedy_sel.iter().map(|&i| row_slices[i]).collect();
        let greedy_hv = compute_hypervolume(&greedy_rows, &refp);

        // Verify (1 - 1/e) approximation
        let bound = 1.0 - 1.0f64 / std::f64::consts::E;
        assert!(
            greedy_hv >= bound * best_hv,
            "greedy hv {} < (1-1/e)*opt {} (opt {})",
            greedy_hv,
            bound * best_hv,
            best_hv
        );
    }

}
