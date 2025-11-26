#![feature(test)]

extern crate test;

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use test::Bencher;

    use rustuna_core::storage::InMemoryStorage;
    use rustuna_core::study::{create_study, Direction};
    use rustuna_samplers::tpe::TpeSampler;

    #[bench]
    fn bench_tpe(b: &mut Bencher) {
        b.iter(|| {
            let directions = vec![Direction::Minimize];
            let storage = InMemoryStorage::new();
            let sampler = Arc::new(Mutex::new(TpeSampler::new()));
            let mut study = create_study("dummy", storage, directions).unwrap();
            study
                .optimize(
                    |mut t| {
                        let x = t.suggest_float("x", 0.0, 10.0)?;
                        let y = t.suggest_float("y", 0.0, 10.0)?;
                        let z = t.suggest_int("z", 0, 10)?;

                        let value = (x - 3.0).powi(2) + (y - 5.0).powi(2) + (z as f64);
                        Ok(vec![value])
                    },
                    sampler,
                    100,
                )
                .unwrap();
        })
    }
}
