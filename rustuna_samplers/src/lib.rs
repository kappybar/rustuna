//! Sampler implementations for Rustuna.
//!
//! This crate provides concrete implementations of the `rustuna_core::sampler::Sampler` trait,
//! including TPE-based samplers and NSGA-II for multi-objective optimization.

pub mod nsgaii;
pub mod tpe;
