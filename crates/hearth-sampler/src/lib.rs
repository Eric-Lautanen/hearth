use std::cell::RefCell;

use hearth_core::SamplerConfig;
use rand::{rngs::SmallRng, Rng, SeedableRng};

thread_local! {
    static RNG: RefCell<SmallRng> = RefCell::new(SmallRng::from_os_rng());
}

fn softmax(logits: &mut [f32]) {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits
        .iter_mut()
        .map(|v| {
            *v = (*v - max_val).exp();
            *v
        })
        .sum();
    if sum > 0.0 {
        for v in logits.iter_mut() {
            *v /= sum;
        }
    }
}

fn apply_repetition_penalty(logits: &mut [f32], past_tokens: &[u32], penalty: f32) {
    if (penalty - 1.0).abs() < 1e-6 {
        return;
    }
    for &token in past_tokens {
        let idx = token as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature > 0.0 && (temperature - 1.0).abs() > 1e-6 {
        let inv_temp = 1.0 / temperature;
        for v in logits.iter_mut() {
            *v *= inv_temp;
        }
    }
}

fn apply_top_k(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() {
        return;
    }
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    // Safety: k < logits.len() is guaranteed by the guard above
    let (_, threshold, _) = indexed.select_nth_unstable_by(k, |(_, a), (_, b)| {
        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
    });
    let threshold_val = threshold.1;
    for v in logits.iter_mut() {
        if *v < threshold_val {
            *v = f32::NEG_INFINITY;
        }
    }
}

fn apply_top_p(probs: &mut [f32], p: f32) {
    if p >= 1.0 || p <= 0.0 {
        return;
    }
    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut cum_sum = 0.0;
    let mut cutoff_idx = indices.len();
    for (rank, &i) in indices.iter().enumerate() {
        cum_sum += probs[i];
        if cum_sum >= p {
            cutoff_idx = rank + 1;
            break;
        }
    }
    // Zero out everything outside the top-p nucleus
    for &i in &indices[cutoff_idx..] {
        probs[i] = 0.0;
    }
}

fn apply_min_p(logits: &mut [f32], min_p: f32) {
    if min_p <= 0.0 || min_p >= 1.0 {
        return;
    }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let threshold = max_val * min_p;
    for v in logits.iter_mut() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
}

fn apply_typical_p(logits: &mut [f32], typical_p: f32) {
    if typical_p <= 0.0 || typical_p >= 1.0 {
        return;
    }
    // Typical-p: keep tokens whose entropy contribution is below threshold
    // Uses the negative log-probability as the "surprisal" measure
    let entropy: f32 = logits
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();
    let threshold = -entropy * typical_p;
    for v in logits.iter_mut() {
        if *v > 0.0 && -*v * v.ln() > threshold {
            *v = f32::NEG_INFINITY;
        }
    }
}

fn sample_from_distribution(probs: &[f32]) -> u32 {
    let r = RNG.with(|rng| rng.borrow_mut().random::<f32>());
    let mut cum = 0.0f32;
    let mut best_id = 0usize;
    let mut best_prob = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        if p > best_prob {
            best_prob = p;
            best_id = i;
        }
        cum += p;
        if r < cum {
            return i as u32;
        }
    }
    // Fallback: return argmax (handles float precision shortfall)
    best_id as u32
}

/// Sample a token from logits using the given config.
///
/// Order of operations (standard HuggingFace/llama.cpp order):
/// 1. Repetition penalty (on logits)
/// 2. Temperature scaling (on logits)
/// 3. Top-K filtering (on logits)
/// 4. Top-P filtering (on logits)
/// 5. Min-P filtering (on logits)
/// 6. Typical-P filtering (on logits)
/// 7. Softmax → probabilities
/// 8. Sample from distribution
pub fn sample(logits: &mut [f32], config: &SamplerConfig, past_tokens: &[u32]) -> u32 {
    // Seed RNG if a non-zero seed is provided
    if config.seed != 0 {
        RNG.with(|rng| {
            *rng.borrow_mut() = SmallRng::seed_from_u64(config.seed);
        });
    }

    // 1. Repetition penalty (on logits)
    apply_repetition_penalty(logits, past_tokens, config.repeat_pen);

    // 2. Greedy shortcut
    if config.temperature == 0.0 {
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);
        return argmax;
    }

    // 3. Temperature (on logits)
    apply_temperature(logits, config.temperature);

    // 4. Top-K on logits (before softmax — filters low-logit tokens to -inf)
    apply_top_k(logits, config.top_k);

    // 5. Min-P on logits
    apply_min_p(logits, config.min_p);

    // 6. Softmax → probabilities
    softmax(logits);

    // 7. Top-P nucleus filtering (must run on probabilities that sum to 1)
    apply_top_p(logits, config.top_p);

    // 8. Typical-P (on probabilities)
    if config.typical_p > 0.0 {
        apply_typical_p(logits, config.typical_p);
    }

    // 9. Re-normalize after any filtering that zeroed tokens
    let sum: f32 = logits.iter().sum();
    if sum > 0.0 {
        for v in logits.iter_mut() {
            *v /= sum;
        }
    }

    // 10. Sample
    sample_from_distribution(logits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_argmax() {
        let mut logits = vec![-1.0, -2.0, 5.0, 0.5];
        let config = SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let token = sample(&mut logits, &config, &[]);
        assert_eq!(token, 2);
    }

    #[test]
    fn test_temperature_zero_equiv_greedy() {
        let mut logits = vec![0.1, 0.2, 0.7, 0.05];
        let config = SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let token = sample(&mut logits, &config, &[]);
        assert_eq!(token, 2);
    }

    #[test]
    fn test_sample_with_temperature() {
        let mut logits = vec![0.1, 0.2, 0.7, 0.05];
        let config = SamplerConfig {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repeat_pen: 1.0,
            ..Default::default()
        };
        let token = sample(&mut logits, &config, &[]);
        assert!(token < 4);
    }

    #[test]
    fn test_repetition_penalty() {
        let mut logits = vec![10.0, 5.0, 3.0];
        let config = SamplerConfig {
            temperature: 1.0,
            repeat_pen: 2.0,
            ..Default::default()
        };
        let token = sample(&mut logits, &config, &[0]);
        assert!(token < 3);
    }

    #[test]
    fn test_seeded_reproducibility() {
        let mut logits_a = vec![0.1, 0.2, 0.7, 0.05];
        let mut logits_b = vec![0.1, 0.2, 0.7, 0.05];
        let config = SamplerConfig {
            temperature: 1.0,
            seed: 42,
            ..Default::default()
        };
        let a = sample(&mut logits_a, &config, &[]);
        let b = sample(&mut logits_b, &config, &[]);
        assert_eq!(a, b, "Same seed should produce same token");
    }
}
