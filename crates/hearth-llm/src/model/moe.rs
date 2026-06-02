use super::LlamaModel;

impl LlamaModel {
    pub(crate) fn moe_forward(
        &self,
        p: &str,
        residual: &[f32],
        sc: &mut crate::model::scratch::ForwardScratch,
        rb: &mut Vec<f32>,
        layer: usize,
    ) -> Result<(), String> {
        let d = self.config.d_model as usize;
        let ffn_dim = self.config.d_ffn as usize;
        let n_experts = self.config.n_experts as usize;
        let n_experts_per_tok = self.config.n_experts_per_tok as usize;

        if sc.moe_gate.len() < n_experts {
            sc.moe_gate.resize(n_experts, 0.0f32);
        }
        if sc.moe_ffn.len() < ffn_dim {
            sc.moe_ffn.resize(ffn_dim, 0.0f32);
        }

        self.matmul(
            &format!("{}.ffn_gate_inp.weight", p),
            residual,
            &mut sc.moe_gate[..n_experts],
            rb,
            layer,
            None,
        )?;

        let mut expert_indices: Vec<usize> = (0..n_experts).collect();
        expert_indices.sort_by(|&a, &b| {
            sc.moe_gate[b]
                .partial_cmp(&sc.moe_gate[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_k: Vec<usize> = expert_indices[..n_experts_per_tok].to_vec();

        let max_gate = top_k
            .iter()
            .map(|&i| sc.moe_gate[i])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut weights = vec![0.0f32; n_experts_per_tok];
        let mut sum_w = 0.0f32;
        for (k, &ei) in top_k.iter().enumerate() {
            weights[k] = (sc.moe_gate[ei] - max_gate).exp();
            sum_w += weights[k];
        }
        if sum_w > 0.0 {
            for w in weights.iter_mut() {
                *w /= sum_w;
            }
        }

        sc.q_buf[..d].fill(0.0f32);

        for (k, &ei) in top_k.iter().enumerate() {
            self.matmul(
                &format!("{}.ffn_gate_exps.{}.weight", p, ei),
                residual,
                &mut sc.gate[..ffn_dim],
                rb,
                layer,
                None,
            )?;
            self.matmul(
                &format!("{}.ffn_up_exps.{}.weight", p, ei),
                residual,
                &mut sc.up[..ffn_dim],
                rb,
                layer,
                None,
            )?;

            crate::ops::silu(&mut sc.gate[..ffn_dim]);
            crate::ops::mul_elem(
                &sc.gate[..ffn_dim],
                &sc.up[..ffn_dim],
                &mut sc.ffn_tmp[..ffn_dim],
            );

            self.matmul(
                &format!("{}.ffn_down_exps.{}.weight", p, ei),
                &sc.ffn_tmp[..ffn_dim],
                &mut sc.moe_ffn[..ffn_dim],
                rb,
                layer,
                None,
            )?;

            let w = weights[k];
            for i in 0..d {
                sc.q_buf[i] += w * sc.moe_ffn[i];
            }
        }

        Ok(())
    }
}
