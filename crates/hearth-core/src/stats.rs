pub struct RunStats {
    pub elapsed_ms: u128,
    /// Tokens actually generated. The prompt is reported separately: a one-word
    /// reply to a 30-token chat prompt used to arrive as "31 tokens", which made
    /// an immediately-stopped generation look like a long reply.
    pub tokens: usize,
    /// Prompt tokens processed before the first generated token.
    pub prompt_tokens: usize,
    /// Generated tokens per second, averaged over the whole run (prefill
    /// included), so a slow model's real wait is visible rather than hidden.
    pub tokens_per_s: f32,
}
