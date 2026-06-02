/// State machine for filtering `<think>` tags from a stream of text tokens.
///
/// Handles both:
/// - Standard models that emit `<think>` / `</think>` inline in text tokens
/// - Models (like Bonsai) that emit `<think>` as a separate special token
///
/// Usage:
/// ```ignore
/// let mut filter = ThinkFilter::new();
/// for token in token_stream {
///     let clean = filter.process(&token);
///     if !clean.is_empty() {
///         output(clean);
///     }
/// }
/// ```
pub struct ThinkFilter {
    in_think: bool,
}

impl ThinkFilter {
    pub fn new() -> Self {
        Self { in_think: false }
    }

    /// Process a single text token, returning only the text that should be
    /// shown to the user (think-tag content is stripped).
    pub fn process(&mut self, token: &str) -> String {
        if self.in_think {
            if let Some(pos) = token.find("</think>") {
                self.in_think = false;
                token[pos + 8..].to_string()
            } else {
                String::new()
            }
        } else {
            if let Some(pos) = token.find("<think>") {
                self.in_think = true;
                token[..pos].to_string()
            } else {
                token.to_string()
            }
        }
    }
}

impl Default for ThinkFilter {
    fn default() -> Self {
        Self::new()
    }
}
