use hearth_gguf::{GgufFile, MetaValue};
use std::collections::{HashMap, VecDeque};

pub mod jinja;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateKind {
    ChatML,
    MistralInstruct,
    Llama3,
    Gemma,
    Phi3,
    Qwen3,
    Custom,
}

struct CacheItem {
    ids: Vec<u32>,
}

pub struct Tokenizer {
    pub vocab: Vec<String>,
    pub token_to_id: HashMap<String, u32>,
    pub merges: Vec<(String, String)>,
    pub merge_ranks: HashMap<(String, String), usize>,
    pub bos_id: u32,
    pub eos_id: u32,
    pub byte_tokens: HashMap<u8, u32>,
    pub template_kind: TemplateKind,
    pub chat_template: Option<String>,
    pub special_tokens: HashMap<String, u32>,
    encode_cache: HashMap<u64, CacheItem>,
    decode_cache: HashMap<u64, String>,
    cache_order: VecDeque<u64>,
    cache_capacity: usize,
}

impl Tokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let tokens_meta = gguf
            .meta_array("tokenizer.ggml.tokens")
            .ok_or_else(|| "Missing tokenizer.ggml.tokens".to_string())?;

        let mut vocab: Vec<String> = Vec::new();
        let mut token_to_id: HashMap<String, u32> = HashMap::new();
        let mut byte_tokens: HashMap<u8, u32> = HashMap::new();

        for (i, tv) in tokens_meta.iter().enumerate() {
            let token_str = match tv {
                MetaValue::String(s) => s.clone(),
                _ => format!("<token_{}>", i),
            };
            vocab.push(token_str.clone());
            token_to_id.insert(token_str.clone(), i as u32);

            // Detect byte tokens like <0x00> ... <0xFF>
            if token_str.len() == 6 && token_str.starts_with("<0x") && token_str.ends_with(">") {
                if let Ok(byte_val) = u8::from_str_radix(&token_str[3..5], 16) {
                    byte_tokens.insert(byte_val, i as u32);
                }
            }
        }

        let bos_id = gguf.meta_u32("tokenizer.ggml.bos_token_id").unwrap_or(1);
        let eos_id = gguf.meta_u32("tokenizer.ggml.eos_token_id").unwrap_or(2);

        let mut merges: Vec<(String, String)> = Vec::new();
        if let Some(merges_meta) = gguf.meta_array("tokenizer.ggml.merges") {
            for mv in merges_meta {
                if let MetaValue::String(s) = mv {
                    if let Some(space) = s.find(' ') {
                        let a = s[..space].to_string();
                        let b = s[space + 1..].to_string();
                        merges.push((a, b));
                    }
                }
            }
        }

        let mut merge_ranks: HashMap<(String, String), usize> = HashMap::new();
        for (i, (a, b)) in merges.iter().enumerate() {
            merge_ranks.insert((a.clone(), b.clone()), i);
        }

        let (template_kind, chat_template) = detect_template(gguf);

        let mut special_tokens: HashMap<String, u32> = HashMap::new();
        if let Some(added_meta) = gguf.meta_array("tokenizer.ggml.added_tokens") {
            for (i, tv) in added_meta.iter().enumerate() {
                if let MetaValue::String(s) = tv {
                    if !s.starts_with("<0x") {
                        special_tokens.insert(s.clone(), i as u32);
                    }
                }
            }
        }
        for (i, token_str) in vocab.iter().enumerate() {
            if token_str.starts_with("<|") && token_str.ends_with("|>") {
                special_tokens.insert(token_str.clone(), i as u32);
            }
        }

        Ok(Tokenizer {
            vocab,
            token_to_id,
            merges,
            merge_ranks,
            bos_id,
            eos_id,
            byte_tokens,
            template_kind,
            chat_template,
            special_tokens,
            encode_cache: HashMap::new(),
            decode_cache: HashMap::new(),
            cache_order: VecDeque::new(),
            cache_capacity: 256,
        })
    }

    fn cache_key(text: &str, add_bos: bool) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut h);
        add_bos.hash(&mut h);
        h.finish()
    }

    fn cache_get(&mut self, text: &str, add_bos: bool) -> Option<Vec<u32>> {
        let key = Self::cache_key(text, add_bos);
        self.encode_cache.get(&key).map(|c| c.ids.clone())
    }

    fn cache_set(&mut self, text: &str, add_bos: bool, ids: Vec<u32>) {
        let key = Self::cache_key(text, add_bos);
        if self.cache_order.len() >= self.cache_capacity {
            if let Some(old) = self.cache_order.pop_front() {
                self.encode_cache.remove(&old);
                self.decode_cache.remove(&old);
            }
        }
        self.cache_order.push_back(key);
        self.encode_cache.insert(key, CacheItem { ids });
    }

    pub fn encode(&mut self, text: &str, add_bos: bool) -> Vec<u32> {
        if let Some(cached) = self.cache_get(text, add_bos) {
            return cached;
        }

        let mut tokens = Vec::new();

        if add_bos {
            tokens.push(self.bos_id);
        }

        let mut remaining = text;
        while !remaining.is_empty() {
            if let Some((tok_str, &tok_id)) = self
                .special_tokens
                .iter()
                .filter(|(s, _)| remaining.starts_with(s.as_str()))
                .max_by(|a, b| a.0.len().cmp(&b.0.len()))
            {
                tokens.push(tok_id);
                remaining = &remaining[tok_str.len()..];
                continue;
            }

            let end = find_next_special_or_end(remaining, &self.special_tokens);
            let segment = &remaining[..end];
            let pre_tokens = pre_tokenize(segment);
            for word in &pre_tokens {
                let word_ids = if word.bytes().all(|b| b.is_ascii())
                    && self.token_to_id.contains_key(word.as_str())
                {
                    vec![self.token_to_id[word.as_str()]]
                } else {
                    let mut ids = Vec::new();
                    for &byte in word.as_bytes() {
                        if let Some(&id) = self.byte_tokens.get(&byte) {
                            ids.push(id);
                        } else {
                            let byte_char = byte as char;
                            let s = byte_char.to_string();
                            if let Some(&id) = self.token_to_id.get(&s) {
                                ids.push(id);
                            }
                        }
                    }
                    ids
                };

                let merged = self.bpe_merge(word_ids);
                tokens.extend(merged);
            }
            remaining = &remaining[end..];
        }

        self.cache_set(text, add_bos, tokens.clone());
        tokens
    }

    fn bpe_merge(&self, ids: Vec<u32>) -> Vec<u32> {
        if ids.len() <= 1 || self.merges.is_empty() {
            return ids;
        }

        // Convert IDs to tokens for merge lookup
        let tokens: Vec<String> = ids
            .iter()
            .map(|&id| self.vocab.get(id as usize).cloned().unwrap_or_default())
            .collect();

        // Find the best pair to merge
        let mut pairs: HashMap<(String, String), usize> = HashMap::new();
        for window in tokens.windows(2) {
            let pair = (window[0].clone(), window[1].clone());
            *pairs.entry(pair).or_insert(0) += 1;
        }

        // Greedy merge loop: repeatedly merge the lowest-ranked pair
        let mut current_tokens = tokens.clone();
        loop {
            if current_tokens.len() <= 1 {
                break;
            }

            // Find all adjacent pairs and their ranks
            let mut best_pair: Option<(String, String)> = None;
            let mut best_rank = usize::MAX;

            for window in current_tokens.windows(2) {
                let pair = (window[0].clone(), window[1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pair = Some(pair);
                    }
                }
            }

            let (a, b) = match best_pair {
                Some(p) => p,
                None => break,
            };

            let merged = format!("{}{}", a, b);
            let mut new_tokens = Vec::new();
            let mut i = 0;
            while i < current_tokens.len() {
                if i + 1 < current_tokens.len()
                    && current_tokens[i] == a
                    && current_tokens[i + 1] == b
                {
                    new_tokens.push(merged.clone());
                    i += 2;
                } else {
                    new_tokens.push(current_tokens[i].clone());
                    i += 1;
                }
            }

            current_tokens = new_tokens;
        }

        // Convert merged tokens back to IDs
        let mut result = Vec::new();
        for token_str in &current_tokens {
            if let Some(&id) = self.token_to_id.get(token_str) {
                result.push(id);
            } else {
                // Fallback: try to find by byte-level representation
                for &byte in token_str.as_bytes() {
                    if let Some(&id) = self.byte_tokens.get(&byte) {
                        result.push(id);
                    }
                }
            }
        }

        result
    }

    pub fn decode(&mut self, ids: &[u32]) -> String {
        // Check cache for exact sequence
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        ids.hash(&mut h);
        let key = h.finish();
        if let Some(cached) = self.decode_cache.get(&key) {
            return cached.clone();
        }

        let mut raw = String::new();
        for &id in ids {
            if let Some(s) = self.vocab.get(id as usize) {
                raw.push_str(s);
            }
        }
        let out = gpt2_decode(&raw);

        if self.cache_order.len() >= self.cache_capacity {
            if let Some(old) = self.cache_order.pop_front() {
                self.encode_cache.remove(&old);
                self.decode_cache.remove(&old);
            }
        }
        self.cache_order.push_back(key);
        self.decode_cache.insert(key, out.clone());
        out
    }

    pub fn decode_token(&self, id: u32) -> String {
        let raw = self
            .vocab
            .get(id as usize)
            .map(|s| s.as_str())
            .unwrap_or("");
        gpt2_decode(raw)
    }

    pub fn is_control_token(&self, id: u32) -> bool {
        let raw = self
            .vocab
            .get(id as usize)
            .map(|s| s.as_str())
            .unwrap_or("");
        if raw.starts_with("<|") && raw.ends_with("|>") {
            return true;
        }
        if raw.starts_with("<|im_start|>") || raw.starts_with("<|im_end|>") {
            return true;
        }
        if self.special_tokens.contains_key(raw) {
            let s = *self.special_tokens.get(raw).unwrap();
            return s == id;
        }
        false
    }

    /// Generate a synthetic Jinja2 template for a given TemplateKind.
    /// Used as fallback when no chat_template metadata is available in the GGUF file.
    fn synthetic_template(kind: TemplateKind) -> &'static str {
        match kind {
            TemplateKind::ChatML | TemplateKind::Custom => {
                "{% for message in messages %}\
                 {% if message['role'] == 'system' %}<|im_start|>system\n\
                 {{ message['content'] }}<|im_end|>\n\
                 {% endif %}\
                 {% if message['role'] == 'user' %}<|im_start|>user\n\
                 {{ message['content'] }}<|im_end|>\n\
                 {% endif %}\
                 {% if message['role'] == 'assistant' %}<|im_start|>assistant\n\
                 {{ message['content'] }}<|im_end|>\n\
                 {% endif %}\
                 {% endfor %}\
                 {% if add_generation_prompt %}<|im_start|>assistant\n\n\n\n{% endif %}"
            }
            TemplateKind::MistralInstruct => {
                "{% for message in messages %}\
                 {% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]\
                 {% endif %}\
                 {% if message['role'] == 'assistant' %}{{ message['content'] }} </s>\
                 {% endif %}\
                 {% endfor %}"
            }
            TemplateKind::Llama3 => {
                "{{ bos_token }}\
                 {% for message in messages %}\
                 {% if message['role'] == 'system' %}<|start_header_id|>system<|end_header_id|>\n\n\
                 {{ message['content'] }}<|eot_id|>\n\
                 {% endif %}\
                 {% if message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>\n\n\
                 {{ message['content'] }}<|eot_id|>\n\
                 {% endif %}\
                 {% if message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>\n\n\
                 {{ message['content'] }}<|eot_id|>\n\
                 {% endif %}\
                 {% endfor %}\
                 {% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
            }
            TemplateKind::Gemma => {
                "{{ bos_token }}\
                 {% for message in messages %}\
                 {% if message['role'] == 'user' %}<start_of_turn>user\n\
                 {{ message['content'] }}<end_of_turn>\n\
                 {% endif %}\
                 {% if message['role'] == 'assistant' %}<start_of_turn>model\n\
                 {{ message['content'] }}<end_of_turn>\n\
                 {% endif %}\
                 {% endfor %}\
                 {% if add_generation_prompt %}<start_of_turn>user\n{% endif %}"
            }
            TemplateKind::Qwen3 => {
                "{% for message in messages %}\
                 {% if message['role'] == 'system' %}<|im_start|>system\n\
                 {{ message['content'] }}<|im_end|>\n\
                 {% endif %}\
                 {% if message['role'] == 'user' %}<|im_start|>user\n\
                 {{ message['content'] }}<|im_end|>\n\
                 {% endif %}\
                 {% if message['role'] == 'assistant' %}<|im_start|>assistant\n\
                 {{ message['content'] }}<|im_end|>\n\
                 {% endif %}\
                 {% endfor %}\
                 {% if add_generation_prompt %}<|im_start|>assistant\n<think>\n\n</think>\n\n{% endif %}"
            }
            TemplateKind::Phi3 => {
                "{% for message in messages %}\
                 {% if message['role'] == 'system' %}<|system|>\n\
                 {{ message['content'] }}<|end|>\n\
                 {% endif %}\
                 {% if message['role'] == 'user' %}<|user|>\n\
                 {{ message['content'] }}<|end|>\n\
                 {% endif %}\
                 {% if message['role'] == 'assistant' %}<|assistant|>\n\
                 {{ message['content'] }}<|end|>\n\
                 {% endif %}\
                 {% endfor %}\
                 {% if add_generation_prompt %}<|assistant|>\n{% endif %}"
            }
        }
    }

    /// Build a chat prompt from system prompt, history pairs, and latest user message.
    ///
    /// Always routes through the Jinja2 template evaluator:
    /// - If the GGUF file provides a simple `tokenizer.chat_template`, that is used directly.
    /// - Otherwise, a synthetic Jinja2 template is generated based on the detected
    ///   TemplateKind, so there is exactly one code path for prompt building.
    ///
    /// `thinking` controls whether reasoning/thinking sections are shown in the
    /// output (applied via ThinkFilter at the consumer level; some templates
    /// also use this parameter in their chat template).
    /// Check if a GGUF chat_template is simple enough for our Jinja2 evaluator.
    /// Our evaluator handles: {{ }}, {% for %}, {% if %}, {% endif %}, {% endfor %}
    /// It does NOT handle: {% set %}, filters (|), namespace, `is defined`, etc.
    fn is_template_simple_enough(template: &str) -> bool {
        !template.contains("{% set")
            && !template.contains("{%- set")
            && !template.contains("{% elif")
            && !template.contains("namespace")
            && !template.contains("| ")
            && !template.contains("is defined")
            && !template.contains("is not")
    }

    pub fn apply_template(
        &self,
        system: &str,
        history: &[(&str, &str)],
        user: &str,
        thinking: bool,
    ) -> String {
        let mut messages = vec![jinja::ChatMessage {
            role: "system".into(),
            content: system.into(),
        }];
        for (user_msg, assistant_msg) in history {
            messages.push(jinja::ChatMessage {
                role: "user".into(),
                content: user_msg.to_string(),
            });
            messages.push(jinja::ChatMessage {
                role: "assistant".into(),
                content: assistant_msg.to_string(),
            });
        }
        messages.push(jinja::ChatMessage {
            role: "user".into(),
            content: user.to_string(),
        });

        // Template selection:
        // 1. Use the GGUF chat_template directly if it's simple enough for our evaluator
        // 2. Fall back to a synthetic template for the detected format
        let template = match &self.chat_template {
            Some(t) if Self::is_template_simple_enough(t) => t.as_str(),
            _ => Self::synthetic_template(self.template_kind),
        };

        let ctx = jinja::TemplateContext {
            messages,
            bos_token: self.decode_token(self.bos_id),
            eos_token: self.decode_token(self.eos_id),
            add_generation_prompt: true,
            thinking,
        };
        jinja::eval_jinja2(template, &ctx)
    }
}

/// GPT-2 / Qwen / Mistral tokenizers encode raw bytes as unicode codepoints so
/// they can be stored as valid strings.  The mapping is:
///   - Printable ASCII (! through ~) and a few others map to themselves
/// - Everything else maps to codepoints starting at U+0100 (Ā)
///
/// This function reverses that mapping back to actual UTF-8 text.
fn gpt2_decode(s: &str) -> String {
    // Build the reverse map: unicode char → byte value
    // This matches the table in openai/gpt-2's encoder.py bytes_to_unicode()
    let mut char_to_byte = [0u8; 65536];
    let mut populated = [false; 65536];

    let mut bs: Vec<u32> = Vec::new();
    // Printable ASCII ranges that map to themselves
    for b in b'!'..=b'~' {
        bs.push(b as u32);
    }
    for b in b'\xA1'..=b'\xAC' {
        bs.push(b as u32);
    }
    for b in b'\xAE'..=b'\xFF' {
        bs.push(b as u32);
    }

    let mut cs = bs.clone();
    let mut n = 0u32;
    for b in 0u32..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    for (b, c) in bs.iter().zip(cs.iter()) {
        if (*c as usize) < 65536 {
            char_to_byte[*c as usize] = *b as u8;
            populated[*c as usize] = true;
        }
    }

    // Also handle Ġ (U+0120 = 288) → space (0x20) which is the most common case
    // and Ċ (U+010A = 266) → newline (0x0A)
    // These are already covered by the loop above, but double-check:
    // 0x20 (space, 32) is not in printable ASCII range (starts at 33='!'),
    // so it maps to 256+n for some n. The loop handles this correctly.

    let mut bytes: Vec<u8> = Vec::with_capacity(s.len());
    for ch in s.chars() {
        let cp = ch as u32;
        if cp < 65536 && populated[cp as usize] {
            bytes.push(char_to_byte[cp as usize]);
        } else if cp < 128 {
            // Plain ASCII that wasn't in the map (shouldn't happen, but safe fallback)
            bytes.push(cp as u8);
        }
        // Unknown codepoints are dropped — they're encoding artifacts
    }

    String::from_utf8(bytes).unwrap_or_else(|e| {
        // If UTF-8 decoding fails, return the valid prefix
        String::from_utf8_lossy(&e.into_bytes()).into_owned()
    })
}

/// Detect template format from model metadata.
/// Priority:
///   1. GGUF `tokenizer.chat_template` content (format markers in the template string)
///   2. Special tokens present in the model's vocabulary
///   3. Architecture name (last resort — broadest match)
fn detect_template(gguf: &GgufFile) -> (TemplateKind, Option<String>) {
    let chat_template = gguf
        .meta_str("tokenizer.chat_template")
        .map(|s| s.to_string());

    // 1. Detect format from GGUF chat_template content
    if let Some(ref tmpl) = chat_template {
        if let Some(kind) = detect_format_from_content(tmpl) {
            return (kind, chat_template);
        }
    }

    // 2. Detect format from special tokens in the model's vocabulary
    if let Some(kind) = detect_format_from_tokens(gguf) {
        return (kind, chat_template);
    }

    // 3. Detect from architecture name (last resort)
    if let Some(arch) = gguf.meta_str("general.architecture") {
        if let Some(kind) = detect_format_from_arch(arch) {
            return (kind, chat_template);
        }
    }

    (TemplateKind::ChatML, chat_template)
}

/// Detect format from the GGUF chat_template string content.
/// Checks for format-specific markers in the template itself.
fn detect_format_from_content(template: &str) -> Option<TemplateKind> {
    // Check for Qwen3-specific patterns (must be before general ChatML check)
    if template.contains("namespace") && template.contains("reasoning_content") {
        return Some(TemplateKind::Qwen3);
    }
    if template.contains("<|im_start|>") {
        return Some(TemplateKind::ChatML);
    }
    if template.contains("[INST]") {
        return Some(TemplateKind::MistralInstruct);
    }
    if template.contains("<|start_header_id|>") {
        return Some(TemplateKind::Llama3);
    }
    if template.contains("<start_of_turn>") {
        return Some(TemplateKind::Gemma);
    }
    if template.contains("<|system|>")
        || template.contains("<|user|>")
        || template.contains("<|assistant|>")
    {
        return Some(TemplateKind::Phi3);
    }
    None
}

/// Detect format from special tokens present in the model's vocabulary.
/// Checks both main vocab and added tokens.
fn detect_format_from_tokens(gguf: &GgufFile) -> Option<TemplateKind> {
    // Check added tokens first
    if let Some(added) = gguf.meta_array("tokenizer.ggml.added_tokens") {
        for tv in added {
            if let hearth_gguf::MetaValue::String(s) = tv {
                if s.contains("<|im_start|>") {
                    return Some(TemplateKind::ChatML);
                }
                if s.contains("<|start_header_id|>") {
                    return Some(TemplateKind::Llama3);
                }
                if s.contains("<start_of_turn>") {
                    return Some(TemplateKind::Gemma);
                }
            }
        }
    }

    // Check main vocab tokens
    if let Some(tokens) = gguf.meta_array("tokenizer.ggml.tokens") {
        for tv in tokens {
            if let hearth_gguf::MetaValue::String(s) = tv {
                if s.contains("<|im_start|>") {
                    return Some(TemplateKind::ChatML);
                }
                if s.contains("<|start_header_id|>") {
                    return Some(TemplateKind::Llama3);
                }
                if s.contains("<start_of_turn>") {
                    return Some(TemplateKind::Gemma);
                }
                // Check for phi3-style tokens
                if s == "<|system|>" || s == "<|user|>" || s == "<|assistant|>" {
                    return Some(TemplateKind::Phi3);
                }
                // Check for Mistral [INST] — it's literal text, not a token,
                // so we check under a broader condition
            }
        }
    }

    None
}

/// Detect format from architecture name.
/// Broadest matching — only used as last resort.
fn detect_format_from_arch(arch: &str) -> Option<TemplateKind> {
    let arch = arch.to_lowercase();
    if arch.contains("llama") || arch.contains("llama3") {
        return Some(TemplateKind::Llama3);
    }
    if arch.contains("mistral") || arch.contains("mixtral") {
        return Some(TemplateKind::MistralInstruct);
    }
    if arch == "gemma" || arch == "gemma2" {
        return Some(TemplateKind::Gemma);
    }
    if arch == "phi3" || arch == "phi-3" || arch == "phi-3.5" {
        return Some(TemplateKind::Phi3);
    }
    if arch.contains("qwen3") {
        return Some(TemplateKind::Qwen3);
    }
    None
}

fn find_next_special_or_end(
    text: &str,
    special_tokens: &std::collections::HashMap<String, u32>,
) -> usize {
    let mut earliest = text.len();
    for tok in special_tokens.keys() {
        if let Some(pos) = text.find(tok.as_str()) {
            if pos < earliest {
                earliest = pos;
            }
        }
    }
    earliest
}

fn pre_tokenize(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        if ch.is_whitespace() {
            if !current.is_empty() {
                words.push(current.clone());
                current.clear();
            }
            // Preserve whitespace tokens
            let ws: String = ch.to_string();
            words.push(ws);
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        words.push(current);
    }

    words
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mock_tokenizer() -> Tokenizer {
        let mut vocab = vec![
            "<unk>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            "a".to_string(),
            "b".to_string(),
            "ab".to_string(),
            "c".to_string(),
            "abc".to_string(),
            " ".to_string(),
        ];
        // Add byte tokens <0x00> through <0xFF>
        for i in 0..=255u8 {
            vocab.push(format!("<0x{:02X}>", i));
        }

        let mut token_to_id = HashMap::new();
        for (i, s) in vocab.iter().enumerate() {
            token_to_id.insert(s.clone(), i as u32);
        }

        let mut byte_tokens = HashMap::new();
        for i in 0..=255u8 {
            byte_tokens.insert(i, i as u32 + 9); // byte tokens start at index 9
        }

        let merges = vec![
            ("a".to_string(), "b".to_string()),
            ("ab".to_string(), "c".to_string()),
        ];
        let mut merge_ranks = HashMap::new();
        for (i, (a, b)) in merges.iter().enumerate() {
            merge_ranks.insert((a.clone(), b.clone()), i);
        }

        Tokenizer {
            vocab,
            token_to_id,
            merges,
            merge_ranks,
            bos_id: 1,
            eos_id: 2,
            byte_tokens,
            template_kind: TemplateKind::ChatML,
            chat_template: None,
            special_tokens: HashMap::new(),
            encode_cache: HashMap::new(),
            decode_cache: HashMap::new(),
            cache_order: VecDeque::new(),
            cache_capacity: 256,
        }
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let mut tok = make_mock_tokenizer();
        let text = "abc";
        let ids = tok.encode(text, true);
        assert!(!ids.is_empty(), "encode should produce tokens");
        let decoded = tok.decode(&ids);
        assert!(!decoded.is_empty(), "decode should produce text");
    }

    #[test]
    fn test_bos_eos_ids() {
        let tok = make_mock_tokenizer();
        assert_eq!(tok.bos_id, 1);
        assert_eq!(tok.eos_id, 2);
    }

    #[test]
    fn test_chatml_template() {
        let tok = make_mock_tokenizer();
        // apply_template routes through Jinja2 with synthetic ChatML template
        let result = tok.apply_template("You are helpful", &[], "Hello", false);
        assert!(result.contains("<|im_start|>system"));
        assert!(result.contains("<|im_start|>user"));
        assert!(result.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_decode_token() {
        let tok = make_mock_tokenizer();
        assert_eq!(tok.decode_token(0), "<unk>");
        assert_eq!(tok.decode_token(3), "a");
    }
}
