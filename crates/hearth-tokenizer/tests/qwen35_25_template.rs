//! The Ternary Bonsai 2.5 (Qwen3.5) chat template, rendered against the exact
//! output of the reference Jinja engine: jinja2 3.1 inside
//! `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)`, which
//! is the environment `transformers` builds for `apply_chat_template`.
//!
//! The expected files are the model card's render of that template. This model
//! used to fall back to the synthetic Qwen3 template because its template uses
//! `set`, `namespace` and `reasoning_effort`; if the evaluator drifts, the
//! prompt the flagship model sees drifts with it.

use hearth_tokenizer::jinja::{eval_jinja2, ChatMessage, TemplateContext};

const TEMPLATE: &str = include_str!("fixtures/qwen35_25/template.jinja");
const SYSTEM: &str = "You are a helpful assistant.";

fn context(messages: &[(&str, &str)], thinking: bool, effort: Option<&str>) -> TemplateContext {
    TemplateContext {
        messages: messages
            .iter()
            .map(|(role, content)| ChatMessage {
                role: role.to_string(),
                content: content.to_string(),
            })
            .collect(),
        bos_token: String::new(),
        eos_token: String::new(),
        add_generation_prompt: true,
        thinking,
        enable_thinking: thinking,
        reasoning_effort: effort.map(str::to_string),
    }
}

fn assert_matches_reference(
    messages: &[(&str, &str)],
    thinking: bool,
    effort: Option<&str>,
    expected: &str,
) {
    let rendered = eval_jinja2(TEMPLATE, &context(messages, thinking, effort))
        .expect("the 2.5 template must render");
    assert_eq!(rendered, expected);
}

#[test]
fn system_and_user_thinking() {
    assert_matches_reference(
        &[("system", SYSTEM), ("user", "Hello")],
        true,
        None,
        include_str!("fixtures/qwen35_25/system_user.thinking.txt"),
    );
}

#[test]
fn system_and_user_without_thinking() {
    assert_matches_reference(
        &[("system", SYSTEM), ("user", "Hello")],
        false,
        None,
        include_str!("fixtures/qwen35_25/system_user.nothinking.txt"),
    );
}

#[test]
fn empty_system_still_gets_the_reasoning_line() {
    assert_matches_reference(
        &[("system", ""), ("user", "Hello")],
        true,
        None,
        include_str!("fixtures/qwen35_25/no_system.thinking.txt"),
    );
}

#[test]
fn history_gets_a_closed_think_block() {
    assert_matches_reference(
        &[
            ("system", "You are helpful"),
            ("user", "What is 2+2?"),
            ("assistant", "Four."),
            ("user", "And 3+3?"),
        ],
        true,
        None,
        include_str!("fixtures/qwen35_25/history.thinking.txt"),
    );
}

#[test]
fn reasoning_effort_low_is_honoured() {
    assert_matches_reference(
        &[("system", "You are helpful"), ("user", "Hello")],
        true,
        Some("low"),
        include_str!("fixtures/qwen35_25/effort_low.thinking.txt"),
    );
}
