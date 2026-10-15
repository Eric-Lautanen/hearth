#[derive(Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub struct TemplateContext {
    pub messages: Vec<ChatMessage>,
    pub bos_token: String,
    pub eos_token: String,
    pub add_generation_prompt: bool,
    /// Some templates render extra content (e.g. `<think>` tags) based on this.
    pub thinking: bool,
    /// Qwen3-style: when false, pre-close the think block so the model starts
    /// answering directly instead of reasoning inline.
    pub enable_thinking: bool,
}

pub fn eval_jinja2(template: &str, ctx: &TemplateContext) -> String {
    let mut out = String::new();
    let mut pos = 0;
    let bytes = template.as_bytes();

    while pos < bytes.len() {
        if pos + 1 < bytes.len() && &bytes[pos..pos + 2] == b"{{" {
            let end = find_close(&template[pos..], "}}");
            if let Some(end) = end {
                let expr = template[pos + 2..pos + end].trim();
                out.push_str(&eval_expr(expr, ctx));
                pos += end + 2;
                continue;
            }
        }
        if pos + 1 < bytes.len() && &bytes[pos..pos + 2] == b"{%" {
            let end = find_close(&template[pos..], "%}");
            if let Some(end) = end {
                let stmt = template[pos + 2..pos + end].trim();
                pos += end + 2;
                pos = eval_stmt(stmt, template, pos, ctx, &mut out);
                continue;
            }
        }
        if pos + 1 < bytes.len() && &bytes[pos..pos + 2] == b"{#" {
            if let Some(end) = find_close(&template[pos..], "#}") {
                pos += end + 2;
                continue;
            }
        }
        out.push(bytes[pos] as char);
        pos += 1;
    }

    out
}

fn find_close(template: &str, close: &str) -> Option<usize> {
    template.find(close)
}

fn eval_expr(expr: &str, ctx: &TemplateContext) -> String {
    let expr = expr.trim();
    if expr == "bos_token" {
        return ctx.bos_token.clone();
    }
    if expr == "eos_token" {
        return ctx.eos_token.clone();
    }
    if expr.starts_with("message['") || expr.starts_with("message[\"") {
        let key = extract_dict_key(expr);
        if let Some(msg) = ctx.messages.last() {
            match key {
                "role" => return msg.role.clone(),
                "content" => return msg.content.clone(),
                _ => return String::new(),
            }
        }
    }
    if expr.starts_with("loop.index") {
        return "1".to_string();
    }
    String::new()
}

fn extract_dict_key(expr: &str) -> &str {
    let start = if expr.starts_with("message['") || expr.starts_with("message[\"") {
        9
    } else {
        return "";
    };
    let end_quote = if expr.chars().nth(start) == Some('\'') || expr.chars().nth(start) == Some('"')
    {
        start
    } else {
        let rest = &expr[start..];
        if let Some(i) = rest.find("']") {
            start + i
        } else if let Some(i) = rest.find("\"]") {
            start + i
        } else {
            return "";
        }
    };
    if end_quote > start {
        &expr[start..end_quote]
    } else {
        ""
    }
}

fn eval_stmt(
    stmt: &str,
    template: &str,
    after_pos: usize,
    ctx: &TemplateContext,
    out: &mut String,
) -> usize {
    let stmt = stmt.trim();

    if stmt.starts_with("for ") {
        if let Some(rest) = stmt.strip_prefix("for ") {
            if let Some(in_pos) = rest.find(" in ") {
                let var = rest[..in_pos].trim();
                let iter_expr = rest[in_pos + 4..].trim_end();
                let iter_expr = iter_expr.trim_end_matches(':');
                return eval_for(var, iter_expr, template, after_pos, ctx, out);
            }
        }
    }

    if stmt.starts_with("if ") {
        let cond = stmt.strip_prefix("if ").unwrap_or("").trim_end_matches(':');
        return eval_if(cond, template, after_pos, ctx, out);
    }

    if stmt == "else" || stmt.starts_with("elif ") {
        if let Some(end_pos) = find_end_block(template, after_pos) {
            return end_pos;
        }
        if let Some(end_pos) = template[after_pos..].find("{% endif %}") {
            return after_pos + end_pos + "{% endif %}".len();
        }
    }

    after_pos
}

fn eval_for(
    _var: &str,
    iter_expr: &str,
    template: &str,
    after_pos: usize,
    ctx: &TemplateContext,
    out: &mut String,
) -> usize {
    let end_pos = if let Some(ep) = find_for_end(template, after_pos) {
        ep
    } else {
        return after_pos;
    };
    let body = &template[after_pos..end_pos];

    let iter_count = if iter_expr == "messages" {
        ctx.messages.len()
    } else {
        0
    };

    // Render the body once per message through the full evaluator with a
    // single-message context (`message[...]` reads the last message), so
    // nested ifs/interpolations inside the loop work instead of leaking as
    // literal text.
    for msg in ctx.messages.iter().take(iter_count) {
        let loop_ctx = TemplateContext {
            messages: vec![msg.clone()],
            bos_token: ctx.bos_token.clone(),
            eos_token: ctx.eos_token.clone(),
            add_generation_prompt: ctx.add_generation_prompt,
            thinking: ctx.thinking,
            enable_thinking: ctx.enable_thinking,
        };
        out.push_str(&eval_jinja2(body, &loop_ctx));
    }

    end_pos + "{% endfor %}".len()
}

fn eval_if(
    cond: &str,
    template: &str,
    after_pos: usize,
    ctx: &TemplateContext,
    out: &mut String,
) -> usize {
    // Split the if/elif/else/endif chain honoring nested conditionals, then
    // render the winning branch through the full evaluator so nested `{{ }}`
    // and `{% %}` constructs work instead of leaking as literal text.
    let Some((branches, body_end)) = parse_if_branches(cond, template, after_pos) else {
        return after_pos;
    };
    for (branch_cond, range) in &branches {
        let take = match branch_cond {
            Some(c) => eval_cond_global(c, ctx),
            None => true, // {% else %}
        };
        if take {
            out.push_str(&eval_jinja2(&template[range.0..range.1], ctx));
            break;
        }
    }
    body_end
}

/// One branch of an `{% if %}` chain: its condition (`None` for `{% else %}`)
/// and the body's byte range in the template.
type IfBranch = (Option<String>, (usize, usize));

/// Splits an `{% if %}…{% endif %}` chain into branches, tracking nesting so
/// inner if/endif pairs don't terminate the outer block. Returns one branch
/// pair per branch plus the position just past the matching `{% endif %}`.
fn parse_if_branches(
    first_cond: &str,
    template: &str,
    after_pos: usize,
) -> Option<(Vec<IfBranch>, usize)> {
    let bytes = template.as_bytes();
    let mut depth = 1usize;
    let mut pos = after_pos;
    let mut branch_start = after_pos;
    let mut current_cond = Some(first_cond.trim().to_string());
    let mut branches: Vec<(Option<String>, (usize, usize))> = Vec::new();

    while pos < bytes.len() {
        if pos + 1 < bytes.len() && &bytes[pos..pos + 2] == b"{%" {
            if let Some(end) = find_close(&template[pos..], "%}") {
                let stmt = template[pos + 2..pos + end].trim();
                if stmt.starts_with("if ") {
                    depth += 1;
                } else if stmt == "endif" {
                    depth -= 1;
                    if depth == 0 {
                        branches.push((current_cond.take(), (branch_start, pos)));
                        return Some((branches, pos + end + 2));
                    }
                } else if depth == 1 && stmt == "else" {
                    branches.push((current_cond.take(), (branch_start, pos)));
                    current_cond = None;
                    branch_start = pos + end + 2;
                } else if depth == 1 && stmt.starts_with("elif ") {
                    branches.push((current_cond.take(), (branch_start, pos)));
                    current_cond =
                        Some(stmt.strip_prefix("elif ").unwrap_or("").trim().to_string());
                    branch_start = pos + end + 2;
                }
                pos += end + 2;
                continue;
            }
        }
        pos += 1;
    }
    None
}

fn eval_cond(cond: &str, msg: &ChatMessage) -> bool {
    let cond = cond.trim();
    if cond.contains("message['role']") || cond.contains("message[\"role\"]") {
        if let Some(eq_pos) = cond.find("==") {
            let expected = cond[eq_pos + 2..]
                .trim()
                .trim_matches('\'')
                .trim_matches('"');
            return msg.role == expected;
        }
        if let Some(ne_pos) = cond.find("!=") {
            let expected = cond[ne_pos + 2..]
                .trim()
                .trim_matches('\'')
                .trim_matches('"');
            return msg.role != expected;
        }
    }
    if (cond.contains("message['content']") || cond.contains("message[\"content\"]"))
        && (cond.contains("is not none") || cond.contains("is defined"))
    {
        return !msg.content.is_empty();
    }
    false
}

fn eval_cond_global(cond: &str, ctx: &TemplateContext) -> bool {
    let cond = cond.trim();
    // Message-relative conditions (used inside `{% for message in messages %}`
    // loops, where the context holds exactly one message).
    if cond.contains("message['") || cond.contains("message[\"") {
        return ctx.messages.last().is_some_and(|msg| eval_cond(cond, msg));
    }
    if cond == "add_generation_prompt" {
        return ctx.add_generation_prompt;
    }
    if cond == "enable_thinking" {
        return ctx.enable_thinking;
    }
    if cond == "true" {
        return true;
    }
    if cond == "false" {
        return false;
    }
    if cond.contains("and") {
        let parts: Vec<&str> = cond.splitn(2, " and ").collect();
        if parts.len() == 2 {
            return eval_cond_global(parts[0].trim(), ctx)
                && eval_cond_global(parts[1].trim(), ctx);
        }
    }
    if cond.contains("or") {
        let parts: Vec<&str> = cond.splitn(2, " or ").collect();
        if parts.len() == 2 {
            return eval_cond_global(parts[0].trim(), ctx)
                || eval_cond_global(parts[1].trim(), ctx);
        }
    }
    if cond.contains("not ") {
        let inner = cond.strip_prefix("not ").unwrap_or(cond);
        return !eval_cond_global(inner, ctx);
    }
    if cond.contains("messages|length") || cond.contains("messages | length") {
        if let Some(eq_pos) = cond.find("==") {
            let expected: usize = cond[eq_pos + 2..].trim().parse().unwrap_or(0);
            return ctx.messages.len() == expected;
        }
        if let Some(gt_pos) = cond.find(">") {
            let expected: usize = cond[gt_pos + 1..].trim().parse().unwrap_or(0);
            return ctx.messages.len() > expected;
        }
    }
    true
}

fn find_for_end(template: &str, start: usize) -> Option<usize> {
    let mut depth = 1;
    let mut pos = start;
    while pos < template.len() {
        if pos + 1 < template.len() && &template.as_bytes()[pos..pos + 2] == b"{%" {
            if let Some(end) = find_close(&template[pos..], "%}") {
                let stmt = template[pos + 2..pos + end].trim();
                if stmt.starts_with("for ") {
                    depth += 1;
                } else if stmt == "endfor" {
                    depth -= 1;
                    if depth == 0 {
                        return Some(pos);
                    }
                }
                pos += end + 2;
                continue;
            }
        }
        pos += 1;
    }
    None
}

fn find_end_block(template: &str, start: usize) -> Option<usize> {
    let mut pos = start;
    while pos < template.len() {
        if pos + 1 < template.len() && &template.as_bytes()[pos..pos + 2] == b"{%" {
            if let Some(end) = find_close(&template[pos..], "%}") {
                let stmt = template[pos + 2..pos + end].trim();
                if stmt == "endif"
                    || stmt == "endfor"
                    || stmt == "else"
                    || stmt.starts_with("elif ")
                {
                    return Some(pos);
                }
                pos += end + 2;
                continue;
            }
        }
        pos += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_interpolation() {
        let ctx = TemplateContext {
            messages: vec![],
            bos_token: "<s>".to_string(),
            eos_token: "</s>".to_string(),
            add_generation_prompt: true,
            thinking: false,
            enable_thinking: true,
        };
        let result = eval_jinja2("{{ bos_token }}hello{{ eos_token }}", &ctx);
        assert_eq!(result, "<s>hello</s>");
    }

    #[test]
    fn test_for_loop() {
        let ctx = TemplateContext {
            messages: vec![
                ChatMessage {
                    role: "user".to_string(),
                    content: "hi".to_string(),
                },
                ChatMessage {
                    role: "assistant".to_string(),
                    content: "hello".to_string(),
                },
            ],
            bos_token: "<s>".to_string(),
            eos_token: "</s>".to_string(),
            add_generation_prompt: true,
            thinking: false,
            enable_thinking: true,
        };
        let template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}";
        let result = eval_jinja2(template, &ctx);
        assert!(result.contains("user: hi"));
        assert!(result.contains("assistant: hello"));
    }

    #[test]
    fn test_if_condition() {
        let ctx = TemplateContext {
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "test".to_string(),
            }],
            bos_token: "".to_string(),
            eos_token: "".to_string(),
            add_generation_prompt: true,
            thinking: false,
            enable_thinking: true,
        };
        let template = "{% if add_generation_prompt %}GEN{% endif %}";
        let result = eval_jinja2(template, &ctx);
        assert!(result.contains("GEN"));
    }
}
