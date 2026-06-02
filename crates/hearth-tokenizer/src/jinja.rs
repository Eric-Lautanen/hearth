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
    pub thinking: bool,
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
    var: &str,
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

    for idx in 0..iter_count {
        let loop_ctx = TemplateContext {
            messages: ctx.messages.clone(),
            bos_token: ctx.bos_token.clone(),
            eos_token: ctx.eos_token.clone(),
            add_generation_prompt: ctx.add_generation_prompt,
            thinking: ctx.thinking,
        };
        let _ = (var, idx);
        let mut pos = 0;
        let body_bytes = body.as_bytes();
        while pos < body_bytes.len() {
            if pos + 1 < body_bytes.len() && &body_bytes[pos..pos + 2] == b"{{" {
                if let Some(end) = find_close(&body[pos..], "}}") {
                    let expr = body[pos + 2..pos + end].trim();
                    if expr == "message['role']" || expr == "message[\"role\"]" {
                        out.push_str(&loop_ctx.messages[idx].role);
                    } else if expr == "message['content']" || expr == "message[\"content\"]" {
                        out.push_str(&loop_ctx.messages[idx].content);
                    } else if expr == "bos_token" {
                        out.push_str(&loop_ctx.bos_token);
                    } else if expr == "eos_token" {
                        out.push_str(&loop_ctx.eos_token);
                    } else if expr == "loop.index" {
                        out.push_str(&format!("{}", idx + 1));
                    }
                    pos += end + 2;
                    continue;
                }
            }
            if pos + 1 < body_bytes.len() && &body_bytes[pos..pos + 2] == b"{%" {
                if let Some(end) = find_close(&body[pos..], "%}") {
                    let stmt = body[pos + 2..pos + end].trim();
                    if stmt.starts_with("if ") {
                        let cond = stmt.strip_prefix("if ").unwrap_or("").trim_end_matches(':');
                        let cond_result = eval_cond(cond, &loop_ctx.messages[idx]);
                        let if_body_end = find_if_end_simple(body, pos + end + 2);
                        let if_body = &body[pos + end + 2..if_body_end];
                        if cond_result {
                            let mut ip = 0;
                            let ib = if_body.as_bytes();
                            while ip < ib.len() {
                                if ip + 1 < ib.len() && &ib[ip..ip + 2] == b"{{" {
                                    if let Some(e) = find_close(&if_body[ip..], "}}") {
                                        let expr = if_body[ip + 2..ip + e].trim();
                                        if expr == "message['role']" || expr == "message[\"role\"]"
                                        {
                                            out.push_str(&loop_ctx.messages[idx].role);
                                        } else if expr == "message['content']"
                                            || expr == "message[\"content\"]"
                                        {
                                            out.push_str(&loop_ctx.messages[idx].content);
                                        } else if expr == "bos_token" {
                                            out.push_str(&loop_ctx.bos_token);
                                        } else if expr == "eos_token" {
                                            out.push_str(&loop_ctx.eos_token);
                                        }
                                        ip += e + 2;
                                        continue;
                                    }
                                }
                                out.push(ib[ip] as char);
                                ip += 1;
                            }
                        }
                        pos = if_body_end;
                        if pos < body_bytes.len() && body[pos..].starts_with("{% endif %}") {
                            pos += "{% endif %}".len();
                        }
                        continue;
                    }
                    pos += end + 2;
                    continue;
                }
            }
            if pos + 1 < body_bytes.len() && &body_bytes[pos..pos + 2] == b"{#" {
                if let Some(end) = find_close(&body[pos..], "#}") {
                    pos += end + 2;
                    continue;
                }
            }
            out.push(body_bytes[pos] as char);
            pos += 1;
        }
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
    let cond_result = eval_cond_global(cond, ctx);

    let else_pos = template[after_pos..].find("{% else %}");
    let endif_pos = template[after_pos..].find("{% endif %}");

    let (if_end, body_end) = match (else_pos, endif_pos) {
        (Some(ep), Some(eip)) if ep < eip => {
            (after_pos + ep, after_pos + eip + "{% endif %}".len())
        }
        (None, Some(eip)) => (after_pos + eip, after_pos + eip + "{% endif %}".len()),
        _ => return after_pos,
    };

    if cond_result {
        let body = &template[after_pos..if_end];
        out.push_str(body);
    } else if let Some(ep) = else_pos {
        if let Some(eip) = endif_pos {
            if ep < eip {
                let else_body = &template[after_pos + ep + "{% else %}".len()..after_pos + eip];
                out.push_str(else_body);
            }
        }
    }

    body_end
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
    if cond == "add_generation_prompt" {
        return ctx.add_generation_prompt;
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

fn find_if_end_simple(body: &str, start: usize) -> usize {
    let mut depth = 1;
    let mut pos = start;
    while pos < body.len() {
        if pos + 1 < body.len() && &body.as_bytes()[pos..pos + 2] == b"{%" {
            if let Some(end) = find_close(&body[pos..], "%}") {
                let stmt = body[pos + 2..pos + end].trim();
                if stmt.starts_with("if ") {
                    depth += 1;
                } else if stmt == "endif" {
                    depth -= 1;
                    if depth == 0 {
                        return pos;
                    }
                }
                pos += end + 2;
                continue;
            }
        }
        pos += 1;
    }
    body.len()
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
        };
        let template = "{% if add_generation_prompt %}GEN{% endif %}";
        let result = eval_jinja2(template, &ctx);
        assert!(result.contains("GEN"));
    }
}
