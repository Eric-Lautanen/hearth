//! A Jinja2 subset evaluator, sized for chat templates.
//!
//! Chat templates in the wild are written for the Jinja2 environment
//! `transformers` builds (`trim_blocks=True`, `lstrip_blocks=True`), so this
//! module implements that dialect rather than inventing one:
//!
//! - statements: `for` (with `else`), `if`/`elif`/`else`, `set` (plain and
//!   `ns.attr`), `macro`
//! - expressions: `or`/`and`/`not`, comparisons (`==`, `!=`, `<`, `>`, `<=`,
//!   `>=`, `in`, `not in`), `is`/`is not` tests, `~`, `+`/`-`/`*`/`//`/`%`,
//!   filters, attribute and subscript access, slices (`messages[::-1]`), calls
//!   with positional and keyword arguments, and inline `a if cond else b`
//! - filters: `trim`, `length`, `tojson`, `items`, `string`, `safe`,
//!   `default`, `join`, `first`, `last`, `lower`, `upper`, `replace`
//! - whitespace control: `{{-`, `-}}`, `{%-`, `-%}`, plus the two flags the
//!   reference environment sets
//!
//! Values are dynamic (`Value`) so templates can pass messages around, build
//! namespaces, and call macros. Errors are values: [`eval_jinja2`] returns the
//! first unsupported construct or `raise_exception` message instead of
//! rendering a half-filled prompt.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

type R<T> = Result<T, String>;

/// Positional and keyword arguments of a call or filter.
type CallArgs = (Vec<Expr>, Vec<(String, Expr)>);

/// `trim_blocks=True` in the reference environment: a newline directly after a
/// block tag is not part of the output.
const TRIM_BLOCKS: bool = true;
/// `lstrip_blocks=True`: whitespace between the start of a line and a block tag
/// is dropped.
const LSTRIP_BLOCKS: bool = true;

/// Maximum macro call depth. Chat templates nest one or two levels; the cap
/// only exists so a self-recursive template fails instead of overflowing the
/// stack.
const MAX_CALL_DEPTH: usize = 32;

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
    /// Some templates render extra content (e.g. ` thinking` tags) based on this.
    pub thinking: bool,
    /// Qwen3-style: when false, pre-close the think block so the model starts
    /// answering directly instead of reasoning inline.
    pub enable_thinking: bool,
    /// `reasoning_effort` as the model card's template spells it (`xhigh`,
    /// `medium`, `low`). `None` leaves the name undefined, which is exactly
    /// what the reference render does when the caller does not pass it — the
    /// template's `|default('xhigh')` then applies.
    pub reasoning_effort: Option<String>,
}

/// A mutable, ordered string-keyed map. Templates only ever need small maps
/// (messages, namespaces, tool call arguments), and insertion order is what
/// `|items` and `tojson` iterate.
#[derive(Clone, Debug, Default)]
struct Map {
    entries: Vec<(String, Value)>,
}

impl Map {
    fn with(entries: Vec<(String, Value)>) -> Self {
        Map { entries }
    }

    fn get(&self, key: &str) -> Option<&Value> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    fn set(&mut self, key: &str, value: Value) {
        if let Some(slot) = self.entries.iter_mut().find(|(k, _)| k == key) {
            slot.1 = value;
        } else {
            self.entries.push((key.to_string(), value));
        }
    }
}

#[derive(Clone, Debug)]
enum Value {
    /// An unset name. Distinct from `Null`: `is defined` is false, `default`
    /// replaces it, and rendering it yields nothing.
    Undefined,
    Null,
    Bool(bool),
    Int(i64),
    Str(Rc<str>),
    List(Rc<Vec<Value>>),
    Map(Rc<RefCell<Map>>),
}

impl Value {
    fn str(s: impl Into<String>) -> Self {
        Value::Str(Rc::from(s.into()))
    }

    fn list(items: Vec<Value>) -> Self {
        Value::List(Rc::new(items))
    }

    fn map(entries: Vec<(String, Value)>) -> Self {
        Value::Map(Rc::new(RefCell::new(Map::with(entries))))
    }

    fn truthy(&self) -> bool {
        match self {
            Value::Undefined | Value::Null => false,
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Str(s) => !s.is_empty(),
            Value::List(l) => !l.is_empty(),
            Value::Map(m) => !m.borrow().entries.is_empty(),
        }
    }

    /// What `{{ value }}` prints: strings verbatim, everything else as Python's
    /// `str()` would.
    fn text(&self) -> String {
        match self {
            Value::Str(s) => s.to_string(),
            other => other.repr(),
        }
    }

    /// Python's `repr()`, for values that appear inside a rendered expression
    /// rather than as text.
    fn repr(&self) -> String {
        match self {
            Value::Undefined => String::new(),
            Value::Null => "None".to_string(),
            Value::Bool(true) => "True".to_string(),
            Value::Bool(false) => "False".to_string(),
            Value::Int(i) => i.to_string(),
            Value::Str(s) => format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'")),
            Value::List(items) => {
                let parts: Vec<String> = items.iter().map(|v| v.repr()).collect();
                format!("[{}]", parts.join(", "))
            }
            Value::Map(map) => {
                let map = map.borrow();
                let parts: Vec<String> = map
                    .entries
                    .iter()
                    .map(|(k, v)| format!("'{k}': {}", v.repr()))
                    .collect();
                format!("{{{}}}", parts.join(", "))
            }
        }
    }

    fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::Bool(b) => Some(i64::from(*b)),
            _ => None,
        }
    }

    fn items_of(&self) -> Option<Vec<Value>> {
        match self {
            Value::List(items) => Some(items.as_ref().clone()),
            Value::Map(map) => Some(
                map.borrow()
                    .entries
                    .iter()
                    .map(|(k, _)| Value::str(k.clone()))
                    .collect(),
            ),
            Value::Str(s) => Some(s.chars().map(|c| Value::str(c.to_string())).collect()),
            Value::Undefined | Value::Null => Some(Vec::new()),
            Value::Int(_) | Value::Bool(_) => None,
        }
    }
}

fn values_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Undefined, Value::Undefined) => true,
        (Value::Null, Value::Null) => true,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Int(_), Value::Bool(_)) | (Value::Bool(_), Value::Int(_)) => {
            a.as_int() == b.as_int()
        }
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::List(x), Value::List(y)) => {
            x.len() == y.len() && x.iter().zip(y.iter()).all(|(l, r)| values_eq(l, r))
        }
        (Value::Map(x), Value::Map(y)) => {
            let (x, y) = (x.borrow(), y.borrow());
            x.entries.len() == y.entries.len()
                && x.entries
                    .iter()
                    .all(|(k, v)| y.get(k).is_some_and(|other| values_eq(v, other)))
        }
        _ => false,
    }
}

fn json_of(value: &Value) -> String {
    match value {
        Value::Str(s) => {
            let mut out = String::with_capacity(s.len() + 2);
            out.push('"');
            for ch in s.chars() {
                match ch {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                    c => out.push(c),
                }
            }
            out.push('"');
            out
        }
        Value::Bool(true) => "true".to_string(),
        Value::Bool(false) => "false".to_string(),
        Value::Int(i) => i.to_string(),
        Value::Null | Value::Undefined => "null".to_string(),
        Value::List(items) => {
            let parts: Vec<String> = items.iter().map(json_of).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::Map(map) => {
            let map = map.borrow();
            let parts: Vec<String> = map
                .entries
                .iter()
                .map(|(k, v)| format!("{}: {}", json_of(&Value::str(k.clone())), json_of(v)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

// ---------------------------------------------------------------------------
// Template tokenizer
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum TagKind {
    Output,
    Statement,
    Comment,
}

enum Seg {
    Text(String),
    Output(String),
    Statement(String),
}

/// Applies the trim a tag owes to the text that follows it: `-%}`/`-}}`
/// strips all leading whitespace, `trim_blocks` strips exactly one newline.
fn apply_right_strip(text: &mut String, strip: Option<bool>) {
    match strip {
        Some(true) => {
            let kept = text.trim_start();
            if kept.len() != text.len() {
                *text = kept.to_string();
            }
        }
        Some(false) => {
            if let Some(rest) = text
                .strip_prefix("\r\n")
                .or_else(|| text.strip_prefix('\n'))
            {
                *text = rest.to_string();
            }
        }
        None => {}
    }
}

/// Splits the template into text and tag segments, applying whitespace control
/// exactly where Jinja does: `{%-`/`{{-` trim what came before, `-%}`/`-}}`
/// trim what comes after, and the `trim_blocks`/`lstrip_blocks` flags handle
/// the rest.
fn tokenize_template(src: &str) -> R<Vec<Seg>> {
    let bytes = src.as_bytes();
    let mut segs: Vec<Seg> = Vec::new();
    let mut text = String::new();
    let mut pos = 0usize;
    // Right-side trimming owed to the text segment that comes next:
    // `Some(true)` strips all whitespace, `Some(false)` strips one newline.
    let mut strip_next: Option<bool> = None;

    while pos < bytes.len() {
        let kind = if bytes[pos..].starts_with(b"{{") {
            Some(TagKind::Output)
        } else if bytes[pos..].starts_with(b"{%") {
            Some(TagKind::Statement)
        } else if bytes[pos..].starts_with(b"{#") {
            Some(TagKind::Comment)
        } else {
            None
        };

        let Some(kind) = kind else {
            let ch = src[pos..].chars().next().ok_or("bad template encoding")?;
            text.push(ch);
            pos += ch.len_utf8();
            continue;
        };

        apply_right_strip(&mut text, strip_next.take());

        let strip_left = bytes.get(pos + 2) == Some(&b'-');
        let body_start = pos + 2 + usize::from(strip_left);
        let close = match kind {
            TagKind::Output => "}}",
            TagKind::Statement => "%}",
            TagKind::Comment => "#}",
        };
        let close_rel = src[body_start..]
            .find(close)
            .ok_or_else(|| format!("unclosed `{close}` tag"))?;
        let close_pos = body_start + close_rel;
        let raw_body = &src[body_start..close_pos];
        let trimmed_body = raw_body.trim_end();
        let strip_right = trimmed_body.ends_with('-');
        let body = if strip_right {
            trimmed_body[..trimmed_body.len() - 1].trim()
        } else {
            raw_body.trim()
        };

        if !text.is_empty() {
            if kind == TagKind::Statement && LSTRIP_BLOCKS {
                match text.rfind('\n') {
                    Some(nl) if text[nl + 1..].chars().all(|c| c == ' ' || c == '\t') => {
                        text.truncate(nl + 1);
                    }
                    None if text.chars().all(|c| c == ' ' || c == '\t') => text.clear(),
                    _ => {}
                }
            }
            if strip_left {
                let kept = text.trim_end().len();
                text.truncate(kept);
            }
            if !text.is_empty() {
                segs.push(Seg::Text(std::mem::take(&mut text)));
            }
        }

        pos = close_pos + 2;
        match kind {
            TagKind::Output => segs.push(Seg::Output(body.to_string())),
            TagKind::Statement => segs.push(Seg::Statement(body.to_string())),
            TagKind::Comment => {}
        }
        strip_next = if strip_right {
            Some(true)
        } else if kind == TagKind::Statement && TRIM_BLOCKS {
            Some(false)
        } else {
            None
        };
    }

    // A template that ends in text still owes the trim from the last tag.
    apply_right_strip(&mut text, strip_next.take());
    if !text.is_empty() {
        segs.push(Seg::Text(text));
    }
    Ok(segs)
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Int(i64),
    Str(String),
    Name(String),
    Op(String),
}

const TWO_CHAR_OPS: [&str; 6] = ["**", "//", "==", "!=", "<=", ">="];
const ONE_CHAR_OPS: [&str; 17] = [
    "(", ")", "[", "]", "{", "}", ".", ",", ":", "|", "~", "+", "-", "*", "/", "%", "<",
];

fn lex_expr(src: &str) -> R<Vec<Tok>> {
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0usize;
    let mut out = Vec::new();

    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }

        if c == '\'' || c == '"' {
            let quote = c;
            i += 1;
            let mut s = String::new();
            while i < chars.len() && chars[i] != quote {
                if chars[i] == '\\' && i + 1 < chars.len() {
                    i += 1;
                    s.push(match chars[i] {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        '0' => '\0',
                        other => other,
                    });
                } else {
                    s.push(chars[i]);
                }
                i += 1;
            }
            if i >= chars.len() {
                return Err(format!("unterminated string in `{src}`"));
            }
            i += 1;
            out.push(Tok::Str(s));
            continue;
        }

        if c.is_ascii_digit() {
            let start = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            let digits: String = chars[start..i].iter().collect();
            let n = digits
                .parse::<i64>()
                .map_err(|_| format!("bad number `{digits}`"))?;
            out.push(Tok::Int(n));
            continue;
        }

        if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            out.push(Tok::Name(chars[start..i].iter().collect()));
            continue;
        }

        let two: String = chars[i..chars.len().min(i + 2)].iter().collect();
        if let Some(op) = TWO_CHAR_OPS.iter().find(|op| **op == two) {
            out.push(Tok::Op((*op).to_string()));
            i += 2;
            continue;
        }
        let one = c.to_string();
        if ONE_CHAR_OPS.contains(&one.as_str()) || one == ">" || one == "=" {
            out.push(Tok::Op(one));
            i += 1;
            continue;
        }

        return Err(format!("unexpected character `{c}` in `{src}`"));
    }

    Ok(out)
}

#[derive(Clone, Debug)]
enum UnOp {
    Not,
    Neg,
}

#[derive(Clone, Copy, Debug)]
enum BinOp {
    Add,
    Sub,
    Mul,
    FloorDiv,
    Mod,
    Concat,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    In,
    NotIn,
}

#[derive(Clone, Debug)]
enum Expr {
    Lit(Value),
    Name(String),
    Attr(Box<Expr>, String),
    Index(Box<Expr>, Box<Expr>),
    Slice {
        target: Box<Expr>,
        start: Option<Box<Expr>>,
        stop: Option<Box<Expr>>,
        step: Option<Box<Expr>>,
    },
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
        kwargs: Vec<(String, Expr)>,
    },
    Filter {
        value: Box<Expr>,
        name: String,
        args: Vec<Expr>,
        kwargs: Vec<(String, Expr)>,
    },
    Unary(UnOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Cond {
        then: Box<Expr>,
        test: Box<Expr>,
        otherwise: Option<Box<Expr>>,
    },
    ListLit(Vec<Expr>),
    Test {
        value: Box<Expr>,
        name: String,
        negated: bool,
    },
}

struct ExprParser {
    toks: Vec<Tok>,
    pos: usize,
}

impl ExprParser {
    fn new(toks: Vec<Tok>) -> Self {
        ExprParser { toks, pos: 0 }
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn eat_op(&mut self, op: &str) -> bool {
        if let Some(Tok::Op(found)) = self.peek() {
            if found == op {
                self.pos += 1;
                return true;
            }
        }
        false
    }

    fn eat_name(&mut self, name: &str) -> bool {
        if let Some(Tok::Name(found)) = self.peek() {
            if found == name {
                self.pos += 1;
                return true;
            }
        }
        false
    }

    fn expect_op(&mut self, op: &str) -> R<()> {
        if self.eat_op(op) {
            Ok(())
        } else {
            Err(format!("expected `{op}`"))
        }
    }

    fn name(&mut self) -> R<String> {
        match self.peek() {
            Some(Tok::Name(n)) => {
                let n = n.clone();
                self.pos += 1;
                Ok(n)
            }
            other => Err(format!("expected a name, found {other:?}")),
        }
    }

    /// Inline conditional is the lowest-precedence form: `a if cond else b`.
    fn expression(&mut self) -> R<Expr> {
        let then = self.or_expr()?;
        if self.eat_name("if") {
            let test = self.or_expr()?;
            let otherwise = if self.eat_name("else") {
                Some(Box::new(self.expression()?))
            } else {
                None
            };
            return Ok(Expr::Cond {
                then: Box::new(then),
                test: Box::new(test),
                otherwise,
            });
        }
        Ok(then)
    }

    fn or_expr(&mut self) -> R<Expr> {
        let mut left = self.and_expr()?;
        while self.eat_name("or") {
            let right = self.and_expr()?;
            left = Expr::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn and_expr(&mut self) -> R<Expr> {
        let mut left = self.not_expr()?;
        while self.eat_name("and") {
            let right = self.not_expr()?;
            left = Expr::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn not_expr(&mut self) -> R<Expr> {
        if self.eat_name("not") {
            let inner = self.not_expr()?;
            return Ok(Expr::Unary(UnOp::Not, Box::new(inner)));
        }
        self.compare()
    }

    fn compare(&mut self) -> R<Expr> {
        let mut left = self.concat()?;
        loop {
            if self.eat_name("is") {
                let negated = self.eat_name("not");
                let name = self.name()?;
                left = Expr::Test {
                    value: Box::new(left),
                    name,
                    negated,
                };
                continue;
            }
            if self.eat_name("not") {
                if self.eat_name("in") {
                    let right = self.concat()?;
                    left = Expr::Binary(BinOp::NotIn, Box::new(left), Box::new(right));
                    continue;
                }
                return Err("`not` must be followed by `in` here".to_string());
            }
            if self.eat_name("in") {
                let right = self.concat()?;
                left = Expr::Binary(BinOp::In, Box::new(left), Box::new(right));
                continue;
            }
            let op = if self.eat_op("==") {
                BinOp::Eq
            } else if self.eat_op("!=") {
                BinOp::Ne
            } else if self.eat_op("<=") {
                BinOp::Le
            } else if self.eat_op(">=") {
                BinOp::Ge
            } else if self.eat_op("<") {
                BinOp::Lt
            } else if self.eat_op(">") {
                BinOp::Gt
            } else {
                break;
            };
            let right = self.concat()?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn concat(&mut self) -> R<Expr> {
        let mut left = self.add_expr()?;
        while self.eat_op("~") {
            let right = self.add_expr()?;
            left = Expr::Binary(BinOp::Concat, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn add_expr(&mut self) -> R<Expr> {
        let mut left = self.mul_expr()?;
        loop {
            let op = if self.eat_op("+") {
                BinOp::Add
            } else if self.eat_op("-") {
                BinOp::Sub
            } else {
                break;
            };
            let right = self.mul_expr()?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn mul_expr(&mut self) -> R<Expr> {
        let mut left = self.unary()?;
        loop {
            let op = if self.eat_op("*") {
                BinOp::Mul
            } else if self.eat_op("//") {
                BinOp::FloorDiv
            } else if self.eat_op("%") {
                BinOp::Mod
            } else {
                break;
            };
            let right = self.unary()?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn unary(&mut self) -> R<Expr> {
        if self.eat_op("-") {
            let inner = self.unary()?;
            return Ok(Expr::Unary(UnOp::Neg, Box::new(inner)));
        }
        if self.eat_op("+") {
            return self.unary();
        }
        self.filtered()
    }

    /// Filters bind tighter than arithmetic: `messages|length - 1` is
    /// `(messages|length) - 1`, which is how the reference parses it.
    fn filtered(&mut self) -> R<Expr> {
        let mut value = self.postfix()?;
        while self.eat_op("|") {
            let name = self.name()?;
            let (args, kwargs) = if self.eat_op("(") {
                self.call_args()?
            } else {
                (Vec::new(), Vec::new())
            };
            value = Expr::Filter {
                value: Box::new(value),
                name,
                args,
                kwargs,
            };
        }
        Ok(value)
    }

    fn postfix(&mut self) -> R<Expr> {
        let mut value = self.primary()?;
        loop {
            if self.eat_op(".") {
                let field = self.name()?;
                value = Expr::Attr(Box::new(value), field);
                continue;
            }
            if self.eat_op("[") {
                value = self.subscript(value)?;
                continue;
            }
            if self.eat_op("(") {
                let (args, kwargs) = self.call_args()?;
                value = Expr::Call {
                    callee: Box::new(value),
                    args,
                    kwargs,
                };
                continue;
            }
            break;
        }
        Ok(value)
    }

    fn subscript(&mut self, target: Expr) -> R<Expr> {
        let start = if self.eat_op(":") {
            None
        } else {
            let e = self.expression()?;
            if !self.eat_op(":") {
                self.expect_op("]")?;
                return Ok(Expr::Index(Box::new(target), Box::new(e)));
            }
            Some(Box::new(e))
        };
        let stop = if self.eat_op("]") || self.eat_op(":") {
            None
        } else {
            let e = self.expression()?;
            if self.eat_op("]") {
                return Ok(Expr::Slice {
                    target: Box::new(target),
                    start,
                    stop: Some(Box::new(e)),
                    step: None,
                });
            }
            self.expect_op(":")?;
            Some(Box::new(e))
        };
        let step = if self.eat_op("]") {
            None
        } else {
            let e = self.expression()?;
            self.expect_op("]")?;
            Some(Box::new(e))
        };
        Ok(Expr::Slice {
            target: Box::new(target),
            start,
            stop,
            step,
        })
    }

    /// Parses arguments after an opening `(` up to its `)`.
    fn call_args(&mut self) -> R<CallArgs> {
        let mut args = Vec::new();
        let mut kwargs = Vec::new();
        if self.eat_op(")") {
            return Ok((args, kwargs));
        }
        loop {
            // `name=expr` is a keyword argument; `name == expr` is not.
            let keyword = match (self.peek(), self.toks.get(self.pos + 1)) {
                (Some(Tok::Name(n)), Some(Tok::Op(op))) if op == "=" => Some(n.clone()),
                _ => None,
            };
            if let Some(name) = keyword {
                self.pos += 2;
                let value = self.expression()?;
                kwargs.push((name, value));
            } else {
                args.push(self.expression()?);
            }
            if self.eat_op(",") {
                continue;
            }
            self.expect_op(")")?;
            return Ok((args, kwargs));
        }
    }

    fn primary(&mut self) -> R<Expr> {
        match self.peek().cloned() {
            Some(Tok::Int(n)) => {
                self.pos += 1;
                Ok(Expr::Lit(Value::Int(n)))
            }
            Some(Tok::Str(s)) => {
                self.pos += 1;
                Ok(Expr::Lit(Value::str(s)))
            }
            Some(Tok::Name(n)) => {
                self.pos += 1;
                Ok(match n.as_str() {
                    "true" | "True" => Expr::Lit(Value::Bool(true)),
                    "false" | "False" => Expr::Lit(Value::Bool(false)),
                    "none" | "None" | "null" => Expr::Lit(Value::Null),
                    _ => Expr::Name(n),
                })
            }
            Some(Tok::Op(op)) if op == "(" => {
                self.pos += 1;
                let inner = self.expression()?;
                if self.eat_op(",") {
                    let mut items = vec![inner];
                    while !self.eat_op(")") {
                        items.push(self.expression()?);
                        if !self.eat_op(",") {
                            self.expect_op(")")?;
                            break;
                        }
                    }
                    return Ok(Expr::ListLit(items));
                }
                self.expect_op(")")?;
                Ok(inner)
            }
            Some(Tok::Op(op)) if op == "[" => {
                self.pos += 1;
                let mut items = Vec::new();
                while !self.eat_op("]") {
                    items.push(self.expression()?);
                    if !self.eat_op(",") {
                        self.expect_op("]")?;
                        break;
                    }
                }
                Ok(Expr::ListLit(items))
            }
            other => Err(format!("unexpected token {other:?}")),
        }
    }
}

fn parse_expr(src: &str) -> R<Expr> {
    let toks = lex_expr(src)?;
    if toks.is_empty() {
        return Err(format!("empty expression `{src}`"));
    }
    let mut parser = ExprParser::new(toks);
    let expr = parser.expression()?;
    if parser.pos != parser.toks.len() {
        return Err(format!("trailing tokens in `{src}`"));
    }
    Ok(expr)
}

// ---------------------------------------------------------------------------
// Template tree
// ---------------------------------------------------------------------------

struct MacroDef {
    name: String,
    params: Vec<(String, Option<Expr>)>,
    body: Vec<Node>,
}

enum SetTarget {
    Name(String),
    Attr { base: String, field: String },
}

/// One branch of an `{% if %}` chain: its condition (`None` for `{% else %}`)
/// and its body.
type Branch = (Option<Expr>, Vec<Node>);

enum Node {
    Text(String),
    Output(Expr),
    If(Vec<Branch>),
    For {
        vars: Vec<String>,
        iter: Expr,
        body: Vec<Node>,
        otherwise: Vec<Node>,
    },
    Set {
        target: SetTarget,
        value: Expr,
    },
    Macro(Rc<MacroDef>),
}

struct TemplateParser {
    segs: Vec<Seg>,
    pos: usize,
}

/// Splits `text` on `sep` at nesting depth 0, so argument lists inside a
/// parenthesised expression stay together.
fn split_top_level(text: &str, sep: char) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut depth = 0i32;
    let mut quote: Option<char> = None;
    for ch in text.chars() {
        if let Some(q) = quote {
            current.push(ch);
            if ch == q {
                quote = None;
            }
            continue;
        }
        match ch {
            '\'' | '"' => {
                quote = Some(ch);
                current.push(ch);
            }
            '(' | '[' | '{' => {
                depth += 1;
                current.push(ch);
            }
            ')' | ']' | '}' => {
                depth -= 1;
                current.push(ch);
            }
            c if c == sep && depth == 0 => parts.push(std::mem::take(&mut current)),
            c => current.push(c),
        }
    }
    if !current.trim().is_empty() {
        parts.push(current);
    }
    parts
}

fn head_word(stmt: &str) -> &str {
    stmt.split_whitespace().next().unwrap_or("")
}

impl TemplateParser {
    fn new(segs: Vec<Seg>) -> Self {
        TemplateParser { segs, pos: 0 }
    }

    fn parse_all(mut self) -> R<Vec<Node>> {
        let (nodes, terminator) = self.block(&[])?;
        if let Some(stmt) = terminator {
            return Err(format!("unexpected `{stmt}`"));
        }
        Ok(nodes)
    }

    /// Parses nodes until one of `stops` (a statement head word) is reached.
    /// The stop statement is consumed and returned.
    fn block(&mut self, stops: &[&str]) -> R<(Vec<Node>, Option<String>)> {
        let mut nodes = Vec::new();
        while self.pos < self.segs.len() {
            // Clone the segment first: the statement handlers below borrow
            // `self` mutably to parse their bodies, which the segment borrow
            // would otherwise block.
            let seg = match &self.segs[self.pos] {
                Seg::Text(text) => Seg::Text(text.clone()),
                Seg::Output(src) => Seg::Output(src.clone()),
                Seg::Statement(src) => Seg::Statement(src.clone()),
            };
            match &seg {
                Seg::Text(text) => {
                    nodes.push(Node::Text(text.clone()));
                    self.pos += 1;
                }
                Seg::Output(src) => {
                    nodes.push(Node::Output(parse_expr(src)?));
                    self.pos += 1;
                }
                Seg::Statement(src) => {
                    let head = head_word(src);
                    if stops.contains(&head) {
                        let stmt = src.clone();
                        self.pos += 1;
                        return Ok((nodes, Some(stmt)));
                    }
                    match head {
                        "for" => nodes.push(self.parse_for(src)?),
                        "if" => nodes.push(self.parse_if(src)?),
                        "set" => {
                            self.pos += 1;
                            nodes.push(self.parse_set(src)?);
                        }
                        "macro" => nodes.push(self.parse_macro(src)?),
                        "endif" | "endfor" | "else" | "elif" | "endmacro" => {
                            return Err(format!("unexpected `{src}`"));
                        }
                        other => return Err(format!("unsupported template statement `{other}`")),
                    }
                }
            }
        }
        Ok((nodes, None))
    }

    fn parse_if(&mut self, src: &str) -> R<Node> {
        self.pos += 1;
        let cond_src = src["if".len()..].trim();
        let mut branches: Vec<Branch> = Vec::new();
        let mut current = Some(parse_expr(cond_src)?);

        loop {
            let (body, terminator) = self.block(&["endif", "else", "elif"])?;
            branches.push((current.take(), body));
            match terminator {
                Some(stmt) if head_word(&stmt) == "elif" => {
                    let cond = stmt["elif".len()..].trim();
                    current = Some(parse_expr(cond)?);
                }
                Some(stmt) if head_word(&stmt) == "else" => {
                    let (else_body, end) = self.block(&["endif"])?;
                    if end.is_none() {
                        return Err("missing `{% endif %}`".to_string());
                    }
                    branches.push((None, else_body));
                    break;
                }
                Some(_) => break,
                None => return Err("missing `{% endif %}`".to_string()),
            }
        }
        Ok(Node::If(branches))
    }

    fn parse_for(&mut self, src: &str) -> R<Node> {
        self.pos += 1;
        let rest = src["for".len()..].trim();
        let (vars_src, iter_src) = rest
            .split_once(" in ")
            .ok_or_else(|| format!("malformed `{src}`"))?;
        let vars: Vec<String> = vars_src
            .split(',')
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .collect();
        if vars.is_empty() {
            return Err(format!("malformed `{src}`"));
        }
        let iter = parse_expr(iter_src.trim())?;

        let (body, terminator) = self.block(&["endfor", "else"])?;
        let otherwise = if terminator.is_some_and(|s| head_word(&s) == "else") {
            let (else_body, end) = self.block(&["endfor"])?;
            if end.is_none() {
                return Err("missing `{% endfor %}`".to_string());
            }
            else_body
        } else {
            Vec::new()
        };
        Ok(Node::For {
            vars,
            iter,
            body,
            otherwise,
        })
    }

    fn parse_set(&mut self, src: &str) -> R<Node> {
        let rest = src["set".len()..].trim();
        let (target_src, value_src) = rest
            .split_once('=')
            .ok_or_else(|| format!("malformed `{src}`"))?;
        let target_src = target_src.trim();
        let target = match target_src.split_once('.') {
            Some((base, field)) => SetTarget::Attr {
                base: base.trim().to_string(),
                field: field.trim().to_string(),
            },
            None => SetTarget::Name(target_src.to_string()),
        };
        Ok(Node::Set {
            target,
            value: parse_expr(value_src.trim())?,
        })
    }

    fn parse_macro(&mut self, src: &str) -> R<Node> {
        self.pos += 1;
        let rest = src["macro".len()..].trim();
        let open = rest.find('(').ok_or_else(|| format!("malformed `{src}`"))?;
        let close = rest
            .rfind(')')
            .ok_or_else(|| format!("malformed `{src}`"))?;
        let name = rest[..open].trim().to_string();
        let params_src = &rest[open + 1..close];
        let mut params = Vec::new();
        for part in split_top_level(params_src, ',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            match part.split_once('=') {
                Some((pname, default)) => {
                    params.push((pname.trim().to_string(), Some(parse_expr(default.trim())?)))
                }
                None => params.push((part.to_string(), None)),
            }
        }
        let (body, end) = self.block(&["endmacro"])?;
        if end.is_none() {
            return Err("missing `{% endmacro %}`".to_string());
        }
        Ok(Node::Macro(Rc::new(MacroDef { name, params, body })))
    }
}

// ---------------------------------------------------------------------------
// Interpreter
// ---------------------------------------------------------------------------

struct Interp {
    scopes: Vec<Vec<(String, Value)>>,
    macros: HashMap<String, Rc<MacroDef>>,
    depth: usize,
}

impl Interp {
    fn new() -> Self {
        Interp {
            scopes: vec![Vec::new()],
            macros: HashMap::new(),
            depth: 0,
        }
    }

    fn lookup(&self, name: &str) -> Value {
        for scope in self.scopes.iter().rev() {
            if let Some((_, value)) = scope.iter().rev().find(|(k, _)| k == name) {
                return value.clone();
            }
        }
        Value::Undefined
    }

    fn assign(&mut self, name: &str, value: Value) {
        if let Some(scope) = self.scopes.last_mut() {
            if let Some(slot) = scope.iter_mut().find(|(k, _)| k == name) {
                slot.1 = value;
            } else {
                scope.push((name.to_string(), value));
            }
        }
    }

    fn render(&mut self, nodes: &[Node], out: &mut String) -> R<()> {
        for node in nodes {
            match node {
                Node::Text(text) => out.push_str(text),
                Node::Output(expr) => {
                    let value = self.eval(expr)?;
                    out.push_str(&value.text());
                }
                Node::If(branches) => {
                    for (cond, body) in branches {
                        let take = match cond {
                            Some(expr) => self.eval(expr)?.truthy(),
                            None => true,
                        };
                        if take {
                            self.render(body, out)?;
                            break;
                        }
                    }
                }
                Node::For {
                    vars,
                    iter,
                    body,
                    otherwise,
                } => {
                    let value = self.eval(iter)?;
                    let items = value
                        .items_of()
                        .ok_or_else(|| format!("`{}` is not iterable", value.repr()))?;
                    if items.is_empty() {
                        self.render(otherwise, out)?;
                        continue;
                    }
                    let len = items.len();
                    for (index, item) in items.into_iter().enumerate() {
                        let mut scope = Vec::new();
                        bind_loop_vars(vars, &item, &mut scope)?;
                        scope.push(("loop".to_string(), loop_value(index, len)));
                        self.scopes.push(scope);
                        let result = self.render(body, out);
                        self.scopes.pop();
                        result?;
                    }
                }
                Node::Set { target, value } => {
                    let value = self.eval(value)?;
                    match target {
                        SetTarget::Name(name) => self.assign(name, value),
                        SetTarget::Attr { base, field } => match self.lookup(base) {
                            Value::Map(map) => map.borrow_mut().set(field, value),
                            other => {
                                return Err(format!("cannot set `{field}` on {}", other.repr()))
                            }
                        },
                    }
                }
                Node::Macro(def) => {
                    self.macros.insert(def.name.clone(), def.clone());
                }
            }
        }
        Ok(())
    }

    fn eval(&mut self, expr: &Expr) -> R<Value> {
        match expr {
            Expr::Lit(value) => Ok(value.clone()),
            Expr::Name(name) => Ok(self.lookup(name)),
            Expr::Attr(base, field) => {
                let base = self.eval(base)?;
                Ok(attr(&base, field))
            }
            Expr::Index(target, index) => {
                let target = self.eval(target)?;
                let index = self.eval(index)?;
                Ok(subscript(&target, &index))
            }
            Expr::Slice {
                target,
                start,
                stop,
                step,
            } => {
                let target = self.eval(target)?;
                let start = self.eval_opt(start)?;
                let stop = self.eval_opt(stop)?;
                let step = self.eval_opt(step)?;
                slice(&target, start.as_ref(), stop.as_ref(), step.as_ref())
            }
            Expr::Call {
                callee,
                args,
                kwargs,
            } => self.call(callee, args, kwargs),
            Expr::Filter {
                value,
                name,
                args,
                kwargs,
            } => {
                let value = self.eval(value)?;
                let args = self.eval_args(args)?;
                let kwargs = self.eval_kwargs(kwargs)?;
                apply_filter(name, value, &args, &kwargs)
            }
            Expr::Unary(op, inner) => {
                let value = self.eval(inner)?;
                match op {
                    UnOp::Not => Ok(Value::Bool(!value.truthy())),
                    UnOp::Neg => match value.as_int() {
                        Some(n) => Ok(Value::Int(-n)),
                        None => Err(format!("cannot negate {}", value.repr())),
                    },
                }
            }
            Expr::And(left, right) => {
                let left = self.eval(left)?;
                if left.truthy() {
                    self.eval(right)
                } else {
                    Ok(left)
                }
            }
            Expr::Or(left, right) => {
                let left = self.eval(left)?;
                if left.truthy() {
                    Ok(left)
                } else {
                    self.eval(right)
                }
            }
            Expr::Binary(op, left, right) => {
                let left = self.eval(left)?;
                let right = self.eval(right)?;
                binary(*op, &left, &right)
            }
            Expr::Cond {
                then,
                test,
                otherwise,
            } => {
                if self.eval(test)?.truthy() {
                    self.eval(then)
                } else {
                    match otherwise {
                        Some(expr) => self.eval(expr),
                        None => Ok(Value::Undefined),
                    }
                }
            }
            Expr::ListLit(items) => {
                let mut values = Vec::with_capacity(items.len());
                for item in items {
                    values.push(self.eval(item)?);
                }
                Ok(Value::list(values))
            }
            Expr::Test {
                value,
                name,
                negated,
            } => {
                let value = self.eval(value)?;
                let result = test(&value, name);
                Ok(Value::Bool(if *negated { !result } else { result }))
            }
        }
    }

    fn eval_opt(&mut self, expr: &Option<Box<Expr>>) -> R<Option<Value>> {
        match expr {
            Some(expr) => Ok(Some(self.eval(expr)?)),
            None => Ok(None),
        }
    }

    fn eval_args(&mut self, args: &[Expr]) -> R<Vec<Value>> {
        let mut values = Vec::with_capacity(args.len());
        for arg in args {
            values.push(self.eval(arg)?);
        }
        Ok(values)
    }

    fn eval_kwargs(&mut self, kwargs: &[(String, Expr)]) -> R<Vec<(String, Value)>> {
        let mut values = Vec::with_capacity(kwargs.len());
        for (name, expr) in kwargs {
            values.push((name.clone(), self.eval(expr)?));
        }
        Ok(values)
    }

    fn call(&mut self, callee: &Expr, args: &[Expr], kwargs: &[(String, Expr)]) -> R<Value> {
        let values = self.eval_args(args)?;
        let named = self.eval_kwargs(kwargs)?;
        match callee {
            Expr::Name(name) => {
                if let Some(def) = self.macros.get(name).cloned() {
                    return self.invoke_macro(&def, &values, &named);
                }
                call_builtin(name, &values, &named)
            }
            Expr::Attr(obj, method) => {
                let receiver = self.eval(obj)?;
                call_method(&receiver, method, &values)
            }
            other => Err(format!("cannot call {other:?}")),
        }
    }

    fn invoke_macro(
        &mut self,
        def: &MacroDef,
        args: &[Value],
        kwargs: &[(String, Value)],
    ) -> R<Value> {
        if self.depth >= MAX_CALL_DEPTH {
            return Err(format!("macro `{}` recursed too deeply", def.name));
        }
        let mut scope = Vec::with_capacity(def.params.len());
        for (index, (name, default)) in def.params.iter().enumerate() {
            let value = if let Some(value) = args.get(index) {
                value.clone()
            } else if let Some((_, value)) = kwargs.iter().find(|(k, _)| k == name) {
                value.clone()
            } else if let Some(expr) = default {
                self.eval(expr)?
            } else {
                Value::Undefined
            };
            scope.push((name.clone(), value));
        }
        self.depth += 1;
        self.scopes.push(scope);
        let mut out = String::new();
        let result = self.render(&def.body, &mut out);
        self.scopes.pop();
        self.depth -= 1;
        result?;
        Ok(Value::str(out))
    }
}

fn bind_loop_vars(vars: &[String], item: &Value, scope: &mut Vec<(String, Value)>) -> R<()> {
    if vars.len() == 1 {
        scope.push((vars[0].clone(), item.clone()));
        return Ok(());
    }
    let parts = item
        .items_of()
        .ok_or_else(|| format!("cannot unpack {}", item.repr()))?;
    for (name, value) in vars.iter().zip(parts) {
        scope.push((name.clone(), value));
    }
    Ok(())
}

fn loop_value(index: usize, len: usize) -> Value {
    Value::map(vec![
        ("index".to_string(), Value::Int(index as i64 + 1)),
        ("index0".to_string(), Value::Int(index as i64)),
        ("revindex".to_string(), Value::Int((len - index) as i64)),
        (
            "revindex0".to_string(),
            Value::Int((len - index - 1) as i64),
        ),
        ("first".to_string(), Value::Bool(index == 0)),
        ("last".to_string(), Value::Bool(index + 1 == len)),
        ("length".to_string(), Value::Int(len as i64)),
    ])
}

fn attr(value: &Value, field: &str) -> Value {
    match value {
        Value::Map(map) => map.borrow().get(field).cloned().unwrap_or(Value::Undefined),
        _ => Value::Undefined,
    }
}

fn subscript(target: &Value, index: &Value) -> Value {
    match (target, index) {
        (Value::Map(map), Value::Str(key)) => {
            map.borrow().get(key).cloned().unwrap_or(Value::Undefined)
        }
        (Value::List(items), Value::Int(i)) => {
            index_into(items, *i).cloned().unwrap_or(Value::Undefined)
        }
        (Value::Str(s), Value::Int(i)) => {
            let chars: Vec<char> = s.chars().collect();
            index_into(&chars, *i)
                .map(|c| Value::str(c.to_string()))
                .unwrap_or(Value::Undefined)
        }
        _ => Value::Undefined,
    }
}

fn index_into<T>(items: &[T], index: i64) -> Option<&T> {
    let len = items.len() as i64;
    let resolved = if index < 0 { len + index } else { index };
    if resolved < 0 || resolved >= len {
        return None;
    }
    items.get(resolved as usize)
}

fn slice(
    target: &Value,
    start: Option<&Value>,
    stop: Option<&Value>,
    step: Option<&Value>,
) -> R<Value> {
    let step = step.and_then(Value::as_int).unwrap_or(1);
    if step == 0 {
        return Err("slice step cannot be zero".to_string());
    }
    let as_text = match target {
        Value::Str(s) => Some(s.to_string()),
        _ => None,
    };
    let items: Vec<Value> = match target {
        Value::Str(s) => s.chars().map(|c| Value::str(c.to_string())).collect(),
        other => other
            .items_of()
            .ok_or_else(|| format!("cannot slice {}", other.repr()))?,
    };
    let len = items.len() as i64;
    let resolve = |v: Option<&Value>, default: i64| -> i64 {
        match v.and_then(Value::as_int) {
            Some(n) if n < 0 => (len + n).clamp(0, len),
            Some(n) => n.clamp(0, len),
            None => default,
        }
    };
    let (mut from, mut to) = if step > 0 {
        (resolve(start, 0), resolve(stop, len))
    } else {
        (resolve(stop, len - 1), resolve(start, -1))
    };
    let mut out = Vec::new();
    while if step > 0 { from < to } else { from > to } {
        if let Some(item) = items.get(from as usize) {
            out.push(item.clone());
        }
        from += step;
        if from < -1 || from > len {
            break;
        }
        // Reassign to keep the loop bound honest for negative steps.
        if step < 0 && from < 0 {
            break;
        }
        if step > 0 && from >= len {
            break;
        }
    }
    let _ = (&mut to,);
    if let Some(_text) = as_text {
        let joined: String = out.iter().map(Value::text).collect();
        return Ok(Value::str(joined));
    }
    Ok(Value::list(out))
}

fn binary(op: BinOp, left: &Value, right: &Value) -> R<Value> {
    match op {
        BinOp::Add => match (left, right) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
            (Value::Str(a), Value::Str(b)) => Ok(Value::str(format!("{a}{b}"))),
            (Value::List(a), Value::List(b)) => {
                let mut items = a.as_ref().clone();
                items.extend(b.as_ref().iter().cloned());
                Ok(Value::list(items))
            }
            _ => Err(format!("cannot add {} and {}", left.repr(), right.repr())),
        },
        BinOp::Concat => Ok(Value::str(format!("{}{}", left.text(), right.text()))),
        BinOp::Sub | BinOp::Mul | BinOp::FloorDiv | BinOp::Mod => {
            let (a, b) = match (left.as_int(), right.as_int()) {
                (Some(a), Some(b)) => (a, b),
                _ => {
                    return Err(format!(
                        "arithmetic needs numbers, got {} and {}",
                        left.repr(),
                        right.repr()
                    ))
                }
            };
            match op {
                BinOp::Sub => Ok(Value::Int(a - b)),
                BinOp::Mul => Ok(Value::Int(a * b)),
                BinOp::FloorDiv => {
                    if b == 0 {
                        Err("division by zero".to_string())
                    } else {
                        Ok(Value::Int(a.div_euclid(b)))
                    }
                }
                _ => {
                    if b == 0 {
                        Err("division by zero".to_string())
                    } else {
                        Ok(Value::Int(a.rem_euclid(b)))
                    }
                }
            }
        }
        BinOp::Eq => Ok(Value::Bool(values_eq(left, right))),
        BinOp::Ne => Ok(Value::Bool(!values_eq(left, right))),
        BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
            let ordering = match (left, right) {
                (Value::Int(a), Value::Int(b)) => a.partial_cmp(b),
                (Value::Str(a), Value::Str(b)) => a.partial_cmp(b),
                _ => None,
            };
            let ordering = ordering
                .ok_or_else(|| format!("cannot compare {} and {}", left.repr(), right.repr()))?;
            let result = match op {
                BinOp::Lt => ordering.is_lt(),
                BinOp::Gt => ordering.is_gt(),
                BinOp::Le => ordering.is_le(),
                _ => ordering.is_ge(),
            };
            Ok(Value::Bool(result))
        }
        BinOp::In | BinOp::NotIn => {
            let found = match (left, right) {
                (Value::Str(needle), Value::Str(haystack)) => haystack.contains(&**needle),
                (Value::Str(key), Value::Map(map)) => map.borrow().get(key).is_some(),
                (needle, other) => other
                    .items_of()
                    .is_some_and(|items| items.iter().any(|item| values_eq(item, needle))),
            };
            Ok(Value::Bool(if matches!(op, BinOp::NotIn) {
                !found
            } else {
                found
            }))
        }
    }
}

fn test(value: &Value, name: &str) -> bool {
    match name {
        "defined" => !matches!(value, Value::Undefined),
        "undefined" => matches!(value, Value::Undefined),
        "none" => matches!(value, Value::Null),
        "string" => matches!(value, Value::Str(_)),
        "number" => matches!(value, Value::Int(_)),
        "boolean" | "bool" => matches!(value, Value::Bool(_)),
        "mapping" => matches!(value, Value::Map(_)),
        "iterable" => matches!(value, Value::Str(_) | Value::List(_) | Value::Map(_)),
        "sequence" => matches!(value, Value::Str(_) | Value::List(_)),
        "true" => matches!(value, Value::Bool(true)),
        "false" => matches!(value, Value::Bool(false)),
        "even" => value.as_int().is_some_and(|n| n % 2 == 0),
        "odd" => value.as_int().is_some_and(|n| n % 2 != 0),
        _ => false,
    }
}

fn call_builtin(name: &str, args: &[Value], kwargs: &[(String, Value)]) -> R<Value> {
    match name {
        // Every namespace field arrives as a keyword argument, and templates both
        // read and assign them (`ns.last_query_index`), so the map keeps them.
        "namespace" => Ok(Value::map(kwargs.to_vec())),
        "range" => {
            let (start, stop) = match args {
                [stop] => (0, stop.as_int().unwrap_or(0)),
                [start, stop, ..] => (start.as_int().unwrap_or(0), stop.as_int().unwrap_or(0)),
                _ => (0, 0),
            };
            Ok(Value::list((start..stop).map(Value::Int).collect()))
        }
        "string" => Ok(Value::str(
            args.first().map(Value::text).unwrap_or_default(),
        )),
        "length" | "len" | "count" => Ok(Value::Int(
            args.first()
                .and_then(Value::items_of)
                .map(|items| items.len() as i64)
                .unwrap_or(0),
        )),
        // `raise_exception` is how a template rejects a message it cannot
        // render; surface its text as the error.
        "raise_exception" => Err(args
            .first()
            .map(Value::text)
            .unwrap_or_else(|| "template raised an exception".to_string())),
        other => Err(format!("unknown function `{other}`")),
    }
}

fn call_method(receiver: &Value, method: &str, args: &[Value]) -> R<Value> {
    match receiver {
        Value::Str(s) => {
            let text = s.to_string();
            match method {
                "startswith" => Ok(Value::Bool(
                    args.first()
                        .map(Value::text)
                        .is_some_and(|prefix| text.starts_with(&prefix)),
                )),
                "endswith" => Ok(Value::Bool(
                    args.first()
                        .map(Value::text)
                        .is_some_and(|suffix| text.ends_with(&suffix)),
                )),
                "strip" => Ok(Value::str(text.trim())),
                "lstrip" => Ok(Value::str(text.trim_start())),
                "rstrip" => Ok(Value::str(text.trim_end())),
                "lower" => Ok(Value::str(text.to_lowercase())),
                "upper" => Ok(Value::str(text.to_uppercase())),
                "replace" => {
                    let from = args.first().map(Value::text).unwrap_or_default();
                    let to = args.get(1).map(Value::text).unwrap_or_default();
                    Ok(Value::str(text.replace(&from, &to)))
                }
                "split" => {
                    let parts: Vec<Value> = match args.first() {
                        Some(sep) => text.split(&sep.text()).map(Value::str).collect(),
                        None => text.split_whitespace().map(Value::str).collect(),
                    };
                    Ok(Value::list(parts))
                }
                other => Err(format!("unknown string method `{other}`")),
            }
        }
        Value::Map(map) => {
            let map = map.borrow();
            match method {
                "items" => Ok(Value::list(
                    map.entries
                        .iter()
                        .map(|(k, v)| Value::list(vec![Value::str(k.clone()), v.clone()]))
                        .collect(),
                )),
                "keys" => Ok(Value::list(
                    map.entries
                        .iter()
                        .map(|(k, _)| Value::str(k.clone()))
                        .collect(),
                )),
                "values" => Ok(Value::list(
                    map.entries.iter().map(|(_, v)| v.clone()).collect(),
                )),
                "get" => {
                    let key = args.first().map(Value::text).unwrap_or_default();
                    Ok(map
                        .get(&key)
                        .cloned()
                        .unwrap_or_else(|| args.get(1).cloned().unwrap_or(Value::Null)))
                }
                other => Err(format!("unknown mapping method `{other}`")),
            }
        }
        Value::List(items) => match method {
            "count" => {
                let needle = args.first().cloned().unwrap_or(Value::Null);
                Ok(Value::Int(
                    items.iter().filter(|item| values_eq(item, &needle)).count() as i64,
                ))
            }
            other => Err(format!("unknown list method `{other}`")),
        },
        other => Err(format!("{} has no method `{method}`", other.repr())),
    }
}

fn apply_filter(name: &str, value: Value, args: &[Value], kwargs: &[(String, Value)]) -> R<Value> {
    let _ = kwargs;
    match name {
        "trim" => Ok(Value::str(value.text().trim())),
        "lower" => Ok(Value::str(value.text().to_lowercase())),
        "upper" => Ok(Value::str(value.text().to_uppercase())),
        "string" => Ok(Value::str(value.text())),
        "safe" => Ok(value),
        "length" | "count" => {
            if let Value::Str(s) = &value {
                return Ok(Value::Int(s.chars().count() as i64));
            }
            Ok(Value::Int(
                value
                    .items_of()
                    .map(|items| items.len() as i64)
                    .unwrap_or(0),
            ))
        }
        "tojson" => Ok(Value::str(json_of(&value))),
        // Jinja's `items` is the mapping form: a list of `[key, value]` pairs.
        "items" => match &value {
            Value::Map(map) => {
                let map = map.borrow();
                Ok(Value::list(
                    map.entries
                        .iter()
                        .map(|(k, v)| Value::list(vec![Value::str(k.clone()), v.clone()]))
                        .collect(),
                ))
            }
            other => Ok(Value::list(
                other
                    .items_of()
                    .unwrap_or_default()
                    .into_iter()
                    .enumerate()
                    .map(|(index, v)| Value::list(vec![Value::Int(index as i64), v]))
                    .collect(),
            )),
        },
        "first" => Ok(value
            .items_of()
            .and_then(|items| items.first().cloned())
            .unwrap_or(Value::Undefined)),
        "last" => Ok(value
            .items_of()
            .and_then(|items| items.last().cloned())
            .unwrap_or(Value::Undefined)),
        "join" => {
            let sep = args.first().map(Value::text).unwrap_or_default();
            let parts: Vec<String> = value
                .items_of()
                .unwrap_or_default()
                .iter()
                .map(Value::text)
                .collect();
            Ok(Value::str(parts.join(&sep)))
        }
        "replace" => {
            let from = args.first().map(Value::text).unwrap_or_default();
            let to = args.get(1).map(Value::text).unwrap_or_default();
            Ok(Value::str(value.text().replace(&from, &to)))
        }
        "default" | "d" => {
            let fallback = args.first().cloned().unwrap_or(Value::Null);
            let use_fallback_when_falsy = args.get(1).is_some_and(Value::truthy);
            let replace =
                matches!(value, Value::Undefined) || (use_fallback_when_falsy && !value.truthy());
            Ok(if replace { fallback } else { value })
        }
        other => Err(format!("unknown filter `{other}`")),
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Renders `template` against `ctx`. Returns the first construct the evaluator
/// cannot handle (or a `raise_exception` message) instead of emitting a
/// partially rendered prompt.
pub fn eval_jinja2(template: &str, ctx: &TemplateContext) -> R<String> {
    let segs = tokenize_template(template)?;
    let nodes = TemplateParser::new(segs).parse_all()?;

    let mut interp = Interp::new();
    let messages: Vec<Value> = ctx
        .messages
        .iter()
        .map(|m| {
            Value::map(vec![
                ("role".to_string(), Value::str(m.role.clone())),
                ("content".to_string(), Value::str(m.content.clone())),
            ])
        })
        .collect();

    let mut root = vec![
        ("messages".to_string(), Value::list(messages)),
        ("bos_token".to_string(), Value::str(ctx.bos_token.clone())),
        ("eos_token".to_string(), Value::str(ctx.eos_token.clone())),
        (
            "add_generation_prompt".to_string(),
            Value::Bool(ctx.add_generation_prompt),
        ),
        ("thinking".to_string(), Value::Bool(ctx.thinking)),
        (
            "enable_thinking".to_string(),
            Value::Bool(ctx.enable_thinking),
        ),
    ];
    if let Some(effort) = &ctx.reasoning_effort {
        root.push(("reasoning_effort".to_string(), Value::str(effort.clone())));
    }
    interp.scopes[0] = root;

    let mut out = String::new();
    interp.render(&nodes, &mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(messages: &[(&str, &str)], thinking: bool) -> TemplateContext {
        TemplateContext {
            messages: messages
                .iter()
                .map(|(role, content)| ChatMessage {
                    role: role.to_string(),
                    content: content.to_string(),
                })
                .collect(),
            bos_token: "<s>".to_string(),
            eos_token: "</s>".to_string(),
            add_generation_prompt: true,
            thinking,
            enable_thinking: thinking,
            reasoning_effort: None,
        }
    }

    #[test]
    fn test_simple_interpolation() {
        let c = ctx(&[], true);
        assert_eq!(
            eval_jinja2("{{ bos_token }}hello{{ eos_token }}", &c).unwrap(),
            "<s>hello</s>"
        );
    }

    #[test]
    fn test_for_loop() {
        let c = ctx(&[("user", "hi"), ("assistant", "hello")], true);
        let template = "{% for message in messages %}{{ message['role'] }}: \
                        {{ message['content'] }}\n{% endfor %}";
        let result = eval_jinja2(template, &c).unwrap();
        assert!(result.contains("user: hi"), "{result:?}");
        assert!(result.contains("assistant: hello"), "{result:?}");
    }

    #[test]
    fn test_if_condition() {
        let c = ctx(&[("user", "test")], true);
        let result = eval_jinja2("{% if add_generation_prompt %}GEN{% endif %}", &c).unwrap();
        assert_eq!(result, "GEN");
    }

    #[test]
    fn test_set_and_namespace() {
        let c = ctx(&[("system", "s"), ("user", "u")], true);
        let template = "{%- set ns = namespace(last=-1) %}\
                        {%- for m in messages %}\
                        {%- if m.role == 'user' %}{% set ns.last = loop.index0 %}{% endif %}\
                        {%- endfor %}[{{ ns.last }}]";
        assert_eq!(eval_jinja2(template, &c).unwrap(), "[1]");
    }

    #[test]
    fn test_reasoning_effort_default_and_override() {
        let template = "{%- if reasoning_effort is undefined %}\
                        {{- reasoning_effort|default('xhigh') }}\
                        {%- else %}{{ reasoning_effort }}{%- endif %}";
        let mut c = ctx(&[("user", "u")], true);
        assert_eq!(eval_jinja2(template, &c).unwrap(), "xhigh");
        c.reasoning_effort = Some("low".to_string());
        assert_eq!(eval_jinja2(template, &c).unwrap(), "low");
    }

    #[test]
    fn test_macro_and_filters() {
        let c = ctx(&[("user", "  padded  ")], true);
        let template = "{%- macro show(text) %}[{{ text }}]{%- endmacro %}\
                        {{- show(messages[0].content)|trim }}";
        assert_eq!(eval_jinja2(template, &c).unwrap(), "[  padded  ]");
    }

    #[test]
    fn test_reverse_slice_and_loop_state() {
        let c = ctx(&[("user", "a"), ("assistant", "b"), ("user", "c")], true);
        let template = "{%- for m in messages[::-1] %}{{ m.content }}{% endfor %}";
        assert_eq!(eval_jinja2(template, &c).unwrap(), "cba");
        let template = "{%- for m in messages %}\
                        {{- loop.index0 }}{% if loop.first %}F{% endif %}\
                        {%- if loop.last %}L{% endif %}{%- endfor %}";
        assert_eq!(eval_jinja2(template, &c).unwrap(), "0F12L");
    }

    #[test]
    fn test_whitespace_control_matches_reference() {
        // trim_blocks drops the newline after `%}`, `-%}` drops the leading
        // whitespace of the next line, and `{{-` drops the preceding one.
        let c = ctx(&[("user", "u")], true);
        let template = "a\n{%- if add_generation_prompt %}\n b \n{%- endif %}\nc";
        // Verified against jinja2 3.1 with trim_blocks/lstrip_blocks, which is
        // the environment the reference render uses.
        assert_eq!(eval_jinja2(template, &c).unwrap(), "a bc");
    }

    #[test]
    fn test_namespace_fields_and_mapping_filters() {
        let c = ctx(&[("user", "u")], true);
        let template = "{%- set ns = namespace(a=1, b='x') %}\
                        {%- set ns.a = ns.a + 1 %}\
                        {%- for key, value in ns|items %}{{ key }}={{ value }};{% endfor %}\
                        {{- ns|tojson }}";
        assert_eq!(
            eval_jinja2(template, &c).unwrap(),
            "a=2;b=x;{\"a\": 2, \"b\": \"x\"}"
        );
    }

    #[test]
    fn test_raise_exception_is_an_error() {
        let c = ctx(&[("user", "u")], true);
        let err = eval_jinja2("{{ raise_exception('nope') }}", &c).unwrap_err();
        assert_eq!(err, "nope");
    }

    #[test]
    fn test_string_methods_and_inline_conditional() {
        let c = ctx(&[("user", "u")], true);
        let template = "{{- 'yes' if 'abc'.startswith('a') else 'no' }}\
                        {{- '!' if not 'abc'.endswith('z') else '' }}";
        assert_eq!(eval_jinja2(template, &c).unwrap(), "yes!");
    }

    #[test]
    fn test_unsupported_construct_is_reported() {
        let c = ctx(&[("user", "u")], true);
        let err = eval_jinja2("{% include 'other' %}", &c).unwrap_err();
        assert!(err.contains("unsupported"), "{err}");
    }
}
