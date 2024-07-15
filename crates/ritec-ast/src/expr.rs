use ritec_diagnostic::{Diagnostic, Span};
use ritec_parse::{Delim, Token, TokenStream};

use crate::{parse_item, parse_pattern, parse_type, Item, Pattern, Type};

#[derive(Clone, Debug)]
pub struct VoidExpr {
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct ItemExpr {
    pub item: Item,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct LitIntExpr {
    pub value: u64,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct LitFloatExpr {
    pub value: f64,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct FieldInit {
    pub field: String,
    pub value: Expr,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct StructExpr {
    pub item: Item,
    pub fields: Vec<FieldInit>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct ParenExpr {
    pub expr: Box<Expr>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct FieldExpr {
    pub base: Box<Expr>,
    pub field: String,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct CallExpr {
    pub callee: Box<Expr>,
    pub arguments: Vec<Expr>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum UnaryOp {
    Neg,
    Not,
    Ref,
    Deref,
}

impl UnaryOp {
    pub fn from_token(token: Token) -> Option<Self> {
        match token {
            Token::Minus => Some(UnaryOp::Neg),
            Token::Not => Some(UnaryOp::Not),
            Token::And => Some(UnaryOp::Ref),
            Token::Star => Some(UnaryOp::Deref),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct UnaryExpr {
    pub op: UnaryOp,
    pub expr: Box<Expr>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    And,
    Or,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl BinaryOp {
    pub fn from_token(token: Token) -> Option<Self> {
        match token {
            Token::Plus => Some(BinaryOp::Add),
            Token::Minus => Some(BinaryOp::Sub),
            Token::Star => Some(BinaryOp::Mul),
            Token::Slash => Some(BinaryOp::Div),
            Token::Percent => Some(BinaryOp::Mod),
            Token::And => Some(BinaryOp::And),
            Token::Or => Some(BinaryOp::Or),
            Token::Eq => Some(BinaryOp::Eq),
            Token::NotEq => Some(BinaryOp::Ne),
            Token::Lt => Some(BinaryOp::Lt),
            Token::LtEq => Some(BinaryOp::Le),
            Token::Gt => Some(BinaryOp::Gt),
            Token::GtEq => Some(BinaryOp::Ge),
            _ => None,
        }
    }

    pub fn precedence(&self) -> u8 {
        match self {
            BinaryOp::Or => 1,
            BinaryOp::And => 2,
            BinaryOp::Eq | BinaryOp::Ne => 3,
            BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => 4,
            BinaryOp::Add | BinaryOp::Sub => 5,
            BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => 6,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BinaryExpr {
    pub op: BinaryOp,
    pub lhs: Box<Expr>,
    pub rhs: Box<Expr>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct LetExpr {
    pub mutable: bool,
    pub name: String,
    pub type_: Option<Type>,
    pub value: Option<Box<Expr>>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct LoopExpr {
    pub body: Box<Expr>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub body: Expr,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct MatchExpr {
    pub value: Box<Expr>,
    pub arms: Vec<MatchArm>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct BlockExpr {
    pub exprs: Vec<Expr>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum Expr {
    Void(VoidExpr),
    Item(ItemExpr),
    LitInt(LitIntExpr),
    LitFloat(LitFloatExpr),
    Struct(StructExpr),
    Paren(ParenExpr),
    Field(FieldExpr),
    Call(CallExpr),
    Unary(UnaryExpr),
    Binary(BinaryExpr),
    Let(LetExpr),
    Loop(LoopExpr),
    Match(MatchExpr),
    Block(BlockExpr),
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Void(expr) => expr.span,
            Expr::Item(expr) => expr.span,
            Expr::LitInt(expr) => expr.span,
            Expr::LitFloat(expr) => expr.span,
            Expr::Struct(expr) => expr.span,
            Expr::Paren(expr) => expr.span,
            Expr::Field(expr) => expr.span,
            Expr::Call(expr) => expr.span,
            Expr::Unary(expr) => expr.span,
            Expr::Binary(expr) => expr.span,
            Expr::Let(expr) => expr.span,
            Expr::Loop(expr) => expr.span,
            Expr::Match(expr) => expr.span,
            Expr::Block(expr) => expr.span,
        }
    }
}

pub fn parse_void_expr(stream: &mut TokenStream) -> Result<VoidExpr, Diagnostic> {
    let span = stream.expect(Token::Void)?;

    Ok(VoidExpr { span })
}

pub fn parse_paren_expr(stream: &mut TokenStream) -> Result<ParenExpr, Diagnostic> {
    let start = stream.expect(Token::Paren(Delim::Open))?;

    let expr = parse_expr(stream)?;

    let end = stream.expect(Token::Paren(Delim::Close))?;

    Ok(ParenExpr {
        expr: Box::new(expr),
        span: start.join(end),
    })
}

pub fn parse_field_init(stream: &mut TokenStream) -> Result<FieldInit, Diagnostic> {
    let (field, span) = stream.expect_ident_spanned()?;
    stream.expect(Token::Colon)?;
    let value = parse_expr(stream)?;

    Ok(FieldInit { field, value, span })
}

pub fn parse_struct_expr(stream: &mut TokenStream, item: Item) -> Result<Expr, Diagnostic> {
    let start = stream.expect(Token::Brace(Delim::Open))?;

    let has_newlines = stream.take(Token::Newline);

    if has_newlines {
        stream.expect(Token::Indent)?;
    }

    let mut fields = Vec::new();

    loop {
        if has_newlines && stream.is(Token::Dedent) {
            break;
        }

        if stream.is(Token::Brace(Delim::Close)) {
            break;
        }

        let field = parse_field_init(stream)?;
        fields.push(field);

        if !stream.take(Token::Comma) {
            break;
        }

        if has_newlines {
            stream.expect(Token::Newline)?;
        }
    }

    if has_newlines {
        stream.take(Token::Newline);
        stream.expect(Token::Dedent)?;
    }

    let end = stream.expect(Token::Brace(Delim::Close))?;

    let span = start.join(end);
    Ok(Expr::Struct(StructExpr { item, fields, span }))
}

pub fn parse_item_expr(stream: &mut TokenStream) -> Result<Expr, Diagnostic> {
    let item = parse_item(stream, true)?;

    if stream.is(Token::Brace(Delim::Open)) {
        parse_struct_expr(stream, item)
    } else {
        let span = item.span;
        Ok(Expr::Item(ItemExpr { item, span }))
    }
}

pub fn parse_lit_int_expr(stream: &mut TokenStream) -> Result<Expr, Diagnostic> {
    let (value, span) = stream.expect_integer_spanned()?;

    Ok(Expr::LitInt(LitIntExpr { value, span }))
}

pub fn parse_lit_float_expr(stream: &mut TokenStream) -> Result<Expr, Diagnostic> {
    let (value, span) = stream.expect_float_spanned()?;

    Ok(Expr::LitFloat(LitFloatExpr { value, span }))
}

pub fn parse_term_expr(stream: &mut TokenStream) -> Result<Expr, Diagnostic> {
    let (token, span) = stream.peek();

    match token {
        Token::Void => Ok(Expr::Void(parse_void_expr(stream)?)),
        Token::Paren(Delim::Open) => Ok(Expr::Paren(parse_paren_expr(stream)?)),
        Token::Ident(_)
        | Token::ColonColon
        | Token::Quote
        | Token::SelfLower
        | Token::SelfUpper => parse_item_expr(stream),
        Token::Integer(_) => parse_lit_int_expr(stream),
        Token::Float(_) => parse_lit_float_expr(stream),
        _ => {
            let message = format!("expected term, found {}", token);
            let diagnostic = Diagnostic::new(message).with_span(span);

            Err(diagnostic)
        }
    }
}

pub fn parse_field_expr(stream: &mut TokenStream) -> Result<Expr, Diagnostic> {
    let base = parse_term_expr(stream)?;

    if stream.take(Token::Dot) {
        let (field, span) = stream.expect_ident_spanned()?;
        let span = base.span().join(span);

        Ok(Expr::Field(FieldExpr {
            base: Box::new(base),
            field,
            span,
        }))
    } else {
        Ok(base)
    }
}

pub fn parse_call_expr(stream: &mut TokenStream) -> Result<Expr, Diagnostic> {
    let callee = parse_field_expr(stream)?;

    if stream.take(Token::Paren(Delim::Open)) {
        let mut arguments = Vec::new();

        loop {
            if stream.is(Token::Paren(Delim::Close)) {
                break;
            }

            let argument = parse_expr(stream)?;
            arguments.push(argument);

            if !stream.take(Token::Comma) {
                break;
            }
        }

        let last = stream.expect(Token::Paren(Delim::Close))?;
        let span = callee.span().join(last);

        Ok(Expr::Call(CallExpr {
            callee: Box::new(callee),
            arguments,
            span,
        }))
    } else {
        Ok(callee)
    }
}

pub fn parse_unary_expr(stream: &mut TokenStream) -> Result<Expr, Diagnostic> {
    let (token, span) = stream.peek();

    if let Some(op) = UnaryOp::from_token(token) {
        stream.consume();

        let expr = parse_unary_expr(stream)?;

        Ok(Expr::Unary(UnaryExpr {
            op,
            expr: Box::new(expr),
            span,
        }))
    } else {
        parse_call_expr(stream)
    }
}

pub fn parse_binary_expr(stream: &mut TokenStream, min_precedence: u8) -> Result<Expr, Diagnostic> {
    // NOTE: Copilot wrote whole function this thing in it's entirety

    let mut lhs = parse_unary_expr(stream)?;

    loop {
        let (token, _) = stream.peek();

        if let Some(op) = BinaryOp::from_token(token) {
            let precedence = op.precedence();

            if precedence < min_precedence {
                break;
            }

            stream.consume();

            let mut rhs = parse_binary_expr(stream, precedence + 1)?;

            loop {
                let (next_token, _) = stream.peek();

                if let Some(next_op) = BinaryOp::from_token(next_token) {
                    let next_precedence = next_op.precedence();

                    if precedence < next_precedence {
                        rhs = parse_binary_expr(stream, next_precedence + 1)?;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }

            let span = lhs.span().join(rhs.span());
            lhs = Expr::Binary(BinaryExpr {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span,
            });
        } else {
            break;
        }
    }

    Ok(lhs)
}

pub fn parse_let_expr(stream: &mut TokenStream) -> Result<LetExpr, Diagnostic> {
    stream.expect(Token::Let)?;

    let mutable = stream.take(Token::Mut);

    let (name, span) = stream.expect_ident_spanned()?;

    let type_ = if stream.take(Token::Colon) {
        Some(parse_type(stream)?)
    } else {
        None
    };

    let value = if stream.take(Token::Eq) {
        Some(Box::new(parse_expr(stream)?))
    } else {
        None
    };

    Ok(LetExpr {
        mutable,
        name,
        type_,
        value,
        span,
    })
}

pub fn parse_block_expr(stream: &mut TokenStream) -> Result<BlockExpr, Diagnostic> {
    let start = stream.expect(Token::Indent)?;

    let mut exprs = Vec::new();

    loop {
        if stream.is(Token::Dedent) {
            break;
        }

        let expr = parse_expr(stream)?;
        exprs.push(expr);

        stream.take(Token::Newline);
    }

    let end = stream.expect(Token::Dedent)?;

    let span = start.join(end);
    Ok(BlockExpr { exprs, span })
}

pub fn parse_loop_expr(stream: &mut TokenStream) -> Result<LoopExpr, Diagnostic> {
    let span = stream.expect(Token::Loop)?;

    let body = parse_expr(stream)?;

    Ok(LoopExpr {
        body: Box::new(body),
        span,
    })
}

fn is_match_arm(stream: &mut TokenStream) -> bool {
    stream.is(Token::Newline) && stream.nth_is(1, Token::Or)
}

fn parse_match_arm(stream: &mut TokenStream) -> Result<MatchArm, Diagnostic> {
    stream.expect(Token::Or)?;

    let pattern = parse_pattern(stream)?;
    let body = parse_body(stream)?;
    let span = pattern.span().join(body.span());

    Ok(MatchArm {
        pattern,
        body,
        span,
    })
}

pub fn parse_match_expr(stream: &mut TokenStream) -> Result<MatchExpr, Diagnostic> {
    let span = stream.expect(Token::Match)?;

    let value = parse_expr(stream)?;
    let span = span.join(value.span());

    let mut arms = Vec::new();

    while is_match_arm(stream) {
        stream.expect(Token::Newline)?;
        let arm = parse_match_arm(stream)?;
        arms.push(arm);
    }

    Ok(MatchExpr {
        value: Box::new(value),
        arms,
        span,
    })
}

pub fn parse_expr(stream: &mut TokenStream) -> Result<Expr, Diagnostic> {
    let (token, span) = stream.peek();

    match token {
        Token::Void
        | Token::Paren(Delim::Open)
        | Token::Minus
        | Token::Not
        | Token::And
        | Token::Star
        | Token::ColonColon
        | Token::Quote
        | Token::SelfLower
        | Token::SelfUpper
        | Token::Ident(_)
        | Token::Integer(_)
        | Token::Float(_) => parse_binary_expr(stream, 0),
        Token::Let => Ok(Expr::Let(parse_let_expr(stream)?)),
        Token::Loop => Ok(Expr::Loop(parse_loop_expr(stream)?)),
        Token::Match => Ok(Expr::Match(parse_match_expr(stream)?)),
        Token::Newline => {
            stream.consume();
            Ok(Expr::Block(parse_block_expr(stream)?))
        }
        _ => {
            let message = format!("expected expression, found {}", token);
            let diagnostic = Diagnostic::new(message).with_span(span);

            Err(diagnostic)
        }
    }
}

pub fn parse_body(stream: &mut TokenStream) -> Result<Expr, Diagnostic> {
    if stream.take(Token::FatArrow) {
        return parse_expr(stream);
    }

    stream.expect(Token::Newline)?;

    let (token, span) = stream.peek();

    match token {
        Token::Indent => Ok(Expr::Block(parse_block_expr(stream)?)),
        Token::Eof => Ok(Expr::Void(VoidExpr { span })),
        _ => {
            let message = format!("expected body, found {}", token);
            let diagnostic = Diagnostic::new(message).with_span(span);

            Err(diagnostic)
        }
    }
}
