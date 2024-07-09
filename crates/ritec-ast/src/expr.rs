use ritec_diagnostic::{Diagnostic, Span};
use ritec_parse::{Token, TokenStream};

use crate::{parse_type, Type};

#[derive(Clone, Debug)]
pub struct VoidExpr {
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct LetExpr {
    pub name: String,
    pub type_: Option<Type>,
    pub value: Option<Box<Expr>>,
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
    Let(LetExpr),
    Block(BlockExpr),
}

pub fn parse_let_expr(stream: &mut TokenStream) -> Result<LetExpr, Diagnostic> {
    stream.expect(Token::Let)?;

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

        if !stream.take(Token::Newline) {
            break;
        }
    }

    let end = stream.expect(Token::Dedent)?;

    Ok(BlockExpr {
        exprs,
        span: start.join(end),
    })
}

pub fn parse_expr(stream: &mut TokenStream) -> Result<Expr, Diagnostic> {
    let (token, span) = stream.peek();

    match token {
        Token::Let => Ok(Expr::Let(parse_let_expr(stream)?)),
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
