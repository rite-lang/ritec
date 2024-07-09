use ritec_diagnostic::{Diagnostic, Span};
use ritec_parse::{Delim, Token, TokenStream};

use crate::{parse_body, parse_generic, parse_type, Expr, Generic, Type};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Visibility {
    Public,
    Private,
}

#[derive(Clone, Debug)]
pub struct FunctionArgument {
    pub name: String,
    pub type_: Type,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct FunctionDecl {
    pub vis: Visibility,
    pub name: String,
    pub generics: Vec<Generic>,
    pub arguments: Vec<FunctionArgument>,
    pub output: Option<Type>,
    pub body: Expr,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum Decl {
    Function(FunctionDecl),
}

#[derive(Clone, Debug)]
pub struct Module {
    pub decls: Vec<Decl>,
}

pub fn parse_visibility(stream: &mut TokenStream) -> Result<Visibility, Diagnostic> {
    let (token, _) = stream.peek();

    match token {
        Token::Pub => {
            stream.consume();
            Ok(Visibility::Public)
        }
        _ => Ok(Visibility::Private),
    }
}

pub fn parse_function_argument(stream: &mut TokenStream) -> Result<FunctionArgument, Diagnostic> {
    let (name, span) = stream.expect_ident_spanned()?;
    stream.expect(Token::Colon)?;
    let type_ = parse_type(stream)?;

    Ok(FunctionArgument { name, type_, span })
}

pub fn parse_function_decl(stream: &mut TokenStream) -> Result<FunctionDecl, Diagnostic> {
    let vis = parse_visibility(stream)?;

    stream.expect(Token::Fn)?;

    let (name, span) = stream.expect_ident_spanned()?;

    let mut generics = Vec::new();

    if stream.take(Token::Lt) {
        loop {
            let generic = parse_generic(stream)?;
            generics.push(generic);

            if !stream.take(Token::Comma) {
                break;
            }
        }

        stream.expect(Token::Gt)?;
    }

    let mut arguments = Vec::new();

    stream.expect(Token::Paren(Delim::Open))?;

    loop {
        if stream.is(Token::Paren(Delim::Close)) {
            break;
        }

        arguments.push(parse_function_argument(stream)?);

        if !stream.take(Token::Comma) {
            break;
        }
    }

    stream.expect(Token::Paren(Delim::Close))?;

    let output = if stream.take(Token::Arrow) {
        Some(parse_type(stream)?)
    } else {
        None
    };

    let body = parse_body(stream)?;

    Ok(FunctionDecl {
        vis,
        name,
        generics,
        arguments,
        output,
        body,
        span,
    })
}

pub fn parse_decl(stream: &mut TokenStream) -> Result<Decl, Diagnostic> {
    let (token, span) = stream.peek();

    match token {
        Token::Fn => Ok(Decl::Function(parse_function_decl(stream)?)),
        _ => {
            let message = format!("expected declaration, found {}", token);
            let diagnostic = Diagnostic::new(message).with_span(span);
            Err(diagnostic)
        }
    }
}

pub fn parse_module(stream: &mut TokenStream) -> Result<Module, Diagnostic> {
    let mut decls = Vec::new();

    while !stream.is_empty() {
        decls.push(parse_decl(stream)?);
        stream.take(Token::Newline);
    }

    Ok(Module { decls })
}
