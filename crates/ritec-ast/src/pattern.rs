use ritec_diagnostic::{Diagnostic, Span};
use ritec_parse::{Delim, Token, TokenStream};

use crate::{parse_path, Path};

#[derive(Clone, Debug)]
pub struct ItemPat {
    pub item: Path,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct TuplePat {
    pub item: Option<Path>,
    pub patterns: Vec<Pat>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum Pat {
    Item(ItemPat),
    Tuple(TuplePat),
}

impl Pat {
    pub fn span(&self) -> Span {
        match self {
            Pat::Item(item) => item.span,
            Pat::Tuple(tuple) => tuple.span,
        }
    }
}

fn parse_tuple_pattern(
    stream: &mut TokenStream,
    item: Option<Path>,
) -> Result<TuplePat, Diagnostic> {
    let start = stream.expect(Token::Paren(Delim::Open))?;

    let mut patterns = Vec::new();

    loop {
        if stream.is(Token::Paren(Delim::Close)) {
            break;
        }

        let pattern = parse_pattern(stream)?;

        patterns.push(pattern);

        if !stream.take(Token::Comma) {
            break;
        }
    }

    let end = stream.expect(Token::Paren(Delim::Close))?;

    Ok(TuplePat {
        item,
        patterns,
        span: start.join(end),
    })
}

pub fn parse_pattern(stream: &mut TokenStream) -> Result<Pat, Diagnostic> {
    if stream.is(Token::Paren(Delim::Open)) {
        return Ok(Pat::Tuple(parse_tuple_pattern(stream, None)?));
    }

    let item = parse_path(stream, true)?;

    if stream.is(Token::Paren(Delim::Open)) {
        return Ok(Pat::Tuple(parse_tuple_pattern(stream, Some(item))?));
    }

    let span = item.span;
    Ok(Pat::Item(ItemPat { item, span }))
}
