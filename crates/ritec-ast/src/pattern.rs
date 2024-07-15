use ritec_diagnostic::{Diagnostic, Span};
use ritec_parse::{Delim, Token, TokenStream};

use crate::{parse_item, Item};

#[derive(Clone, Debug)]
pub struct ItemPattern {
    pub item: Item,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct TuplePattern {
    pub item: Option<Item>,
    pub patterns: Vec<Pattern>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum Pattern {
    Item(ItemPattern),
    Tuple(TuplePattern),
}

impl Pattern {
    pub fn span(&self) -> Span {
        match self {
            Pattern::Item(item) => item.span,
            Pattern::Tuple(tuple) => tuple.span,
        }
    }
}

fn parse_tuple_pattern(
    stream: &mut TokenStream,
    item: Option<Item>,
) -> Result<TuplePattern, Diagnostic> {
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

    Ok(TuplePattern {
        item,
        patterns,
        span: start.join(end),
    })
}

pub fn parse_pattern(stream: &mut TokenStream) -> Result<Pattern, Diagnostic> {
    if stream.is(Token::Paren(Delim::Open)) {
        return Ok(Pattern::Tuple(parse_tuple_pattern(stream, None)?));
    }

    let item = parse_item(stream, true)?;

    if stream.is(Token::Paren(Delim::Open)) {
        return Ok(Pattern::Tuple(parse_tuple_pattern(stream, Some(item))?));
    }

    let span = item.span;
    Ok(Pattern::Item(ItemPattern { item, span }))
}
