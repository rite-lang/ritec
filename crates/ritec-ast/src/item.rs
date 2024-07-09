use ritec_diagnostic::{Diagnostic, Span};
use ritec_parse::{Token, TokenStream};

use crate::{parse_generic, Generic};

#[derive(Clone, Debug)]
pub struct NamedSegment {
    pub name: String,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum ItemSegment {
    Named(NamedSegment),
    Generic(Generic),
}

impl ItemSegment {
    pub fn span(&self) -> Span {
        match self {
            ItemSegment::Named(segment) => segment.span,
            ItemSegment::Generic(generic) => generic.span,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Item {
    pub segments: Vec<ItemSegment>,
    pub span: Span,
}

pub fn parse_named_segment(stream: &mut TokenStream) -> Result<NamedSegment, Diagnostic> {
    let (name, span) = stream.expect_ident_spanned()?;

    Ok(NamedSegment { name, span })
}

pub fn parse_item_segment(stream: &mut TokenStream) -> Result<ItemSegment, Diagnostic> {
    let (name, span) = stream.peek();

    match name {
        Token::Ident(_) => Ok(ItemSegment::Named(parse_named_segment(stream)?)),
        Token::Quote => Ok(ItemSegment::Generic(parse_generic(stream)?)),
        _ => Err(Diagnostic::new("expected identifier").with_span(span)),
    }
}

pub fn parse_item(stream: &mut TokenStream) -> Result<Item, Diagnostic> {
    let mut segments = Vec::new();

    let first_segment = parse_item_segment(stream)?;
    let mut span = first_segment.span();

    segments.push(first_segment);

    loop {
        if stream.take(Token::ColonColon) {
            let segment = parse_item_segment(stream)?;
            span = span.join(segment.span());
            segments.push(segment);
        } else {
            break;
        }
    }

    Ok(Item { segments, span })
}
