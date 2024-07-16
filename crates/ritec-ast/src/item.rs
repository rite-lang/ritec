use ritec_diagnostic::{Diagnostic, Span};
use ritec_parse::{Token, TokenStream};

use crate::{parse_generic, parse_type, Generic, Type};

#[derive(Clone, Debug)]
pub struct NamedSegment {
    pub name: String,
    pub generics: Vec<Type>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct AssocSegment {
    pub implementor: Type,
    pub trait_item: Item,
    pub name: String,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct SelfLowerSegment {
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct SelfUpperSegment {
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum ItemSegment {
    Named(NamedSegment),
    Assoc(AssocSegment),
    Generic(Generic),
    SelfLower(SelfLowerSegment),
    SelfUpper(SelfUpperSegment),
}

impl ItemSegment {
    pub fn span(&self) -> Span {
        match self {
            ItemSegment::Named(segment) => segment.span,
            ItemSegment::Assoc(segment) => segment.span,
            ItemSegment::Generic(generic) => generic.span,
            ItemSegment::SelfLower(segment) => segment.span,
            ItemSegment::SelfUpper(segment) => segment.span,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Item {
    pub segments: Vec<ItemSegment>,
    pub span: Span,
}

impl Item {
    pub fn ident(&self) -> Option<&str> {
        if self.segments.len() != 1 {
            return None;
        }

        match self.segments.first() {
            Some(ItemSegment::Named(segment)) if segment.generics.is_empty() => Some(&segment.name),
            _ => None,
        }
    }
}

pub fn parse_named_segment(
    stream: &mut TokenStream,
    allow_generics: bool,
) -> Result<NamedSegment, Diagnostic> {
    let (name, span) = stream.expect_ident_spanned()?;

    let mut generics = Vec::new();

    if allow_generics && stream.take(Token::Lt) {
        loop {
            if stream.is(Token::Gt) {
                break;
            }

            let ty = parse_type(stream)?;
            generics.push(ty);

            if !stream.take(Token::Comma) {
                break;
            }
        }

        stream.expect(Token::Gt)?;
    }

    Ok(NamedSegment {
        name,
        generics,
        span,
    })
}

pub fn parse_assoc_segment(stream: &mut TokenStream) -> Result<AssocSegment, Diagnostic> {
    let start = stream.expect(Token::Lt)?;

    let implementor = parse_type(stream)?;

    stream.expect(Token::As)?;

    let trait_item = parse_item(stream, true)?;

    stream.expect(Token::Gt)?;

    stream.expect(Token::ColonColon)?;

    let (name, span) = stream.expect_ident_spanned()?;

    Ok(AssocSegment {
        implementor,
        trait_item,
        name,
        span: start.join(span),
    })
}

pub fn parse_item_segment(
    stream: &mut TokenStream,
    allow_generics: bool,
) -> Result<ItemSegment, Diagnostic> {
    let (name, span) = stream.peek();

    match name {
        Token::Ident(_) => Ok(ItemSegment::Named(parse_named_segment(
            stream,
            allow_generics,
        )?)),
        Token::Lt => Ok(ItemSegment::Assoc(parse_assoc_segment(stream)?)),
        Token::Quote => Ok(ItemSegment::Generic(parse_generic(stream)?)),
        Token::SelfLower => {
            stream.consume();
            Ok(ItemSegment::SelfLower(SelfLowerSegment { span }))
        }
        Token::SelfUpper => {
            stream.consume();
            Ok(ItemSegment::SelfUpper(SelfUpperSegment { span }))
        }
        _ => Err(Diagnostic::new("expected identifier").with_span(span)),
    }
}

pub fn parse_item(stream: &mut TokenStream, allow_generics: bool) -> Result<Item, Diagnostic> {
    let mut segments = Vec::new();

    let first_segment = parse_item_segment(stream, allow_generics)?;
    let mut span = first_segment.span();

    segments.push(first_segment);

    loop {
        if stream.take(Token::ColonColon) {
            let segment = parse_item_segment(stream, allow_generics)?;
            span = span.join(segment.span());
            segments.push(segment);
        } else {
            break;
        }
    }

    Ok(Item { segments, span })
}
