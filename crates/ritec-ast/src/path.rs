use ritec_diagnostic::{Diagnostic, Span};
use ritec_parse::{Token, TokenStream};

use crate::{parse_generic, parse_type, Generic, Type};

/// name::name<'a, 'b>::name <- Three named segments, middle has generics
#[derive(Clone, Debug)]
pub struct NamedSegment {
    pub name: String,
    pub generics: Vec<Type>,
    pub span: Span,
}

/// <Type as Trait>::name <- AssocSegment
#[derive(Clone, Debug)]
pub struct AssocSegment {
    pub implementor: Type,
    pub trait_path: Path,
    pub name: String,
    pub span: Span,
}

/// self
#[derive(Clone, Debug)]
pub struct SelfLowerSegment {
    pub span: Span,
}

/// Self
/// fn func(self) -> Self <-- Type in other contexts
#[derive(Clone, Debug)]
pub struct SelfUpperSegment {
    pub span: Span,
}

/// Any segment
#[derive(Clone, Debug)]
pub enum PathSegment {
    Assoc(AssocSegment),
    Named(NamedSegment),
    Generic(Generic),
    SelfLower(SelfLowerSegment),
    SelfUpper(SelfUpperSegment),
}

impl PathSegment {
    pub fn span(&self) -> Span {
        match self {
            PathSegment::Named(segment) => segment.span,
            PathSegment::Assoc(segment) => segment.span,
            PathSegment::Generic(generic) => generic.span,
            PathSegment::SelfLower(segment) => segment.span,
            PathSegment::SelfUpper(segment) => segment.span,
        }
    }
}

/// A list of paths
#[derive(Clone, Debug)]
pub struct Path {
    pub segments: Vec<PathSegment>,
    pub span: Span,
}

impl Path {
    pub fn ident(&self) -> Option<&str> {
        if self.segments.len() != 1 {
            return None;
        }

        match self.segments.first() {
            Some(PathSegment::Named(segment)) if segment.generics.is_empty() => Some(&segment.name),
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

    let trait_path = parse_path(stream, true)?;

    stream.expect(Token::Gt)?;

    stream.expect(Token::ColonColon)?;

    let (name, span) = stream.expect_ident_spanned()?;

    Ok(AssocSegment {
        implementor,
        trait_path,
        name,
        span: start.join(span),
    })
}

pub fn parse_path_segment(
    stream: &mut TokenStream,
    allow_generics: bool,
) -> Result<PathSegment, Diagnostic> {
    let (name, span) = stream.peek();

    match name {
        Token::Ident(_) => Ok(PathSegment::Named(parse_named_segment(
            stream,
            allow_generics,
        )?)),
        Token::Lt => Ok(PathSegment::Assoc(parse_assoc_segment(stream)?)),
        Token::Quote => Ok(PathSegment::Generic(parse_generic(stream)?)),
        Token::SelfLower => {
            stream.consume();
            Ok(PathSegment::SelfLower(SelfLowerSegment { span }))
        }
        Token::SelfUpper => {
            stream.consume();
            Ok(PathSegment::SelfUpper(SelfUpperSegment { span }))
        }
        _ => Err(Diagnostic::new("expected identifier").with_span(span)),
    }
}

pub fn parse_path(stream: &mut TokenStream, allow_generics: bool) -> Result<Path, Diagnostic> {
    let mut segments = Vec::new();

    let first_segment = parse_path_segment(stream, allow_generics)?;
    let mut span = first_segment.span();

    segments.push(first_segment);

    loop {
        if stream.take(Token::ColonColon) {
            let segment = parse_path_segment(stream, allow_generics)?;
            span = span.join(segment.span());
            segments.push(segment);
        } else {
            break;
        }
    }

    Ok(Path { segments, span })
}
