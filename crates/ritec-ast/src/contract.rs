use ritec_diagnostic::{Diagnostic, Span};
use ritec_parse::{Token, TokenStream};

use crate::{parse_path, parse_type, Path, Type};

#[derive(Clone, Debug)]
pub struct AssocBound {
    pub name: String,
    pub type_: Type,
}

#[derive(Clone, Debug)]
pub struct TraitBound {
    pub item: Path,
    pub generics: Vec<Type>,
    pub types: Vec<AssocBound>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Clause {
    pub bindee: Type,
    pub bounds: Vec<TraitBound>,
}

#[derive(Clone, Debug, Default)]
pub struct Contract {
    pub clauses: Vec<Clause>,
}

fn is_assoc_bound(stream: &mut TokenStream) -> bool {
    let is_name = matches!(stream.peek_nth(0).0, Token::Ident(_));
    let is_eq = matches!(stream.peek_nth(1).0, Token::Eq);

    is_name && is_eq
}

pub fn parse_assoc_bound(stream: &mut TokenStream) -> Result<AssocBound, Diagnostic> {
    let name = stream.expect_ident()?;
    stream.expect(Token::Eq)?;
    let type_ = parse_type(stream)?;

    Ok(AssocBound { name, type_ })
}

pub fn parse_trait_bound(stream: &mut TokenStream) -> Result<TraitBound, Diagnostic> {
    let item = parse_path(stream, false)?;

    let mut generics = Vec::new();
    let mut types = Vec::new();

    if stream.take(Token::Lt) {
        loop {
            if stream.is(Token::Gt) {
                break;
            }

            if is_assoc_bound(stream) {
                let assoc = parse_assoc_bound(stream)?;
                types.push(assoc);
            } else {
                let ty = parse_type(stream)?;
                generics.push(ty);
            }

            if !stream.take(Token::Comma) {
                break;
            }
        }

        stream.expect(Token::Gt)?;
    }

    let span = item.span.join(stream.peek().1);

    Ok(TraitBound {
        item,
        generics,
        types,
        span,
    })
}

pub fn parse_clause(stream: &mut TokenStream) -> Result<Clause, Diagnostic> {
    let bindee = parse_type(stream)?;

    stream.expect(Token::Colon)?;

    let mut bounds = Vec::new();

    loop {
        if stream.is(Token::Newline) {
            break;
        }

        let bound = parse_trait_bound(stream)?;
        bounds.push(bound);

        if !stream.take(Token::Comma) {
            break;
        }
    }

    stream.expect(Token::Newline)?;

    Ok(Clause { bindee, bounds })
}

pub fn parse_contract(stream: &mut TokenStream) -> Result<Contract, Diagnostic> {
    stream.expect(Token::Newline)?;

    let mut bounds = Vec::new();

    loop {
        if !stream.is(Token::Or) {
            break;
        }

        stream.consume();

        let clause = parse_clause(stream)?;
        bounds.push(clause);
    }

    Ok(Contract { clauses: bounds })
}
