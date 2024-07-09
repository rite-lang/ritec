use ritec_diagnostic::{Diagnostic, Span};
use ritec_parse::{Token, TokenStream};

#[derive(Clone, Debug)]
pub struct Generic {
    pub name: String,
    pub span: Span,
}

pub fn parse_generic(stream: &mut TokenStream) -> Result<Generic, Diagnostic> {
    let start = stream.expect(Token::Quote)?;
    let (name, end) = stream.expect_ident_spanned()?;

    Ok(Generic {
        name,
        span: start.join(end),
    })
}
