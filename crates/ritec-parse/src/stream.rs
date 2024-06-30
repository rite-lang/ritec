use std::sync::Arc;

use ritec_span::Span;

use crate::Token;

#[derive(Clone, Debug, PartialEq)]
pub struct TokenStream {
    tokens: Arc<[(Token, Span)]>,
    index: usize,
    span: Span,
}

impl TokenStream {
    pub fn new(tokens: impl Into<Arc<[(Token, Span)]>>, span: Span) -> Self {
        Self {
            tokens: tokens.into(),
            index: 0,
            span,
        }
    }

    pub fn peek(&self) -> (Token, Span) {
        let eof_span = Span::new(self.span.hi, self.span.hi);

        match self.tokens.get(self.index) {
            Some(token) => token.clone(),
            None => (Token::Eof, eof_span),
        }
    }

    pub fn consume(&mut self) -> (Token, Span) {
        let eof_span = Span::new(self.span.hi, self.span.hi);

        match self.tokens.get(self.index) {
            Some(token) => {
                self.index += 1;
                token.clone()
            }
            None => (Token::Eof, eof_span),
        }
    }
}
