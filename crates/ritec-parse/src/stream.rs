use std::sync::Arc;

use ritec_diagnostic::Diagnostic;
use ritec_span::Span;

use crate::{ParseError, Token};

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

    pub fn eof_span(&self) -> Span {
        Span::new(self.span.hi, self.span.hi)
    }

    pub fn peek(&self) -> (Token, Span) {
        match self.tokens.get(self.index) {
            Some(token) => token.clone(),
            None => (Token::Eof, self.eof_span()),
        }
    }

    pub fn peek_nth(&self, n: usize) -> (Token, Span) {
        match self.tokens.get(self.index + n) {
            Some(token) => token.clone(),
            None => (Token::Eof, self.eof_span()),
        }
    }

    pub fn is(&self, token: Token) -> bool {
        self.peek().0 == token
    }

    pub fn consume(&mut self) -> (Token, Span) {
        match self.tokens.get(self.index) {
            Some(token) => {
                self.index += 1;
                token.clone()
            }
            None => (Token::Eof, self.eof_span()),
        }
    }

    pub fn consume_nth(&mut self, n: usize) -> (Token, Span) {
        match self.tokens.get(self.index + n) {
            Some(token) => {
                self.index += n + 1;
                token.clone()
            }
            None => (Token::Eof, self.eof_span()),
        }
    }

    pub fn expect(&mut self, token: Token) -> Result<Span, ParseError> {
        let (actual, span) = self.peek();

        if actual == token {
            Ok(span)
        } else {
            let message = format!("expected {}, found {}", token, actual);
            let diagnostic = Diagnostic::new(message).with_span(span);

            Err(ParseError::from(diagnostic))
        }
    }

    pub fn spanned<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<T, ParseError>,
    ) -> Result<(T, Span), ParseError> {
        let start = self.peek().1.lo;

        let result = f(self)?;

        let end = match self.index.checked_sub(1) {
            Some(index) => self.tokens[index].1.hi,
            None => start,
        };

        Ok((result, Span::new(start, end)))
    }
}
