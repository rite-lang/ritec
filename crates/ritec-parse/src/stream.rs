use std::sync::Arc;

use ritec_diagnostic::{Diagnostic, Span};
use ritec_source::SourceId;
use crate::Token;

#[derive(Clone, Debug, PartialEq)]
pub struct TokenStream {
    tokens: Arc<[(Token, Span)]>,
    index: usize,
    span: Span,
    source_id: SourceId,
}

impl TokenStream {
    pub fn new(tokens: impl Into<Arc<[(Token, Span)]>>, span: Span, source_id: SourceId) -> Self {
        Self {
            tokens: tokens.into(),
            index: 0,
            span,
            source_id,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.index >= self.tokens.len()
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

    pub fn nth_is(&self, n: usize, token: Token) -> bool {
        self.peek_nth(n).0 == token
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

    pub fn consume_n(&mut self, n: usize) {
        self.index += n;
    }

    pub fn take_spanned(&mut self, token: Token) -> Option<Span> {
        if self.is(token) {
            Some(self.consume().1)
        } else {
            None
        }
    }

    pub fn take(&mut self, token: Token) -> bool {
        self.take_spanned(token).is_some()
    }

    pub fn expect(&mut self, token: Token) -> Result<Span, Diagnostic> {
        let (actual, span) = self.consume();

        if actual == token {
            Ok(span)
        } else {
            let message = format!("expected {}, found {}", token, actual);
            let diagnostic = Diagnostic::new(message).with_span(span);

            Err(diagnostic)
        }
    }

    pub fn expect_ident_spanned(&mut self) -> Result<(String, Span), Diagnostic> {
        let (token, span) = self.consume();

        match token {
            Token::Ident(ident) => Ok((ident, span)),
            actual => {
                let message = format!("expected identifier, found {}", actual);
                let diagnostic = Diagnostic::new(message).with_span(span);

                Err(diagnostic)
            }
        }
    }

    pub fn expect_ident(&mut self) -> Result<String, Diagnostic> {
        let (ident, _) = self.expect_ident_spanned()?;
        Ok(ident)
    }

    pub fn expect_string_spanned(&mut self) -> Result<(String, Span), Diagnostic> {
        let (token, span) = self.consume();

        match token {
            Token::String(string) => Ok((string, span)),
            actual => {
                let message = format!("expected string, found {}", actual);
                let diagnostic = Diagnostic::new(message).with_span(span);

                Err(diagnostic)
            }
        }
    }

    pub fn expect_string(&mut self) -> Result<String, Diagnostic> {
        let (string, _) = self.expect_string_spanned()?;
        Ok(string)
    }

    pub fn expect_integer_spanned(&mut self) -> Result<(u64, Span), Diagnostic> {
        let (token, span) = self.consume();

        match token {
            Token::Integer(integer) => Ok((integer, span)),
            actual => {
                let message = format!("expected integer, found {}", actual);
                let diagnostic = Diagnostic::new(message).with_span(span);

                Err(diagnostic)
            }
        }
    }

    pub fn expect_integer(&mut self) -> Result<u64, Diagnostic> {
        let (integer, _) = self.expect_integer_spanned()?;
        Ok(integer)
    }

    pub fn expect_float_spanned(&mut self) -> Result<(f64, Span), Diagnostic> {
        let (token, span) = self.consume();

        match token {
            Token::Float(float) => Ok((float, span)),
            actual => {
                let message = format!("expected float, found {}", actual);
                let diagnostic = Diagnostic::new(message).with_span(span);

                Err(diagnostic)
            }
        }
    }

    pub fn expect_float(&mut self) -> Result<f64, Diagnostic> {
        let (float, _) = self.expect_float_spanned()?;
        Ok(float)
    }

    pub fn spanned<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<T, Diagnostic>,
    ) -> Result<(T, Span), Diagnostic> {
        let start = self.peek().1.lo;

        let result = f(self)?;

        let end = match self.index.checked_sub(1) {
            Some(index) => self.tokens[index].1.hi,
            None => start,
        };

        Ok((result, Span::new(start, end)))
    }

    pub fn source_id(&self) -> SourceId {
        self.source_id
    }
}
