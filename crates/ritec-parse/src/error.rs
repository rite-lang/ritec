use ritec_diagnostic::{Diagnostic, Span};

#[derive(Debug)]
pub struct ParseError {
    diagnostic: Diagnostic,
}

impl ParseError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            diagnostic: Diagnostic::new(message),
        }
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.diagnostic = self.diagnostic.with_span(span);
        self
    }

    pub fn diagnostic(&self) -> &Diagnostic {
        &self.diagnostic
    }
}

impl From<Diagnostic> for ParseError {
    fn from(diagnostic: Diagnostic) -> Self {
        Self { diagnostic }
    }
}
