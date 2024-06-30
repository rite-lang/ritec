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
}
