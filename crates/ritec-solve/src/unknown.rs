use std::sync::atomic::{AtomicUsize, Ordering};

use ritec_span::Span;

/// A unique identifier for an unknown type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Uid {
    id: usize,
}

impl Default for Uid {
    fn default() -> Self {
        Self::new()
    }
}

impl Uid {
    /// Create a new unique identifier.
    pub fn new() -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

        Self {
            id: NEXT_ID.fetch_add(1, Ordering::SeqCst),
        }
    }
}

/// An unknown type.
#[derive(Clone, Debug, PartialEq)]
pub struct Unknown {
    /// A unique identifier for this unknown type.
    pub uid: Uid,

    /// The span of this unknown type.
    pub span: Span,
}
