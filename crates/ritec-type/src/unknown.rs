use std::{
    fmt::Display,
    sync::atomic::{AtomicUsize, Ordering},
};

use ritec_diagnostic::Span;

use crate::WhereId;

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

impl Display for Uid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id)
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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Unknown {
    pub where_id: WhereId,

    /// A unique identifier for this unknown type.
    pub uid: Uid,

    /// The span of this unknown type.
    pub span: Span,
}

impl Display for Unknown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "_{} in {}", self.uid, self.where_id)?;

        Ok(())
    }
}
