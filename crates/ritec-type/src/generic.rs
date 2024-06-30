use std::sync::atomic::{AtomicUsize, Ordering};

/// A unique identifier for a generic type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Generic {
    id: usize,
}

impl Default for Generic {
    fn default() -> Self {
        Self::new()
    }
}

impl Generic {
    /// Create a new unique identifier.
    pub fn new() -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

        Self {
            id: NEXT_ID.fetch_add(1, Ordering::SeqCst),
        }
    }
}
