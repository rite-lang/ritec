use crate::{Partial, Projected, Unknown};

/// A type variable.
#[derive(Clone, Debug, PartialEq)]
pub enum Variable {
    /// An unknown type.
    Unknown(Unknown),

    /// A partially known type.
    Partial(Partial),

    /// A projected type.
    Projected(Projected),
}
