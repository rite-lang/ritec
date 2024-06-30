use std::fmt::Display;

use crate::{Partial, Projected, Unknown};

/// A type variable.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Variable {
    /// An unknown type.
    Unknown(Unknown),

    /// A partially known type.
    Partial(Partial),

    /// A projected type.
    Projected(Projected),
}

impl Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Variable::Unknown(unknown) => write!(f, "{}", unknown),
            Variable::Partial(partial) => write!(f, "{}", partial),
            Variable::Projected(projected) => write!(f, "{}", projected),
        }
    }
}
