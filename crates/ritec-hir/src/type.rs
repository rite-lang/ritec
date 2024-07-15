use std::fmt::Display;

use ritec_diagnostic::Span;

use crate::{Generic, Item, Partial, Projected, Uid, Unknown};

/// A type variable.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    /// An unknown type.
    Unknown(Unknown),

    /// A partially known type.
    Partial(Partial),

    /// A projected type.
    Projected(Projected),

    /// A type that accepts all types.
    Generic(Generic),

    SelfType,
}

impl Type {
    pub const VOID: Self = Self::Partial(Partial {
        item: Item::Void,
        params: Vec::new(),
    });

    pub fn unknown(span: Span) -> Self {
        Self::Unknown(Unknown {
            uid: Uid::new(),
            span,
        })
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Unknown(unknown) => write!(f, "{}", unknown),
            Type::Partial(partial) => write!(f, "{}", partial),
            Type::Projected(projected) => write!(f, "{}", projected),
            Type::Generic(generic) => write!(f, "{}", generic),
            Type::SelfType => write!(f, "Self"),
        }
    }
}
