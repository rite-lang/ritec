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
}

impl Type {
    pub const VOID: Self = Self::Partial(Partial {
        item: Item::Void,
        params: Vec::new(),
    });

    pub const BOOL: Self = Self::Partial(Partial {
        item: Item::Bool,
        params: Vec::new(),
    });

    pub const U8: Self = Self::int(false, Some(8));
    pub const U16: Self = Self::int(false, Some(16));
    pub const U32: Self = Self::int(false, Some(32));
    pub const U64: Self = Self::int(false, Some(64));
    pub const U128: Self = Self::int(false, Some(128));
    pub const I8: Self = Self::int(true, Some(8));
    pub const I16: Self = Self::int(true, Some(16));
    pub const I32: Self = Self::int(true, Some(32));
    pub const I64: Self = Self::int(true, Some(64));
    pub const I128: Self = Self::int(true, Some(128));
    pub const USIZE: Self = Self::int(false, None);
    pub const ISIZE: Self = Self::int(true, None);

    pub const F32: Self = Self::Partial(Partial {
        item: Item::Float { width: 32 },
        params: Vec::new(),
    });

    pub const F64: Self = Self::Partial(Partial {
        item: Item::Float { width: 64 },
        params: Vec::new(),
    });

    pub const fn int(signed: bool, width: Option<u16>) -> Self {
        Self::Partial(Partial {
            item: Item::Int { signed, width },
            params: Vec::new(),
        })
    }

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
        }
    }
}
