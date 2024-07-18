use std::fmt::Display;

use crate::{EnumId, Partial, StructId, Type};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Item {
    Void,
    Bool,
    Int { signed: bool, width: Option<u16> },
    Float { width: u16 },
    Pointer { mutable: bool },
    Tuple,
    Slice,
    Function,
    Struct(StructId),
    Enum(EnumId),
}

impl Display for Item {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Void => write!(f, "void"),
            Self::Bool => write!(f, "bool"),
            Self::Int { signed, width } => {
                if *signed {
                    write!(f, "i")
                } else {
                    write!(f, "u")
                }?;

                if let Some(width) = width {
                    write!(f, "{}", width)?;
                } else {
                    write!(f, "size")?;
                }

                Ok(())
            }
            Self::Float { width } => write!(f, "f{}", width),
            Self::Pointer { mutable } => {
                if *mutable {
                    write!(f, "*mut")
                } else {
                    write!(f, "*")
                }
            }
            Self::Tuple => write!(f, "tuple"),
            Self::Slice => write!(f, "slice"),
            Self::Function => write!(f, "fn"),
            Self::Struct(id) => write!(f, "{:?}", id),
            Self::Enum(id) => write!(f, "{:?}", id),
        }
    }
}

/// A fully known type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Known {
    /// The item of this known type.
    pub item: Item,

    /// The parameters of this known type.
    pub params: Vec<Known>,
}

impl Known {
    pub fn to_type(&self) -> Type {
        Type::Partial(Partial {
            item: self.item.clone(),
            params: self.params.iter().map(Known::to_type).collect(),
        })
    }
}

impl Display for Known {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.item)?;

        if !self.params.is_empty() {
            write!(f, "<")?;

            let params: Vec<_> = self.params.iter().map(ToString::to_string).collect();
            write!(f, "{}", params.join(", "))?;

            write!(f, ">")?;
        }

        Ok(())
    }
}
