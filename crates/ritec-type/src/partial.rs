use std::fmt::Display;

use crate::Variable;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Item {
    Void,
    Bool,
    Pointer { mutable: bool },
    Tuple,
    Slice,
    Function,
}

impl Display for Item {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Void => write!(f, "void"),
            Self::Bool => write!(f, "bool"),
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
        }
    }
}

/// A partially known type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Partial {
    pub item: Item,
    pub params: Vec<Variable>,
}

impl Display for Partial {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<_> = self.params.iter().map(ToString::to_string).collect();
        write!(_f, "{}[{}]", self.item, params.join(", "))
    }
}
