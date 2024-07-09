use std::fmt::Display;

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

/// A fully known type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Known {
    /// The item of this known type.
    pub item: Item,

    /// The parameters of this known type.
    pub params: Vec<Known>,
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
