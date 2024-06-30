use std::fmt::Display;

use crate::Variable;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Item {}

impl Display for Item {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            _ => todo!(),
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
