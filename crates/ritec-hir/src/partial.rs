use std::fmt::Display;

use crate::{Item, Type};

/// A partially known type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Partial {
    pub item: Item,
    pub params: Vec<Type>,
}

impl Display for Partial {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<_> = self.params.iter().map(ToString::to_string).collect();
        write!(_f, "{}[{}]", self.item, params.join(", "))
    }
}
