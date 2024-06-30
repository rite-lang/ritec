use crate::Variable;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Item {}

/// A partially known type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Partial {
    pub item: Item,
    pub params: Vec<Variable>,
}
