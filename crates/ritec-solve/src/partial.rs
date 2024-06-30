use crate::Variable;

#[derive(Clone, Debug, PartialEq)]
pub enum Item {}

/// A partially known type.
#[derive(Clone, Debug, PartialEq)]
pub struct Partial {
    pub item: Item,
    pub params: Vec<Variable>,
}
