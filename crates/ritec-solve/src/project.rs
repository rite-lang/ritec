use crate::Variable;

/// An associated type.
#[derive(Clone, Debug, PartialEq)]
pub struct Assoc {
    pub trait_: todo!(),
}

/// A type projection.
#[derive(Clone, Debug, PartialEq)]
pub enum Projection {
    /// An associated type.
    Assoc(Assoc),
}

/// A projected type.
#[derive(Clone, Debug, PartialEq)]
pub struct Projected {
    /// The base type of this projected type.
    pub base: Box<Variable>,

    /// The projection of this projected type.
    pub projection: Projection,
}
