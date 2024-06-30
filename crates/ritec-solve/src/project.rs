use crate::{TraitId, Variable};

/// A type projection.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Projection {
    /// An associated type.
    Associated {
        /// The trait that defines the associated type.
        trait_: TraitId,

        /// The index of the associated type.
        index: usize,
    },
}

/// A projected type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Projected {
    /// The base type of this projected type.
    pub base: Box<Variable>,

    /// The projection of this projected type.
    pub projection: Projection,
}
