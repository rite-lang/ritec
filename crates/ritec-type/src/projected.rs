use std::fmt::Display;

use crate::{TraitId, Variable, WhereId};

/// A type projection.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Projection {
    /// An associated type.
    Associated {
        /// The trait that defines the associated type.
        trait_: TraitId,

        /// The generics that specialize the trait.
        generics: Vec<Variable>,

        /// The index of the associated type.
        index: usize,
    },
}

impl Display for Projection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Projection::Associated {
                trait_,
                generics,
                index,
            } => {
                let generics: Vec<_> = generics.iter().map(ToString::to_string).collect();

                write!(f, "{}<{}> assoc {}", trait_, generics.join(", "), index)
            }
        }
    }
}

/// A projected type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Projected {
    /// The where clause this is attached to.
    pub where_: WhereId,

    /// The base type of this projected type.
    pub base: Box<Variable>,

    /// The projection of this projected type.
    pub projection: Projection,
}

impl Display for Projected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} as {}", self.base, self.projection)
    }
}
