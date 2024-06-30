use std::fmt::Display;

use crate::{TraitId, Variable, WhereId};

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

impl Display for Projection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Projection::Associated { trait_, index } => write!(f, "<{}>::{}", trait_, index),
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
        write!(f, "{}::{}", self.base, self.projection)
    }
}
