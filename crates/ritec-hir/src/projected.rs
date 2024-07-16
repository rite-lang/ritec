use std::fmt::Display;

use crate::{ContractId, TraitId, Type};

/// A type projection.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Projection {
    /// An associated type.
    AssocType {
        /// The trait that defines the associated type.
        trait_id: TraitId,

        /// The generics that specialize the trait.
        generics: Vec<Type>,

        /// The index of the associated type.
        index: usize,
    },

    AssocMethod {
        name: String,
        generics: Vec<Type>,
    },

    /// A field projection.
    Field { name: String },
}

impl Display for Projection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Projection::AssocType {
                trait_id: trait_,
                generics,
                index,
            } => {
                let generics: Vec<_> = generics.iter().map(ToString::to_string).collect();

                write!(f, "{}<{}> assoc {}", trait_, generics.join(", "), index)
            }
            Projection::AssocMethod { name, generics } => {
                let generics: Vec<_> = generics.iter().map(ToString::to_string).collect();

                write!(f, "method {}<{}>", generics.join(", "), name)
            }
            Projection::Field { name } => write!(f, "{}", name),
        }
    }
}

/// A projected type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Projected {
    /// The where clause this is attached to.
    pub contract: ContractId,

    /// The base type of this projected type.
    pub base: Box<Type>,

    /// The projection of this projected type.
    pub projection: Projection,
}

impl Display for Projected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} as {}", self.base, self.projection)
    }
}
