use std::fmt::Display;

use crate::{TraitId, Type};

/// A trait bound.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TraitBound {
    /// The base type of this bound.
    pub base: Type,

    /// The trait in question.
    pub trait_id: TraitId,

    /// The generics that specialize the trait.
    pub generics: Vec<Type>,

    /// The optionally specified associated types.
    pub types: Vec<Option<Type>>,
}

impl Display for TraitBound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}<", self.trait_id)?;

        let generics: Vec<_> = self.generics.iter().map(ToString::to_string).collect();
        write!(f, "{}", generics.join(", "))?;

        for (i, type_) in self.types.iter().enumerate() {
            match type_ {
                Some(type_) => write!(f, ", {} = {}", i, type_)?,
                None => write!(f, "")?,
            }
        }

        write!(f, ">")?;

        Ok(())
    }
}

/// A where clause.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Contract {
    /// The bounds of the where clause.
    pub bounds: Vec<TraitBound>,
}

impl Default for Contract {
    fn default() -> Self {
        Self::new()
    }
}

impl Contract {
    pub fn new() -> Self {
        Self { bounds: Vec::new() }
    }
}

ritec_arena::arena!(Contracts[ContractId]: Contract);

impl Display for ContractId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.index)
    }
}
