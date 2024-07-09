use std::fmt::Display;

use crate::{TraitId, Variable};

/// A trait bound.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TraitBound {
    /// The base type of this bound.
    pub base: Variable,

    /// The trait in question.
    pub trait_: TraitId,

    /// The generics that specialize the trait.
    pub generics: Vec<Variable>,

    /// The optionally specified associated types.
    pub types: Vec<Option<Variable>>,
}

impl Display for TraitBound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}<", self.trait_)?;

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
pub struct Where {
    /// The parent of the where clause.
    ///
    /// This clause will inherit the bounds of the parent.
    pub parent: Option<WhereId>,

    /// The bounds of the where clause.
    pub bounds: Vec<TraitBound>,
}

impl Default for Where {
    fn default() -> Self {
        Self::new()
    }
}

impl Where {
    pub fn new() -> Self {
        Self {
            parent: None,
            bounds: Vec::new(),
        }
    }
}

ritec_arena::arena!(Wheres[WhereId]: Where);

impl Display for WhereId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.index)
    }
}
