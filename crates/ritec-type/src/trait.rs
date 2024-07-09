use std::fmt::Display;

use crate::{Forall, Variable, WhereId};

/// A trait method.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TraitMethod {
    /// The arguments of the method.
    pub arguments: Vec<Variable>,

    /// The output of the method.
    pub output: Variable,

    /// The where clause of the method.
    pub where_: WhereId,
}

/// A trait.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Trait {
    /// The generics of the trait.
    pub generics: Vec<Forall>,

    /// The where clause of the trait.
    pub where_: WhereId,

    /// The associated types of the trait.
    pub types: u32,
}

ritec_arena::arena!(Traits[TraitId]: Trait);

impl Display for TraitId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "trait{}", self.index)
    }
}
