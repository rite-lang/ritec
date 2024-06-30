use crate::{TraitId, Variable};

/// A trait bound.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TraitBound {
    /// The trait in question.
    pub trait_: TraitId,

    /// The generics that specialize the trait.
    pub generics: Vec<Variable>,

    /// The optionally specified associated types.
    pub types: Vec<Option<Variable>>,
}

/// A bound in a where clause.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Bound {
    /// The base type of this bound.
    pub base: Variable,

    /// The trait bounds of this bound.
    pub traits: Vec<TraitBound>,
}

/// A where clause.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Where {
    /// The bounds of the where clause.
    pub bounds: Vec<Bound>,
}
