use crate::{TraitBound, Variable, Where};

/// An associated type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TraitType {
    /// The bounds of the associated type.
    pub bounds: Vec<TraitBound>,
}

/// A trait method.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TraitMethod {
    /// The arguments of the method.
    pub arguments: Vec<Variable>,

    /// The output of the method.
    pub output: Variable,

    /// The where clause of the method.
    pub where_: Where,
}

/// A trait.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Trait {
    /// The where clause of the trait.
    pub where_: Where,

    /// The associated types of the trait.
    pub types: Vec<TraitType>,
}

ritec_arena::arena!(Traits[TraitId]: Trait);
