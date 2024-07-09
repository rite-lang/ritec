use crate::{TraitId, Variable, WhereId};

/// A trait implementation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TraitImpl {
    /// The trait that is implemented.
    pub trait_: TraitId,

    /// The generics that specialize the trait.
    pub generics: Vec<Variable>,

    /// The where clause of the implementation.
    pub where_: WhereId,

    /// The type that the implementation is for.
    pub for_: Variable,

    /// The associated types of the implementation.
    pub types: Vec<Variable>,
}
