use crate::{ContractId, TraitId, Type};

/// A trait implementation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TraitImpl {
    /// The trait that is implemented.
    pub trait_: TraitId,

    /// The generics that specialize the trait.
    pub generics: Vec<Type>,

    /// The contract of the implementation.
    pub contract: ContractId,

    /// The type that the implementation is for.
    pub for_: Type,

    /// The associated types of the implementation.
    pub types: Vec<Type>,
}
