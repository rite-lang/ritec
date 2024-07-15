use std::fmt::Display;

use crate::{ContractId, Generic, Type};

/// A trait method.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TraitMethod {
    /// The name of the method.
    pub name: String,

    /// The generics of the method.
    pub generics: Vec<Generic>,

    /// The arguments of the method.
    pub arguments: Vec<Type>,

    /// The output of the method.
    pub output: Type,

    /// The where clause of the method.
    pub contract: ContractId,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Assoc {
    pub name: String,
}

/// A trait.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Trait {
    /// The name of the trait.
    pub name: Option<String>,

    /// The generics of the trait.
    pub generics: Vec<Generic>,

    /// The where clause of the trait.
    pub contract: ContractId,

    /// The associated types of the trait.
    pub assocs: Vec<Assoc>,

    /// The methods of the trait.
    pub methods: Vec<TraitMethod>,
}

ritec_arena::arena!(Traits[TraitId]: Trait);

impl Display for TraitId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "trait{}", self.index)
    }
}
