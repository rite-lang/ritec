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
    /// The self type of the trait.
    ///
    /// This is implemented as a generic, that will be specialized.
    pub self_generic: Generic,

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

impl Trait {
    pub fn self_type(&self) -> Type {
        Type::Generic(self.self_generic)
    }

    pub fn assoc_index(&self, name: &str) -> Option<usize> {
        self.assocs.iter().position(|assoc| assoc.name == name)
    }

    pub fn method_index(&self, name: &str) -> Option<usize> {
        self.methods.iter().position(|method| method.name == name)
    }
}

ritec_arena::arena!(Traits[TraitId]: Trait);

impl Display for TraitId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "trait{}", self.index)
    }
}
