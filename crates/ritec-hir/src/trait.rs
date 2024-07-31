use ritec_diagnostic::Span;

use crate::{BodyId, ContractId, Generic, KnownTy, Ty};

ritec_arena::arena!(Traits[TraitId]: TraitDef);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LangTrait {
    Add,
}

/// A definition of a trait.
#[derive(Clone, Debug, PartialEq)]
pub struct TraitDef {
    /// The generic that represents `Self`.
    pub self_generic: Generic,

    /// The optional language trait that the trait represents.
    pub lang_trait: Option<LangTrait>,

    /// The optional name of the trait.
    pub name: Option<String>,

    /// The generics of the trait.
    pub generics: Vec<Generic>,

    /// The contract that implementations of the trait must satisfy.
    pub contract: ContractId,

    /// The associated types of the trait.
    pub assocs: Vec<AssocDef>,

    /// The methods of the trait.
    pub methods: Vec<MethodDef>,

    /// The optional span of the trait.
    pub span: Option<Span>,
}

impl TraitDef {
    pub fn self_ty(&self) -> Ty {
        Ty::Generic(self.self_generic)
    }

    pub fn assoc_index(&self, name: &str) -> Option<usize> {
        self.assocs.iter().position(|assoc| assoc.name == name)
    }

    pub fn method_index(&self, name: &str) -> Option<usize> {
        self.methods.iter().position(|method| method.name == name)
    }
}

/// A definition of an associated type in a trait.
#[derive(Clone, Debug, PartialEq)]
pub struct AssocDef {
    /// The name of the associated type.
    pub name: String,

    /// The optional span of the associated type.
    pub span: Option<Span>,
}

/// A definition of a method in a trait.
#[derive(Clone, Debug, PartialEq)]
pub struct MethodDef {
    /// The name of the method.
    pub name: String,

    /// The generics of the method.
    pub generics: Vec<Generic>,

    /// The arguments of the method.
    pub arguments: Vec<KnownTy>,

    /// The output of the method.
    pub output: KnownTy,

    /// The contract that the method must satisfy.
    pub contract: ContractId,

    /// The optional span of the method.
    pub span: Option<Span>,
}

/// A trait implementation.
#[derive(Clone, Debug, PartialEq)]
pub struct TraitImpl {
    /// The implementor of the trait.
    pub trait_id: TraitId,

    /// The types that specialize the trait.
    pub generics: Vec<KnownTy>,

    /// The type that implements the trait.
    pub implementor: KnownTy,

    /// The Contract that the implementation satisfies.
    pub contract: ContractId,

    /// The associated types of the implementation.
    pub assocs: Vec<AssocImpl>,

    /// The methods of the implementation.
    pub methods: Vec<MethodImpl>,

    /// The optional span of the implementation.
    pub span: Option<Span>,
}

/// An implementation of an associated type.
#[derive(Clone, Debug, PartialEq)]
pub struct AssocImpl {
    /// The type that implements the associated type.
    pub ty: KnownTy,

    /// The optional span of the associated type.
    pub span: Option<Span>,
}

/// An implementation of a method.
#[derive(Clone, Debug, PartialEq)]
pub struct MethodImpl {
    /// The name of the method.
    pub name: String,

    /// The body of the method.
    pub body: BodyId,

    /// The optional span of the method.
    pub span: Option<Span>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Impl {
    pub generics: Vec<Generic>,
    pub implementor: KnownTy,
    pub contract: ContractId,
    pub methods: Vec<MethodImpl>,
}
