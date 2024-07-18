use crate::{BodyId, ContractId, Generic, Item, Partial, TraitId, Type};

/// A trait implementation.
#[derive(Clone, Debug)]
pub struct TraitImpl {
    /// The trait that is implemented.
    pub trait_id: TraitId,

    /// The generics that specialize the trait.
    pub generics: Vec<Type>,

    /// The type that the implementation is for.
    pub implementor: Type,

    /// The contract of the implementation.
    pub contract: ContractId,

    /// The associated types of the implementation.
    pub types: Vec<Type>,

    /// The methods of the implementation.
    pub methods: Vec<Method>,
}

#[derive(Clone, Debug)]
pub struct Impl {
    /// The trait that is implemented.
    pub generics: Vec<Generic>,

    /// The generics that specialize the trait.
    pub implementor: Type,

    /// The contract of the implementation.
    pub contract: ContractId,

    /// The associated types of the implementation.
    pub methods: Vec<Method>,
}

#[derive(Clone, Debug)]
pub struct Method {
    /// The name of the method.
    pub name: String,

    /// Generics
    pub generics: Vec<Generic>,

    /// Arguments
    pub arguments: Vec<Type>,

    /// Return type
    pub output: Type,

    /// The body of the method.
    pub body: BodyId,
}

impl Method {
    pub fn function_type(&self) -> Type {
        let mut params = Vec::new();

        params.push(self.output.clone());

        for argument in &self.arguments {
            params.push(argument.clone());
        }

        Type::Partial(Partial {
            item: Item::Function,
            params,
        })
    }
}
