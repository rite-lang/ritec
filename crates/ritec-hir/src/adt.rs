use crate::{ContractId, Generic, Type};

#[derive(Clone, Debug)]
pub struct Variant {
    pub name: String,
    pub fields: Vec<Type>,
}

#[derive(Clone, Debug)]
pub struct Enum {
    pub contract: ContractId,
    pub generics: Vec<Generic>,
    pub variants: Vec<Variant>,
}

#[derive(Clone, Debug)]
pub struct Field {
    pub name: String,
    pub type_: Type,
}

#[derive(Clone, Debug)]
pub struct Struct {
    pub contract: ContractId,
    pub generics: Vec<Generic>,
    pub fields: Vec<Field>,
}

ritec_arena::arena!(Enums[EnumId]: Enum);
ritec_arena::arena!(Structs[StructId]: Struct);
