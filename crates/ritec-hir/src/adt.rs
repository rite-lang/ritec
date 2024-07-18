use crate::{BodyId, ContractId, Generic, Type};

#[derive(Clone, Debug)]
pub struct Variant {
    pub name: String,
    pub discriminant: u64,
    pub fields: Vec<Type>,
    pub builder: Option<BodyId>,
}

#[derive(Clone, Debug)]
pub struct Enum {
    pub name: Option<String>,
    pub contract: ContractId,
    pub generics: Vec<Generic>,
    pub variants: Vec<Variant>,
}

impl Enum {
    pub fn variant_index(&self, name: &str) -> Option<usize> {
        self.variants.iter().position(|v| v.name == name)
    }
}

#[derive(Clone, Debug)]
pub struct Field {
    pub name: String,
    pub ty: Type,
}

#[derive(Clone, Debug)]
pub struct Struct {
    pub name: Option<String>,
    pub contract: ContractId,
    pub generics: Vec<Generic>,
    pub fields: Vec<Field>,
}

impl Struct {
    pub fn field_index(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|field| field.name == name)
    }
}

ritec_arena::arena!(Enums[EnumId]: Enum);
ritec_arena::arena!(Structs[StructId]: Struct);
