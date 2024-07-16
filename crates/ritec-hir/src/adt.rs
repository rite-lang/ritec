use crate::{ContractId, Generic, Type};

#[derive(Clone, Debug)]
pub struct Variant {
    pub name: String,
    pub fields: Vec<Type>,
}

#[derive(Clone, Debug)]
pub struct Enum {
    pub name: Option<String>,
    pub contract: ContractId,
    pub generics: Vec<Generic>,
    pub variants: Vec<Variant>,
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
