use crate::{BodyId, Type};

#[derive(Clone, Debug)]
pub struct Const {
    pub kind: ConstKind,
    pub ty: Type,
}

impl Const {
    pub const VOID: Self = Self {
        kind: ConstKind::Struct(Vec::new()),
        ty: Type::VOID,
    };
}

#[derive(Clone, Debug)]
pub enum ConstKind {
    Int(i64),
    Float(f64),
    Struct(Vec<Const>),
    Body(BodyId),
}
