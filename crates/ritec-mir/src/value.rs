use crate::{Operand, Type};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
}

#[derive(Clone, Debug)]
pub enum Value {
    Use(Operand),

    Binary(BinaryOp, Operand, Operand),

    Struct(Vec<Operand>),

    Intrinsic(&'static str, Vec<Operand>, Type),
}

impl Value {
    pub const VOID: Self = Self::Use(Operand::VOID);

    pub fn ty(&self) -> Type {
        match self {
            Self::Use(op) => op.ty().clone(),
            Self::Binary(_, lhs, _) => lhs.ty().clone(),
            Self::Struct(ops) => {
                let fields = ops.iter().map(|op| op.ty().clone()).collect();
                Type::Struct { fields }
            }
            Self::Intrinsic(_, _, ty) => ty.clone(),
        }
    }
}
