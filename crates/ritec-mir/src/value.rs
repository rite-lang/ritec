use crate::{Operand, Type};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
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

    Binary(BinOp, Operand, Operand),

    Intrinsic(&'static str, Vec<Operand>, Type),
}

impl Value {
    pub const VOID: Self = Self::Use(Operand::VOID);
}
