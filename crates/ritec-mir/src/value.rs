use crate::{Operand, Place, Type};

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
    /// Use a value from an operand.
    Use(Operand),

    /// Perform a binary operation on two operands.
    Binary(BinaryOp, Operand, Operand),

    /// Get the address of a place.
    AddressOf(bool, Place),

    /// Create a struct from a list of operands.
    Struct(Vec<Operand>),

    /// Get the size of a type.
    Sizeof(Type),

    /// Call an intrinsic function.
    Intrinsic(&'static str, Vec<Operand>, Type),
}

impl Value {
    pub const VOID: Self = Self::Use(Operand::VOID);

    pub fn ty(&self) -> Type {
        match self {
            Self::Use(op) => op.ty().clone(),
            Self::Binary(_, lhs, _) => lhs.ty().clone(),
            Self::AddressOf(mutable, place) => Type::Pointer {
                mutable: *mutable,
                pointee: Box::new(place.ty().clone()),
            },
            Self::Struct(ops) => {
                let fields = ops.iter().map(|op| op.ty().clone()).collect();
                Type::Struct { fields }
            }
            Self::Sizeof(_) => Type::Int {
                signed: false,
                width: None,
            },
            Self::Intrinsic(_, _, ty) => ty.clone(),
        }
    }
}
