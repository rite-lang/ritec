use crate::{Const, Place, Type};

#[derive(Clone, Debug)]
pub enum Operand {
    Copy(Place),

    Move(Place),

    Const(Const),
}

impl Operand {
    pub const VOID: Self = Self::Const(Const::VOID);

    pub fn ty(&self) -> &Type {
        match self {
            Self::Copy(place) | Self::Move(place) => place.ty(),
            Self::Const(constant) => &constant.ty,
        }
    }
}
