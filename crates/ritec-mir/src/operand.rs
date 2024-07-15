use crate::{Const, Place};

#[derive(Clone, Debug)]
pub enum Operand {
    Copy(Place),

    Move(Place),

    Const(Const),
}

impl Operand {
    pub const VOID: Self = Self::Const(Const::VOID);
}
