use crate::{BlockId, Operand, Place, Value};

#[derive(Clone, Debug)]
pub enum Terminator {
    Goto(BlockId),

    Return(Value),

    Call {
        callee: Operand,
        arguments: Vec<Operand>,
        destination: Place,
        target: Option<BlockId>,
    },

    Unreachable,
}
