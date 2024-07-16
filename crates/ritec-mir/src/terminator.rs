use crate::{BlockId, Operand, Place, Value};

#[derive(Clone, Debug)]
pub enum Terminator {
    Goto(BlockId),

    Return(Value),

    Switch {
        discriminant: Operand,
        default: BlockId,
        cases: Vec<(u64, BlockId)>,
    },

    Call {
        callee: Operand,
        arguments: Vec<Operand>,
        destination: Place,
        target: Option<BlockId>,
    },

    Unreachable,
}
