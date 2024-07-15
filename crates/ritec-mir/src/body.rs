use crate::{BlockId, Blocks, Local, Locals, Type};

#[derive(Clone, Debug)]
pub struct Body {
    pub arguments: Vec<Local>,
    pub output: Type,
    pub locals: Locals,
    pub blocks: Blocks,
    pub entry: BlockId,
}

ritec_arena::arena!(Bodies[BodyId]: Body);
