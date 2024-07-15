use crate::{Statement, Terminator};

#[derive(Clone, Debug)]
pub struct Block {
    pub statements: Vec<Statement>,
    pub terminator: Terminator,
}

impl Default for Block {
    fn default() -> Self {
        Self::new()
    }
}

impl Block {
    pub fn new() -> Self {
        Self {
            statements: Vec::new(),
            terminator: Terminator::Unreachable,
        }
    }
}

ritec_arena::arena!(Blocks[BlockId]: Block);
