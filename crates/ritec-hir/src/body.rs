use crate::{ContractId, Expr, Generic, Type};

ritec_arena::arena!(Locals[LocalId]: Local);
ritec_arena::arena!(Bodies[BodyId]: Body);

#[derive(Clone, Debug)]
pub struct Local {
    pub mutable: bool,
    pub name: Option<String>,
    pub ty: Type,
}

#[derive(Clone, Debug)]
pub struct Body {
    pub name: Option<String>,
    pub arguments: Vec<LocalId>,
    pub output: Type,
    pub generics: Vec<Generic>,
    pub contract: ContractId,
    pub locals: Locals,
    pub expr: Expr,
}

impl Body {
    pub fn is_generic(&self) -> bool {
        !self.generics.is_empty()
    }
    pub fn get_args(&self) -> Vec<Type> {
        self.arguments.iter().map(|id| self.locals[*id].ty.clone()).collect()
    }
}
