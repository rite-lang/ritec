use ritec_diagnostic::Diagnostic;

use crate::{ContractId, Expr, Generic, KnownTy, Spec, Ty};

ritec_arena::arena!(Locals[LocalId]: Local);
ritec_arena::arena!(Bodies[BodyId]: Body);

#[derive(Clone, Debug)]
pub struct Local {
    pub mutable: bool,
    pub name: Option<String>,
    pub ty: Ty,
}

#[derive(Clone, Debug)]
pub struct Body {
    pub name: Option<String>,
    pub arguments: Vec<Argument>,
    pub output: KnownTy,
    pub generics: Vec<Generic>,
    pub contract: ContractId,
    pub locals: Locals,
    pub expr: Expr,
}

impl Body {
    pub fn is_generic(&self) -> bool {
        !self.generics.is_empty()
    }

    pub fn get_args(&self) -> Vec<Ty> {
        self.arguments.iter().map(|a| a.ty.to_ty()).collect()
    }

    pub fn func_ty(&self, generics: &[Ty]) -> Result<Ty, Diagnostic> {
        let spec = Spec::specified(&self.generics, generics)?;
        Ok(spec.specialize(&self.output))
    }
}

#[derive(Clone, Debug)]
pub struct Argument {
    pub name: Option<String>,
    pub local: LocalId,
    pub ty: KnownTy,
}
