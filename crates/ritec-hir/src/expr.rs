use ritec_diagnostic::Span;

use crate::{BodyId, LocalId, TraitId, Type};

#[derive(Clone, Debug)]
pub enum Constant {
    Void,
    Int(u64),
    Float(f64),
    Func(BodyId, Vec<Type>),
    Method {
        implementor: Type,
        trait_id: TraitId,
        generics: Vec<Type>,
        index: usize,
    },
}

#[derive(Clone, Debug)]
pub enum ExprKind {
    Const(Constant),
    Local(LocalId),
    Let(LocalId, Box<Expr>),
    Assign(Box<Expr>, Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    Block(Vec<Expr>),
    Intrinsic(&'static str, Vec<Expr>),
}

#[derive(Clone, Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Option<Span>,
    pub ty: Type,
}

impl Expr {
    pub fn void(span: Span) -> Self {
        Self {
            kind: ExprKind::Const(Constant::Void),
            span: Some(span),
            ty: Type::VOID,
        }
    }
}
