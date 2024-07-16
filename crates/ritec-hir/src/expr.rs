use ritec_diagnostic::Span;

use crate::{BodyId, LocalId, StructId, TraitId, Type};

#[derive(Clone, Debug)]
pub enum Constant {
    Void,
    Int(u64),
    Float(f64),
    Null,
    Func(BodyId, Vec<Type>),
    Method {
        implementor: Type,
        trait_id: TraitId,
        trait_generics: Vec<Type>,
        method_generics: Vec<Type>,
        index: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Debug)]
pub enum Pat {
    Binding(LocalId),
}

#[derive(Clone, Debug)]
pub struct Arm {
    pub pat: Pat,
    pub expr: Expr,
}

#[derive(Clone, Debug)]
pub enum ExprKind {
    Const(Constant),
    Local(LocalId),
    Let(LocalId, Box<Expr>),
    Assign(Box<Expr>, Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    Struct(StructId, Vec<Type>, Vec<Expr>),
    Field(Box<Expr>, String),
    Ref(Box<Expr>),
    Deref(Box<Expr>),
    Sizeof(Type),
    If(Box<Expr>, Box<Expr>, Option<Box<Expr>>),
    Match(Box<Expr>, Vec<Arm>),
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
