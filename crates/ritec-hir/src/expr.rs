use ritec_diagnostic::Span;

use crate::{BodyId, EnumId, LocalId, StructId, TraitId, Ty};

#[derive(Clone, Debug)]
pub enum Const {
    Void,
    Int(u64),
    Float(f64),
    Null,
    Func(BodyId, Vec<Ty>),
    Method {
        implementor: Ty,
        trait_id: TraitId,
        trait_generics: Vec<Ty>,
        method_generics: Vec<Ty>,
        index: usize,
    },
    AssocMethod {
        implementor: Ty,
        name: String,
        generics: Vec<Ty>,
        arguments: Option<Vec<Ty>>,
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
    Variant(EnumId, usize, Vec<Pat>),
    Binding(LocalId),
}

#[derive(Clone, Debug)]
pub struct Arm {
    pub pat: Pat,
    pub expr: Expr,
}

#[derive(Clone, Debug)]
pub enum ExprKind {
    Const(Const),
    Local(LocalId),
    Let(LocalId, Box<Expr>),
    Cast(Box<Expr>),
    Assign(Box<Expr>, Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    Struct(StructId, Vec<Ty>, Vec<Expr>),
    Variant(EnumId, Vec<Ty>, usize, Vec<Expr>),
    Field(Box<Expr>, String),
    Ref(Box<Expr>),
    Deref(Box<Expr>),
    Sizeof(Ty),
    If(Box<Expr>, Box<Expr>, Option<Box<Expr>>),
    Match(Box<Expr>, Vec<Arm>),
    Block(Vec<Expr>),
    Intrinsic(&'static str, Vec<Expr>),
}

#[derive(Clone, Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Option<Span>,
    pub ty: Ty,
}

impl Expr {
    pub fn void(span: Span) -> Self {
        Self {
            kind: ExprKind::Const(Const::Void),
            span: Some(span),
            ty: Ty::VOID,
        }
    }
}
