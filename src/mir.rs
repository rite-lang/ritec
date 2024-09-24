use crate::{
    ast::BinOp,
    number::{Base, IntKind},
};

#[derive(Debug)]
pub struct Mir {
    pub funcs: Vec<Func>,
}

#[derive(Debug)]
pub struct Func {
    pub input: Vec<Ty>,
    pub output: Ty,
    pub locals: Vec<Ty>,
    pub body: Expr,
}

impl Default for Func {
    fn default() -> Self {
        Self {
            input: Vec::new(),
            output: Ty::Void,
            locals: Vec::new(),
            body: Expr {
                kind: ExprKind::Const(Constant::Void),
                ty: Ty::Void,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ty {
    Void,
    Bool,
    Int(IntKind),
    List(Box<Ty>),
    Tuple(Vec<Ty>),
    Func(Vec<Ty>, Box<Ty>),
    Adt(Vec<Variant>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Variant {
    pub fields: Vec<Ty>,
}

#[derive(Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub ty: Ty,
}

#[derive(Debug)]
pub enum ExprKind {
    Const(Constant),
    Func(usize, Vec<Expr>),
    Local(usize),
    Argument(usize),
    Captured(usize),
    List(Vec<Expr>, Option<Box<Expr>>),
    ListHead(Box<Expr>),
    ListTail(Box<Expr>),
    Block(Vec<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Let(usize, Box<Expr>),
    Adt(usize, Vec<Expr>),
    Field(Box<Expr>, usize),
    VariantField(Box<Expr>, usize, usize),
    Match(Box<Expr>, Match),
}

#[derive(Debug)]
pub enum Match {
    Bool(Box<Expr>, Box<Expr>),
    List(Box<Expr>, Box<Expr>),
    Adt(Vec<Option<Expr>>, Option<Box<Expr>>),
}

#[derive(Clone, Debug)]
pub enum Constant {
    Void,
    Bool(bool),
    Int(bool, Base, Vec<u8>),
}
