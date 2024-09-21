use crate::number::{Base, IntKind};

#[derive(Debug)]
pub struct Mir {
    pub funcs: Vec<Func>,
}

#[derive(Debug)]
pub struct Func {
    pub input: Vec<Ty>,
    pub output: Ty,
    pub body: Expr,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ty {
    Void,
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
    Local(usize),
    Argument(usize),
    List(Vec<Expr>),
    Block(Vec<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    Let(usize, Box<Expr>),
    Adt(usize, Vec<Expr>),
    Field(Box<Expr>, usize),
}

#[derive(Clone, Debug)]
pub enum Constant {
    Void,
    Int(bool, Base, Vec<u8>),
    Func(usize),
}
