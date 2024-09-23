use crate::{
    number::{Base, IntKind},
    span::Span,
};

#[derive(Debug)]
pub struct Module {
    pub decls: Vec<Decl>,
}

#[derive(Debug)]
pub enum Decl {
    Func(Func),
    Type(Type),
}

#[derive(Debug)]
pub struct Func {
    pub vis: Vis,
    pub name: &'static str,
    pub input: Vec<Argument>,
    pub output: Option<Ty>,
    pub body: Option<Expr>,
    pub span: Span,
}

#[derive(Debug)]
pub enum Type {
    Adt(Adt),
}

#[derive(Debug)]
pub struct Adt {
    pub vis: Vis,
    pub name: &'static str,
    pub variants: Vec<Variant>,
    pub span: Span,
}

#[derive(Debug)]
pub struct Variant {
    pub name: &'static str,
    pub fields: Vec<Argument>,
    pub span: Span,
}

#[derive(Debug)]
pub struct Argument {
    pub name: &'static str,
    pub ty: Option<Ty>,
    pub span: Span,
}

#[derive(Debug)]
pub enum Vis {
    Public,
    Private,
}

#[derive(Debug)]
pub enum Ty {
    Void,
    Bool,
    Int(IntKind),
    Tuple(Vec<Ty>),
    Item(Path),
    Generic(Generic),
}

#[derive(Debug)]
pub struct Generic {
    pub name: &'static str,
    pub span: Span,
}

#[derive(Debug)]
pub enum Expr {
    Int(bool, Base, Vec<u8>, Span),
    Bool(bool, Span),
    Paren(Box<Expr>, Span),
    Item(Path),
    Tuple(Vec<Expr>),
    List(Vec<Expr>, Option<Box<Expr>>, Span),
    Block(Vec<Expr>),
    Field(Box<Expr>, &'static str),
    Call(Box<Expr>, Vec<Option<Expr>>),
    Pipe(Box<Expr>, Vec<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Let(&'static str, Box<Expr>),
    Match(Box<Expr>, Vec<Arm>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl BinOp {
    pub const fn precedence(self) -> u8 {
        match self {
            BinOp::Or => 1,
            BinOp::And => 2,
            BinOp::Eq | BinOp::Ne => 3,
            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => 4,
            BinOp::Add | BinOp::Sub => 5,
            BinOp::Mul | BinOp::Div | BinOp::Rem => 6,
        }
    }
}

#[derive(Debug)]
pub struct Arm {
    pub pat: Pat,
    pub expr: Expr,
    pub span: Span,
}

#[derive(Debug)]
pub struct Pat {
    pub kind: PatKind,
    pub span: Span,
}

#[derive(Debug)]
pub enum PatKind {
    Bind(Option<&'static str>),
    Bool(bool),
    Tuple(Vec<Pat>),
    Variant(Path, Vec<Pat>),
}

#[derive(Debug)]
pub struct Path {
    pub segments: Vec<&'static str>,
    pub span: Span,
}
