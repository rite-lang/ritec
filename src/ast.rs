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
    Paren(Box<Expr>, Span),
    Item(Path),
    Tuple(Vec<Expr>),
    List(Vec<Expr>),
    Block(Vec<Expr>),
    Field(Box<Expr>, &'static str),
    Call(Box<Expr>, Vec<Option<Expr>>),
    Pipe(Box<Expr>, Vec<Expr>),
    Let(&'static str, Box<Expr>),
}

#[derive(Debug)]
pub struct Path {
    pub segments: Vec<&'static str>,
    pub span: Span,
}
