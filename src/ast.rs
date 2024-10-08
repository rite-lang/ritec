use crate::decorator::Decorator;
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
    Import(Import),
    Func(Func),
    Type(Type),
}

#[derive(Debug)]
pub struct Import {
    pub vis: Vis,
    pub path: Path,
    pub span: Span,
}

#[derive(Debug)]
pub struct Func {
    pub decorators: Vec<Decorator>,
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
    Single(Single),
}

#[derive(Debug)]
pub struct Adt {
    pub decorators: Vec<Decorator>,
    pub vis: Vis,
    pub name: &'static str,
    pub generics: Option<Vec<Generic>>,
    pub variants: Vec<Variant>,
    pub span: Span,
}

#[derive(Debug)]
pub struct Single {
    pub decorators: Vec<Decorator>,
    pub vis: Vis,
    pub name: &'static str,
    pub generics: Option<Vec<Generic>>,
    pub fields: Vec<Field>,
    pub span: Span,
}

#[derive(Debug)]
pub struct Variant {
    pub name: &'static str,
    pub fields: Vec<Field>,
    pub span: Span,
}

#[derive(Debug)]
pub struct Argument {
    pub name: &'static str,
    pub ty: Option<Ty>,
    pub span: Span,
}

#[derive(Debug)]
pub struct Field {
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
    Str,
    Inferred,
    Int(IntKind),
    Ref(Box<Ty>),
    Tuple(Vec<Ty>),
    Item(Path, Option<Vec<Ty>>),
    List(Box<Ty>),
    Func(Vec<Ty>, Option<Box<Ty>>),
    Generic(Generic),
}

#[derive(Debug)]
pub struct Generic {
    pub name: &'static str,
    pub span: Span,
}

#[derive(Debug)]
pub enum CallArgument {
    Positional(Option<Expr>),
    Named(&'static str, Expr),
}

#[derive(Debug)]
pub enum Expr {
    Void(Span),
    String(&'static str, Span),
    Int(bool, Base, Vec<u8>, Span),
    Bool(bool, Span),
    Paren(Box<Expr>, Span),
    Item(Path),
    Tuple(Vec<Expr>),
    List(Vec<Expr>, Option<Box<Expr>>, Span),
    Block(Vec<Expr>),
    As(Box<Expr>, Ty),
    Field(Box<Expr>, &'static str),
    Call(Box<Expr>, Vec<CallArgument>, Option<Box<Expr>>),
    Pipe(Box<Expr>, Vec<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>, Span),
    Unary(UnOp, Box<Expr>, Span),
    Let(Pat, Option<Ty>, Box<Expr>),
    Mut(&'static str, Option<Ty>, Box<Expr>),
    LetAssert(Pat, Option<Ty>, Box<Expr>),
    Assign(Box<Expr>, Box<Expr>),
    Match(Box<Expr>, Vec<Arm>, Span),
    Return(Option<Box<Expr>>, Span),
    Closure(Vec<Argument>, Box<Expr>),
    Panic(&'static str, Span),
    Assert(Box<Expr>, Option<&'static str>, Span),
    Try(Box<Expr>, Span),
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Void(span) => *span,
            Expr::String(_, span) => *span,
            Expr::Int(_, _, _, span) => *span,
            Expr::Bool(_, span) => *span,
            Expr::Paren(_, span) => *span,
            Expr::Item(path) => path.span,
            Expr::Tuple(exprs) => {
                let start = exprs.first().unwrap().span();
                let end = exprs.last().unwrap().span();
                start.join(end)
            }
            Expr::List(_, _, span) => *span,
            Expr::Block(exprs) => {
                let start = exprs.first().unwrap().span();
                let end = exprs.last().unwrap().span();
                start.join(end)
            }
            Expr::As(expr, _) => expr.span(),
            Expr::Field(expr, _) => expr.span(),
            Expr::Call(func, args, _) => {
                let mut start = func.span();

                for arg in args.iter() {
                    match arg {
                        CallArgument::Positional(Some(expr)) => {
                            start = start.join(expr.span());
                        }
                        CallArgument::Positional(None) => {}
                        CallArgument::Named(_, expr) => {
                            start = start.join(expr.span());
                        }
                    }
                }

                start
            }
            Expr::Pipe(lhs, rhs) => {
                let mut start = lhs.span();

                for expr in rhs.iter() {
                    start = start.join(expr.span());
                }

                start
            }
            Expr::Binary(_, lhs, rhs, _) => {
                let start = lhs.span();
                let end = rhs.span();
                start.join(end)
            }
            Expr::Unary(_, expr, span) => {
                let inner = expr.span();
                span.join(inner)
            }
            Expr::Let(_, _, expr) => expr.span(),
            Expr::Mut(_, _, expr) => expr.span(),
            Expr::LetAssert(_, _, expr) => expr.span(),
            Expr::Assign(lhs, rhs) => {
                let start = lhs.span();
                let end = rhs.span();
                start.join(end)
            }
            Expr::Match(_, _, span) => *span,
            Expr::Return(_, span) => *span,
            Expr::Closure(args, expr) => {
                let end = expr.span();
                let start = args.first().map_or(end, |arg| arg.span);
                start.join(end)
            }
            Expr::Panic(_, span) => *span,
            Expr::Assert(expr, _, span) => expr.span().join(*span),
            Expr::Try(expr, span) => expr.span().join(*span),
        }
    }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnOp {
    Ref,
    Deref,
    Neg,
    Not,
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
    List(Vec<Pat>, Option<Option<Box<Pat>>>),
}

#[derive(Debug)]
pub struct Path {
    pub segments: Vec<&'static str>,
    pub span: Span,
}

impl Path {
    pub fn item(&self) -> &'static str {
        self.segments.last().unwrap()
    }
}
