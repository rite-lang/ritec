use std::{
    collections::HashMap,
    hash::Hash,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::{
    ast::BinOp,
    infer::TyEnv,
    number::{Base, FloatKind, IntKind},
    span::Span,
};

#[derive(Debug, Default)]
pub struct Unit {
    /// A list of modules in the unit.
    pub modules: Vec<Module>,

    /// A list of functions in the unit.
    pub funcs: Vec<Func>,

    /// A list of ADTs in the unit.
    pub adts: Vec<Adt>,

    /// A list of constraints between types.
    ///
    /// Each constraint requires that two types are equal.
    pub env: TyEnv,
}

#[derive(Debug)]
pub struct Module {
    /// The name of the module.
    pub name: &'static str,

    /// The submodules of the module.
    pub modules: HashMap<&'static str, usize>,

    /// The functions in the module.
    pub funcs: HashMap<&'static str, usize>,

    /// The ADTs in the module.
    pub adts: HashMap<&'static str, usize>,
}

impl Module {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            modules: HashMap::new(),
            funcs: HashMap::new(),
            adts: HashMap::new(),
        }
    }
}

#[derive(Debug)]
pub struct Func {
    pub name: &'static str,
    pub generics: Vec<Generic>,
    pub input: Vec<Argument>,
    pub output: Ty,
    pub locals: Vec<Local>,
    pub body: Expr,
}

#[derive(Debug)]
pub struct Local {
    pub name: &'static str,
    pub ty: Ty,
}

#[derive(Debug)]
pub struct Adt {
    pub name: &'static str,
    pub generics: Vec<Generic>,
    pub variants: Vec<Variant>,
    pub span: Span,
}

#[derive(Debug)]
pub struct Variant {
    pub name: &'static str,
    pub fields: Vec<Argument>,
}

#[derive(Clone, Debug)]
pub struct Argument {
    pub name: &'static str,
    pub ty: Ty,
    pub span: Span,
}

#[derive(Debug)]
pub struct Generic {
    pub name: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ty {
    Inferred(Tid, Inferred, Option<usize>),
    Partial(Part, Vec<Ty>),
    Field(Box<Ty>, &'static str),
    Tuple(Box<Ty>, usize),
    Call(Box<Ty>, Vec<Ty>),
    Pipe(Box<Ty>, Box<Ty>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tid {
    id: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Inferred {
    Any,
    Int(IntKind),
    Float(FloatKind),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Part {
    Void,
    Bool,
    List,
    Tuple,
    Func,
    Int(IntKind),
    Generic(usize),
    Adt(usize),
}

#[derive(Clone, Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub ty: Ty,
}

#[derive(Clone, Debug)]
pub enum ExprKind {
    Void,
    Int(bool, Base, Vec<u8>),
    Bool(bool),
    Func(usize),
    Variant(usize, usize),
    Local(usize),
    Argument(usize),
    Tuple(Vec<Expr>),
    List(Vec<Expr>, Option<Box<Expr>>),
    ListHead(Box<Expr>),
    ListTail(Box<Expr>),
    Block(Vec<Expr>),
    Field(Box<Expr>, &'static str),
    VariantField(Box<Expr>, usize, usize),
    TupleField(Box<Expr>, usize),
    Call(Box<Expr>, Vec<Expr>),
    Pipe(Box<Expr>, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Let(usize, Box<Expr>),
    Match(Box<Expr>, Match),
}

#[derive(Clone, Debug)]
pub enum Match {
    Bool(Box<Expr>, Box<Expr>),
    Adt(usize, Vec<Option<Expr>>, Option<Box<Expr>>),
    List(Box<Expr>, Box<Expr>),
}

impl Unit {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_module(&mut self, module: Module) -> usize {
        let id = self.modules.len();
        self.modules.push(module);
        id
    }

    pub fn push_func(&mut self, func: Func) -> usize {
        let id = self.funcs.len();
        self.funcs.push(func);
        id
    }

    pub fn push_adt(&mut self, adt: Adt) -> usize {
        let id = self.adts.len();
        self.adts.push(adt);
        id
    }

    pub fn unify(&mut self, a: Ty, b: Ty) {
        self.env.unify(a, b);
    }

    pub fn normalize(&mut self, ty: Ty) {
        self.env.normalize(ty);
    }
}

impl Adt {
    pub fn find_field(&self, name: &str) -> miette::Result<(usize, Ty)> {
        let mut index = None;
        let mut ty = None;

        for variant in &self.variants {
            let Some(field) = variant.fields.iter().position(|field| field.name == name) else {
                return Err(miette::miette!(
                    "field not found `{}` in variant `{}`",
                    name,
                    variant.name
                ));
            };

            if index.is_none() {
                index = Some(field);
                ty = Some(variant.fields[field].ty.clone());
                continue;
            }

            if index != Some(field) {
                return Err(miette::miette!("ambiguous field"));
            }

            if ty != Some(variant.fields[field].ty.clone()) {
                return Err(miette::miette!("ambiguous field"));
            }
        }

        match (index, ty) {
            (Some(index), Some(ty)) => Ok((index, ty)),
            (_, _) => Err(miette::miette!("field not found")),
        }
    }
}

impl Ty {
    pub fn any() -> Self {
        Ty::Inferred(Tid::new(), Inferred::Any, None)
    }

    pub fn inferred(inferred: Inferred) -> Self {
        Ty::Inferred(Tid::new(), inferred, None)
    }

    pub const fn void() -> Self {
        Ty::Partial(Part::Void, Vec::new())
    }

    pub const fn bool() -> Self {
        Ty::Partial(Part::Bool, Vec::new())
    }

    pub const fn int(kind: IntKind) -> Self {
        Ty::Partial(Part::Int(kind), Vec::new())
    }

    pub fn specialize(&self, generics: &[Ty]) -> Ty {
        match self {
            Ty::Inferred(_, _, _) => self.clone(),
            Ty::Partial(Part::Generic(index), args) => {
                assert!(args.is_empty());
                generics[*index].clone()
            }
            Ty::Partial(part, args) => {
                let args = args.iter().map(|arg| arg.specialize(generics)).collect();
                Ty::Partial(*part, args)
            }
            Ty::Field(adt, field) => {
                let base = adt.specialize(generics);
                Ty::Field(Box::new(base), field)
            }
            Ty::Tuple(base, index) => {
                let base = base.specialize(generics);
                Ty::Tuple(Box::new(base), *index)
            }
            Ty::Call(callee, arguments) => {
                let callee = callee.specialize(generics);
                let arguments = arguments
                    .iter()
                    .map(|arg| arg.specialize(generics))
                    .collect();
                Ty::Call(Box::new(callee), arguments)
            }
            Ty::Pipe(lhs, rhs) => {
                let lhs = lhs.specialize(generics);
                let rhs = rhs.specialize(generics);
                Ty::Pipe(Box::new(lhs), Box::new(rhs))
            }
        }
    }
}

impl std::fmt::Display for Ty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ty::Inferred(tid, kind, _) => match kind {
                Inferred::Any => write!(f, "_{}", tid.id),
                Inferred::Int(kind) => write!(f, "{{{}}}", kind),
                Inferred::Float(_) => write!(f, "_"),
            },
            Ty::Partial(part, args) => write!(f, "{}", format_partial(part, args)),
            Ty::Field(ty, field) => write!(f, "{}.{}", ty, field),
            Ty::Tuple(ty, index) => write!(f, "{}.{}", ty, index),
            Ty::Call(func, args) => {
                let args = args
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");

                write!(f, "({})<{}>", func, args)
            }
            Ty::Pipe(a, b) => write!(f, "{} |> {}", a, b),
        }
    }
}

pub fn format_partial(part: &Part, args: &[Ty]) -> String {
    match part {
        Part::Void => String::from("void"),
        Part::Bool => String::from("bool"),
        Part::List => format!("[{}]", args[0]),
        Part::Tuple => {
            let args = args
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(" * ");

            format!("({})", args)
        }
        Part::Func => {
            let input = args
                .iter()
                .take(args.len() - 1)
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            let output = args.last().unwrap().to_string();
            format!("fn({}) -> {}", input, output)
        }
        Part::Int(kind) => format!("{}", kind),
        Part::Generic(id) => format!(
            "{}<{}>",
            id,
            args.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Part::Adt(id) => format!("Type<{}>", id),
    }
}

impl Tid {
    pub fn new() -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);

        Self { id }
    }
}
