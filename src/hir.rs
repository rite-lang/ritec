use std::{
    collections::HashMap,
    hash::Hash,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::decorator::Decorator;
use crate::{
    ast::BinOp,
    infer::TyEnv,
    number::{Base, IntKind},
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

    /// The index std result type.
    pub result: Option<usize>,

    /// A list of constraints between types.
    ///
    /// Each constraint requires that two types are equal.
    pub env: TyEnv,
}

#[derive(Debug)]
pub struct Module {
    /// The name of the module.
    pub name: &'static str,

    /// The imports in the module.
    pub imports: HashMap<&'static str, Import>,
}

impl Module {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            imports: HashMap::new(),
        }
    }

    pub fn get_import(&self, name: &str, is_defining: bool) -> Option<&Import> {
        let import = self.imports.get(name)?;

        if !is_defining && import.vis == Vis::Private {
            return None;
        }

        Some(import)
    }
}

#[derive(Clone, Debug)]
pub struct Import {
    pub vis: Vis,
    pub kind: ImportKind,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum ImportKind {
    Module(usize),
    Func(usize),
    Adt(usize),
}

#[derive(Debug)]
pub struct Func {
    pub decorators: Vec<Decorator>,
    pub name: &'static str,
    pub generics: Vec<Generic>,
    pub input: Vec<Argument>,
    pub output: Ty,
    pub locals: Vec<Local>,
    pub captures: Vec<Ty>,
    pub body: Expr,
}

#[derive(Clone, Debug)]
pub struct Local {
    pub mutable: bool,
    pub name: &'static str,
    pub ty: Ty,
}

#[derive(Debug)]
pub struct Adt {
    pub decorators: Vec<Decorator>,
    pub name: &'static str,
    pub generics: Vec<Generic>,
    pub variants: Vec<Variant>,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Vis {
    Public,
    Private,
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

#[derive(Clone, Debug)]
pub struct Generic {
    pub name: &'static str,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ty {
    Inferred(Tid, Inferred, Option<usize>, Span),
    Partial(Part, Vec<Ty>, Span),
    Field(Box<Ty>, &'static str, Span),
    Tuple(Box<Ty>, usize, Span),
    Call(Box<Ty>, Vec<Option<Ty>>, Span),
    Pipe(Box<Ty>, Box<Ty>, Vec<Option<Ty>>, Span),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tid {
    id: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Inferred {
    Any,
    Unsigned,
    Signed,
    Float,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Part {
    Void,
    Bool,
    List,
    Tuple,
    Func,
    Str,
    Ref,
    Int(IntKind),
    Generic(usize, Option<usize>),
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
    String(&'static str),
    Int(bool, Base, Vec<u8>),
    Bool(bool),
    Func(usize),
    Variant(usize, usize),
    Local(usize),
    Argument(usize),
    Capture(usize),
    Tuple(Vec<Expr>),
    List(Vec<Expr>, Option<Box<Expr>>),
    ListHead(Box<Expr>),
    ListTail(Box<Expr>),
    ListEmpty(Box<Expr>),
    Block(Vec<Expr>),
    Field(Box<Expr>, &'static str),
    VariantField(Box<Expr>, usize, usize),
    TupleField(Box<Expr>, usize),
    IsVariant(Box<Expr>, usize),
    VariantNew(usize, usize, Vec<Expr>),
    Call(Box<Expr>, Vec<Option<Expr>>),
    Pipe(Box<Expr>, Box<Expr>, Vec<Option<Expr>>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    Unary(UnOp, Box<Expr>),
    Ref(Box<Expr>),
    Deref(Box<Expr>),
    Let(usize, Box<Expr>),
    Assign(Box<Expr>, Box<Expr>),
    Closure(Vec<Local>, Vec<Expr>, Box<Expr>),
    Match(Box<Expr>, Match),
    Panic(&'static str),
    Return(Box<Expr>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Clone, Debug)]
pub enum Match {
    Bool(Box<Expr>, Box<Expr>),
    Adt(usize, Vec<Option<Expr>>, Option<Box<Expr>>),
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

    pub fn unify(&mut self, a: Ty, b: Ty, span: Span) {
        self.env.unify(a, b, span);
    }

    pub fn normalize(&mut self, ty: Ty, span: Span) {
        self.env.normalize(ty, span);
    }
}

impl Adt {
    pub fn find_variant(&self, name: &str) -> miette::Result<usize> {
        self.variants
            .iter()
            .position(|variant| variant.name == name)
            .ok_or_else(|| miette::miette!("variant not found `{}` in ADT `{}`", name, self.name))
    }

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
    pub fn any(span: Span) -> Self {
        Ty::Inferred(Tid::new(), Inferred::Any, None, span)
    }

    pub fn inferred(inferred: Inferred, span: Span) -> Self {
        Ty::Inferred(Tid::new(), inferred, None, span)
    }

    pub const fn void(span: Span) -> Self {
        Ty::Partial(Part::Void, Vec::new(), span)
    }

    pub const fn bool(span: Span) -> Self {
        Ty::Partial(Part::Bool, Vec::new(), span)
    }

    pub const fn string(span: Span) -> Self {
        Ty::Partial(Part::Str, Vec::new(), span)
    }

    pub const fn int(kind: IntKind, span: Span) -> Self {
        Ty::Partial(Part::Int(kind), Vec::new(), span)
    }

    pub fn new_ref(ty: Ty, span: Span) -> Self {
        Ty::Partial(Part::Ref, vec![ty], span)
    }

    pub fn specialize(&self, generics: &[Ty]) -> Ty {
        match self {
            Ty::Inferred(_, _, _, _) => self.clone(),
            Ty::Partial(Part::Generic(index, _), args, _) => {
                assert!(args.is_empty());
                generics[*index].clone()
            }
            Ty::Partial(part, args, span) => {
                let args = args.iter().map(|arg| arg.specialize(generics)).collect();
                Ty::Partial(*part, args, *span)
            }
            Ty::Field(adt, field, span) => {
                let base = adt.specialize(generics);
                Ty::Field(Box::new(base), field, *span)
            }
            Ty::Tuple(base, index, span) => {
                let base = base.specialize(generics);
                Ty::Tuple(Box::new(base), *index, *span)
            }
            Ty::Call(callee, arguments, span) => {
                let callee = callee.specialize(generics);
                let arguments = arguments
                    .iter()
                    .map(|arg| arg.as_ref().map(|arg| arg.specialize(generics)))
                    .collect();
                Ty::Call(Box::new(callee), arguments, *span)
            }
            Ty::Pipe(lhs, rhs, arguments, span) => {
                let lhs = lhs.specialize(generics);
                let rhs = rhs.specialize(generics);

                let arguments = arguments
                    .iter()
                    .map(|arg| arg.as_ref().map(|arg| arg.specialize(generics)))
                    .collect();

                Ty::Pipe(Box::new(lhs), Box::new(rhs), arguments, *span)
            }
        }
    }

    pub fn format(&self, unit: &Unit) -> String {
        match self {
            Ty::Inferred(_, kind, _, _) => match kind {
                Inferred::Any => String::from("_"),
                Inferred::Unsigned => String::from("{unsigned}"),
                Inferred::Signed => String::from("{signed}"),
                Inferred::Float => String::from("{float}"),
            },
            Ty::Partial(part, args, _) => Self::format_partial(unit, part, args),
            Ty::Field(_, _, _) => todo!(),
            Ty::Tuple(_, _, _) => todo!(),
            Ty::Call(_, _, _) => String::from("call"),
            Ty::Pipe(_, _, _, _) => String::from("pipe"),
        }
    }

    pub fn format_partial(unit: &Unit, part: &Part, args: &[Ty]) -> String {
        match part {
            Part::Void => String::from("void"),
            Part::Bool => String::from("bool"),
            Part::Str => String::from("str"),
            Part::List => format!("[{}]", args[0].format(unit)),
            Part::Tuple => {
                let args = args
                    .iter()
                    .map(|arg| arg.format(unit))
                    .collect::<Vec<_>>()
                    .join(" * ");

                format!("({})", args)
            }
            Part::Func => {
                let input = args
                    .iter()
                    .take(args.len().saturating_sub(1))
                    .map(|arg| arg.format(unit))
                    .collect::<Vec<_>>()
                    .join(", ");

                let output = args
                    .last()
                    .map_or(String::from("wat"), |arg| arg.format(unit));

                format!("fn({}) -> {}", input, output)
            }
            Part::Int(kind) => format!("{}", kind),
            Part::Ref => format!("&{}", args[0].format(unit)),
            Part::Generic(id, _) => format!("<{}>", id),
            Part::Adt(id) => {
                let name = &unit.adts[*id].name;

                if args.is_empty() {
                    return name.to_string();
                }

                let args = args
                    .iter()
                    .map(|arg| arg.format(unit))
                    .collect::<Vec<_>>()
                    .join(", ");

                format!("{}<{}>", name, args)
            }
        }
    }
}

impl Tid {
    pub fn new() -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);

        Self { id }
    }
}
