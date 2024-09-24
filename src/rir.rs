use crate::{
    ast::BinOp,
    number::{Base, IntKind},
    span::Span,
};

#[derive(Debug)]
pub struct Unit<T = Ty> {
    pub funcs: Vec<Func<T>>,
    pub adts: Vec<Adt<T>>,
}

impl<T> Default for Unit<T> {
    fn default() -> Self {
        Self {
            funcs: Vec::new(),
            adts: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Func<T = Ty> {
    pub name: String,
    pub generics: Vec<Generic>,
    pub input: Vec<Argument<T>>,
    pub output: T,
    pub locals: Vec<Local<T>>,
    pub captures: Vec<Capture<T>>,
    pub body: Block<T>,
}

impl Default for Func {
    fn default() -> Self {
        Self {
            name: String::new(),
            generics: Vec::new(),
            input: Vec::new(),
            output: Ty::Void,
            locals: Vec::new(),
            captures: Vec::new(),
            body: Block::new(),
        }
    }
}

impl Func {
    pub fn ty(&self) -> Ty {
        Ty::Func(
            self.input.iter().map(|arg| arg.ty.clone()).collect(),
            Box::new(self.output.clone()),
        )
    }
}

#[derive(Clone, Debug)]
pub struct Local<T = Ty> {
    pub ty: T,
}

#[derive(Clone, Debug)]
pub struct Capture<T = Ty> {
    pub ty: T,
}

#[derive(Debug)]
pub struct Adt<T = Ty> {
    pub name: String,
    pub generics: Vec<Generic>,
    pub variants: Vec<Variant<T>>,
}

impl<T> Default for Adt<T> {
    fn default() -> Self {
        Self {
            name: String::new(),
            generics: Vec::new(),
            variants: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub struct Variant<T = Ty> {
    pub fields: Vec<Argument<T>>,
}

#[derive(Clone, Debug)]
pub struct Generic {}

#[derive(Clone, Debug)]
pub struct Argument<T = Ty> {
    pub ty: T,
    pub span: Option<Span>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Ty {
    Void,
    Bool,
    Str,
    Mut(Box<Ty>),
    Int(IntKind),
    List(Box<Ty>),
    Tuple(Vec<Ty>),
    Func(Vec<Ty>, Box<Ty>),
    Adt(usize, Vec<Ty>),
    Generic(usize),
}

impl Ty {
    pub fn is_mut(&self) -> bool {
        matches!(self, Ty::Mut(_))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Specific {
    Void,
    Bool,
    Str,
    Mut(Box<Specific>),
    Int(IntKind),
    List(Box<Specific>),
    Tuple(Vec<Specific>),
    Func(Vec<Specific>, Box<Specific>),
    Adt(usize),
}

#[derive(Clone, Debug)]
pub struct Block<T = Ty> {
    pub statements: Vec<Statement<T>>,
}

impl<T> Block<T> {
    pub fn new() -> Self {
        Self {
            statements: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Statement<T = Ty> {
    Use {
        value: Value<T>,
    },
    Return {
        value: Option<Value<T>>,
    },
    Assign {
        place: Place<T>,
        value: Value<T>,
    },
    MatchBool {
        input: Operand<T>,
        r#true: Block<T>,
        r#false: Block<T>,
    },
    MatchList {
        input: Operand<T>,
        some: Block<T>,
        none: Block<T>,
    },
    MatchAdt {
        input: Operand<T>,
        variants: Vec<Option<Block<T>>>,
        default: Option<Block<T>>,
    },
}

#[derive(Clone, Debug)]
pub enum Value<T = Ty> {
    Use(Operand<T>),
    Func(usize, Vec<Operand<T>>, Vec<T>),
    List(Vec<Operand<T>>, Option<Operand<T>>),
    ListHead(Operand<T>),
    ListTail(Operand<T>),
    Binary(BinOp, Operand<T>, Operand<T>),
    Call(Operand<T>, Vec<Operand<T>>),
    Mut(Place<T>),
    Tuple(Vec<Operand<T>>),
    Adt(usize, Vec<Operand<T>>),
}

#[derive(Clone, Debug)]
pub enum Operand<T = Ty> {
    Copy(Place<T>),
    Move(Place<T>),
    Constant(Constant),
}

#[derive(Clone, Debug)]
pub enum Constant {
    Void,
    Bool(bool),
    Int(bool, Base, Vec<u8>),
    String(&'static str),
}

#[derive(Clone, Debug)]
pub struct Place<T = Ty> {
    pub location: Location,
    pub projection: Vec<Projection<T>>,
    pub ty: T,
}

impl<T> Place<T> {
    pub fn ty(&self) -> &T {
        self.projection.last().map(|p| &p.ty).unwrap_or(&self.ty)
    }
}

#[derive(Clone, Debug)]
pub enum Location {
    Local(usize),
    Argument(usize),
    Capture(usize),
}

#[derive(Clone, Debug)]
pub struct Projection<T = Ty> {
    pub kind: ProjectionKind,
    pub ty: T,
    pub span: Option<Span>,
}

#[derive(Clone, Debug)]
pub enum ProjectionKind {
    Field {
        variant: Option<usize>,
        field: usize,
    },
    Deref,
}
