use std::fmt::Display;

use crate::decorator::Decorator;
use crate::{
    ast::BinOp,
    hir::UnOp,
    number::{Base, FloatKind, IntKind},
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
    pub decorators: Vec<Decorator>,
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
            decorators: Vec::new(),
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
    pub decorators: Vec<Decorator>,
    pub name: String,
    pub generics: Vec<Generic>,
    pub variants: Vec<Variant<T>>,
}

impl<T> Default for Adt<T> {
    fn default() -> Self {
        Self {
            decorators: Vec::new(),
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
    Ref(Box<Ty>),
    Int(IntKind),
    Float(FloatKind),
    List(Box<Ty>),
    Tuple(Vec<Ty>),
    Func(Vec<Ty>, Box<Ty>),
    Adt(usize, Vec<Ty>),
    Generic(usize),
}

impl Ty {
    pub fn is_mut(&self) -> bool {
        matches!(self, Ty::Ref(_))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Specific {
    Void,
    Bool,
    Str,
    Ref(Box<Specific>),
    Int(IntKind),
    Float(FloatKind),
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
    Panic {
        message: &'static str,
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
    MatchAdt {
        input: Operand<T>,
        variants: Vec<Option<Block<T>>>,
        default: Option<Block<T>>,
    },
}

#[derive(Clone, Debug)]
pub enum Value<T = Ty> {
    Use(Operand<T>),
    Cast(Cast, Operand<T>),
    Func(usize, Vec<Operand<T>>, Vec<T>),
    List(Vec<Operand<T>>, Option<Operand<T>>),
    ListHead(Operand<T>),
    ListTail(Operand<T>),
    ListEmpty(Operand<T>),
    Binary(BinOp, Operand<T>, Operand<T>),
    Unary(UnOp, Operand<T>),
    IsVariant(Operand<T>, usize),
    Call(Operand<T>, Vec<Operand<T>>),
    Ref(Place<T>),
    Tuple(Vec<Operand<T>>),
    Adt(usize, Vec<Operand<T>>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cast {
    Int(IntKind),
    Float(FloatKind),
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
    Int(bool, Base, Vec<u8>, IntKind),
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

impl Display for Ty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ty::Void => write!(f, "void"),
            Ty::Bool => write!(f, "bool"),
            Ty::Str => write!(f, "str"),
            Ty::Ref(ty) => write!(f, "&{}", ty),
            Ty::Int(kind) => write!(f, "{:?}", kind),
            Ty::Float(kind) => write!(f, "{:?}", kind),
            Ty::List(ty) => write!(f, "[{}]", ty),
            Ty::Tuple(tys) => {
                write!(f, "(")?;
                for (i, ty) in tys.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", ty)?;
                }
                write!(f, ")")
            }
            Ty::Func(input, output) => {
                write!(f, "(")?;
                for (i, ty) in input.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", ty)?;
                }
                write!(f, ") -> {}", output)
            }
            Ty::Adt(adt, tys) => {
                write!(f, "adt {}", adt)?;
                if !tys.is_empty() {
                    write!(f, "<")?;
                    for (i, ty) in tys.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", ty)?;
                    }
                    write!(f, ">")?;
                }
                Ok(())
            }
            Ty::Generic(generic) => write!(f, "generic {}", generic),
        }
    }
}

impl Display for Specific {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Specific::Void => write!(f, "void"),
            Specific::Bool => write!(f, "bool"),
            Specific::Str => write!(f, "str"),
            Specific::Ref(specific) => write!(f, "&{}", specific),
            Specific::Int(kind) => write!(f, "{:?}", kind),
            Specific::Float(kind) => write!(f, "{:?}", kind),
            Specific::List(specific) => write!(f, "[{}]", specific),
            Specific::Tuple(specifics) => {
                write!(f, "(")?;
                for (i, specific) in specifics.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", specific)?;
                }
                write!(f, ")")
            }
            Specific::Func(input, output) => {
                write!(f, "(")?;
                for (i, specific) in input.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", specific)?;
                }
                write!(f, ") -> {}", output)
            }
            Specific::Adt(adt) => write!(f, "adt {}", adt),
        }
    }
}

pub fn dump_unit<T: Display>(unit: &Unit<T>) {
    let mut dumper = Dumper::new();
    dumper.dump_unit(unit);
}

pub struct Dumper {
    indent: usize,
}

impl Dumper {
    pub fn new() -> Self {
        Self { indent: 0 }
    }

    pub fn dump_unit<T: Display>(&mut self, unit: &Unit<T>) {
        for func in &unit.funcs {
            self.dump_func(func);
        }
        for adt in &unit.adts {
            self.dump_adt(adt);
        }
    }

    pub fn dump_func<T: Display>(&mut self, func: &Func<T>) {
        self.dump_indent();
        print!("func {}(", func.name);
        for (i, arg) in func.input.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{}", arg.ty);
        }
        println!(") -> {}", func.output);
        self.dump_block(&func.body);
    }

    pub fn dump_adt<T: Display>(&mut self, adt: &Adt<T>) {
        self.dump_indent();
        print!("adt {}", adt.name);
        if !adt.generics.is_empty() {
            print!("<");
            for (i, generic) in adt.generics.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("{:?}", generic);
            }
            print!(">");
        }
        println!(" {{");
        self.indent();
        for variant in &adt.variants {
            self.dump_indent();
            print!("variant {{");
            for (i, field) in variant.fields.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("{}", field.ty);
            }
            println!("}}");
        }
        self.dedent();
        self.dump_indent();
        println!("}}");
    }

    pub fn dump_block<T: Display>(&mut self, block: &Block<T>) {
        self.dump_indent();
        println!("{{");
        self.indent();
        for statement in &block.statements {
            self.dump_statement(statement);
        }
        self.dedent();
        self.dump_indent();
        println!("}}");
    }

    pub fn dump_statement<T: Display>(&mut self, statement: &Statement<T>) {
        match statement {
            Statement::Use { value } => {
                self.dump_indent();
                print!("use ");
                self.dump_value(value);
                println!(";");
            }
            Statement::Return { value } => {
                self.dump_indent();
                print!("return");
                if let Some(value) = value {
                    print!(" ");
                    self.dump_value(value);
                }
                println!(";");
            }
            Statement::Panic { message } => {
                self.dump_indent();
                println!("panic \"{}\";", message);
            }
            Statement::Assign { place, value } => {
                self.dump_indent();
                self.dump_place(place);
                print!(" = ");
                self.dump_value(value);
                println!(";");
            }
            Statement::MatchBool {
                input,
                r#true,
                r#false,
            } => {
                self.dump_indent();
                print!("match ");
                self.dump_operand(input);
                println!(" {{");
                self.indent();
                self.dump_indent();
                println!("true =>");
                self.dump_block(r#true);
                self.dump_indent();
                println!("false =>");
                self.dump_block(r#false);
                self.dedent();
                self.dump_indent();
                println!("}}");
            }
            Statement::MatchAdt {
                input,
                variants,
                default,
            } => {
                self.dump_indent();
                print!("match ");
                self.dump_operand(input);
                println!(" {{");
                self.indent();
                for (i, variant) in variants.iter().enumerate() {
                    self.dump_indent();
                    if let Some(variant) = variant {
                        print!("variant {} =>", i);
                        self.dump_block(variant);
                    } else {
                        println!("variant {} => {{}}", i);
                    }
                }
                if let Some(default) = default {
                    self.dump_indent();
                    println!("_ =>");
                    self.dump_block(default);
                }
                self.dedent();
                self.dump_indent();
                println!("}}");
            }
        }
    }

    pub fn dump_indent(&self) {
        for _ in 0..self.indent {
            print!("  ");
        }
    }

    pub fn indent(&mut self) {
        self.indent += 1;
    }

    pub fn dedent(&mut self) {
        self.indent -= 1;
    }

    pub fn dump_value<T: Display>(&self, value: &Value<T>) {
        match value {
            Value::Use(operand) => self.dump_operand(operand),
            Value::Cast(cast, operand) => {
                print!("{:?} ", cast);
                self.dump_operand(operand);
            }
            Value::Func(func, captures, tys) => {
                print!("func {}(", func);
                for (i, capture) in captures.iter().enumerate() {
                    if i > 0 {
                        print!(", ");
                    }
                    self.dump_operand(capture);
                }
                print!(") -> ");
                for (i, ty) in tys.iter().enumerate() {
                    if i > 0 {
                        print!(", ");
                    }
                    print!("{}", ty);
                }
            }
            Value::List(head, tail) => {
                print!("[");

                for (i, head) in head.iter().enumerate() {
                    if i > 0 {
                        print!(", ");
                    }

                    self.dump_operand(head);
                }

                if let Some(tail) = tail {
                    print!(", ");
                    self.dump_operand(tail);
                }
                print!("]");
            }
            Value::ListHead(operand) => {
                print!("head ");
                self.dump_operand(operand);
            }
            Value::ListTail(operand) => {
                print!("tail ");
                self.dump_operand(operand);
            }
            Value::ListEmpty(operand) => {
                print!("empty ");
                self.dump_operand(operand);
            }
            Value::Binary(op, lhs, rhs) => {
                print!("{:?} ", op);
                self.dump_operand(lhs);
                print!(" ");
                self.dump_operand(rhs);
            }
            Value::Unary(op, operand) => {
                print!("{:?} ", op);
                self.dump_operand(operand);
            }
            Value::IsVariant(operand, variant) => {
                print!("is_variant ");
                self.dump_operand(operand);
                print!(" {}", variant);
            }
            Value::Call(func, args) => {
                self.dump_operand(func);
                print!("(");
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        print!(", ");
                    }
                    self.dump_operand(arg);
                }
                print!(")");
            }
            Value::Ref(place) => {
                print!("&");
                self.dump_place(place);
            }
            Value::Tuple(operands) => {
                print!("(");
                for (i, operand) in operands.iter().enumerate() {
                    if i > 0 {
                        print!(", ");
                    }
                    self.dump_operand(operand);
                }
                print!(")");
            }
            Value::Adt(adt, operands) => {
                print!("adt {}(", adt);
                for (i, operand) in operands.iter().enumerate() {
                    if i > 0 {
                        print!(", ");
                    }
                    self.dump_operand(operand);
                }
                print!(")");
            }
        }
    }

    pub fn dump_operand<T: Display>(&self, operand: &Operand<T>) {
        match operand {
            Operand::Copy(place) => {
                print!("copy ");
                self.dump_place(place);
            }
            Operand::Move(place) => {
                print!("move ");
                self.dump_place(place);
            }
            Operand::Constant(constant) => self.dump_constant(constant),
        }
    }

    pub fn dump_constant(&self, constant: &Constant) {
        match constant {
            Constant::Void => print!("void"),
            Constant::Bool(value) => print!("{}", value),
            Constant::Int(signed, base, value, kind) => {
                if *signed {
                    print!("-");
                }
                print!("{:?}", base);
                for byte in value {
                    print!("{:02x}", byte);
                }
                print!(" {:?}", kind);
            }
            Constant::String(value) => print!("{:?}", value),
        }
    }

    pub fn dump_place<T: Display>(&self, place: &Place<T>) {
        match &place.location {
            Location::Local(local) => print!("local {}", local),
            Location::Argument(argument) => print!("argument {}", argument),
            Location::Capture(capture) => print!("capture {}", capture),
        }

        for projection in &place.projection {
            match &projection.kind {
                ProjectionKind::Field { variant, field } => {
                    if let Some(variant) = variant {
                        print!(".variant {} ", variant);
                    }
                    print!(".field {}", field);
                }
                ProjectionKind::Deref => {
                    print!(".*");
                }
            }
        }

        print!(": {}", place.ty);
    }
}
