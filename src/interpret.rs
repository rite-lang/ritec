use crate::{
    ast::BinOp,
    hir::UnOp,
    rir::{
        self, Block, Constant, Location, Operand, Place, Projection, ProjectionKind, Specific,
        Statement, Unit,
    },
};
use std::collections::HashMap;
use std::{cell::RefCell, iter::Peekable, rc::Rc};

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Void,
    Int(i64),
    Bool(bool),
    Func(usize, Vec<Value>),
    List(Option<Box<List>>),
    Adt(usize, Vec<Value>),
    String(String),
    Ref(Rc<RefCell<Value>>),
}

impl Value {
    pub fn list_from_vec(values: Vec<Value>) -> Value {
        let mut list = None;

        for value in values.into_iter().rev() {
            list = Some(Box::new(List {
                head: value,
                tail: list,
            }));
        }

        Value::List(list)
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Value::Void => write!(f, "void"),
            Value::Int(n) => write!(f, "{}", n),
            Value::Bool(b) => write!(f, "{}", b),
            Value::String(s) => write!(f, "\"{}\"", s),
            Value::Func(func, captured) => write!(
                f,
                "Func({}, [{}])",
                func,
                captured
                    .iter()
                    .map(|value| value.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Value::List(None) => write!(f, "[]"),
            Value::List(Some(list)) => write!(f, "[{}]", list),
            Value::Adt(variant, fields) => {
                write!(f, "{{{}|", variant)?;

                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }

                    write!(f, "{}", field)?;
                }

                write!(f, "}}")
            }
            Value::Ref(value) => write!(f, "&{}", value.borrow()),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct List {
    head: Value,
    tail: Option<Box<List>>,
}

impl std::fmt::Display for List {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.head)?;

        if let Some(tail) = &self.tail {
            write!(f, ", {}", tail)?;
        }

        Ok(())
    }
}

#[derive(Debug)]
struct Frame {
    locals: Vec<Value>,
    arguments: Vec<Value>,
    captured: Vec<Value>,
}

// Standard library bindings.
fn string_to_bytes(mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::String(s) = args.pop().unwrap() else {
        panic!("expected string literal")
    };

    let bytes = s
        .as_bytes()
        .iter()
        .map(|&b| Value::Int(b as i64))
        .collect::<Vec<_>>();

    Value::list_from_vec(bytes)
}

fn string_from_bytes(mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::List(list) = args.pop().unwrap() else {
        panic!("expected list")
    };

    let mut s = String::new();

    let mut list = list;

    while let Some(l) = list {
        let Value::Int(b) = l.head else {
            panic!("expected integer")
        };

        s.push(b as u8 as char);
        list = l.tail;
    }

    Value::String(s)
}

fn string_concat(mut args: Vec<Value>) -> Value {
    let mut s = String::new();

    for arg in args {
        match arg {
            Value::String(s2) => s.push_str(&s2),
            _ => panic!("expected string"),
        }
    }

    Value::String(s)
}

pub struct Interpreter<'a> {
    mir: &'a Unit<Specific>,
    builtins: HashMap<usize, fn(Vec<Value>) -> Value>,
}

impl<'a> Interpreter<'a> {
    pub fn new(mir: &'a Unit<Specific>) -> Self {
        Self {
            mir,
            builtins: Self::find_builtins(mir),
        }
    }

    /// Find builtin functions in the MIR based on the `extern intrinsic` decorator.
    fn find_builtins(mir: &'a Unit<Specific>) -> HashMap<usize, fn(Vec<Value>) -> Value> {
        let mut builtins = HashMap::new();
        for (index, value) in mir.funcs.iter().enumerate() {
            // Find extern intrinsic decorator.
            let extern_decorator = value
                .decorators
                .iter()
                .find(|d| d.name == "extern" && d.args.len() == 2 && d.args[0] == "intrinsic");

            match extern_decorator {
                Some(extern_decorator) => {
                    let name = extern_decorator.args[1].as_str();
                    let func = match name {
                        "string_to_bytes" => string_to_bytes,
                        "string_from_bytes" => string_from_bytes,
                        "string_concat" => string_concat,
                        // We allow intrinsics to have pure rite fallbacks.
                        _ => continue,
                    };

                    builtins.insert(index, func);
                }
                None => {}
            }
        }

        builtins
    }

    pub fn interpret(&self, main: usize) -> Value {
        let mut frame = Frame {
            locals: vec![Value::Void; self.mir.funcs[main].locals.len()],
            arguments: Vec::new(),
            captured: Vec::new(),
        };

        self.interpret_block(&mut frame, &self.mir.funcs[main].body)
            .unwrap()
    }

    fn interpret_block(&self, frame: &mut Frame, block: &Block<Specific>) -> Option<Value> {
        for statement in block.statements.iter() {
            match statement {
                Statement::Use { value } => {
                    self.interpret_value(frame, value);
                }
                Statement::Return { value } => {
                    return match value {
                        Some(value) => Some(self.interpret_value(frame, value)),
                        None => Some(Value::Void),
                    }
                }
                Statement::Panic { message } => {
                    panic!("{}", message);
                }
                Statement::Assign { place, value } => {
                    let value = self.interpret_value(frame, value);
                    self.assign_place(frame, place, value);
                }
                Statement::MatchBool {
                    input,
                    r#true,
                    r#false,
                } => {
                    let Value::Bool(input) = self.interpret_operand(frame, input) else {
                        panic!("expected boolean")
                    };

                    match input {
                        true => {
                            if let Some(value) = self.interpret_block(frame, r#true) {
                                return Some(value);
                            }
                        }
                        false => {
                            if let Some(value) = self.interpret_block(frame, r#false) {
                                return Some(value);
                            }
                        }
                    }
                }
                Statement::MatchAdt {
                    input,
                    variants,
                    default,
                } => {
                    let Value::Adt(variant, _) = self.interpret_operand(frame, input) else {
                        panic!("expected adt")
                    };

                    if let Some(block) = &variants[variant] {
                        if let Some(value) = self.interpret_block(frame, block) {
                            return Some(value);
                        }
                    } else if let Some(block) = default {
                        if let Some(value) = self.interpret_block(frame, block) {
                            return Some(value);
                        }
                    }
                }
            }
        }

        None
    }

    fn interpret_value(&self, frame: &mut Frame, value: &rir::Value<Specific>) -> Value {
        match value {
            rir::Value::Use(operand) => self.interpret_operand(frame, operand),
            rir::Value::Func(index, captures, _) => {
                let captures = captures
                    .iter()
                    .map(|op| self.interpret_operand(frame, op))
                    .collect();

                Value::Func(*index, captures)
            }
            rir::Value::List(items, tail) => {
                let mut list = match tail {
                    Some(tail) => {
                        let Value::List(tail) = self.interpret_operand(frame, tail) else {
                            panic!("expected list")
                        };

                        tail
                    }
                    None => None,
                };

                for item in items.iter().rev() {
                    let item = self.interpret_operand(frame, item);
                    list = Some(Box::new(List {
                        head: item,
                        tail: list,
                    }));
                }

                Value::List(list)
            }
            rir::Value::ListHead(list) => {
                let Value::List(list) = self.interpret_operand(frame, list) else {
                    panic!("expected list")
                };

                list.unwrap().head
            }
            rir::Value::ListTail(tail) => {
                let Value::List(list) = self.interpret_operand(frame, tail) else {
                    panic!("expected list")
                };

                Value::List(list.unwrap().tail)
            }
            rir::Value::ListEmpty(tail) => {
                let Value::List(list) = self.interpret_operand(frame, tail) else {
                    panic!("expected list")
                };

                Value::Bool(list.is_none())
            }
            rir::Value::Binary(op, lhs, rhs) => {
                let lhs = self.interpret_operand(frame, lhs);
                let rhs = self.interpret_operand(frame, rhs);

                match op {
                    BinOp::Add => {
                        let (Value::Int(lhs), Value::Int(rhs)) = (lhs, rhs) else {
                            panic!("expected integers")
                        };

                        Value::Int(lhs + rhs)
                    }
                    BinOp::Sub => {
                        let (Value::Int(lhs), Value::Int(rhs)) = (lhs, rhs) else {
                            panic!("expected integers")
                        };

                        Value::Int(lhs - rhs)
                    }
                    BinOp::Mul => {
                        let (Value::Int(lhs), Value::Int(rhs)) = (lhs, rhs) else {
                            panic!("expected integers")
                        };

                        Value::Int(lhs * rhs)
                    }
                    BinOp::Div => {
                        let (Value::Int(lhs), Value::Int(rhs)) = (lhs, rhs) else {
                            panic!("expected integers")
                        };

                        Value::Int(lhs / rhs)
                    }
                    BinOp::Rem => {
                        let (Value::Int(lhs), Value::Int(rhs)) = (lhs, rhs) else {
                            panic!("expected integers")
                        };

                        Value::Int(lhs % rhs)
                    }
                    BinOp::And => {
                        let (Value::Bool(lhs), Value::Bool(rhs)) = (lhs, rhs) else {
                            panic!("expected booleans")
                        };

                        Value::Bool(lhs && rhs)
                    }
                    BinOp::Or => {
                        let (Value::Bool(lhs), Value::Bool(rhs)) = (lhs, rhs) else {
                            panic!("expected booleans")
                        };

                        Value::Bool(lhs || rhs)
                    }
                    BinOp::Eq => Value::Bool(lhs == rhs),
                    BinOp::Ne => Value::Bool(lhs != rhs),
                    BinOp::Lt => {
                        let (Value::Int(lhs), Value::Int(rhs)) = (lhs, rhs) else {
                            panic!("expected integers")
                        };

                        Value::Bool(lhs < rhs)
                    }
                    BinOp::Le => {
                        let (Value::Int(lhs), Value::Int(rhs)) = (lhs, rhs) else {
                            panic!("expected integers")
                        };

                        Value::Bool(lhs <= rhs)
                    }
                    BinOp::Gt => {
                        let (Value::Int(lhs), Value::Int(rhs)) = (lhs, rhs) else {
                            panic!("expected integers")
                        };

                        Value::Bool(lhs > rhs)
                    }
                    BinOp::Ge => {
                        let (Value::Int(lhs), Value::Int(rhs)) = (lhs, rhs) else {
                            panic!("expected integers")
                        };

                        Value::Bool(lhs >= rhs)
                    }
                }
            }
            rir::Value::Unary(op, operand) => {
                let operand = self.interpret_operand(frame, operand);

                match op {
                    UnOp::Neg => {
                        let Value::Int(operand) = operand else {
                            panic!("expected integer")
                        };

                        Value::Int(-operand)
                    }
                    UnOp::Not => {
                        let Value::Bool(operand) = operand else {
                            panic!("expected boolean")
                        };

                        Value::Bool(!operand)
                    }
                }
            }
            rir::Value::VariantTag(value) => {
                let Value::Adt(variant, _) = self.interpret_operand(frame, value) else {
                    panic!("expected adt")
                };

                Value::Int(variant as i64)
            }
            rir::Value::Call(func, args) => {
                let Value::Func(func, captured) = self.interpret_operand(frame, func) else {
                    panic!("expected function")
                };

                let args = args
                    .iter()
                    .map(|op| self.interpret_operand(frame, op))
                    .collect();

                let mut frame = Frame {
                    locals: vec![Value::Void; self.mir.funcs[func].locals.len()],
                    arguments: args,
                    captured,
                };

                if let Some(builtin_func) = self.builtins.get(&func) {
                    return builtin_func(frame.arguments);
                }

                self.interpret_block(&mut frame, &self.mir.funcs[func].body)
                    .unwrap()
            }
            rir::Value::Ref(place) => {
                let value = self.interpret_copy_place(frame, place);
                Value::Ref(Rc::new(RefCell::new(value)))
            }
            rir::Value::Tuple(items) => {
                let items = items
                    .iter()
                    .map(|op| self.interpret_operand(frame, op))
                    .collect();

                Value::Adt(0, items)
            }
            rir::Value::Adt(variant, items) => {
                let items = items
                    .iter()
                    .map(|op| self.interpret_operand(frame, op))
                    .collect();

                Value::Adt(*variant, items)
            }
        }
    }

    fn interpret_operand(&self, frame: &mut Frame, operand: &Operand<Specific>) -> Value {
        match operand {
            Operand::Copy(place) => self.interpret_copy_place(frame, place),
            Operand::Move(place) => self.interpret_copy_place(frame, place),
            Operand::Constant(constant) => self.interpret_constant(constant),
        }
    }

    fn interpret_copy_place(&self, frame: &mut Frame, place: &Place<Specific>) -> Value {
        let mut value = match place.location {
            Location::Local(i) => frame.locals[i].clone(),
            Location::Argument(i) => frame.arguments[i].clone(),
            Location::Capture(i) => frame.captured[i].clone(),
        };

        for projection in place.projection.iter() {
            match projection.kind {
                ProjectionKind::Field { field, .. } => match &value {
                    Value::Adt(_, fields) => value = fields[field].clone(),
                    _ => todo!(),
                },
                ProjectionKind::Deref => {
                    let Value::Ref(mut_value) = value else {
                        panic!("expected mutable reference");
                    };

                    value = mut_value.borrow().clone();
                }
            }
        }

        value
    }

    fn assign_place(&self, frame: &mut Frame, place: &Place<Specific>, value: Value) {
        let target = match place.location {
            Location::Local(i) => &mut frame.locals[i],
            Location::Argument(i) => &mut frame.arguments[i],
            Location::Capture(i) => &mut frame.captured[i],
        };

        fn recurse<'a>(
            target: &mut Value,
            value: Value,
            mut projection: Peekable<impl Iterator<Item=&'a Projection<Specific>>>,
        ) {
            match projection.next() {
                Some(proj) => match proj.kind {
                    ProjectionKind::Field { field, .. } => match target {
                        Value::Adt(_, fields) => recurse(&mut fields[field], value, projection),
                        _ => todo!(),
                    },
                    ProjectionKind::Deref => {
                        let Value::Ref(ref target) = target else {
                            panic!("expected mutable reference");
                        };

                        recurse(&mut target.borrow_mut(), value, projection);
                    }
                },
                None => *target = value,
            }
        }

        recurse(target, value, place.projection.iter().peekable());
    }

    fn interpret_constant(&self, constant: &Constant) -> Value {
        match constant {
            Constant::Void => Value::Void,
            Constant::Bool(b) => Value::Bool(*b),
            Constant::Int(negative, base, digits) => {
                let mut n = 0;

                for &digit in digits.iter() {
                    n = n * base.radix() as i64 + digit as i64;
                }

                if *negative {
                    n = -n;
                }

                Value::Int(n)
            }
            Constant::String(s) => Value::String(s.to_string()),
        }
    }
}
