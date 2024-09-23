use crate::{ast::BinOp, mir};

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Void,
    Int(i64),
    Bool(bool),
    Func(usize),
    List(Option<Box<List>>),
    Adt(usize, Vec<Value>),
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Value::Void => write!(f, "void"),
            Value::Int(n) => write!(f, "{}", n),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Func(func) => write!(f, "func {}", func),
            Value::List(None) => write!(f, "[]"),
            Value::List(Some(list)) => write!(f, "[{}]", list),
            Value::Adt(variant, fields) => {
                write!(f, "Adt({}, [", variant)?;

                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }

                    write!(f, "{}", field)?;
                }

                write!(f, "])")
            }
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
}

pub struct Interpreter<'a> {
    mir: &'a mir::Mir,
}

impl<'a> Interpreter<'a> {
    pub fn new(mir: &'a mir::Mir) -> Self {
        Self { mir }
    }

    pub fn run(&self, main: usize) -> Value {
        let func = &self.mir.funcs[main];

        let mut frame = Frame {
            locals: vec![Value::Void; func.locals.len()],
            arguments: Vec::new(),
        };

        self.eval(&mut frame, &func.body)
    }

    fn eval(&self, frame: &mut Frame, expr: &mir::Expr) -> Value {
        match &expr.kind {
            mir::ExprKind::Const(constant) => match constant {
                mir::Constant::Void => Value::Void,
                mir::Constant::Int(negative, base, value) => {
                    let mut n = 0;

                    for &digit in value {
                        n = n * base.radix() as i64 + digit as i64;
                    }

                    if *negative {
                        n = -n;
                    }

                    Value::Int(n)
                }
                mir::Constant::Bool(value) => Value::Bool(*value),
                mir::Constant::Func(func) => Value::Func(*func),
            },
            mir::ExprKind::Local(index) => frame.locals[*index].clone(),
            mir::ExprKind::Argument(index) => frame.arguments[*index].clone(),
            mir::ExprKind::List(items, rest) => {
                let mut values = match rest {
                    Some(rest) => {
                        let Value::List(rest) = self.eval(frame, rest) else {
                            panic!("expected list");
                        };

                        rest
                    }
                    None => None,
                };

                for item in items.iter().rev() {
                    let value = self.eval(frame, item);
                    values = Some(Box::new(List {
                        head: value,
                        tail: values,
                    }));
                }

                Value::List(values)
            }
            mir::ExprKind::ListHead(list) => {
                let Value::List(Some(list)) = self.eval(frame, list) else {
                    panic!("expected list");
                };

                list.head.clone()
            }
            mir::ExprKind::ListTail(list) => {
                let Value::List(Some(list)) = self.eval(frame, list) else {
                    panic!("expected list");
                };

                Value::List(list.tail.clone())
            }
            mir::ExprKind::Block(exprs) => {
                let mut value = Value::Void;

                for expr in exprs {
                    value = self.eval(frame, expr);
                }

                value
            }
            mir::ExprKind::Call(func, args) => {
                let Value::Func(func) = self.eval(frame, func) else {
                    panic!("expected function");
                };

                let func = &self.mir.funcs[func];

                let mut arguments = Vec::new();

                for arg in args {
                    arguments.push(self.eval(frame, arg));
                }

                let mut frame = Frame {
                    locals: vec![Value::Void; func.locals.len()],
                    arguments,
                };

                self.eval(&mut frame, &func.body)
            }
            mir::ExprKind::Binary(op, lhs, rhs) => {
                let lhs = self.eval(frame, lhs);
                let rhs = self.eval(frame, rhs);

                match op {
                    BinOp::Add => {
                        let (Value::Int(lhs), Value::Int(rhs)) = (lhs, rhs) else {
                            panic!("expected integer");
                        };

                        Value::Int(lhs + rhs)
                    }
                    BinOp::Sub => {
                        let (Value::Int(lhs), Value::Int(rhs)) = (lhs, rhs) else {
                            panic!("expected integer");
                        };

                        Value::Int(lhs - rhs)
                    }
                    BinOp::Mul => {
                        let (Value::Int(lhs), Value::Int(rhs)) = (lhs, rhs) else {
                            panic!("expected integer");
                        };

                        Value::Int(lhs * rhs)
                    }
                    BinOp::Div => {
                        let (Value::Int(lhs), Value::Int(rhs)) = (lhs, rhs) else {
                            panic!("expected integer");
                        };

                        Value::Int(lhs / rhs)
                    }
                    BinOp::Rem => {
                        let (Value::Int(lhs), Value::Int(rhs)) = (lhs, rhs) else {
                            panic!("expected integer");
                        };

                        Value::Int(lhs % rhs)
                    }
                    BinOp::Eq => Value::Bool(lhs == rhs),
                    BinOp::Ne => Value::Bool(lhs != rhs),
                    BinOp::Lt => {
                        let (Value::Int(lhs), Value::Int(rhs)) = (lhs, rhs) else {
                            panic!("expected integer");
                        };

                        Value::Bool(lhs < rhs)
                    }
                    BinOp::Le => {
                        let (Value::Int(lhs), Value::Int(rhs)) = (lhs, rhs) else {
                            panic!("expected integer");
                        };

                        Value::Bool(lhs <= rhs)
                    }
                    BinOp::Gt => {
                        let (Value::Int(lhs), Value::Int(rhs)) = (lhs, rhs) else {
                            panic!("expected integer");
                        };

                        Value::Bool(lhs > rhs)
                    }
                    BinOp::Ge => {
                        let (Value::Int(lhs), Value::Int(rhs)) = (lhs, rhs) else {
                            panic!("expected integer");
                        };

                        Value::Bool(lhs >= rhs)
                    }
                    BinOp::And => {
                        let (Value::Bool(lhs), Value::Bool(rhs)) = (lhs, rhs) else {
                            panic!("expected boolean");
                        };

                        Value::Bool(lhs && rhs)
                    }
                    BinOp::Or => {
                        let (Value::Bool(lhs), Value::Bool(rhs)) = (lhs, rhs) else {
                            panic!("expected boolean");
                        };

                        Value::Bool(lhs || rhs)
                    }
                }
            }
            mir::ExprKind::Let(index, expr) => {
                let value = self.eval(frame, expr);
                frame.locals[*index] = value;
                Value::Void
            }
            mir::ExprKind::Adt(variant, fields) => {
                let mut values = Vec::new();

                for field in fields {
                    values.push(self.eval(frame, field));
                }

                Value::Adt(*variant, values)
            }
            mir::ExprKind::Field(adt, field) => {
                let Value::Adt(_, fields) = self.eval(frame, adt) else {
                    panic!("expected ADT");
                };

                fields[*field].clone()
            }
            mir::ExprKind::VariantField(adt, variant, field) => {
                let Value::Adt(tag, fields) = self.eval(frame, adt) else {
                    panic!("expected ADT");
                };

                if tag != *variant {
                    panic!("expected variant");
                }

                fields[*field].clone()
            }
            mir::ExprKind::Match(input, r#match) => match r#match {
                mir::Match::Bool(r#true, r#false) => {
                    let Value::Bool(value) = self.eval(frame, input) else {
                        panic!("expected boolean");
                    };

                    match value {
                        true => self.eval(frame, r#true),
                        false => self.eval(frame, r#false),
                    }
                }
                mir::Match::List(some, none) => {
                    let Value::List(list) = self.eval(frame, input) else {
                        panic!("expected list");
                    };

                    match list.is_some() {
                        true => self.eval(frame, some),
                        false => self.eval(frame, none),
                    }
                }
                mir::Match::Adt(variants, default) => {
                    let Value::Adt(tag, _) = self.eval(frame, input) else {
                        panic!("expected ADT");
                    };

                    match variants[tag] {
                        Some(ref body) => self.eval(frame, body),
                        None => match default {
                            Some(default) => self.eval(frame, default),
                            None => panic!("no default branch"),
                        },
                    }
                }
            },
        }
    }
}
