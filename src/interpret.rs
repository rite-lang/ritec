use crate::{ast::BinOp, mir};

#[derive(Clone, Debug)]
pub enum Value {
    Void,
    Int(i64),
    Bool(bool),
    Func(usize),
    List(Vec<Value>),
    Adt(usize, Vec<Value>),
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
            mir::ExprKind::List(_) => todo!(),
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
            mir::ExprKind::Binary(op, lhs, rhs) => match op {
                BinOp::Add => {
                    let Value::Int(lhs) = self.eval(frame, lhs) else {
                        panic!("expected integer");
                    };

                    let Value::Int(rhs) = self.eval(frame, rhs) else {
                        panic!("expected integer");
                    };

                    Value::Int(lhs + rhs)
                }
                BinOp::Sub => {
                    let Value::Int(lhs) = self.eval(frame, lhs) else {
                        panic!("expected integer");
                    };

                    let Value::Int(rhs) = self.eval(frame, rhs) else {
                        panic!("expected integer");
                    };

                    Value::Int(lhs - rhs)
                }
                BinOp::Mul => {
                    let Value::Int(lhs) = self.eval(frame, lhs) else {
                        panic!("expected integer");
                    };

                    let Value::Int(rhs) = self.eval(frame, rhs) else {
                        panic!("expected integer");
                    };

                    Value::Int(lhs * rhs)
                }
                BinOp::Div => {
                    let Value::Int(lhs) = self.eval(frame, lhs) else {
                        panic!("expected integer");
                    };

                    let Value::Int(rhs) = self.eval(frame, rhs) else {
                        panic!("expected integer");
                    };

                    Value::Int(lhs / rhs)
                }
                BinOp::Rem => {
                    let Value::Int(lhs) = self.eval(frame, lhs) else {
                        panic!("expected integer");
                    };

                    let Value::Int(rhs) = self.eval(frame, rhs) else {
                        panic!("expected integer");
                    };

                    Value::Int(lhs % rhs)
                }
                _ => todo!(),
            },
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
            mir::ExprKind::Match(input, r#match) => match r#match {
                mir::Match::Bool(r#true, r#false) => {
                    let Value::Bool(value) = frame.locals[*input].clone() else {
                        panic!("expected boolean");
                    };

                    if value {
                        self.eval(frame, r#true)
                    } else {
                        self.eval(frame, r#false)
                    }
                }
                mir::Match::Adt(variants, default) => {
                    let Value::Adt(tag, fields) = frame.locals[*input].clone() else {
                        panic!("expected ADT");
                    };

                    match variants[tag] {
                        Some((ref locals, ref body)) => {
                            for (i, &index) in locals.iter().enumerate() {
                                frame.locals[index] = fields[i].clone();
                            }

                            self.eval(frame, body)
                        }
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

#[derive(Debug)]
struct Frame {
    locals: Vec<Value>,
    arguments: Vec<Value>,
}
