use crate::mir;

#[derive(Clone, Debug)]
pub enum Value {
    Void,
    Int(i64),
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
            locals: Vec::new(),
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
                    locals: Vec::new(),
                    arguments,
                };

                self.eval(&mut frame, &func.body)
            }
            mir::ExprKind::Let(_, expr) => {
                let value = self.eval(frame, expr);
                frame.locals.push(value);
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
        }
    }
}

#[derive(Debug)]
struct Frame {
    locals: Vec<Value>,
    arguments: Vec<Value>,
}
