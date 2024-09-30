use smallvec::SmallVec;

use crate::ast::BinOp;
use crate::hir::UnOp;
use crate::interpret::builtins::IntrinsicMap;
use crate::interpret::value::{List, Value};
use crate::number::IntKind;
use crate::rir::{
    Constant, Location, Operand, Place, Projection, ProjectionKind, Specific, Statement, Unit,
};
use crate::{match_value_binary_op, match_value_convert, rir};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::env;
use std::iter::Peekable;
use std::rc::Rc;

#[derive(Debug)]
/// Interpreter call stack frame.
/// Holds locals, arguments, and captured values.
struct Frame {
    locals: Vec<Value>,
    arguments: Vec<Value>,
    captured: Rc<SmallVec<[Value; 4]>>,

    func: usize,
    actions: VecDeque<Action>,
    return_address: NestedProgramCounter,
    return_value: Option<Value>,
    return_place: Option<Place<Specific>>,
}

impl Frame {
    fn copy(&mut self, place: &Place<Specific>) -> Value {
        let mut value = match place.location {
            Location::Local(i) => self.locals[i].clone(),
            Location::Argument(i) => self.arguments[i].clone(),
            Location::Capture(i) => self.captured[i].clone(),
        };

        for projection in place.projection.iter() {
            match projection.kind {
                ProjectionKind::Field { field, .. } => match &value {
                    Value::Adt((_, fields)) => value = fields[field].clone(),
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

    fn assign(&mut self, place: &Place<Specific>, value: Value) {
        fn recurse<'a>(
            target: &mut Value,
            value: Value,
            mut projection: Peekable<impl Iterator<Item = &'a Projection<Specific>>>,
        ) {
            match projection.next() {
                Some(proj) => match proj.kind {
                    ProjectionKind::Field { field, .. } => match target {
                        Value::Adt((_, fields)) => {
                            recurse(&mut Rc::make_mut(fields)[field], value, projection)
                        }
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

        let target = match place.location {
            Location::Local(i) => &mut self.locals[i],
            Location::Argument(i) => &mut self.arguments[i],
            Location::Capture(i) => &mut Rc::make_mut(&mut self.captured)[i],
        };

        recurse(target, value, place.projection.iter().peekable());
    }
}

#[derive(Debug)]
enum Action {
    Enter {
        place: Place<Specific>,
        func: usize,
        args: Vec<Value>,
        captures: Rc<SmallVec<[Value; 4]>>,
    },
    Exit,
}

#[derive(Clone, Debug)]
/// Store program counter inside the block
/// And keep track of which block to choose from complex expressions.
enum ProgramCounterPath {
    Root,
    Bool { value: bool },
    Adt { variant: Option<usize> },
}

#[derive(Clone, Debug)]
/// Program counter to keep track
/// Of independent blocks.
struct NestedProgramCounter {
    /// Current workable program counter
    /// This is the same value as the last program counter path
    /// if it has any.
    pcs: Vec<(usize, ProgramCounterPath)>,
    needs_restore: bool,
}

impl NestedProgramCounter {
    fn new() -> Self {
        Self {
            pcs: vec![(0, ProgramCounterPath::Root)],
            needs_restore: false,
        }
    }

    fn current(&self) -> usize {
        self.pcs.last().map(|(pc, _)| *pc).unwrap_or(0)
    }

    fn set(&mut self, spc: usize) {
        self.pcs.last_mut().map(|(pc, _)| *pc = spc);
    }

    fn increment(&mut self) {
        self.pcs.last_mut().map(|(pc, _)| *pc += 1);
    }

    fn store(&self) -> Self {
        Self {
            pcs: self.pcs.clone(),
            needs_restore: true,
        }
    }

    fn reset(&mut self) {
        self.pcs.clear();
        self.pcs.push((0, ProgramCounterPath::Root));
    }

    fn is_at_end(&self, block: &rir::Block<Specific>) -> bool {
        self.pcs.len() == 1 && self.current() >= block.statements.len()
    }

    fn push(&mut self, pc: usize, path: ProgramCounterPath) {
        self.pcs.push((pc, path));
    }

    fn pop(&mut self) {
        self.pcs.pop();
    }

    /// Extracts the block from the program counter path
    fn extract_block<'a>(&self, block: &'a rir::Block<Specific>) -> &'a rir::Block<Specific> {
        let mut pc = self.pcs.first().unwrap().0;
        let mut block: &'a rir::Block<Specific> = block;

        for (next_pc, path) in self.pcs.iter().skip(1) {
            let statement = &block.statements[pc];

            match (statement, path) {
                (
                    Statement::MatchBool { ref r#true, .. },
                    ProgramCounterPath::Bool { value: true, .. },
                ) => {
                    block = r#true;
                }
                (
                    Statement::MatchBool { ref r#false, .. },
                    ProgramCounterPath::Bool { value: false, .. },
                ) => {
                    block = r#false;
                }
                (
                    Statement::MatchAdt {
                        ref variants,
                        ref default,
                        ..
                    },
                    ProgramCounterPath::Adt { variant, .. },
                ) => {
                    if let Some(variant) = variant {
                        block = &variants[*variant].as_ref().unwrap();
                    } else {
                        block = default.as_ref().unwrap();
                    }
                }
                _ => panic!("unexpected program counter path"),
            }

            pc = *next_pc;
        }

        block
    }

    /// Restores program counter together with a block
    /// Clears out block indices by extracting the actual block to work on
    /// and sets the program counter to the correct one where we need to continue
    /// inside the returned block.
    fn restore_block<'a>(&mut self, block: &'a rir::Block<Specific>) -> &'a rir::Block<Specific> {
        if self.pcs.len() == 1 || !self.needs_restore {
            return block;
        }

        self.needs_restore = false;

        let block = self.extract_block(block);

        block
    }
}

#[derive(Debug)]
enum ControlFlow {
    Continue,
    StackTrace(String),
    Yield,
}

pub struct Interpreter<'a> {
    rir: &'a Unit<Specific>,
    builtins: IntrinsicMap,
}

impl<'a> Interpreter<'a> {
    pub fn new(rir: &'a Unit<Specific>) -> Self {
        Self {
            rir,
            builtins: IntrinsicMap::new(rir),
        }
    }

    pub(crate) fn interpret(&self, main: usize) -> Value {
        let args = env::args().map(Value::String).collect();
        let args = Value::list_from_vec(args);

        // Setup initial frame we do this manually to only
        // create a single frame without another as overhead.
        let frame = Frame {
            locals: vec![Value::void(); self.rir.funcs[main].locals.len()],
            arguments: vec![args],
            captured: Rc::new(SmallVec::new()),
            func: main,
            actions: VecDeque::new(),
            return_address: NestedProgramCounter::new(),
            return_value: None,
            return_place: None,
        };

        let mut pc = NestedProgramCounter::new();
        let mut stack = vec![frame];
        let mut last_value = Value::void();

        loop {
            let Some(frame) = stack.last_mut() else {
                break;
            };

            // If no actions are pending we continue interpreting the current function.
            let Some(action) = frame.actions.pop_front() else {
                let func = &self.rir.funcs[frame.func];

                // If we reached the end of the function, exit.
                if pc.is_at_end(&func.body) {
                    panic!(
                        "End of function without return invalid program counter! {:?}",
                        pc
                    );
                }

                if let Some(builtin) = self.builtins.map.get(&frame.func) {
                    frame.return_value = Some(builtin(func, frame.arguments.clone()));
                    frame.actions.push_back(Action::Exit);
                    continue;
                }

                let block = pc.restore_block(&func.body);

                match self.interpret_block(frame, block, &mut pc) {
                    // End of block, continue to the next block.
                    Err(ControlFlow::Yield) => {}
                    Err(ControlFlow::StackTrace(message)) => {
                        for frame in stack.iter() {
                            let func = &self.rir.funcs[frame.func];

                            println!("Function: {}", func.name);
                            println!("Pc: {}", frame.return_address.current());
                        }

                        panic!("Stack trace: {}", message);
                    }
                    _ => {
                        if pc.pcs.len() > 1 {
                            pc.pop();
                            pc.increment();
                            pc.needs_restore = true;
                        }
                    }
                }

                continue;
            };

            match action {
                Action::Enter {
                    place,
                    func,
                    args,
                    captures,
                } => {
                    let specific_func = &self.rir.funcs[func];
                    let locals_count = specific_func.locals.len();

                    // The saved program counter will start at the next instruction.
                    let mut saved_pc = pc.store();
                    saved_pc.increment();

                    stack.push(Frame {
                        locals: vec![Value::void(); locals_count],
                        arguments: args,
                        captured: captures,
                        func,
                        actions: VecDeque::new(),
                        return_address: saved_pc,
                        return_value: None,
                        return_place: Some(place.clone()),
                    });

                    // We fully reset the program counter to call the next function.
                    pc.reset();
                }
                Action::Exit => {
                    let frame = stack.pop().unwrap();
                    // We store the return value from the last frame.
                    last_value = frame.return_value.unwrap();

                    // We restore the program counter from the previous frame.
                    pc = frame.return_address;

                    // Assign the return value to the return place.
                    if let (Some(place), Some(current_frame)) =
                        (frame.return_place, stack.last_mut())
                    {
                        current_frame.assign(&place, last_value.clone());
                    }
                }
            }
        }

        last_value
    }

    fn interpret_block(
        &self,
        frame: &mut Frame,
        block: &rir::Block<Specific>,
        pc: &mut NestedProgramCounter,
    ) -> Result<Value, ControlFlow> {
        for (i, statement) in block.statements.iter().enumerate() {
            if i < pc.current() {
                continue;
            }

            // We increment the program counter after each statement
            // as we use the program counter to index into the block.
            pc.set(i);

            match statement {
                Statement::Use { value } => {
                    self.interpret_value(frame, value);
                }
                Statement::Call {
                    place, args, func, ..
                } => {
                    let Value::Func((func, captures)) = self.interpret_operand(frame, func) else {
                        panic!("expected function")
                    };

                    let args = args
                        .iter()
                        .map(|op| self.interpret_operand(frame, op))
                        .collect();

                    frame.actions.push_back(Action::Enter {
                        place: place.clone(),
                        func,
                        args,
                        captures,
                    });

                    return Err(ControlFlow::Yield);
                }
                Statement::Return { value } => {
                    if let Some(value1) = value {
                        let value = self.interpret_value(frame, value1);
                        frame.return_value = Some(value);
                        frame.actions.push_back(Action::Exit);
                        return Err(ControlFlow::Yield);
                    }
                }
                Statement::Panic { message } => {
                    panic!("{}", message);
                }
                Statement::Assign { place, value } => {
                    let value = self.interpret_value(frame, value);
                    frame.assign(place, value);
                }
                Statement::MatchBool {
                    input,
                    r#true,
                    r#false,
                } => {
                    let value = self.interpret_operand(frame, input);
                    let Value::Bool(input) = value else {
                        panic!("expected boolean got {:?}", value)
                    };

                    pc.push(0, ProgramCounterPath::Bool { value: input });

                    let block = match input {
                        true => r#true,
                        false => r#false,
                    };

                    match self.interpret_block(frame, block, pc) {
                        Ok(value) => {
                            pc.pop();
                            return Ok(value);
                        }
                        Err(err) => match err {
                            ControlFlow::Yield => {
                                return Err(ControlFlow::Yield);
                            }
                            ControlFlow::Continue => {
                                pc.pop();
                            }
                            ControlFlow::StackTrace(m) => {
                                return Err(ControlFlow::StackTrace(m));
                            }
                        },
                    }
                }
                Statement::MatchAdt {
                    input,
                    variants,
                    default,
                } => {
                    let Value::Adt((variant, _)) = self.interpret_operand(frame, input) else {
                        return Err(ControlFlow::StackTrace(format!(
                            "expected adt got {:?}",
                            input
                        )));
                    };

                    // Get the block to execute based on the variant
                    let block = match (&variants[variant], default) {
                        (Some(block), _) => {
                            pc.push(
                                0,
                                ProgramCounterPath::Adt {
                                    variant: Some(variant),
                                },
                            );

                            block
                        }
                        (None, Some(block)) => {
                            pc.push(0, ProgramCounterPath::Adt { variant: None });
                            block
                        }
                        _ => panic!("expected either variant or default"),
                    };

                    match self.interpret_block(frame, block, pc) {
                        Ok(value) => {
                            pc.pop();
                            return Ok(value);
                        }
                        Err(err) => match err {
                            ControlFlow::Yield => {
                                return Err(ControlFlow::Yield);
                            }
                            ControlFlow::Continue => {
                                pc.pop();
                            }
                            ControlFlow::StackTrace(m) => {
                                return Err(ControlFlow::StackTrace(m));
                            }
                        },
                    }
                }
            };
        }

        Err(ControlFlow::Continue)
    }

    /// Resolve values from IR to runtime values (and perform logical operations on them).
    fn interpret_value(&self, frame: &mut Frame, value: &rir::Value<Specific>) -> Value {
        match value {
            rir::Value::Use(operand) => self.interpret_operand(frame, operand),
            rir::Value::Cast(_, operand) => self.interpret_operand(frame, operand),
            rir::Value::Func(index, captures, _) => {
                let captures = captures
                    .iter()
                    .map(|op| self.interpret_operand(frame, op))
                    .collect();

                Value::Func((*index, Rc::new(captures)))
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
                    list = Some(Rc::new(List {
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

                list.unwrap().head.clone()
            }
            rir::Value::ListTail(tail) => {
                let Value::List(list) = self.interpret_operand(frame, tail) else {
                    panic!("expected list")
                };

                Value::List(list.unwrap().tail.clone())
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
                        match_value_binary_op!((lhs, rhs), Value, lhs, rhs, { lhs + rhs })
                    }
                    BinOp::Sub => {
                        match_value_binary_op!((lhs, rhs), Value, lhs, rhs, { lhs - rhs })
                    }
                    BinOp::Mul => {
                        match_value_binary_op!((lhs, rhs), Value, lhs, rhs, { lhs * rhs })
                    }
                    BinOp::Div => {
                        match_value_binary_op!((lhs, rhs), Value, lhs, rhs, { lhs / rhs })
                    }
                    BinOp::Rem => {
                        match_value_binary_op!((lhs, rhs), Value, lhs, rhs, { lhs % rhs })
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
                        match_value_binary_op!((lhs, rhs), Value, Bool, lhs, rhs, { lhs < rhs })
                    }
                    BinOp::Le => {
                        match_value_binary_op!((lhs, rhs), Value, Bool, lhs, rhs, { lhs <= rhs })
                    }
                    BinOp::Gt => {
                        match_value_binary_op!((lhs, rhs), Value, Bool, lhs, rhs, { lhs > rhs })
                    }
                    BinOp::Ge => {
                        match_value_binary_op!((lhs, rhs), Value, Bool, lhs, rhs, { lhs >= rhs })
                    }
                }
            }
            rir::Value::Unary(op, operand) => {
                let operand = self.interpret_operand(frame, operand);

                match op {
                    UnOp::Neg => {
                        match_value_convert!(
                            operand,
                            Value,
                            Value,
                            o,
                            { -o },
                            I8,
                            I16,
                            I32,
                            I64,
                            Int
                        )
                    }
                    UnOp::Not => {
                        let Value::Bool(operand) = operand else {
                            panic!("expected boolean")
                        };

                        Value::Bool(!operand)
                    }
                }
            }
            rir::Value::IsVariant(value, variant) => {
                let Value::Adt((current, _)) = self.interpret_operand(frame, value) else {
                    panic!("expected adt")
                };

                Value::Bool(current == *variant)
            }

            rir::Value::Ref(place) => {
                let value = frame.copy(place);
                Value::Ref(Rc::new(RefCell::new(value)))
            }
            rir::Value::Tuple(items) => {
                let items = items
                    .iter()
                    .map(|op| self.interpret_operand(frame, op))
                    .collect();

                Value::Adt((0, Rc::new(items)))
            }
            rir::Value::Adt(variant, items) => {
                let items = items
                    .iter()
                    .map(|op| self.interpret_operand(frame, op))
                    .collect();

                Value::Adt((*variant, Rc::new(items)))
            }
        }
    }

    fn interpret_operand(&self, frame: &mut Frame, operand: &Operand<Specific>) -> Value {
        match operand {
            Operand::Copy(place) => frame.copy(place),
            Operand::Move(place) => frame.copy(place),
            Operand::Constant(constant) => self.interpret_constant(constant),
        }
    }

    fn interpret_constant(&self, constant: &Constant<Specific>) -> Value {
        match constant {
            Constant::Void => Value::void(),
            Constant::Bool(b) => Value::Bool(*b),
            Constant::Int(negative, base, digits, ty) => {
                let mut n = 0;

                for &digit in digits.iter() {
                    n = n * base.radix() as isize + digit as isize;
                }

                if *negative {
                    n = -n;
                }

                let rir::Specific::Int(kind) = ty else {
                    panic!("expected integer kind")
                };

                match kind {
                    IntKind::U8 => Value::U8(n as u8),
                    IntKind::U16 => Value::U16(n as u16),
                    IntKind::U32 => Value::U32(n as u32),
                    IntKind::U64 => Value::U64(n as u64),
                    IntKind::I8 => Value::I8(n as i8),
                    IntKind::I16 => Value::I16(n as i16),
                    IntKind::I32 => Value::I32(n as i32),
                    IntKind::I64 => Value::I64(n as i64),
                    IntKind::Int => Value::Int(n),
                }
            }
            Constant::String(s) => Value::String(s.to_string()),
        }
    }
}
