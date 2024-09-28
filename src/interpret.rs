use crate::number::IntKind;
use crate::{
    ast::BinOp,
    hir::UnOp,
    rir::{
        self, Block, Constant, Location, Operand, Place, Projection, ProjectionKind, Specific,
        Statement, Unit,
    },
};
use std::{
    cell::RefCell,
    cmp, env,
    io::{self, Read, Write},
    iter::Peekable,
    rc::Rc,
    sync::{
        atomic::{self, AtomicUsize},
        Arc, Mutex,
    },
};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
};

#[derive(Clone, Debug)]
pub struct RiteFile {
    id: usize,
    file: Arc<Mutex<Option<fs::File>>>,
}

impl RiteFile {
    fn new(file: fs::File) -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

        let id = NEXT_ID.fetch_add(1, atomic::Ordering::Relaxed);

        Self {
            id,
            file: Arc::new(Mutex::new(Some(file))),
        }
    }
}

impl PartialEq for RiteFile {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for RiteFile {}

impl PartialOrd for RiteFile {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RiteFile {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Array {
    Void(Vec<()>),
    Bool(Vec<bool>),
    Func(Vec<(usize, Vec<Value>)>),
    List(Vec<Option<Box<List>>>),
    Adt(Vec<(usize, Vec<Value>)>),
    String(Vec<String>),
    Ref(Vec<Rc<RefCell<Value>>>),
    Dict(Vec<BTreeMap<Value, Value>>),
    Array(Vec<Array>),
    File(Vec<RiteFile>),

    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    Int(Vec<isize>),
}


// Call a function with shared return type on inner vec on array.
macro_rules! array_vec_call {
    ($expr:expr, $name:ident, $func:block) => {
        match $expr {
            Array::Void($name) => $func,
            Array::Bool($name) => $func,
            Array::Func($name) => $func,
            Array::List($name) => $func,
            Array::Adt($name) => $func,
            Array::String($name) => $func,
            Array::Ref($name) => $func,
            Array::Dict($name) => $func,
            Array::Array($name) => $func,
            Array::File($name) => $func,
            Array::U8($name) => $func,
            Array::U16($name) => $func,
            Array::U32($name) => $func,
            Array::U64($name) => $func,
            Array::I8($name) => $func,
            Array::I16($name) => $func,
            Array::I32($name) => $func,
            Array::I64($name) => $func,
            Array::Int($name) => $func,
        }
    };
}

// Transform an array with an expression on the inner vec.
macro_rules! array_vec_transform_self {
    ($m:expr, $name:ident, $expr:expr) => {
        match $m {
            Array::Void($name) => Array::Void($expr),
            Array::Bool($name) => Array::Bool($expr),
            Array::Func($name) => Array::Func($expr),
            Array::List($name) => Array::List($expr),
            Array::Adt($name) => Array::Adt($expr),
            Array::String($name) => Array::String($expr),
            Array::Ref($name) => Array::Ref($expr),
            Array::Dict($name) => Array::Dict($expr),
            Array::Array($name) => Array::Array($expr),
            Array::File($name) => Array::File($expr),
            Array::U8($name) => Array::U8($expr),
            Array::U16($name) => Array::U16($expr),
            Array::U32($name) => Array::U32($expr),
            Array::U64($name) => Array::U64($expr),
            Array::I8($name) => Array::I8($expr),
            Array::I16($name) => Array::I16($expr),
            Array::I32($name) => Array::I32($expr),
            Array::I64($name) => Array::I64($expr),
            Array::Int($name) => Array::Int($expr),
        }
    };
}

// Match an array with a value and call a block with the value.
macro_rules! array_vec_and_value {
    ($array:expr, $value:expr, $vec:ident, $name: ident, $block:block) => {
        match ($array, $value) {
            (Array::Void($vec), Value::Void) => { let $name = (); $block }
            (Array::Bool($vec), Value::Bool($name)) => { $block }
            (Array::Func($vec), Value::Func(func, captured)) => { let $name = (func, captured); $block }
            (Array::List($vec), Value::List($name)) => { $block }
            (Array::Adt($vec), Value::Adt(variant, fields)) => { let $name = (variant, fields); $block }
            (Array::String($vec), Value::String($name)) => { $block }
            (Array::Ref($vec), Value::Ref($name)) => { $block }
            (Array::Dict($vec), Value::Dict($name)) => { $block }
            (Array::Array($vec), Value::Array($name)) => { $block }
            (Array::File($vec), Value::File($name)) => { $block }
            (Array::U8($vec), Value::Int(Int::U8($name))) => { $block }
            (Array::U16($vec), Value::Int(Int::U16($name))) => { $block }
            (Array::U32($vec), Value::Int(Int::U32($name))) => { $block }
            (Array::U64($vec), Value::Int(Int::U64($name))) => { $block }
            (Array::I8($vec), Value::Int(Int::I8($name))) => { $block }
            (Array::I16($vec), Value::Int(Int::I16($name))) => { $block }
            (Array::I32($vec), Value::Int(Int::I32($name))) => { $block }
            (Array::I64($vec), Value::Int(Int::I64($name))) => { $block }
            (Array::Int($vec), Value::Int(Int::Int($name))) => { $block }

            _ => panic!("mismatched types"),
        }
    };
}

impl Array {
    pub fn new(length: usize, default: Value) -> Self {
        match default {
            Value::Void => {
                let v = ();
                Array::Void(vec![v; length])
            }
            Value::Int(k) => match k {
                Int::U8(v) => Array::U8(vec![v; length]),
                Int::U16(v) => Array::U16(vec![v; length]),
                Int::U32(v) => Array::U32(vec![v; length]),
                Int::U64(v) => Array::U64(vec![v; length]),
                Int::I8(v) => Array::I8(vec![v; length]),
                Int::I16(v) => Array::I16(vec![v; length]),
                Int::I32(v) => Array::I32(vec![v; length]),
                Int::I64(v) => Array::I64(vec![v; length]),
                Int::Int(v) => Array::Int(vec![v; length]),
            }
            Value::Bool(v) => Array::Bool(vec![v; length]),
            Value::Func(a, b) => {
                let v = (a, b);
                Array::Func(vec![v; length])
            }
            Value::List(v) => Array::List(vec![v; length]),
            Value::Adt(a, b) => {
                let v = (a, b);
                Array::Adt(vec![v; length])
            }
            Value::String(v) => Array::String(vec![v; length]),
            Value::Ref(v) => Array::Ref(vec![v; length]),
            Value::Dict(v) => Array::Dict(vec![v; length]),
            Value::Array(v) => Array::Array(vec![v; length]),
            Value::File(v) => Array::File(vec![v; length]),
        }
    }

    pub fn len(&self) -> usize {
        array_vec_call!(self, v, { v.len() })
    }

    pub fn resize(&mut self, new_len: usize, default: Value) {
        array_vec_and_value!(self, default, v, d, {
            v.resize(new_len, d);
        })
    }

    pub fn truncate(&mut self, len: usize) {
        array_vec_call!(self, v, { v.truncate(len) })
    }

    pub fn slice(&self, start: usize, end: usize) -> Array {
        array_vec_transform_self!(self, v, v[start..end].to_vec())
    }

    pub fn get(&self, index: usize) -> Option<Value> {
        if index >= self.len() {
            return None;
        }

        match self {
            Array::Void(v) => {
                v.get(index).map(|_| Value::Void)
            }
            Array::Bool(v) => {
                v.get(index).map(|&b| Value::Bool(b))
            }
            Array::Func(v) => {
                v.get(index).map(|(func, captured)| Value::Func(*func, captured.clone()))
            }
            Array::List(v) => {
                v.get(index).map(|list| Value::List(list.clone()))
            }
            Array::Adt(v) => {
                v.get(index).map(|(variant, fields)| Value::Adt(*variant, fields.clone()))
            }
            Array::String(v) => {
                v.get(index).map(|s| Value::String(s.clone()))
            }
            Array::Ref(v) => {
                v.get(index).map(|r| Value::Ref(r.clone()))
            }
            Array::Dict(v) => {
                v.get(index).map(|dict| Value::Dict(dict.clone()))
            }
            Array::Array(v) => {
                v.get(index).map(|arr| Value::Array(arr.clone()))
            }
            Array::File(v) => {
                v.get(index).map(|file| Value::File(file.clone()))
            }
            Array::U8(v) => {
                v.get(index).map(|&n| Value::Int(Int::U8(n)))
            }
            Array::U16(v) => {
                v.get(index).map(|&n| Value::Int(Int::U16(n)))
            }
            Array::U32(v) => {
                v.get(index).map(|&n| Value::Int(Int::U32(n)))
            }
            Array::U64(v) => {
                v.get(index).map(|&n| Value::Int(Int::U64(n)))
            }
            Array::I8(v) => {
                v.get(index).map(|&n| Value::Int(Int::I8(n)))
            }
            Array::I16(v) => {
                v.get(index).map(|&n| Value::Int(Int::I16(n)))
            }
            Array::I32(v) => {
                v.get(index).map(|&n| Value::Int(Int::I32(n)))
            }
            Array::I64(v) => {
                v.get(index).map(|&n| Value::Int(Int::I64(n)))
            }
            Array::Int(v) => {
                v.get(index).map(|&n| Value::Int(Int::Int(n)))
            }
        }
    }

    pub fn set(&mut self, index: usize, value: Value) {
        if index >= self.len() {
            return;
        }

        array_vec_and_value!(self, value, v, d, {
            v[index] = d;
        })
    }
}

impl std::fmt::Display for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "[")?;

        for i in 0..self.len() {
            if i > 0 {
                write!(f, ", ")?;
            }

            write!(f, "{}", self.get(i).unwrap())?;
        }

        write!(f, "]")
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Int {
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    Int(isize),
}

// Macro for implementing binary operations on Int.
macro_rules! int_op {
    ($name:ident, $op:tt) => {
        fn $name(self, rhs: Int) -> Int {
            match (self, rhs) {
                (Int::U8(lhs), Int::U8(rhs)) => Int::U8(lhs $op rhs),
                (Int::U16(lhs), Int::U16(rhs)) => Int::U16(lhs $op rhs),
                (Int::U32(lhs), Int::U32(rhs)) => Int::U32(lhs $op rhs),
                (Int::U64(lhs), Int::U64(rhs)) => Int::U64(lhs $op rhs),
                (Int::I8(lhs), Int::I8(rhs)) => Int::I8(lhs $op rhs),
                (Int::I16(lhs), Int::I16(rhs)) => Int::I16(lhs $op rhs),
                (Int::I32(lhs), Int::I32(rhs)) => Int::I32(lhs $op rhs),
                (Int::I64(lhs), Int::I64(rhs)) => Int::I64(lhs $op rhs),
                (Int::Int(lhs), Int::Int(rhs)) => Int::Int(lhs $op rhs),
                _ => panic!("mismatched integer types"),
            }
        }
    };
}

impl std::ops::Add for Int {
    type Output = Int;
    int_op!(add, +);
}

impl std::ops::Sub for Int {
    type Output = Int;
    int_op!(sub, -);
}

impl std::ops::Mul for Int {
    type Output = Int;
    int_op!(mul, *);
}

impl std::ops::Div for Int {
    type Output = Int;
    int_op!(div, /);
}

impl std::ops::Rem for Int {
    type Output = Int;
    int_op!(rem, %);
}

impl std::ops::Neg for Int {
    type Output = Int;

    fn neg(self) -> Int {
        match self {
            Int::I8(n) => Int::I8(-n),
            Int::I16(n) => Int::I16(-n),
            Int::I32(n) => Int::I32(-n),
            Int::I64(n) => Int::I64(-n),
            Int::Int(n) => Int::Int(-n),
            _ => panic!("mismatched integer types"),
        }
    }
}

impl std::fmt::Display for Int {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Int::U8(n) => write!(f, "{}", n),
            Int::U16(n) => write!(f, "{}", n),
            Int::U32(n) => write!(f, "{}", n),
            Int::U64(n) => write!(f, "{}", n),
            Int::I8(n) => write!(f, "{}", n),
            Int::I16(n) => write!(f, "{}", n),
            Int::I32(n) => write!(f, "{}", n),
            Int::I64(n) => write!(f, "{}", n),
            Int::Int(n) => write!(f, "{}", n),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Value {
    Void,
    Int(Int),
    Bool(bool),
    Func(usize, Vec<Value>),
    List(Option<Box<List>>),
    Adt(usize, Vec<Value>),
    String(String),
    Ref(Rc<RefCell<Value>>),
    Dict(BTreeMap<Value, Value>),
    Array(Array),
    File(RiteFile),
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
            Value::Dict(dict) => {
                if dict.is_empty() {
                    return write!(f, "{{}}");
                }

                write!(f, "{{ ")?;

                for (i, (key, value)) in dict.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }

                    write!(f, "{}: {}", key, value)?;
                }

                write!(f, " }}")
            }
            Value::Array(arr) => write!(f, "{}", arr),
            Value::File(file) => write!(f, "File({})", file.id),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
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
fn string_bytes(mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::String(s) = args.pop().unwrap() else {
        panic!("expected string literal")
    };

    let bytes = s.as_bytes().iter().map(|&b| b).collect::<Vec<_>>();

    Value::Array(Array::U8(bytes))
}

fn string_from_bytes(mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::Array(Array::U8(bytes)) = args.pop().unwrap() else {
        panic!("expected u8 array")
    };

    let s = String::from_utf8(bytes).unwrap();

    Value::String(s)
}

fn string_length(mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::String(s) = args.pop().unwrap() else {
        panic!("expected string")
    };

    Value::Int(Int::Int(s.chars().count() as isize))
}

fn string_slice(mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 3);

    let Value::Int(Int::Int(end)) = args.pop().unwrap() else {
        panic!("expected integer")
    };

    let Value::Int(Int::Int(start)) = args.pop().unwrap() else {
        panic!("expected integer")
    };

    let Value::String(s) = args.pop().unwrap() else {
        panic!("expected string")
    };

    // Find char indices for the start and end of the range.
    let start = s
        .char_indices()
        .nth(start as usize)
        .map_or(s.len(), |(i, _)| i);
    let end = s
        .char_indices()
        .nth(end as usize)
        .map_or(s.len(), |(i, _)| i);

    // Return an empty string if the range is invalid.
    let s = s[start..end.max(start)].to_string();

    Value::String(s)
}

fn string_graphemes(mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::String(s) = args.pop().unwrap() else {
        panic!("expected string")
    };

    let chars = s.chars().map(|c| Value::String(String::from(c))).collect();
    Value::list_from_vec(chars)
}

fn string_split(mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 2);

    let Value::String(sep) = args.pop().unwrap() else {
        panic!("expected string")
    };

    let Value::String(s) = args.pop().unwrap() else {
        panic!("expected string")
    };

    let parts = s
        .split(&sep)
        .map(|s| Value::String(s.to_string()))
        .collect();

    Value::list_from_vec(parts)
}

fn string_concat(args: Vec<Value>) -> Value {
    let mut s = String::new();

    for arg in args {
        match arg {
            Value::String(s2) => s.push_str(&s2),
            _ => panic!("expected string"),
        }
    }

    Value::String(s)
}

fn dict_new(args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 0);
    Value::Dict(BTreeMap::new())
}

fn dict_length(mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::Dict(dict) = args.pop().unwrap() else {
        panic!("expected dictionary")
    };

    Value::Int(Int::Int(dict.len() as isize))
}

fn dict_get(mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 2);

    let key = args.pop().unwrap();

    let Value::Dict(dict) = args.pop().unwrap() else {
        panic!("expected dictionary")
    };

    match dict.get(&key) {
        Some(value) => Value::Adt(0, vec![value.clone()]),
        None => Value::Adt(1, vec![Value::Void]),
    }
}

fn dict_insert(mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 3);

    let value = args.pop().unwrap();
    let key = args.pop().unwrap();

    let Value::Dict(mut dict) = args.pop().unwrap() else {
        panic!("expected dictionary")
    };

    dict.insert(key, value);

    Value::Dict(dict)
}

fn dict_remove(mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 2);

    let key = args.pop().unwrap();

    let Value::Dict(mut dict) = args.pop().unwrap() else {
        panic!("expected dictionary")
    };

    dict.remove(&key);

    Value::Dict(dict)
}

fn dict_pairs(mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::Dict(dict) = args.pop().unwrap() else {
        panic!("expected dictionary")
    };

    let pairs = dict
        .iter()
        .map(|(key, value)| Value::Adt(0, vec![key.clone(), value.clone()]))
        .collect();

    Value::list_from_vec(pairs)
}

fn dict_keys(mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::Dict(dict) = args.pop().unwrap() else {
        panic!("expected dictionary")
    };

    let keys = dict.keys().cloned().collect();
    Value::list_from_vec(keys)
}

fn dict_values(mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::Dict(dict) = args.pop().unwrap() else {
        panic!("expected dictionary")
    };

    let values = dict.values().cloned().collect();
    Value::list_from_vec(values)
}

fn array_new(mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 2);

    let default = args.pop().unwrap();

    let Value::Int(Int::Int(len)) = args.pop().unwrap() else {
        panic!("expected integer")
    };

    let len = len as usize;

    Value::Array(Array::new(len, default))
}

fn array_empty(args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 0);
    Value::Array(Array::Void(Vec::new()))
}

fn array_extend(mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 3);

    let Value::Int(Int::Int(len2)) = args.pop().unwrap() else {
        panic!("expected integer")
    };

    let Value::Array(mut arr) = args.pop().unwrap() else {
        panic!("expected array")
    };

    let default = args.pop().unwrap();

    arr.resize(arr.len() + len2 as usize, default);

    Value::Array(arr)
}

fn array_truncate(mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 2);

    let Value::Int(Int::Int(len2)) = args.pop().unwrap() else {
        panic!("expected integer")
    };

    let Value::Array(mut arr) = args.pop().unwrap() else {
        panic!("expected array")
    };

    arr.truncate(arr.len() - len2 as usize);

    Value::Array(arr)
}

fn array_length(mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::Array(arr) = args.pop().unwrap() else {
        panic!("expected array")
    };

    Value::Int(Int::Int(arr.len() as isize))
}

fn array_get(mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 2);

    let Value::Int(Int::Int(index)) = args.pop().unwrap() else {
        panic!("expected integer")
    };

    let Value::Array(arr) = args.pop().unwrap() else {
        panic!("expected array")
    };

    match arr.get(index as usize) {
        Some(value) => Value::Adt(0, vec![value]),
        None => Value::Adt(1, vec![Value::Void]),
    }
}

fn array_set(mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 3);

    let value = args.pop().unwrap();

    let Value::Int(Int::Int(index)) = args.pop().unwrap() else {
        panic!("expected integer")
    };

    let Value::Array(mut arr) = args.pop().unwrap() else {
        panic!("expected array")
    };

    arr.set(index as usize, value);

    Value::Array(arr)
}

fn array_slice(mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 3);

    let Value::Int(Int::Int(end)) = args.pop().unwrap() else {
        panic!("expected integer")
    };

    let Value::Int(Int::Int(start)) = args.pop().unwrap() else {
        panic!("expected integer")
    };

    let Value::Array(arr) = args.pop().unwrap() else {
        panic!("expected array")
    };

    let start = start as usize;
    let end = end as usize;

    Value::Array(arr.slice(start, end))
}

fn rite_io_error(kind: io::ErrorKind) -> Value {
    let err = match kind {
        io::ErrorKind::NotFound => 0,
        io::ErrorKind::PermissionDenied => 1,
        _ => 2,
    };

    Value::Adt(1, vec![Value::Adt(err, Vec::new())])
}

fn rite_io_error_other() -> Value {
    Value::Adt(1, vec![Value::Adt(2, Vec::new())])
}

fn fs_open(mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::String(path) = args.pop().unwrap() else {
        panic!("expected string")
    };

    let mut opts = fs::OpenOptions::new();
    opts.read(true).write(true).create(true);

    match opts.open(&path) {
        Ok(file) => {
            let file = Value::File(RiteFile::new(file));
            Value::Adt(0, vec![file])
        }
        Err(err) => rite_io_error(err.kind()),
    }
}

fn fs_close(mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::File(file) = args.pop().unwrap() else {
        panic!("expected file")
    };

    file.file.lock().unwrap().take();

    Value::Adt(0, vec![Value::Void])
}

fn fs_read_all(mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::File(file) = args.pop().unwrap() else {
        panic!("expected file")
    };

    let mut file = file.file.lock().unwrap();

    let mut contents = Vec::new();

    match file.as_mut() {
        Some(file) => match file.read_to_end(&mut contents) {
            Ok(_) => {
                Value::Adt(0, vec![Value::Array(Array::U8(contents))])
            }

            Err(err) => rite_io_error(err.kind()),
        },
        None => rite_io_error_other(),
    }
}

fn fs_write_all(mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 2);

    let Value::Array(Array::U8(bytes)) = args.pop().unwrap() else {
        panic!("expected u8 array")
    };

    let Value::File(file) = args.pop().unwrap() else {
        panic!("expected file")
    };

    let mut file = file.file.lock().unwrap();

    match file.as_mut() {
        Some(file) => match file.write_all(&bytes) {
            Ok(_) => Value::Adt(0, vec![Value::Void]),
            Err(err) => rite_io_error(err.kind()),
        },
        None => rite_io_error_other(),
    }
}

fn io_print(mut args: Vec<Value>) -> Value {
    let Value::String(s) = args.pop().unwrap() else {
        panic!("expected string")
    };

    print!("{}", s);

    Value::Void
}

fn debug_format(mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let arg = args.pop().unwrap();

    // FIXME: this is a disgusting hack...
    if let Value::String(s) = arg {
        return Value::String(s);
    }

    Value::String(format!("{}", arg))
}

pub struct Interpreter<'a> {
    rir: &'a Unit<Specific>,
    builtins: HashMap<usize, fn(Vec<Value>) -> Value>,
}

impl<'a> Interpreter<'a> {
    pub fn new(mir: &'a Unit<Specific>) -> Self {
        Self {
            rir: mir,
            builtins: Self::find_builtins(mir),
        }
    }

    /// Find builtin functions in the MIR based on the `extern intrinsic` decorator.
    fn find_builtins(mir: &'a Unit<Specific>) -> HashMap<usize, fn(Vec<Value>) -> Value> {
        let mut builtins = HashMap::new();
        for (index, value) in mir.funcs.iter().enumerate() {
            // find extern intrinsic decorator.
            let decorator = value.decorators.iter().find(|d| d.name == "language");

            if let Some(extern_decorator) = decorator {
                let name = extern_decorator.args[0].as_str();
                let func = match name {
                    "string:bytes" => string_bytes,
                    "string:from_bytes" => string_from_bytes,
                    "string:concat" => string_concat,
                    "string:length" => string_length,
                    "string:slice" => string_slice,
                    "string:graphemes" => string_graphemes,
                    "string:split" => string_split,
                    "debug:format" => debug_format,
                    "io:print" => io_print,
                    "dict:new" => dict_new,
                    "dict:length" => dict_length,
                    "dict:get" => dict_get,
                    "dict:insert" => dict_insert,
                    "dict:remove" => dict_remove,
                    "dict:pairs" => dict_pairs,
                    "dict:keys" => dict_keys,
                    "dict:values" => dict_values,
                    "array:new" => array_new,
                    "array:empty" => array_empty,
                    "array:extend" => array_extend,
                    "array:truncate" => array_truncate,
                    "array:length" => array_length,
                    "array:get" => array_get,
                    "array:set" => array_set,
                    "array:slice" => array_slice,
                    "fs:open" => fs_open,
                    "fs:close" => fs_close,
                    "fs:read_all" => fs_read_all,
                    "fs:write_all" => fs_write_all,
                    // we allow intrinsics to have pure rite fallbacks.
                    _ => continue,
                };

                builtins.insert(index, func);
            }
        }

        builtins
    }

    pub fn interpret(&self, main: usize) -> Value {
        let args = env::args().map(Value::String).collect();
        let args = Value::list_from_vec(args);

        let mut frame = Frame {
            locals: vec![Value::Void; self.rir.funcs[main].locals.len()],
            arguments: vec![args],
            captured: Vec::new(),
        };

        self.interpret_block(&mut frame, &self.rir.funcs[main].body)
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
            rir::Value::Cast(_, operand) => self.interpret_operand(frame, operand),
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
            rir::Value::IsVariant(value, variant) => {
                let Value::Adt(current, _) = self.interpret_operand(frame, value) else {
                    panic!("expected adt")
                };

                Value::Bool(current == *variant)
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
                    locals: vec![Value::Void; self.rir.funcs[func].locals.len()],
                    arguments: args,
                    captured,
                };

                if let Some(builtin_func) = self.builtins.get(&func) {
                    return builtin_func(frame.arguments);
                }

                self.interpret_block(&mut frame, &self.rir.funcs[func].body)
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
            Constant::Int(negative, base, digits, kind) => {
                let mut n = 0;

                for &digit in digits.iter() {
                    n = n * base.radix() as i64 + digit as i64;
                }

                if *negative {
                    n = -n;
                }

                match kind {
                    IntKind::U8 => Value::Int(Int::U8(n as u8)),
                    IntKind::U16 => Value::Int(Int::U16(n as u16)),
                    IntKind::U32 => Value::Int(Int::U32(n as u32)),
                    IntKind::U64 => Value::Int(Int::U64(n as u64)),
                    IntKind::I8 => Value::Int(Int::I8(n as i8)),
                    IntKind::I16 => Value::Int(Int::I16(n as i16)),
                    IntKind::I32 => Value::Int(Int::I32(n as i32)),
                    IntKind::I64 => Value::Int(Int::I64(n as i64)),
                    IntKind::Int => Value::Int(Int::Int(n as isize)),
                }
            }
            Constant::String(s) => Value::String(s.to_string()),
        }
    }
}
