use crate::number::IntKind;
use crate::rir::Func;
use crate::{
    ast::BinOp,
    hir::UnOp,
    rir::{
        self, Block, Constant, Location, Operand, Place, Projection, ProjectionKind, Specific,
        Statement, Unit,
    },
};
use std::fmt::Formatter;
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

macro_rules! type_map {
    (U8) => { u8 };
    (U16) => { u16 };
    (U32) => { u32 };
    (U64) => { u64 };
    (I8) => { i8 };
    (I16) => { i16 };
    (I32) => { i32 };
    (I64) => { i64 };
    (Int) => { isize };
    (Void) => { () };
    (Bool) => { bool };
    (Func) => { (usize, Vec<Value>) };
    (List) => { Option<Box<List>> };
    (Adt) => { (usize, Vec<Value>) };
    (String) => { String };
    (Ref) => { Rc<RefCell<Value>> };
    (Dict) => { BTreeMap<Value, Value> };
    (Array) => { Array };
    (File) => { RiteFile };
}

macro_rules! variant_map {
  ($macro_name:ident, $($args:tt)*) => {
        $macro_name! {
            $($args)*, // Pass along any extra arguments like $value, $a, $block, etc.
            Void,
            Bool,
            Func,
            List,
            Adt,
            String,
            Ref,
            Dict,
            Array,
            File,
            U8,
            U16,
            U32,
            U64,
            I8,
            I16,
            I32,
            I64,
            Int
        }
    };
}

macro_rules! array_enum {
    ($extra:tt, $($variant:ident),*) => {
        #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
        pub enum Array {
            $(
                $variant(Vec<type_map!($variant)>),
            )*
        }
    };
}

macro_rules! value_enum {
    ($extra:tt, $($variant:ident),*) => {
        #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
        pub enum Value {
            $(
                $variant(type_map!($variant)),
            )*
        }
    };
}

// Generate the `Array` and `Value` enums.
variant_map!(value_enum, extra);
variant_map!(array_enum, extra);

/// Match variant with shared block.
macro_rules! match_value_block {
    ($value:expr, $ty:ident, $a:ident, $block:block, $($variant:ident),*) => {
        {
            let value = $value;
            match value {
                $(
                    $ty::$variant($a) => $block
                )*
                #[allow(unreachable_patterns)]
                _ => panic!("[1] Unexpected variant {:?} when matching {}.", value, stringify!($ty))
            }
        }
    };

    ($value:expr, $ty:ident, $a:ident, $block:block) => {
       variant_map!(match_value_block, $value, $ty, $a, $block)
    };
}

/// Convert a value/array from one to another.
macro_rules! match_value_convert {
    ($value:expr, $x:ident, $y:ident, $a:ident, $block:block, $($variant:ident),*) => {
        {
            let value = $value;
            match value {
                $(
                    $x::$variant($a) => $y::$variant($block),
                )*
                #[allow(unreachable_patterns)]
                _ => panic!("[2] Unexpected variant {:?} when converting {} to {}.", value, stringify!($x), stringify!($y))
            }
        }
    };

    ($value:expr, $x:ident, $y:ident, $a:ident, $block:block) => {
        variant_map!(match_value_convert, $value, $x, $y, $a, $block)
    };
}

/// Match two values of different types with the same variant.
macro_rules! match_two_value_block {
    ($value:expr, $x:ident, $y:ident, $a:ident, $b:ident, $block:block, $($variant:ident),*) => {
        {
            let value = $value;
            match value {
                $(
                    ($x::$variant($a), $y::$variant($b)) => $block
                )*
                #[allow(unreachable_patterns)]
                _ => panic!("[3] Unexpected variant {:?} when matching for ({}, {})", value, stringify!($x), stringify!($y))
            }
        }
    };

     ($value:expr, $x:ident, $y:ident, $a:ident, $b:ident, $block:block) => {
        variant_map!(match_two_value_block, $value, $x, $y, $a, $b, $block)
    };
}

/// Operate on either on ('a, 'a) -> 'a or ('a, 'a) -> fixed.
macro_rules! match_value_binary_op {
    // Match two identical variants of the same type and produce the same variant
    ($value:expr, $ty:ident, $a:ident, $b:ident, $block:block, $($variant:ident),*) => {
        {
            let value = $value;
            match value {
                $(
                   ($ty::$variant($a), $ty::$variant($b)) => $ty::$variant($block),
                )*
                #[allow(unreachable_patterns)]
                _ => panic!("[4] Unexpected variant {:?} when matching for paired {}.", value, stringify!($ty))
            }
        }
    };

    ($value:expr, $ty:ident, $a:ident, $b:ident, $block:block) => {
        match_value_binary_op!(
            $value,
            $ty,
            $a,
            $b,
            $block,
            U8,
            U16,
            U32,
            U64,
            I8,
            I16,
            I32,
            I64,
            Int
        )
    };

    // Binary operations that produce a shared different variant.
    ($value:expr, $ty:ident, $out:ident, $a:ident, $b:ident, $block:block, $($variant:ident),*) => {
        {
        let value = $value;
            match value {
                $(
                   ($ty::$variant($a), $ty::$variant($b)) => $ty::$out($block),
                )*
                #[allow(unreachable_patterns)]
                _ => panic!("[5] Unexpected variant {:?} when matching for paired {}.", value, stringify!($ty))
            }
        }
    };

    ($value:expr, $ty:ident, $out:ident, $a:ident, $b:ident, $block:block) => {
        match_value_binary_op!(
            $value,
            $ty,
            $out,
            $a,
            $b,
            $block,
            U8,
            U16,
            U32,
            U64,
            I8,
            I16,
            I32,
            I64,
            Int
        )
    };
}

impl Value {
    pub fn void() -> Value {
        Value::Void(())
    }

    pub fn ok(value: Value) -> Value {
        Value::Adt((0, vec![value]))
    }

    pub fn err(value: Value) -> Value {
        Value::Adt((1, vec![value]))
    }

    pub fn none() -> Value {
        Value::Adt((1, vec![Value::void()]))
    }

    pub fn tuple(values: Vec<Value>) -> Value {
        Value::Adt((0, values))
    }

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
            Value::Void(_) => write!(f, "void"),
            Value::String(s) => write!(f, "\"{}\"", s),
            Value::Func((func, captured)) => write!(
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
            Value::Adt((variant, fields)) => {
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
            Value::File(file) => write!(f, "File({})", file.id),
            Value::Bool(v) => write!(f, "{}", v),
            Value::Array(v) => write!(f, "{}", v),
            Value::U8(v) => write!(f, "{}", v),
            Value::U16(v) => write!(f, "{}", v),
            Value::U32(v) => write!(f, "{}", v),
            Value::U64(v) => write!(f, "{}", v),
            Value::I8(v) => write!(f, "{}", v),
            Value::I16(v) => write!(f, "{}", v),
            Value::I32(v) => write!(f, "{}", v),
            Value::I64(v) => write!(f, "{}", v),
            Value::Int(v) => write!(f, "{}", v),
        }
    }
}

impl Array {
    pub fn new(len: usize, default: Value) -> Array {
        match_value_convert!(default, Value, Array, default, { vec![default; len] })
    }

    pub fn len(&self) -> usize {
        match_value_block!(self, Self, s, { s.len() })
    }

    pub fn slice(&self, start: usize, end: usize) -> Array {
        match_value_convert!(self, Array, Array, s, { s[start..end].to_vec() })
    }

    pub fn resize(&mut self, len: usize, default: Value) {
        match_two_value_block!((self, default), Array, Value, vec, value, {
            vec.resize(len, value);
        })
    }

    pub fn truncate(&mut self, len: usize) {
        match_value_block!(self, Self, s, {
            s.truncate(len);
        })
    }

    pub fn set(&mut self, index: usize, value: Value) {
        match_two_value_block!((self, value), Array, Value, vec, value, {
            vec[index] = value;
        })
    }

    pub fn get(&self, index: usize) -> Option<Value> {
        if index >= self.len() {
            return None;
        }

        match self {
            Array::Void(_) => return Some(Value::void()),
            _ => {}
        }

        Some(match_value_convert!(
            self,
            Array,
            Value,
            s,
            { s[index].clone() },
            // Exclude void.
            Bool,
            Func,
            List,
            Adt,
            String,
            Ref,
            Dict,
            Array,
            File,
            U8,
            U16,
            U32,
            U64,
            I8,
            I16,
            I32,
            I64,
            Int
        ))
    }
}

impl std::fmt::Display for Array {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match_value_block!(self, Self, s, {
            write!(f, "[")?;

            for (i, value) in s.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }

                write!(f, "{:?}", value)?;
            }

            write!(f, "]")
        })
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
fn string_bytes(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::String(s) = args.pop().unwrap() else {
        panic!("expected string literal")
    };

    let bytes = s.as_bytes().iter().map(|&b| b).collect::<Vec<_>>();

    Value::Array(Array::U8(bytes))
}

fn string_from_bytes(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::Array(Array::U8(bytes)) = args.pop().unwrap() else {
        panic!("expected u8 array")
    };

    let s = String::from_utf8(bytes).unwrap();

    Value::String(s)
}

fn string_length(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::String(s) = args.pop().unwrap() else {
        panic!("expected string")
    };

    Value::Int(s.chars().count() as isize)
}

fn string_slice(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 3);

    let Value::Int(end) = args.pop().unwrap() else {
        panic!("expected integer")
    };

    let Value::Int(start) = args.pop().unwrap() else {
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

fn string_graphemes(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::String(s) = args.pop().unwrap() else {
        panic!("expected string")
    };

    let chars = s.chars().map(|c| Value::String(String::from(c))).collect();
    Value::list_from_vec(chars)
}

fn string_split(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
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

fn string_concat(_: &Func<Specific>, args: Vec<Value>) -> Value {
    let mut s = String::new();

    for arg in args {
        match arg {
            Value::String(s2) => s.push_str(&s2),
            _ => panic!("expected string"),
        }
    }

    Value::String(s)
}

fn dict_new(_: &Func<Specific>, args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 0);
    Value::Dict(BTreeMap::new())
}

fn dict_length(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::Dict(dict) = args.pop().unwrap() else {
        panic!("expected dictionary")
    };

    Value::Int(dict.len() as isize)
}

fn dict_get(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 2);

    let key = args.pop().unwrap();

    let Value::Dict(dict) = args.pop().unwrap() else {
        panic!("expected dictionary")
    };

    match dict.get(&key) {
        Some(value) => Value::ok(value.clone()),
        None => Value::none(),
    }
}

fn dict_insert(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 3);

    let value = args.pop().unwrap();
    let key = args.pop().unwrap();

    let Value::Dict(mut dict) = args.pop().unwrap() else {
        panic!("expected dictionary")
    };

    dict.insert(key, value);

    Value::Dict(dict)
}

fn dict_remove(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 2);

    let key = args.pop().unwrap();

    let Value::Dict(mut dict) = args.pop().unwrap() else {
        panic!("expected dictionary")
    };

    dict.remove(&key);

    Value::Dict(dict)
}

fn dict_pairs(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::Dict(dict) = args.pop().unwrap() else {
        panic!("expected dictionary")
    };

    let pairs = dict
        .iter()
        .map(|(key, value)| Value::Adt((0, vec![key.clone(), value.clone()])))
        .collect();

    Value::list_from_vec(pairs)
}

fn dict_keys(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::Dict(dict) = args.pop().unwrap() else {
        panic!("expected dictionary")
    };

    let keys = dict.keys().cloned().collect();
    Value::list_from_vec(keys)
}

fn dict_values(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::Dict(dict) = args.pop().unwrap() else {
        panic!("expected dictionary")
    };

    let values = dict.values().cloned().collect();
    Value::list_from_vec(values)
}

fn array_new(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 2);

    let default = args.pop().unwrap();

    let Value::Int(len) = args.pop().unwrap() else {
        panic!("expected integer")
    };

    let len = len as usize;

    Value::Array(Array::new(len, default))
}

fn array_empty(signature: &Func<Specific>, args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 0);

    println!("{:?}", signature.output);

    let Specific::Adt(_, generics) = &signature.output else {
        panic!("expected array type");
    };

    assert_eq!(generics.len(), 1);

    let arr = match &generics[0] {
        Specific::Void => Array::Void(Vec::new()),
        Specific::Bool => Array::Bool(Vec::new()),
        Specific::Str => Array::String(Vec::new()),
        Specific::Ref(_) => Array::Ref(Vec::new()),
        Specific::Int(k) => match k {
            IntKind::U8 => Array::U8(Vec::new()),
            IntKind::U16 => Array::U16(Vec::new()),
            IntKind::U32 => Array::U32(Vec::new()),
            IntKind::U64 => Array::U64(Vec::new()),
            IntKind::I8 => Array::I8(Vec::new()),
            IntKind::I16 => Array::I16(Vec::new()),
            IntKind::I32 => Array::I32(Vec::new()),
            IntKind::I64 => Array::I64(Vec::new()),
            IntKind::Int => Array::Int(Vec::new()),
        }
        Specific::List(_) => Array::List(Vec::new()),
        Specific::Tuple(_) => Array::Adt(Vec::new()),
        Specific::Func(_, _) => Array::Func(Vec::new()),
        Specific::Adt(_, _) => Array::Adt(Vec::new()),
        Specific::Float(_) => panic!("floats not supported"),
    };

    Value::Array(arr)
}

fn array_extend(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 3);

    let default = args.pop().unwrap();

    let Value::Int(len2) = args.pop().unwrap() else {
        panic!("expected integer")
    };

    let Value::Array(mut arr) = args.pop().unwrap() else {
        panic!("expected array")
    };

    arr.resize(arr.len() + len2 as usize, default);

    Value::Array(arr)
}

fn array_truncate(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 2);

    let Value::Int(len2) = args.pop().unwrap() else {
        panic!("expected integer")
    };

    let Value::Array(mut arr) = args.pop().unwrap() else {
        panic!("expected array")
    };

    arr.truncate(arr.len() - len2 as usize);

    Value::Array(arr)
}

fn array_length(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::Array(arr) = args.pop().unwrap() else {
        panic!("expected array")
    };

    Value::Int(arr.len() as isize)
}

fn array_get(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 2);

    let Value::Int(index) = args.pop().unwrap() else {
        panic!("expected integer")
    };

    let Value::Array(arr) = args.pop().unwrap() else {
        panic!("expected array")
    };

    match arr.get(index as usize) {
        Some(value) => Value::ok(value),
        None => Value::none(),
    }
}

fn array_set(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 3);

    let value = args.pop().unwrap();

    let Value::Int(index) = args.pop().unwrap() else {
        panic!("expected integer")
    };

    let Value::Array(mut arr) = args.pop().unwrap() else {
        panic!("expected array")
    };

    arr.set(index as usize, value);

    Value::Array(arr)
}

fn array_slice(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 3);

    let Value::Int(end) = args.pop().unwrap() else {
        panic!("expected integer")
    };

    let Value::Int(start) = args.pop().unwrap() else {
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

    Value::err(Value::Adt((err, Vec::new())))
}

fn rite_io_error_other() -> Value {
    Value::err(Value::Adt((2, Vec::new())))
}

fn fs_open(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::String(path) = args.pop().unwrap() else {
        panic!("expected string")
    };

    let mut opts = fs::OpenOptions::new();
    opts.read(true).write(true).create(true);

    match opts.open(&path) {
        Ok(file) => {
            let file = Value::File(RiteFile::new(file));
            Value::ok(file)
        }
        Err(err) => rite_io_error(err.kind()),
    }
}

fn fs_close(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::File(file) = args.pop().unwrap() else {
        panic!("expected file")
    };

    file.file.lock().unwrap().take();

    Value::ok(Value::void())
}

fn fs_read_all(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::File(file) = args.pop().unwrap() else {
        panic!("expected file")
    };

    let mut file = file.file.lock().unwrap();

    let mut contents = Vec::new();

    match file.as_mut() {
        Some(file) => match file.read_to_end(&mut contents) {
            Ok(_) => Value::ok(Value::Array(Array::U8(contents))),

            Err(err) => rite_io_error(err.kind()),
        },
        None => rite_io_error_other(),
    }
}

fn fs_write_all(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
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
            Ok(_) => Value::ok(Value::void()),
            Err(err) => rite_io_error(err.kind()),
        },
        None => rite_io_error_other(),
    }
}

fn fs_list_dir(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::String(path) = args.pop().unwrap() else {
        panic!("expected string")
    };

    match fs::read_dir(&path) {
        Ok(entries) => {
            // Return list of Adt(file = 0, dir = 1, [path]).
            let entries = entries
                .map(|entry| {
                    let entry = entry.unwrap();
                    let path = entry.path().to_string_lossy().to_string();

                    let variant = if entry.file_type().unwrap().is_dir() {
                        1
                    } else {
                        0
                    };

                    Value::Adt((variant, vec![Value::String(path)]))
                })
                .collect();

            Value::ok(Value::list_from_vec(entries))
        }
        Err(err) => rite_io_error(err.kind()),
    }
}

fn io_print(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    let Value::String(s) = args.pop().unwrap() else {
        panic!("expected string")
    };

    print!("{}", s);

    Value::void()
}

fn debug_format(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
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
    stack: Vec<Frame>,
    builtins: HashMap<usize, fn(&Func<Specific>, Vec<Value>) -> Value>,
}

impl<'a> Interpreter<'a> {
    pub fn new(rir: &'a Unit<Specific>) -> Self {
        Self {
            rir,
            stack: Vec::new(),
            builtins: Self::find_builtins(rir),
        }
    }

    /// Find builtin functions in the IR based on the `extern intrinsic` decorator.
    fn find_builtins(
        rir: &'a Unit<Specific>,
    ) -> HashMap<usize, fn(&Func<Specific>, Vec<Value>) -> Value> {
        let mut builtins = HashMap::new();
        for (index, value) in rir.funcs.iter().enumerate() {
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
                    "fs:list_dir" => fs_list_dir,
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
            locals: vec![Value::void(); self.rir.funcs[main].locals.len()],
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
                        None => Some(Value::void()),
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
                    let Value::Adt((variant, _)) = self.interpret_operand(frame, input) else {
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

                Value::Func((*index, captures))
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
            rir::Value::Call(func, args) => {
                let Value::Func((func, captured)) = self.interpret_operand(frame, func) else {
                    panic!("expected function")
                };

                let args = args
                    .iter()
                    .map(|op| self.interpret_operand(frame, op))
                    .collect();

                let mut frame = Frame {
                    locals: vec![Value::void(); self.rir.funcs[func].locals.len()],
                    arguments: args,
                    captured,
                };

                if let Some(builtin_func) = self.builtins.get(&func) {
                    return builtin_func(&self.rir.funcs[func], frame.arguments);
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

                Value::Adt((0, items))
            }
            rir::Value::Adt(variant, items) => {
                let items = items
                    .iter()
                    .map(|op| self.interpret_operand(frame, op))
                    .collect();

                Value::Adt((*variant, items))
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
                        Value::Adt((_, fields)) => recurse(&mut fields[field], value, projection),
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
