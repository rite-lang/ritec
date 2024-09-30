use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt::Formatter;
use std::rc::Rc;
use std::sync::atomic::AtomicUsize;
use std::sync::{atomic, Arc, Mutex};
use std::{cmp, fs};

use smallvec::{smallvec, SmallVec};

#[derive(Clone, Debug)]
pub struct RiteFile {
    id: usize,
    pub(crate) file: Arc<Mutex<Option<fs::File>>>,
}

impl RiteFile {
    pub(crate) fn new(file: fs::File) -> Self {
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
    (Func) => { (usize, Rc<SmallVec<[Value; 4]>>) };
    (List) => { Option<Rc<List>> };
    (Adt) => { (usize, Rc<SmallVec<[Value; 4]>>) };
    (String) => { String };
    (Ref) => { Rc<RefCell<Value>> };
    (Dict) => { Rc<BTreeMap<Value, Value>> };
    (Array) => { Rc<Array> };
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

#[macro_export]
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

#[macro_export]
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
        Value::Adt((0, Rc::new(smallvec![value])))
    }

    pub fn err(value: Value) -> Value {
        Value::Adt((1, Rc::new(smallvec![value])))
    }

    pub fn none() -> Value {
        Value::Adt((1, Rc::new(smallvec![Value::void()])))
    }

    pub fn tuple(values: SmallVec<[Value; 4]>) -> Value {
        Value::Adt((0, Rc::new(values)))
    }

    pub fn list_from_vec(values: Vec<Value>) -> Value {
        let mut list = None;

        for value in values.into_iter().rev() {
            list = Some(Rc::new(List {
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
    pub(crate) head: Value,
    pub(crate) tail: Option<Rc<List>>,
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
