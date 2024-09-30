use crate::interpret::value::{Array, RiteFile, Value};
use crate::number::IntKind;
use crate::rir::{Func, Specific, Unit};
use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Write};
use std::rc::Rc;
use std::{fs, io};

pub type Intrinsic = fn(&Func<Specific>, Vec<Value>) -> Value;

#[repr(transparent)]
pub struct IntrinsicMap {
    pub(crate) map: HashMap<usize, Intrinsic>,
}

impl IntrinsicMap {
    pub fn new(rir: &Unit<Specific>) -> IntrinsicMap {
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
                    "string:is_whitespace" => string_is_whitespace,
                    "string:is_alphabetic" => string_is_alphabetic,
                    "string:is_numeric" => string_is_numeric,
                    "string:is_lowercase" => string_is_lowercase,
                    "string:is_uppercase" => string_is_uppercase,
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

        IntrinsicMap { map: builtins }
    }
}

// Standard library bindings.
fn string_bytes(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::String(s) = args.pop().unwrap() else {
        panic!("expected string literal")
    };

    let bytes = s.as_bytes().iter().map(|&b| b).collect::<Vec<_>>();

    Value::Array(Rc::new(Array::U8(bytes)))
}

fn string_from_bytes(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::Array(arr) = args.pop().unwrap() else {
        panic!("expected array")
    };

    let Array::U8(bytes) = &*arr else {
        panic!("expected u8 array got {:?}", arr)
    };

    let s = String::from_utf8(bytes.clone()).unwrap();

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

fn string_is_whitespace(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::String(s) = args.pop().unwrap() else {
        panic!("expected string")
    };

    let is_whitespace = s.chars().all(char::is_whitespace);

    Value::Bool(is_whitespace)
}

fn string_is_alphabetic(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::String(s) = args.pop().unwrap() else {
        panic!("expected string")
    };

    let is_alphabetic = s.chars().all(char::is_alphabetic);

    Value::Bool(is_alphabetic)
}

fn string_is_numeric(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::String(s) = args.pop().unwrap() else {
        panic!("expected string")
    };

    let is_numeric = s.chars().all(char::is_numeric);

    Value::Bool(is_numeric)
}

fn string_is_lowercase(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::String(s) = args.pop().unwrap() else {
        panic!("expected string")
    };

    let is_lowercase = s.chars().all(char::is_lowercase);

    Value::Bool(is_lowercase)
}

fn string_is_uppercase(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 1);

    let Value::String(s) = args.pop().unwrap() else {
        panic!("expected string")
    };

    let is_uppercase = s.chars().all(char::is_uppercase);

    Value::Bool(is_uppercase)
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
    Value::Dict(Rc::new(BTreeMap::new()))
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

    Rc::make_mut(&mut dict).insert(key, value);

    Value::Dict(dict)
}

fn dict_remove(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 2);

    let key = args.pop().unwrap();

    let Value::Dict(mut dict) = args.pop().unwrap() else {
        panic!("expected dictionary")
    };

    Rc::make_mut(&mut dict).remove(&key);

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

    Value::Array(Rc::new(Array::new(len, default)))
}

fn array_empty(signature: &Func<Specific>, args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 0);

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
        },
        Specific::List(_) => Array::List(Vec::new()),
        Specific::Tuple(_) => Array::Adt(Vec::new()),
        Specific::Func(_, _) => Array::Func(Vec::new()),
        Specific::Adt(_, _) => Array::Adt(Vec::new()),
        Specific::Float(_) => panic!("floats not supported"),
    };

    Value::Array(Rc::new(arr))
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

    let len = arr.len() + len2 as usize;

    Rc::make_mut(&mut arr).resize(len, default);

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

    Rc::make_mut(&mut arr).truncate(len2 as usize);

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

    Rc::make_mut(&mut arr).set(index as usize, value);

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

    Value::Array(Rc::new(arr.slice(start, end)))
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
            Ok(_) => Value::ok(Value::Array(Rc::new(Array::U8(contents)))),

            Err(err) => rite_io_error(err.kind()),
        },
        None => rite_io_error_other(),
    }
}

fn fs_write_all(_: &Func<Specific>, mut args: Vec<Value>) -> Value {
    assert_eq!(args.len(), 2);

    let Value::Array(arr) = args.pop().unwrap() else {
        panic!("expected array")
    };

    let Array::U8(bytes) = &*arr else {
        panic!("expected u8 array")
    };

    let Value::File(file) = args.pop().unwrap() else {
        panic!("expected file")
    };

    let mut file = file.file.lock().unwrap();

    match file.as_mut() {
        Some(file) => match file.write_all(bytes) {
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
