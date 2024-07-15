#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    Bool,

    Int {
        signed: bool,
        width: Option<u16>,
    },

    Float {
        width: u16,
    },

    Pointer {
        mutable: bool,
        pointee: Box<Type>,
    },

    Array {
        element: Box<Type>,
        length: u64,
    },

    Function {
        arguments: Vec<Type>,
        output: Box<Type>,
    },

    Struct {
        fields: Vec<Type>,
    },

    Union {
        variants: Vec<Type>,
    },
}

impl Type {
    pub const VOID: Type = Type::Struct { fields: Vec::new() };

    pub fn deref(&self) -> Option<&Type> {
        match self {
            Type::Pointer { pointee, .. } => Some(pointee),
            _ => None,
        }
    }
}
