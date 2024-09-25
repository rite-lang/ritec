#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IntKind {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    Int,
}

impl IntKind {
    pub fn is_signed(self) -> bool {
        matches!(
            self,
            IntKind::I8 | IntKind::I16 | IntKind::I32 | IntKind::I64 | IntKind::Int
        )
    }
}

impl std::fmt::Display for IntKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IntKind::U8 => write!(f, "u8"),
            IntKind::U16 => write!(f, "u16"),
            IntKind::U32 => write!(f, "u32"),
            IntKind::U64 => write!(f, "u64"),
            IntKind::I8 => write!(f, "i8"),
            IntKind::I16 => write!(f, "i16"),
            IntKind::I32 => write!(f, "i32"),
            IntKind::I64 => write!(f, "i64"),
            IntKind::Int => write!(f, "int"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Base {
    Bin,
    Oct,
    Dec,
    Hex,
}

impl Base {
    pub const fn radix(self) -> u32 {
        match self {
            Base::Bin => 2,
            Base::Oct => 8,
            Base::Dec => 10,
            Base::Hex => 16,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FloatKind {
    F32,
    F64,
}

impl std::fmt::Display for FloatKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FloatKind::F32 => write!(f, "f32"),
            FloatKind::F64 => write!(f, "f64"),
        }
    }
}
