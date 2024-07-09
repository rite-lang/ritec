use ritec_diagnostic::{Diagnostic, Span};
use ritec_parse::{Delim, Token, TokenStream};

use crate::{parse_item, Item};

#[derive(Clone, Debug)]
pub struct VoidType {
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct BoolType {
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct IntType {
    pub signed: bool,
    pub size: Option<u16>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct FloatType {
    pub size: Option<u16>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct PointerType {
    pub mutable: bool,
    pub ty: Box<Type>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct ArrayType {
    pub ty: Box<Type>,
    pub size: u64,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct SliceType {
    pub ty: Box<Type>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct TupleType {
    pub tys: Vec<Type>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct FunctionType {
    pub args: Vec<Type>,
    pub ret: Box<Type>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum Type {
    Void(VoidType),
    Bool(BoolType),
    Int(IntType),
    Float(FloatType),
    Pointer(PointerType),
    Array(ArrayType),
    Slice(SliceType),
    Tuple(TupleType),
    Function(FunctionType),
    Item(Item),
}

impl Type {
    pub fn span(&self) -> Span {
        match self {
            Type::Void(ty) => ty.span,
            Type::Bool(ty) => ty.span,
            Type::Int(ty) => ty.span,
            Type::Float(ty) => ty.span,
            Type::Pointer(ty) => ty.span,
            Type::Array(ty) => ty.span,
            Type::Slice(ty) => ty.span,
            Type::Tuple(ty) => ty.span,
            Type::Function(ty) => ty.span,
            Type::Item(ty) => ty.span,
        }
    }
}

pub fn parse_void_type(stream: &mut TokenStream) -> Result<VoidType, Diagnostic> {
    let span = stream.expect(Token::Void)?;
    Ok(VoidType { span })
}

pub fn parse_bool_type(stream: &mut TokenStream) -> Result<BoolType, Diagnostic> {
    let span = stream.expect(Token::Bool)?;
    Ok(BoolType { span })
}

pub fn parse_int_type(stream: &mut TokenStream) -> Result<IntType, Diagnostic> {
    let (token, span) = stream.consume();

    let (signed, size) = match token {
        Token::U8 => (false, Some(8)),
        Token::U16 => (false, Some(16)),
        Token::U32 => (false, Some(32)),
        Token::U64 => (false, Some(64)),
        Token::U128 => (false, Some(128)),
        Token::Usize => (false, None),
        Token::I8 => (true, Some(8)),
        Token::I16 => (true, Some(16)),
        Token::I32 => (true, Some(32)),
        Token::I64 => (true, Some(64)),
        Token::I128 => (true, Some(128)),
        Token::Isize => (true, None),
        _ => {
            let message = format!("expected integer type, found {:?}", token);
            return Err(Diagnostic::new(message).with_span(span));
        }
    };

    Ok(IntType { signed, size, span })
}

pub fn parse_float_type(stream: &mut TokenStream) -> Result<FloatType, Diagnostic> {
    let (token, span) = stream.consume();

    let size = match token {
        Token::F32 => Some(32),
        Token::F64 => Some(64),
        _ => {
            let message = format!("expected float type, found {:?}", token);
            return Err(Diagnostic::new(message).with_span(span));
        }
    };

    Ok(FloatType { size, span })
}

pub fn parse_pointer_type(stream: &mut TokenStream) -> Result<PointerType, Diagnostic> {
    stream.expect(Token::Star)?;
    let mutable = stream.take(Token::Mut);

    let ty = Box::new(parse_type(stream)?);
    let span = ty.span().join(stream.peek().1);

    Ok(PointerType { mutable, ty, span })
}

pub fn parse_array_type(stream: &mut TokenStream) -> Result<Type, Diagnostic> {
    let start = stream.expect(Token::Bracket(Delim::Open))?;

    let ty = Box::new(parse_type(stream)?);

    if stream.take(Token::Semi) {
        let size = stream.expect_integer()?;
        let end = stream.expect(Token::Bracket(Delim::Close))?;

        Ok(Type::Array(ArrayType {
            ty,
            size,
            span: start.join(end),
        }))
    } else {
        let end = stream.expect(Token::Bracket(Delim::Close))?;

        Ok(Type::Slice(SliceType {
            ty,
            span: start.join(end),
        }))
    }
}

pub fn parse_tuple_type(stream: &mut TokenStream) -> Result<TupleType, Diagnostic> {
    let start = stream.expect(Token::Paren(Delim::Open))?;

    let mut tys = Vec::new();

    loop {
        if stream.is(Token::Paren(Delim::Close)) {
            break;
        }

        tys.push(parse_type(stream)?);

        if !stream.take(Token::Comma) {
            break;
        }
    }

    let end = stream.expect(Token::Paren(Delim::Close))?;

    Ok(TupleType {
        tys,
        span: start.join(end),
    })
}

pub fn parse_function_type(stream: &mut TokenStream) -> Result<FunctionType, Diagnostic> {
    let start = stream.expect(Token::Paren(Delim::Open))?;

    let mut args = Vec::new();

    loop {
        if stream.is(Token::Paren(Delim::Close)) {
            break;
        }

        args.push(parse_type(stream)?);

        if !stream.take(Token::Comma) {
            break;
        }
    }

    let end = stream.expect(Token::Paren(Delim::Close))?;

    stream.expect(Token::Arrow)?;

    let ret = Box::new(parse_type(stream)?);

    Ok(FunctionType {
        args,
        ret,
        span: start.join(end),
    })
}

pub fn parse_type(stream: &mut TokenStream) -> Result<Type, Diagnostic> {
    let (token, _) = stream.peek();

    match token {
        Token::Void => Ok(Type::Void(parse_void_type(stream)?)),
        Token::Bool => Ok(Type::Bool(parse_bool_type(stream)?)),
        Token::U8
        | Token::U16
        | Token::U32
        | Token::U64
        | Token::U128
        | Token::Usize
        | Token::I8
        | Token::I16
        | Token::I32
        | Token::I64
        | Token::I128
        | Token::Isize => Ok(Type::Int(parse_int_type(stream)?)),
        Token::F32 | Token::F64 => Ok(Type::Float(parse_float_type(stream)?)),
        Token::Star => Ok(Type::Pointer(parse_pointer_type(stream)?)),
        Token::Bracket(Delim::Open) => parse_array_type(stream),
        Token::Paren(Delim::Open) => Ok(Type::Tuple(parse_tuple_type(stream)?)),
        Token::Fn => Ok(Type::Function(parse_function_type(stream)?)),
        Token::Ident(_) | Token::Quote => Ok(Type::Item(parse_item(stream)?)),
        _ => {
            let message = format!("expected type, found {:?}", token);
            let span = stream.peek().1;
            Err(Diagnostic::new(message).with_span(span))
        }
    }
}
