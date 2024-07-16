use ritec_diagnostic::{Diagnostic, Span};
use ritec_parse::{Delim, Token, TokenStream};

use crate::{
    parse_block_expr, parse_contract, parse_expr, parse_generic, parse_trait_bound, parse_type,
    Contract, Expr, Generic, Path, TraitBound, Type, VoidExpr,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vis {
    Public,
    Private,
}

#[derive(Clone, Debug)]
pub struct Variant {
    pub name: String,
    pub fields: Vec<Type>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Enum {
    pub vis: Vis,
    pub name: String,
    pub generics: Vec<Generic>,
    pub contract: Contract,
    pub variants: Vec<Variant>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Field {
    pub name: String,
    pub type_: Type,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Struct {
    pub vis: Vis,
    pub name: String,
    pub generics: Vec<Generic>,
    pub contract: Contract,
    pub fields: Vec<Field>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Argument {
    pub mutable: bool,
    pub name: String,
    pub type_: Type,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Function {
    pub vis: Vis,
    pub name: String,
    pub generics: Vec<Generic>,
    pub arguments: Vec<Argument>,
    pub output: Option<Type>,
    pub contract: Contract,
    pub body: Expr,
    pub span: Span,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelfArgument {
    Value,
    Ref,
    MutRef,
}

#[derive(Clone, Debug)]
pub struct Method {
    pub vis: Vis,
    pub name: String,
    pub generics: Vec<Generic>,
    pub self_argument: Option<SelfArgument>,
    pub arguments: Vec<Argument>,
    pub output: Option<Type>,
    pub contract: Contract,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Assoc {
    pub name: String,
    pub span: Span,
    pub bounds: Vec<TraitBound>,
}

#[derive(Clone, Debug)]
pub struct Trait {
    pub vis: Vis,
    pub name: String,
    pub generics: Vec<Generic>,
    pub contract: Contract,
    pub types: Vec<Assoc>,
    pub methods: Vec<Method>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct MethodImpl {
    pub vis: Vis,
    pub name: String,
    pub generics: Vec<Generic>,
    pub self_argument: Option<SelfArgument>,
    pub arguments: Vec<Argument>,
    pub output: Option<Type>,
    pub contract: Contract,
    pub body: Expr,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct AssocImpl {
    pub name: String,
    pub type_: Type,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct TraitImpl {
    pub trait_: Path,
    pub implementor: Type,
    pub contract: Contract,
    pub types: Vec<AssocImpl>,
    pub methods: Vec<MethodImpl>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct Impl {
    pub implementor: Type,
    pub contract: Contract,
    pub methods: Vec<MethodImpl>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct ModuleDecl {
    pub name: String,
    pub module: Option<Module>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum Decl {
    Enum(Enum),
    Struct(Struct),
    Function(Function),
    Trait(Trait),
    TraitImpl(TraitImpl),
    Impl(Impl),
    Module(ModuleDecl),
}

#[derive(Clone, Debug)]
pub struct Module {
    pub decls: Vec<Decl>,
}

pub fn parse_visibility(stream: &mut TokenStream) -> Result<Vis, Diagnostic> {
    let (token, _) = stream.peek();

    match token {
        Token::Pub => {
            stream.consume();
            Ok(Vis::Public)
        }
        _ => Ok(Vis::Private),
    }
}

pub fn parse_generics(stream: &mut TokenStream) -> Result<Vec<Generic>, Diagnostic> {
    let mut generics = Vec::new();

    if stream.take(Token::Lt) {
        loop {
            let generic = parse_generic(stream)?;
            generics.push(generic);

            if !stream.take(Token::Comma) {
                break;
            }
        }

        stream.expect(Token::Gt)?;
    }

    Ok(generics)
}

pub fn parse_variant(stream: &mut TokenStream) -> Result<Variant, Diagnostic> {
    let (name, mut span) = stream.expect_ident_spanned()?;

    let mut fields = Vec::new();

    if stream.take(Token::Paren(Delim::Open)) {
        loop {
            if stream.is(Token::Paren(Delim::Close)) {
                break;
            }

            let field = parse_type(stream)?;
            fields.push(field);

            if !stream.take(Token::Comma) {
                break;
            }
        }

        let end = stream.expect(Token::Paren(Delim::Close))?;
        span = span.join(end);
    }

    Ok(Variant { name, fields, span })
}

pub fn parse_enum_decl(stream: &mut TokenStream) -> Result<Enum, Diagnostic> {
    let vis = parse_visibility(stream)?;

    stream.expect(Token::Enum)?;

    let (name, span) = stream.expect_ident_spanned()?;
    let generics = parse_generics(stream)?;

    let clause = parse_contract(stream)?;

    let mut variants = Vec::new();

    if stream.take(Token::Indent) {
        while !stream.is(Token::Dedent) {
            if stream.is(Token::Newline) {
                stream.consume();
                continue;
            }

            variants.push(parse_variant(stream)?);
        }

        stream.expect(Token::Dedent)?;
    }

    Ok(Enum {
        vis,
        name,
        generics,
        contract: clause,
        variants,
        span,
    })
}

pub fn parse_field(stream: &mut TokenStream) -> Result<Field, Diagnostic> {
    let (name, span) = stream.expect_ident_spanned()?;
    stream.expect(Token::Colon)?;
    let type_ = parse_type(stream)?;

    Ok(Field { name, type_, span })
}

pub fn parse_struct_decl(stream: &mut TokenStream) -> Result<Struct, Diagnostic> {
    let vis = parse_visibility(stream)?;

    stream.expect(Token::Struct)?;

    let (name, span) = stream.expect_ident_spanned()?;
    let generics = parse_generics(stream)?;

    let clause = parse_contract(stream)?;

    let mut fields = Vec::new();

    if stream.take(Token::Indent) {
        while !stream.is(Token::Dedent) {
            if stream.is(Token::Newline) {
                stream.consume();
                continue;
            }

            fields.push(parse_field(stream)?);
        }

        stream.expect(Token::Dedent)?;
    }

    Ok(Struct {
        vis,
        name,
        generics,
        contract: clause,
        fields,
        span,
    })
}

pub fn parse_argument(stream: &mut TokenStream) -> Result<Argument, Diagnostic> {
    let mutable = stream.take(Token::Mut);
    let (name, span) = stream.expect_ident_spanned()?;
    stream.expect(Token::Colon)?;
    let type_ = parse_type(stream)?;

    Ok(Argument {
        mutable,
        name,
        type_,
        span,
    })
}

pub fn parse_arguments(stream: &mut TokenStream) -> Result<Vec<Argument>, Diagnostic> {
    let mut arguments = Vec::new();

    loop {
        if stream.is(Token::Paren(Delim::Close)) {
            break;
        }

        arguments.push(parse_argument(stream)?);

        if !stream.take(Token::Comma) {
            break;
        }
    }

    Ok(arguments)
}

fn parse_function_body(stream: &mut TokenStream) -> Result<(Contract, Expr), Diagnostic> {
    if stream.is(Token::Newline) {
        let (_, span) = stream.peek();
        let contract = parse_contract(stream)?;

        let body = if stream.is(Token::Indent) {
            Expr::Block(parse_block_expr(stream)?)
        } else {
            Expr::Void(VoidExpr { span })
        };

        Ok((contract, body))
    } else {
        stream.expect(Token::FatArrow)?;
        let body = parse_expr(stream)?;
        let contract = parse_contract(stream)?;

        Ok((contract, body))
    }
}

pub fn parse_function_decl(stream: &mut TokenStream) -> Result<Function, Diagnostic> {
    let vis = parse_visibility(stream)?;

    stream.expect(Token::Fn)?;

    let (name, span) = stream.expect_ident_spanned()?;
    let generics = parse_generics(stream)?;

    stream.expect(Token::Paren(Delim::Open))?;
    let arguments = parse_arguments(stream)?;
    stream.expect(Token::Paren(Delim::Close))?;

    let output = if stream.take(Token::Arrow) {
        Some(parse_type(stream)?)
    } else {
        None
    };

    let (contract, body) = parse_function_body(stream)?;

    Ok(Function {
        vis,
        name,
        generics,
        arguments,
        output,
        contract,
        body,
        span,
    })
}

pub fn parse_assoc(stream: &mut TokenStream) -> Result<Assoc, Diagnostic> {
    let (name, span) = stream.expect_ident_spanned()?;

    let mut bounds = Vec::new();

    if stream.take(Token::Colon) {
        loop {
            if stream.is(Token::Newline) {
                break;
            }

            let bound = parse_trait_bound(stream)?;
            bounds.push(bound);

            if !stream.take(Token::Comma) {
                break;
            }
        }
    }

    stream.expect(Token::Newline)?;

    Ok(Assoc { name, span, bounds })
}

pub fn parse_self_argument(stream: &mut TokenStream) -> Result<SelfArgument, Diagnostic> {
    let (token, span) = stream.consume();

    match token {
        Token::And => {
            if stream.take(Token::Mut) {
                stream.expect(Token::SelfLower)?;

                Ok(SelfArgument::MutRef)
            } else {
                stream.expect(Token::SelfLower)?;

                Ok(SelfArgument::Ref)
            }
        }
        Token::SelfLower => Ok(SelfArgument::Value),
        _ => {
            let message = format!("expected self argument, found {}", token);
            let diagnostic = Diagnostic::new(message).with_span(span);
            Err(diagnostic)
        }
    }
}

pub fn parse_method(stream: &mut TokenStream) -> Result<Method, Diagnostic> {
    let vis = parse_visibility(stream)?;

    stream.expect(Token::Fn)?;

    let (name, span) = stream.expect_ident_spanned()?;
    let generics = parse_generics(stream)?;

    stream.expect(Token::Paren(Delim::Open))?;

    let self_argument = if stream.is(Token::SelfLower) || stream.is(Token::And) {
        let self_ = parse_self_argument(stream)?;
        stream.take(Token::Comma);

        Some(self_)
    } else {
        None
    };

    let arguments = parse_arguments(stream)?;
    stream.expect(Token::Paren(Delim::Close))?;

    let output = if stream.take(Token::Arrow) {
        Some(parse_type(stream)?)
    } else {
        None
    };

    let contract = parse_contract(stream)?;

    Ok(Method {
        vis,
        name,
        generics,
        self_argument,
        arguments,
        output,
        contract,
        span,
    })
}

pub fn parse_trait_decl(stream: &mut TokenStream) -> Result<Trait, Diagnostic> {
    let vis = parse_visibility(stream)?;

    stream.expect(Token::Trait)?;

    let (name, span) = stream.expect_ident_spanned()?;
    let generics = parse_generics(stream)?;

    let contract = parse_contract(stream)?;

    let mut types = Vec::new();
    let mut methods = Vec::new();

    if stream.take(Token::Indent) {
        while !stream.is(Token::Dedent) {
            if stream.is(Token::Newline) {
                stream.consume();
                continue;
            }

            if stream.is(Token::Ident(String::from("type"))) {
                stream.consume();
                types.push(parse_assoc(stream)?);
            } else {
                methods.push(parse_method(stream)?);
            }
        }

        stream.expect(Token::Dedent)?;
    }

    Ok(Trait {
        vis,
        name,
        generics,
        contract,
        types,
        methods,
        span,
    })
}

pub fn parse_method_impl(stream: &mut TokenStream) -> Result<MethodImpl, Diagnostic> {
    let vis = parse_visibility(stream)?;

    stream.expect(Token::Fn)?;

    let (name, span) = stream.expect_ident_spanned()?;
    let generics = parse_generics(stream)?;

    stream.expect(Token::Paren(Delim::Open))?;

    let self_argument = if stream.is(Token::SelfLower) || stream.is(Token::And) {
        let self_ = parse_self_argument(stream)?;
        stream.take(Token::Comma);

        Some(self_)
    } else {
        None
    };

    let arguments = parse_arguments(stream)?;
    stream.expect(Token::Paren(Delim::Close))?;

    let output = if stream.take(Token::Arrow) {
        Some(parse_type(stream)?)
    } else {
        None
    };

    let (contract, body) = parse_function_body(stream)?;

    Ok(MethodImpl {
        vis,
        name,
        generics,
        self_argument,
        arguments,
        output,
        contract,
        body,
        span,
    })
}

pub fn parse_assoc_impl(stream: &mut TokenStream) -> Result<AssocImpl, Diagnostic> {
    let (name, span) = stream.expect_ident_spanned()?;
    stream.expect(Token::Eq)?;
    let type_ = parse_type(stream)?;

    Ok(AssocImpl { name, type_, span })
}

fn parse_trait_impl(stream: &mut TokenStream, trait_: Path) -> Result<TraitImpl, Diagnostic> {
    let for_ = parse_type(stream)?;
    let contract = parse_contract(stream)?;

    let mut types = Vec::new();
    let mut methods = Vec::new();

    if stream.take(Token::Indent) {
        while !stream.is(Token::Dedent) {
            if stream.is(Token::Newline) {
                stream.consume();
                continue;
            }

            if stream.is(Token::Ident(String::from("type"))) {
                stream.consume();
                types.push(parse_assoc_impl(stream)?);
            } else {
                methods.push(parse_method_impl(stream)?);
            }
        }

        stream.expect(Token::Dedent)?;
    }

    let span = trait_.span.join(for_.span());
    Ok(TraitImpl {
        trait_,
        implementor: for_,
        contract,
        types,
        methods,
        span,
    })
}

fn parse_impl(stream: &mut TokenStream) -> Result<Decl, Diagnostic> {
    let start = stream.expect(Token::Impl)?;

    let first = parse_type(stream)?;

    if stream.take(Token::For) {
        return match first {
            Type::Item(item) => Ok(Decl::TraitImpl(parse_trait_impl(stream, item)?)),
            _ => {
                let message = format!("expected trait, found {:?}", first);
                let diagnostic = Diagnostic::new(message).with_span(start.join(first.span()));
                Err(diagnostic)
            }
        };
    }

    let implementor = first;
    let contract = parse_contract(stream)?;
    let mut methods = Vec::new();

    if stream.take(Token::Indent) {
        while !stream.is(Token::Dedent) {
            if stream.is(Token::Newline) {
                stream.consume();
                continue;
            }

            methods.push(parse_method_impl(stream)?);
        }

        stream.expect(Token::Dedent)?;
    }

    let span = start.join(implementor.span());

    Ok(Decl::Impl(Impl {
        implementor,
        contract,
        methods,
        span,
    }))
}

pub fn parse_module_decl(stream: &mut TokenStream) -> Result<ModuleDecl, Diagnostic> {
    stream.expect(Token::Mod)?;

    let (name, span) = stream.expect_ident_spanned()?;

    stream.expect(Token::Newline)?;
    stream.expect(Token::Indent)?;

    let mut decls = Vec::new();

    while !stream.is(Token::Dedent) {
        decls.push(parse_decl(stream)?);
        stream.take(Token::Newline);
    }

    stream.expect(Token::Dedent)?;

    Ok(ModuleDecl {
        name,
        module: Some(Module { decls }),
        span,
    })
}

pub fn parse_decl(stream: &mut TokenStream) -> Result<Decl, Diagnostic> {
    let (token, span) = stream.peek();

    match token {
        Token::Enum => Ok(Decl::Enum(parse_enum_decl(stream)?)),
        Token::Struct => Ok(Decl::Struct(parse_struct_decl(stream)?)),
        Token::Fn => Ok(Decl::Function(parse_function_decl(stream)?)),
        Token::Trait => Ok(Decl::Trait(parse_trait_decl(stream)?)),
        Token::Impl => parse_impl(stream),
        Token::Mod => Ok(Decl::Module(parse_module_decl(stream)?)),
        _ => {
            let message = format!("expected declaration, found {}", token);
            let diagnostic = Diagnostic::new(message).with_span(span);
            Err(diagnostic)
        }
    }
}

pub fn parse_module(stream: &mut TokenStream) -> Result<Module, Diagnostic> {
    let mut decls = Vec::new();

    while !stream.is_empty() {
        decls.push(parse_decl(stream)?);
        stream.take(Token::Newline);
    }

    Ok(Module { decls })
}
