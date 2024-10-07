use miette::Severity;

use crate::ast::{CallArgument, Field};
use crate::decorator::Decorator;
use crate::{
    ast::{
        Adt, Argument, Arm, BinOp, Decl, Expr, Func, Generic, Import, Module, Pat, PatKind, Path,
        Single, Ty, Type, UnOp, Variant, Vis,
    },
    number::{Base, IntKind},
    span::Span,
    token::{Token, TokenStream},
};

pub fn parse(tokens: &mut TokenStream) -> miette::Result<Module> {
    while tokens.is(Token::Newline) {
        tokens.consume();
    }

    let mut decls = Vec::new();

    // TODO: Add to decl
    while tokens.is(Token::ModDocComment) || tokens.is(Token::Newline) {
        tokens.consume();
    }

    while !tokens.is_eof() {
        decls.push(parse_decl(tokens)?);

        while tokens.is(Token::Newline) {
            tokens.consume();
        }
    }

    Ok(Module { decls })
}

fn parse_decl(tokens: &mut TokenStream) -> miette::Result<Decl> {
    if tokens.is(Token::Import) || tokens.nth_is(1, Token::Import) {
        return parse_import(tokens).map(Decl::Import);
    }

    // TODO: Add to function and types so LSP can extract comments
    while tokens.is(Token::DocComment) || tokens.is(Token::Newline) {
        tokens.consume();
    }

    let decorators = parse_decorators(tokens)?;

    if tokens.is(Token::Fn) || tokens.nth_is(1, Token::Fn) {
        parse_func_decl(tokens, decorators).map(Decl::Func)
    } else if tokens.is(Token::Type) || tokens.nth_is(1, Token::Type) {
        parse_type_decl(tokens, decorators).map(Decl::Type)
    } else {
        let (_, span) = tokens.peek();

        Err(miette::miette!(
            severity = Severity::Error,
            code = "expected::decl",
            labels = vec![span.label("here")],
            "expected function or type declaration, found {:?}",
            tokens.peek().0,
        )
        .with_source_code(span))
    }
}

fn parse_import(tokens: &mut TokenStream) -> miette::Result<Import> {
    let vis = parse_vis(tokens)?;

    tokens.expect(Token::Import)?;

    let path = parse_path(tokens)?;
    let span = path.span;

    Ok(Import { vis, path, span })
}

fn parse_decorators(tokens: &mut TokenStream) -> miette::Result<Vec<Decorator>> {
    let mut decorators = Vec::new();

    while tokens.is(Token::Pound) {
        decorators.push(parse_decorator(tokens)?);
        tokens.expect(Token::Newline)?;
    }

    Ok(decorators)
}

/// Parse decorator with the syntax
/// #[name]
/// #[name()]
/// #[name("arg0", "arg1")]
fn parse_decorator(tokens: &mut TokenStream) -> miette::Result<Decorator> {
    tokens.expect(Token::Pound)?;
    tokens.expect(Token::LBracket)?;

    let (name, _) = parse_snake(tokens)?;

    let mut args = Vec::new();

    if tokens.is(Token::RBracket) {
        tokens.consume();

        return Ok(Decorator {
            name: name.to_string(),
            args,
        });
    }

    tokens.expect(Token::LParen)?;

    while !tokens.is(Token::RParen) {
        match parse_string_expr(tokens)? {
            Expr::String(value, _) => args.push(value.to_string()),
            _ => {
                return Err(miette::miette!(
                    severity = Severity::Error,
                    code = "expected::string_literal",
                    labels = vec![tokens.peek().1.label("here")],
                    "expected string literal",
                )
                .with_source_code(tokens.peek().1))?
            }
        }

        if tokens.take(Token::Comma).is_none() {
            break;
        }
    }

    tokens.expect(Token::RParen)?;
    tokens.expect(Token::RBracket)?;

    Ok(Decorator {
        name: name.to_string(),
        args,
    })
}

fn parse_func_decl(tokens: &mut TokenStream, decorators: Vec<Decorator>) -> miette::Result<Func> {
    let vis = parse_vis(tokens)?;
    tokens.expect(Token::Fn)?;
    let (name, span) = parse_snake(tokens)?;
    let input = parse_arguments(tokens)?;
    let output = parse_output(tokens)?;
    let body = parse_body(tokens)?;

    Ok(Func {
        decorators,
        vis,
        name,
        input,
        output,
        body,
        span,
    })
}

fn parse_type_decl(tokens: &mut TokenStream, decorators: Vec<Decorator>) -> miette::Result<Type> {
    let vis = parse_vis(tokens)?;
    tokens.expect(Token::Type)?;

    let (name, span) = parse_pascal(tokens)?;

    let generics = match tokens.take(Token::Lt) {
        Some(_) => {
            let mut generics = Vec::new();

            while !tokens.is(Token::Gt) {
                generics.push(parse_generic(tokens)?);

                if tokens.take(Token::Comma).is_none() {
                    break;
                }
            }

            tokens.expect(Token::Gt)?;

            Some(generics)
        }
        None => None,
    };

    if tokens.is(Token::LParen) {
        let fields = parse_fields(tokens)?;

        return Ok(Type::Single(Single {
            decorators,
            vis,
            name,
            generics,
            fields,
            span,
        }));
    }

    if tokens.is(Token::Newline) {
        return Ok(Type::Adt(Adt {
            decorators,
            vis,
            name,
            generics,
            variants: Vec::new(),
            span,
        }));
    }

    tokens.expect(Token::Eq)?;

    let mut variants = Vec::new();

    while tokens.is(Token::Newline) || tokens.is(Token::DocComment) {
        tokens.consume();
    }

    tokens.expect(Token::Indent)?;

    while !tokens.is(Token::Dedent) {
        while tokens.is(Token::Newline) || tokens.is(Token::DocComment) {
            tokens.consume();
        }

        tokens.expect(Token::Pipe)?;

        let (name, span) = parse_pascal(tokens)?;

        let fields = match tokens.is(Token::LParen) {
            true => parse_fields(tokens)?,
            false => Vec::new(),
        };

        variants.push(Variant { name, fields, span });

        while tokens.is(Token::Newline) {
            tokens.consume();
        }
    }

    tokens.expect(Token::Dedent)?;

    Ok(Type::Adt(Adt {
        decorators,
        vis,
        name,
        generics,
        variants,
        span,
    }))
}

fn parse_arguments(tokens: &mut TokenStream) -> miette::Result<Vec<Argument>> {
    let mut arguments = Vec::new();

    tokens.expect(Token::LParen)?;

    if tokens.is(Token::Newline) {
        return parse_arguments_multiline(tokens);
    }

    while !tokens.is(Token::RParen) {
        arguments.push(parse_argument(tokens)?);

        if tokens.take(Token::Comma).is_none() {
            break;
        }
    }

    tokens.expect(Token::RParen)?;

    Ok(arguments)
}

fn parse_arguments_multiline(tokens: &mut TokenStream) -> miette::Result<Vec<Argument>> {
    let mut arguments = Vec::new();

    while tokens.is(Token::Newline) || tokens.is(Token::DocComment) {
        tokens.consume();
    }

    tokens.expect(Token::Indent)?;

    while !tokens.is(Token::Dedent) {
        while tokens.is(Token::Newline) || tokens.is(Token::DocComment) {
            tokens.consume();
        }

        arguments.push(parse_argument(tokens)?);

        tokens.take(Token::Comma);

        while tokens.is(Token::Newline) {
            tokens.consume();
        }
    }

    tokens.expect(Token::Dedent)?;
    tokens.expect(Token::RParen)?;

    Ok(arguments)
}

fn parse_output(tokens: &mut TokenStream) -> miette::Result<Option<Ty>> {
    if tokens.take(Token::Arrow).is_none() {
        return Ok(None);
    }

    Ok(Some(parse_ty(tokens)?))
}

fn parse_argument(tokens: &mut TokenStream) -> miette::Result<Argument> {
    let (name, span) = parse_snake(tokens)?;

    let ty = match tokens.take(Token::Colon) {
        Some(_) => Some(parse_ty(tokens)?),
        None => None,
    };

    Ok(Argument { name, ty, span })
}

fn parse_fields(tokens: &mut TokenStream) -> miette::Result<Vec<Field>> {
    let mut fields = Vec::new();

    tokens.expect(Token::LParen)?;

    if tokens.is(Token::Newline) {
        return parse_fields_multiline(tokens);
    }

    while !tokens.is(Token::RParen) {
        fields.push(parse_field(tokens)?);

        if tokens.take(Token::Comma).is_none() {
            break;
        }
    }

    tokens.expect(Token::RParen)?;

    Ok(fields)
}

fn parse_fields_multiline(tokens: &mut TokenStream) -> miette::Result<Vec<Field>> {
    let mut fields = Vec::new();

    while tokens.is(Token::Newline) || tokens.is(Token::DocComment) {
        tokens.consume();
    }

    tokens.expect(Token::Indent)?;

    while !tokens.is(Token::Dedent) {
        while tokens.is(Token::Newline) || tokens.is(Token::DocComment) {
            tokens.consume();
        }

        fields.push(parse_field(tokens)?);

        tokens.take(Token::Comma);

        while tokens.is(Token::Newline) {
            tokens.consume();
        }
    }

    tokens.expect(Token::Dedent)?;
    tokens.expect(Token::RParen)?;

    Ok(fields)
}

fn parse_field(tokens: &mut TokenStream) -> miette::Result<Field> {
    let (name, span) = parse_snake(tokens)?;

    let ty = match tokens.take(Token::Colon) {
        Some(_) => Some(parse_ty(tokens)?),
        None => None,
    };

    Ok(Field { name, ty, span })
}

fn parse_vis(tokens: &mut TokenStream) -> miette::Result<Vis> {
    if tokens.take(Token::Pub).is_some() {
        Ok(Vis::Public)
    } else {
        Ok(Vis::Private)
    }
}

fn parse_ty(tokens: &mut TokenStream) -> miette::Result<Ty> {
    let term = parse_ty_term(tokens)?;

    match tokens.is(Token::Star) {
        true => parse_tuple_ty(tokens, term),
        false => Ok(term),
    }
}

fn parse_tuple_ty(tokens: &mut TokenStream, first: Ty) -> miette::Result<Ty> {
    let mut tys = vec![first];

    while tokens.take(Token::Star).is_some() {
        tys.push(parse_ty_term(tokens)?);
    }

    Ok(Ty::Tuple(tys))
}

fn parse_ty_term(tokens: &mut TokenStream) -> miette::Result<Ty> {
    let (token, span) = tokens.peek();

    match token {
        Token::U8
        | Token::U16
        | Token::U32
        | Token::U64
        | Token::I8
        | Token::I16
        | Token::I32
        | Token::I64
        | Token::Int => parse_int_ty(tokens),
        Token::LParen => {
            tokens.consume();
            let ty = parse_ty(tokens)?;
            tokens.expect(Token::RParen)?;
            Ok(ty)
        }
        Token::LBracket => {
            tokens.consume();
            let ty = parse_ty(tokens)?;
            tokens.expect(Token::RBracket)?;
            Ok(Ty::List(Box::new(ty)))
        }
        Token::Under => {
            tokens.consume();
            Ok(Ty::Inferred)
        }
        Token::Fn => parse_fn_ty(tokens),
        Token::Void => {
            tokens.consume();
            Ok(Ty::Void)
        }
        Token::Str => {
            tokens.consume();
            Ok(Ty::Str)
        }
        Token::Bool => {
            tokens.consume();
            Ok(Ty::Bool)
        }
        Token::Amp => {
            tokens.consume();
            let ty = parse_ty(tokens)?;
            Ok(Ty::Ref(Box::new(ty)))
        }
        Token::Snake | Token::Pascal | Token::Path => {
            let path = parse_path(tokens)?;

            if tokens.take(Token::Lt).is_none() {
                return Ok(Ty::Item(path, None));
            }

            let mut args = Vec::new();

            while !tokens.is(Token::Gt) {
                args.push(parse_ty(tokens)?);

                if tokens.take(Token::Comma).is_none() {
                    break;
                }
            }

            tokens.expect(Token::Gt)?;

            Ok(Ty::Item(path, Some(args)))
        }
        Token::Quote => parse_generic(tokens).map(Ty::Generic),
        _ => Err(miette::miette!(
            severity = Severity::Error,
            code = "expected::ty",
            labels = vec![span.label("here")],
            "expected type, found {:?}",
            token,
        )
        .with_source_code(span)),
    }
}

fn parse_int_ty(tokens: &mut TokenStream) -> miette::Result<Ty> {
    let (token, span) = tokens.consume();

    match token {
        Token::U8 => Ok(Ty::Int(IntKind::U8)),
        Token::U16 => Ok(Ty::Int(IntKind::U16)),
        Token::U32 => Ok(Ty::Int(IntKind::U32)),
        Token::U64 => Ok(Ty::Int(IntKind::U64)),
        Token::I8 => Ok(Ty::Int(IntKind::I8)),
        Token::I16 => Ok(Ty::Int(IntKind::I16)),
        Token::I32 => Ok(Ty::Int(IntKind::I32)),
        Token::I64 => Ok(Ty::Int(IntKind::I64)),
        Token::Int => Ok(Ty::Int(IntKind::Int)),
        _ => Err(miette::miette!(
            severity = Severity::Error,
            code = "expected::int_ty",
            labels = vec![span.label("here")],
            "expected integer type, found {:?}",
            token,
        )
        .with_source_code(span)),
    }
}

fn parse_fn_ty(tokens: &mut TokenStream) -> miette::Result<Ty> {
    tokens.expect(Token::Fn)?;

    let mut input = Vec::new();

    tokens.expect(Token::LParen)?;

    while !tokens.is(Token::RParen) {
        input.push(parse_ty(tokens)?);

        if tokens.take(Token::Comma).is_none() {
            break;
        }
    }

    tokens.expect(Token::RParen)?;

    let output = match tokens.take(Token::Arrow) {
        Some(_) => Some(Box::new(parse_ty(tokens)?)),
        None => None,
    };

    Ok(Ty::Func(input, output))
}

fn parse_generic(tokens: &mut TokenStream) -> miette::Result<Generic> {
    let start = tokens.expect(Token::Quote)?;
    let (name, span) = parse_snake(tokens)?;

    let span = start.join(span);
    Ok(Generic { name, span })
}

fn parse_body(tokens: &mut TokenStream) -> miette::Result<Option<Expr>> {
    while tokens.is(Token::Newline) {
        tokens.consume();
    }

    if !tokens.is(Token::Indent) {
        return Ok(None);
    }

    parse_block(tokens).map(Some)
}

fn parse_block(tokens: &mut TokenStream) -> miette::Result<Expr> {
    let mut exprs = Vec::new();

    tokens.take(Token::Newline);
    tokens.expect(Token::Indent)?;

    while tokens.is(Token::Newline) {
        tokens.consume();
    }

    while !tokens.is(Token::Dedent) {
        exprs.push(parse_expr(tokens, true)?);

        while tokens.is(Token::Newline) {
            tokens.consume();
        }
    }

    tokens.expect(Token::Dedent)?;

    Ok(Expr::Block(exprs))
}

fn is_block(tokens: &TokenStream) -> bool {
    tokens.is(Token::Newline) && tokens.nth_is(1, Token::Indent)
}

fn parse_expr(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    let (token, _) = tokens.peek();

    match token {
        Token::Let => parse_let_expr(tokens, multiline),
        Token::Mut => parse_mut_expr(tokens, multiline),
        Token::Match => parse_match_expr(tokens, multiline),
        Token::Return => parse_return_expr(tokens),
        Token::Assert => parse_assert_expr(tokens, multiline),
        _ => parse_assign_expr(tokens, multiline),
    }
}

fn parse_let_expr(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    tokens.expect(Token::Let)?;

    if tokens.take(Token::Assert).is_some() {
        let pat = parse_pat(tokens)?;

        let ty = match tokens.take(Token::Colon) {
            Some(_) => Some(parse_ty(tokens)?),
            None => None,
        };

        tokens.expect(Token::Eq)?;

        let value = match is_block(tokens) && multiline {
            true => parse_block(tokens)?,
            false => parse_match_or_pipe_expr(tokens, false)?,
        };

        return Ok(Expr::LetAssert(pat, ty, Box::new(value)));
    }

    let pat = parse_pat(tokens)?;

    let ty = match tokens.take(Token::Colon) {
        Some(_) => Some(parse_ty(tokens)?),
        None => None,
    };

    tokens.expect(Token::Eq)?;

    let value = match is_block(tokens) && multiline {
        true => parse_block(tokens)?,
        false => parse_match_or_pipe_expr(tokens, false)?,
    };

    Ok(Expr::Let(pat, ty, Box::new(value)))
}

fn parse_mut_expr(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    tokens.expect(Token::Mut)?;

    let (name, _) = parse_snake(tokens)?;

    let ty = match tokens.take(Token::Colon) {
        Some(_) => Some(parse_ty(tokens)?),
        None => None,
    };

    tokens.expect(Token::Eq)?;

    let value = match is_block(tokens) && multiline {
        true => parse_block(tokens)?,
        false => parse_match_or_pipe_expr(tokens, false)?,
    };

    Ok(Expr::Mut(name, ty, Box::new(value)))
}

fn parse_match_or_pipe_expr(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    let (token, _) = tokens.peek();

    match token {
        Token::Match => parse_match_expr(tokens, multiline),
        _ => parse_pipe_expr(tokens, multiline),
    }
}

fn parse_match_expr(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    let span = tokens.expect(Token::Match)?;

    let input = parse_expr(tokens, false)?;
    let mut arms = Vec::new();

    if !multiline {
        tokens.expect(Token::Newline)?;
        tokens.expect(Token::Indent)?;
    }

    while is_arm(tokens) {
        if tokens.is(Token::Newline) {
            tokens.consume();
        }

        arms.push(parse_arm(tokens)?);
    }

    if !multiline {
        while tokens.is(Token::Newline) {
            tokens.consume();
        }

        tokens.expect(Token::Dedent)?;
    }

    Ok(Expr::Match(Box::new(input), arms, span))
}

fn is_arm(tokens: &TokenStream) -> bool {
    tokens.is(Token::Newline) && tokens.nth_is(1, Token::Pipe) || tokens.is(Token::Pipe)
}

fn parse_arm(tokens: &mut TokenStream) -> miette::Result<Arm> {
    let start = tokens.expect(Token::Pipe)?;

    let pat = parse_pat(tokens)?;

    let end = tokens.expect(Token::Arrow)?;
    let span = start.join(end);

    if is_block(tokens) {
        let expr = parse_block(tokens)?;
        return Ok(Arm { pat, expr, span });
    }

    let expr = parse_expr(tokens, false)?;

    Ok(Arm { pat, expr, span })
}

fn parse_pat(tokens: &mut TokenStream) -> miette::Result<Pat> {
    let pat = parse_pat_term(tokens)?;

    if !tokens.is(Token::Comma) {
        return Ok(pat);
    }

    let mut span = pat.span;
    let mut pats = vec![pat];

    while tokens.take(Token::Comma).is_some() {
        let pat = parse_pat_term(tokens)?;
        span = span.join(pat.span);

        pats.push(pat);
    }

    let kind = PatKind::Tuple(pats);
    Ok(Pat { kind, span })
}

fn parse_pat_term(tokens: &mut TokenStream) -> miette::Result<Pat> {
    let (token, span) = tokens.peek();

    match token {
        Token::Under => {
            tokens.consume();

            let kind = PatKind::Bind(None);
            Ok(Pat { kind, span })
        }
        Token::LParen => {
            tokens.expect(Token::LParen)?;
            let pat = parse_pat(tokens)?;
            tokens.expect(Token::RParen)?;

            Ok(pat)
        }
        Token::LBracket => {
            let start = tokens.expect(Token::LBracket)?;

            let mut pats = Vec::new();
            let mut rest = None;

            while !tokens.is(Token::RBracket) {
                if tokens.take(Token::DotDot).is_some() {
                    rest = match tokens.is(Token::RBracket) {
                        true => Some(None),
                        false => Some(Some(Box::new(parse_pat_term(tokens)?))),
                    };

                    break;
                }

                pats.push(parse_pat_term(tokens)?);

                if tokens.take(Token::Comma).is_none() {
                    break;
                }
            }

            let end = tokens.expect(Token::RBracket)?;

            let kind = PatKind::List(pats, rest);
            let span = start.join(end);
            Ok(Pat { kind, span })
        }
        Token::Snake | Token::Pascal | Token::Path => {
            let path = parse_path(tokens)?;

            if path.segments.len() == 1 && token == Token::Snake {
                let kind = PatKind::Bind(Some(path.segments[0]));
                return Ok(Pat { kind, span });
            }

            if tokens.take(Token::LParen).is_none() {
                let kind = PatKind::Variant(path, Vec::new());
                return Ok(Pat { kind, span });
            }

            let mut pats = Vec::new();

            while !tokens.is(Token::RParen) {
                pats.push(parse_pat_term(tokens)?);

                if tokens.take(Token::Comma).is_none() {
                    break;
                }
            }

            tokens.expect(Token::RParen)?;

            let kind = PatKind::Variant(path, pats);
            Ok(Pat { kind, span })
        }
        Token::True | Token::False => {
            let kind = PatKind::Bool(token == Token::True);
            tokens.consume();
            Ok(Pat { kind, span })
        }
        _ => Err(miette::miette!(
            severity = Severity::Error,
            code = "expected::pat",
            labels = vec![tokens.peek().1.label("here")],
            "expected pattern, found {:?}",
            token,
        )
        .with_source_code(tokens.peek().1)),
    }
}

fn parse_return_expr(tokens: &mut TokenStream) -> miette::Result<Expr> {
    let span = tokens.expect(Token::Return)?;

    let expr = match tokens.is(Token::Newline) {
        true => None,
        false => Some(Box::new(parse_expr(tokens, false)?)),
    };

    Ok(Expr::Return(expr, span))
}

fn parse_assert_expr(tokens: &mut TokenStream, _multiline: bool) -> miette::Result<Expr> {
    let span = tokens.expect(Token::Assert)?;

    let expr = parse_expr(tokens, false)?;

    let message = tokens.take(Token::String).map(|span| span.as_str());

    Ok(Expr::Assert(Box::new(expr), message, span))
}

fn parse_assign_expr(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    let lhs = parse_pipe_expr(tokens, multiline)?;

    if tokens.take(Token::Eq).is_none() {
        return Ok(lhs);
    }

    let rhs = parse_expr(tokens, multiline)?;

    Ok(Expr::Assign(Box::new(lhs), Box::new(rhs)))
}

fn parse_pipe_expr(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    let input = parse_tuple_expr(tokens, multiline)?;

    if !tokens.is(Token::PipeGt) && !is_pipe_multiline(tokens) && !is_pipe_indent_multiline(tokens)
    {
        return Ok(input);
    }

    if is_pipe_multiline(tokens) && multiline {
        return parse_pipe_expr_multiline(tokens, input);
    }

    if is_pipe_indent_multiline(tokens) {
        return parse_pipe_expr_indent_multiline(tokens, input);
    }

    let mut exprs = Vec::new();

    while tokens.take(Token::PipeGt).is_some() {
        exprs.push(parse_unary_expr(tokens, multiline)?);
    }

    Ok(Expr::Pipe(Box::new(input), exprs))
}

fn parse_pipe_expr_multiline(tokens: &mut TokenStream, input: Expr) -> miette::Result<Expr> {
    let mut exprs = Vec::new();

    while is_pipe_multiline(tokens) {
        tokens.consume();
        tokens.consume();
        exprs.push(parse_binary_expr(tokens, true)?);
    }

    Ok(Expr::Pipe(Box::new(input), exprs))
}

fn parse_pipe_expr_indent_multiline(tokens: &mut TokenStream, input: Expr) -> miette::Result<Expr> {
    let mut exprs = Vec::new();

    tokens.expect(Token::Newline)?;
    tokens.expect(Token::Indent)?;

    while !tokens.is(Token::Dedent) {
        tokens.expect(Token::PipeGt)?;
        exprs.push(parse_binary_expr(tokens, true)?);
        tokens.expect(Token::Newline)?;

        while tokens.is(Token::Newline) {
            tokens.consume();
        }
    }

    tokens.expect(Token::Dedent)?;

    Ok(Expr::Pipe(Box::new(input), exprs))
}

fn is_pipe_multiline(tokens: &TokenStream) -> bool {
    tokens.is(Token::Newline) && tokens.nth_is(1, Token::PipeGt)
}

fn is_pipe_indent_multiline(tokens: &TokenStream) -> bool {
    tokens.is(Token::Newline) && tokens.nth_is(1, Token::Indent) && tokens.nth_is(2, Token::PipeGt)
}

fn parse_tuple_expr(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    let first = parse_binary_expr(tokens, multiline)?;

    let mut items = vec![first];

    while tokens.take(Token::Comma).is_some() {
        let second = parse_binary_expr(tokens, multiline)?;
        items.push(second);
    }

    if items.len() == 1 {
        Ok(items.pop().unwrap())
    } else {
        Ok(Expr::Tuple(items))
    }
}

fn parse_binary_expr(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    let (token, _) = tokens.peek();

    match token {
        Token::Pipe => return parse_closure_expr(tokens),
        Token::PipePipe => return parse_empty_closure_expr(tokens),
        _ => {}
    }

    let lhs = parse_as_expr(tokens, multiline)?;

    let Some(lop) = get_binop(tokens) else {
        return Ok(lhs);
    };

    let (_, lspan) = tokens.consume();

    let rhs = parse_binary_expr(tokens, multiline)?;

    match rhs {
        Expr::Binary(rop, mid, rhs, rspan) => {
            if lop.precedence() > rop.precedence() {
                Ok(Expr::Binary(
                    rop,
                    Box::new(Expr::Binary(lop, Box::new(lhs), mid, lspan)),
                    rhs,
                    rspan,
                ))
            } else {
                Ok(Expr::Binary(
                    lop,
                    Box::new(lhs),
                    Box::new(Expr::Binary(rop, mid, rhs, rspan)),
                    lspan,
                ))
            }
        }
        rhs => Ok(Expr::Binary(lop, Box::new(lhs), Box::new(rhs), lspan)),
    }
}

fn get_binop(tokens: &TokenStream) -> Option<BinOp> {
    let (token, _) = tokens.peek();

    Some(match token {
        Token::Plus => BinOp::Add,
        Token::Minus => BinOp::Sub,
        Token::Star => BinOp::Mul,
        Token::Slash => BinOp::Div,
        Token::Percent => BinOp::Rem,
        Token::AmpAmp => BinOp::And,
        Token::PipePipe => BinOp::Or,
        Token::EqEq => BinOp::Eq,
        Token::NotEq => BinOp::Ne,
        Token::Lt => BinOp::Lt,
        Token::LtEq => BinOp::Le,
        Token::Gt => BinOp::Gt,
        Token::GtEq => BinOp::Ge,
        _ => return None,
    })
}

fn parse_as_expr(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    let lhs = parse_unary_expr(tokens, multiline)?;

    if tokens.take(Token::As).is_none() {
        return Ok(lhs);
    }

    let ty = parse_ty(tokens)?;

    Ok(Expr::As(Box::new(lhs), ty))
}

fn parse_unary_expr(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    let (token, span) = tokens.peek();

    match token {
        Token::Amp => {
            tokens.consume();
            let expr = parse_unary_expr(tokens, multiline)?;
            Ok(Expr::Unary(UnOp::Ref, Box::new(expr), span))
        }
        Token::Star => {
            tokens.consume();
            let expr = parse_unary_expr(tokens, multiline)?;
            Ok(Expr::Unary(UnOp::Deref, Box::new(expr), span))
        }
        Token::Minus => {
            tokens.consume();
            let expr = parse_unary_expr(tokens, multiline)?;
            Ok(Expr::Unary(UnOp::Neg, Box::new(expr), span))
        }
        Token::Not => {
            tokens.consume();
            let expr = parse_unary_expr(tokens, multiline)?;
            Ok(Expr::Unary(UnOp::Not, Box::new(expr), span))
        }
        _ => parse_try_expr(tokens, multiline),
    }
}

fn parse_try_expr(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    let mut expr = parse_call_expr(tokens, multiline)?;

    while let Some(span) = tokens.take(Token::Question) {
        expr = Expr::Try(Box::new(expr), span);
    }

    Ok(expr)
}

fn parse_call_expr(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    let callee = parse_field_expr(tokens, multiline)?;

    if !tokens.is(Token::LParen) {
        return Ok(callee);
    }

    let mut args = Vec::new();

    tokens.expect(Token::LParen)?;

    if is_block(tokens) {
        tokens.expect(Token::Newline)?;
        tokens.expect(Token::Indent)?;

        let mut spread = None;

        while !tokens.is(Token::Dedent) {
            if tokens.take(Token::DotDot).is_some() {
                spread = Some(Box::new(parse_expr(tokens, true)?));
                tokens.expect(Token::Newline)?;
                break;
            }

            let (arg, took_newline) = parse_call_argument(tokens, true)?;
            args.push(arg);

            if !took_newline {
                tokens.expect(Token::Newline)?;
            }
        }

        tokens.expect(Token::Dedent)?;
        tokens.expect(Token::RParen)?;

        return Ok(Expr::Call(Box::new(callee), args, spread));
    }

    let mut spread = None;

    while !tokens.is(Token::RParen) {
        if tokens.take(Token::DotDot).is_some() {
            spread = Some(Box::new(parse_expr(tokens, false)?));
            break;
        }

        let (arg, _) = parse_call_argument(tokens, false)?;
        args.push(arg);

        if tokens.take(Token::Comma).is_none() {
            break;
        }
    }

    tokens.expect(Token::RParen)?;

    Ok(Expr::Call(Box::new(callee), args, spread))
}

fn parse_call_argument(
    tokens: &mut TokenStream,
    is_multiline: bool,
) -> miette::Result<(CallArgument, bool)> {
    if tokens.is(Token::Under) {
        tokens.consume();
        return Ok((CallArgument::Positional(None), false));
    }

    if tokens.is(Token::Snake) && tokens.nth_is(1, Token::Colon) {
        let (name, _) = parse_snake(tokens)?;
        tokens.expect(Token::Colon)?;

        if is_block(tokens) && is_multiline {
            let value = parse_block(tokens)?;
            return Ok((CallArgument::Named(name, value), true));
        }

        let value = match is_multiline {
            true => parse_expr(tokens, true)?,
            false => parse_binary_expr(tokens, false)?,
        };

        return Ok((CallArgument::Named(name, value), false));
    }

    let expr = match is_multiline {
        true => parse_expr(tokens, true)?,
        false => parse_binary_expr(tokens, false)?,
    };

    Ok((CallArgument::Positional(Some(expr)), false))
}

fn parse_field_expr(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    let mut base = parse_term_expr(tokens, multiline)?;

    while tokens.take(Token::Dot).is_some() {
        let (field, _) = parse_snake(tokens)?;
        base = Expr::Field(Box::new(base), field);
    }

    Ok(base)
}

fn parse_term_expr(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    let (token, span) = tokens.peek();

    match token {
        Token::LParen => parse_paren_expr(tokens, multiline),
        Token::Integer => parse_integer_expr(tokens),
        Token::Minus => parse_integer_expr(tokens),
        Token::FormatStart => parse_format_expr(tokens),
        Token::String => parse_string_expr(tokens),
        Token::LBracket => parse_list_expr(tokens, multiline),
        Token::Void => {
            tokens.consume();
            Ok(Expr::Void(span))
        }
        Token::Panic => {
            tokens.consume();

            let message = match tokens.take(Token::String) {
                Some(span) => span.as_str(),
                _ => "explicit panic",
            };

            Ok(Expr::Panic(message, span))
        }
        Token::True | Token::False => parse_bool_expr(tokens),
        Token::Snake | Token::Pascal | Token::Path => parse_path(tokens).map(Expr::Item),
        _ => Err(miette::miette!(
            severity = Severity::Error,
            code = "expected::term",
            labels = vec![span.label("here")],
            "expected term, found {:?}",
            token,
        )
        .with_source_code(span)),
    }
}

fn parse_paren_expr(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    let span = tokens.take(Token::LParen).unwrap();

    let expr = parse_expr(tokens, multiline)?;

    let close = tokens.take(Token::RParen).unwrap();

    Ok(Expr::Paren(Box::new(expr), span.join(close)))
}

fn parse_integer_expr(tokens: &mut TokenStream) -> miette::Result<Expr> {
    let span = tokens.take(Token::Integer).ok_or_else(|| {
        let (token, span) = tokens.peek();

        miette::miette!(
            severity = Severity::Error,
            code = "expected::integer",
            labels = vec![span.label("here")],
            "expected integer, found {:?}",
            token,
        )
        .with_source_code(span)
    })?;

    let mut base = Base::Dec;
    let mut digits = Vec::new();

    let mut string = span.as_str();

    if string.starts_with("0x") {
        base = Base::Hex;
        string = &string[2..];
    } else if string.starts_with("0b") {
        base = Base::Bin;
        string = &string[2..];
    } else if string.starts_with("0o") {
        base = Base::Oct;
        string = &string[2..];
    }

    for c in string.chars() {
        let digit = c.to_digit(base.radix()).unwrap();
        digits.push(digit as u8);
    }

    Ok(Expr::Int(false, base, digits, span))
}

fn parse_string_expr(tokens: &mut TokenStream) -> miette::Result<Expr> {
    let (token, mut span) = tokens.consume();

    span.lo += 1;
    span.hi -= 1;

    match token {
        Token::String => Ok(Expr::String(span.as_str(), span)),
        _ => unreachable!(),
    }
}

fn parse_format_expr(tokens: &mut TokenStream) -> miette::Result<Expr> {
    tokens.expect(Token::FormatStart)?;

    let (_, mut total_span) = tokens.peek();

    let mut parts = Vec::new();

    loop {
        let (token, span) = tokens.peek();
        total_span.hi = span.hi;

        match token {
            Token::FormatEnd => break,
            Token::FormatExprStart => {
                tokens.consume();
                let expr = parse_expr(tokens, false)?;
                tokens.expect(Token::FormatExprEnd)?;

                // Wrap expression in call to `std:debug:repr`
                let repr = Expr::Item(Path {
                    segments: vec!["std", "debug", "format"],
                    span,
                });

                let args = vec![CallArgument::Positional(Some(expr))];

                parts.push(Expr::Call(Box::new(repr), args, None));
            }
            Token::String => {
                tokens.consume();
                let string = span.as_str();
                parts.push(Expr::String(string, span));
            }
            _ => {
                return Err(miette::miette!(
                    severity = Severity::Error,
                    code = "expected::format_string",
                    labels = vec![span.label("here")],
                    "expected format string, found {:?}",
                    token,
                )
                .with_source_code(span))
            }
        }
    }

    tokens.expect(Token::FormatEnd)?;

    // Generate big expression that chains all parts together
    // with std:string:concat start from the end

    let mut expr = parts.pop().unwrap();

    let concat_segments = vec!["std", "string", "concat"];

    while let Some(part) = parts.pop() {
        let concat = Expr::Item(Path {
            segments: concat_segments.clone(),
            span: total_span,
        });

        let args = vec![
            CallArgument::Positional(Some(part)),
            CallArgument::Positional(Some(expr)),
        ];

        expr = Expr::Call(Box::new(concat), args, None);
    }

    Ok(expr)
}

fn parse_list_expr(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    let start = tokens.expect(Token::LBracket)?;

    let mut items = Vec::new();
    let mut rest = None;

    while !tokens.is(Token::RBracket) {
        if tokens.take(Token::DotDot).is_some() {
            rest = Some(Box::new(parse_expr(tokens, multiline)?));
            break;
        }

        items.push(parse_binary_expr(tokens, multiline)?);

        if tokens.take(Token::Comma).is_none() {
            break;
        }
    }

    let end = tokens.expect(Token::RBracket)?;

    Ok(Expr::List(items, rest, start.join(end)))
}

fn parse_closure_expr(tokens: &mut TokenStream) -> miette::Result<Expr> {
    let _start = tokens.expect(Token::Pipe)?;

    let mut args = Vec::new();

    while !tokens.is(Token::Pipe) {
        args.push(parse_argument(tokens)?);

        if tokens.take(Token::Comma).is_none() {
            break;
        }
    }

    let _end = tokens.expect(Token::Pipe)?;

    let body = match is_block(tokens) {
        true => parse_block(tokens)?,
        false => parse_binary_expr(tokens, false)?,
    };

    Ok(Expr::Closure(args, Box::new(body)))
}

fn parse_empty_closure_expr(tokens: &mut TokenStream) -> miette::Result<Expr> {
    let _start = tokens.expect(Token::PipePipe)?;

    let body = match is_block(tokens) {
        true => parse_block(tokens)?,
        false => parse_binary_expr(tokens, false)?,
    };

    Ok(Expr::Closure(Vec::new(), Box::new(body)))
}

fn parse_bool_expr(tokens: &mut TokenStream) -> miette::Result<Expr> {
    let (token, span) = tokens.consume();

    match token {
        Token::True => Ok(Expr::Bool(true, span)),
        Token::False => Ok(Expr::Bool(false, span)),
        _ => unreachable!(),
    }
}

fn parse_path(tokens: &mut TokenStream) -> miette::Result<Path> {
    let (token, span) = tokens.consume();

    match token {
        Token::Snake => Ok(Path {
            segments: vec![span.as_str()],
            span,
        }),
        Token::Pascal => Ok(Path {
            segments: vec![span.as_str()],
            span,
        }),
        Token::Path => Ok(Path {
            segments: span.as_str().split(":").collect(),
            span,
        }),
        _ => Err(miette::miette!(
            severity = Severity::Error,
            code = "expected::path",
            labels = vec![span.label("here")],
            "expected path, found {:?}",
            token,
        )
        .with_source_code(span)),
    }
}

fn parse_snake(tokens: &mut TokenStream) -> miette::Result<(&'static str, Span)> {
    match tokens.consume() {
        (Token::Snake, span) => Ok((span.as_str(), span)),
        (token, span) => Err(miette::miette!(
            severity = Severity::Error,
            code = "expected::snake_case",
            labels = vec![span.label("here")],
            "expected snake_case identifier, found {:?}",
            token,
        )
        .with_source_code(span)),
    }
}

fn parse_pascal(tokens: &mut TokenStream) -> miette::Result<(&'static str, Span)> {
    match tokens.consume() {
        (Token::Pascal, span) => Ok((span.as_str(), span)),
        (token, span) => Err(miette::miette!(
            severity = Severity::Error,
            code = "expected::pascal_case",
            labels = vec![span.label("here")],
            "expected PascalCase identifier, found {:?}",
            token,
        )
        .with_source_code(span)),
    }
}

#[cfg(test)]
mod tests {
    use crate::{lex, token::TokenStream};

    use super::*;

    fn tokens(input: &'static str) -> TokenStream {
        lex::lex("", input).unwrap()
    }

    macro_rules! parse {
        ($parser:expr, $input:expr) => {
            $parser(&mut tokens($input)).unwrap()
        };
    }

    macro_rules! parse_err {
        ($parser:expr, $input:expr) => {
            $parser(&mut tokens($input)).is_err()
        };
    }

    #[test]
    fn snake() {
        assert_eq!(parse!(parse_snake, "foo").0, "foo");
        assert_eq!(parse!(parse_snake, "foo_bar").0, "foo_bar");
        assert!(parse_err!(parse_snake, "Foo"));
    }
}
