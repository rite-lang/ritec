use miette::Severity;

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

    while !tokens.is_eof() {
        decls.push(parse_decl(tokens)?);

        while tokens.is(Token::Newline) {
            tokens.consume();
        }
    }

    Ok(Module { decls })
}

fn parse_decl(tokens: &mut TokenStream) -> miette::Result<Decl> {
    if tokens.is(Token::Import) {
        parse_import(tokens).map(Decl::Import)
    } else if tokens.is(Token::Fn) || tokens.nth_is(1, Token::Fn) {
        parse_func_decl(tokens).map(Decl::Func)
    } else if tokens.is(Token::Type) || tokens.nth_is(1, Token::Type) {
        parse_type_decl(tokens).map(Decl::Type)
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
    tokens.expect(Token::Import)?;

    let path = parse_path(tokens)?;
    let span = path.span;

    Ok(Import { path, span })
}

fn parse_func_decl(tokens: &mut TokenStream) -> miette::Result<Func> {
    let vis = parse_vis(tokens)?;
    tokens.expect(Token::Fn)?;
    let (name, span) = parse_snake(tokens)?;
    let input = parse_arguments(tokens)?;
    let output = parse_output(tokens)?;
    let body = parse_body(tokens)?;

    Ok(Func {
        vis,
        name,
        input,
        output,
        body,
        span,
    })
}

fn parse_type_decl(tokens: &mut TokenStream) -> miette::Result<Type> {
    let vis = parse_vis(tokens)?;
    tokens.expect(Token::Type)?;

    let (name, span) = parse_pascal(tokens)?;

    if tokens.is(Token::LParen) {
        let fields = parse_arguments(tokens)?;

        return Ok(Type::Single(Single {
            vis,
            name,
            fields,
            span,
        }));
    }

    tokens.expect(Token::Eq)?;

    let mut variants = Vec::new();

    tokens.expect(Token::Newline)?;
    tokens.expect(Token::Indent)?;

    while !tokens.is(Token::Dedent) {
        tokens.expect(Token::Pipe)?;

        let (name, span) = parse_pascal(tokens)?;

        let fields = match tokens.is(Token::LParen) {
            true => parse_arguments(tokens)?,
            false => Vec::new(),
        };

        variants.push(Variant { name, fields, span });

        while !tokens.is(Token::Dedent) && tokens.is(Token::Newline) {
            tokens.consume();
        }
    }

    tokens.expect(Token::Dedent)?;

    Ok(Type::Adt(Adt {
        vis,
        name,
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

    tokens.expect(Token::Newline)?;
    tokens.expect(Token::Indent)?;

    while !tokens.is(Token::Dedent) {
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
        Token::Mut => {
            tokens.consume();
            let ty = parse_ty(tokens)?;
            Ok(Ty::Mut(Box::new(ty)))
        }
        Token::Snake | Token::Pascal => {
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
    if !is_block(tokens) {
        return Ok(None);
    }

    parse_block(tokens).map(Some)
}

fn parse_block(tokens: &mut TokenStream) -> miette::Result<Expr> {
    let mut exprs = Vec::new();

    tokens.expect(Token::Newline)?;
    tokens.expect(Token::Indent)?;

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
        Token::Let => parse_let(tokens, multiline),
        Token::Mut => parse_mut(tokens, multiline),
        Token::Match => parse_match(tokens, multiline),
        _ => parse_assign(tokens, multiline),
    }
}

fn parse_let(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    tokens.expect(Token::Let)?;

    let (name, _) = parse_snake(tokens)?;

    tokens.expect(Token::Eq)?;

    let value = match is_block(tokens) {
        true => parse_block(tokens)?,
        false => parse_pipe(tokens, multiline)?,
    };

    Ok(Expr::Let(name, Box::new(value)))
}

fn parse_mut(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    tokens.expect(Token::Mut)?;

    let (name, _) = parse_snake(tokens)?;

    tokens.expect(Token::Eq)?;

    let value = match is_block(tokens) {
        true => parse_block(tokens)?,
        false => parse_expr(tokens, multiline)?,
    };

    Ok(Expr::Mut(name, Box::new(value)))
}

fn parse_match(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    if !multiline {
        return Err(miette::miette!(
            severity = Severity::Error,
            code = "expected::match",
            labels = vec![tokens.peek().1.label("here")],
            "expected match expression",
        )
        .with_source_code(tokens.peek().1));
    }

    let span = tokens.expect(Token::Match)?;

    let input = parse_expr(tokens, false)?;
    let mut arms = Vec::new();

    while is_arm(tokens) {
        if tokens.is(Token::Newline) {
            tokens.consume();
        }

        arms.push(parse_arm(tokens)?);
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

fn is_pat_term(tokens: &TokenStream) -> bool {
    let (token, _) = tokens.peek();

    matches!(
        token,
        Token::Under | Token::Snake | Token::Pascal | Token::True | Token::False
    )
}

fn parse_pat_term(tokens: &mut TokenStream) -> miette::Result<Pat> {
    let (token, span) = tokens.peek();

    match token {
        Token::Under => {
            tokens.consume();

            let kind = PatKind::Bind(None);
            Ok(Pat { kind, span })
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
        Token::Snake | Token::Pascal => {
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

fn parse_assign(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    let lhs = parse_pipe(tokens, multiline)?;

    if tokens.take(Token::Eq).is_none() {
        return Ok(lhs);
    }

    let rhs = parse_expr(tokens, multiline)?;

    Ok(Expr::Assign(Box::new(lhs), Box::new(rhs)))
}

fn parse_pipe(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    let input = parse_tuple(tokens, multiline)?;

    if !tokens.is(Token::PipeGt) && !is_pipe_multiline(tokens) {
        return Ok(input);
    }

    if is_pipe_multiline(tokens) && multiline {
        return parse_pipe_multiline(tokens, input);
    }

    let mut exprs = Vec::new();

    while tokens.take(Token::PipeGt).is_some() {
        exprs.push(parse_unary(tokens, multiline)?);
    }

    Ok(Expr::Pipe(Box::new(input), exprs))
}

fn parse_pipe_multiline(tokens: &mut TokenStream, input: Expr) -> miette::Result<Expr> {
    let mut exprs = Vec::new();

    while is_pipe_multiline(tokens) {
        tokens.consume();
        tokens.consume();
        exprs.push(parse_binary(tokens, true)?);
    }

    Ok(Expr::Pipe(Box::new(input), exprs))
}

fn is_pipe_multiline(tokens: &TokenStream) -> bool {
    tokens.is(Token::Newline) && tokens.nth_is(1, Token::PipeGt)
}

fn parse_tuple(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    let first = parse_binary(tokens, multiline)?;

    let mut items = vec![first];

    while tokens.take(Token::Comma).is_some() {
        let second = parse_binary(tokens, multiline)?;
        items.push(second);
    }

    if items.len() == 1 {
        Ok(items.pop().unwrap())
    } else {
        Ok(Expr::Tuple(items))
    }
}

fn parse_binary(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    let (token, _) = tokens.peek();

    match token {
        Token::Pipe => return parse_closure(tokens),
        Token::PipePipe => return parse_empty_closure(tokens),
        _ => {}
    }

    let lhs = parse_unary(tokens, multiline)?;

    let Some(lop) = get_binop(tokens) else {
        return Ok(lhs);
    };

    let (_, lspan) = tokens.consume();

    let rhs = parse_binary(tokens, multiline)?;

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
        Token::Amp => BinOp::And,
        Token::Pipe => BinOp::Or,
        Token::EqEq => BinOp::Eq,
        Token::NotEq => BinOp::Ne,
        Token::Lt => BinOp::Lt,
        Token::LtEq => BinOp::Le,
        Token::Gt => BinOp::Gt,
        Token::GtEq => BinOp::Ge,
        _ => return None,
    })
}

fn parse_unary(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    let (token, span) = tokens.peek();

    match token {
        Token::Mut => {
            tokens.consume();
            let expr = parse_unary(tokens, multiline)?;
            Ok(Expr::Unary(UnOp::Mut, Box::new(expr), span))
        }
        Token::Star => {
            tokens.consume();
            let expr = parse_unary(tokens, multiline)?;
            Ok(Expr::Unary(UnOp::Deref, Box::new(expr), span))
        }
        _ => parse_call(tokens, multiline),
    }
}

fn parse_call(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    let callee = parse_field(tokens, multiline)?;

    if !tokens.is(Token::LParen) {
        return Ok(callee);
    }

    let mut args = Vec::new();

    tokens.expect(Token::LParen)?;

    while !tokens.is(Token::RParen) {
        match tokens.take(Token::Under).is_some() {
            false => args.push(Some(parse_binary(tokens, false)?)),
            true => args.push(None),
        }

        if tokens.take(Token::Comma).is_none() {
            break;
        }
    }

    tokens.expect(Token::RParen)?;

    Ok(Expr::Call(Box::new(callee), args))
}

fn parse_field(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    let base = parse_term(tokens, multiline)?;

    if !tokens.is(Token::Dot) {
        return Ok(base);
    }

    tokens.expect(Token::Dot)?;

    let (name, _) = parse_snake(tokens)?;

    Ok(Expr::Field(Box::new(base), name))
}

fn parse_term(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    let (token, span) = tokens.peek();

    match token {
        Token::LParen => parse_paren(tokens, multiline),
        Token::Integer => parse_integer(tokens),
        Token::String => parse_string(tokens),
        Token::LBracket => parse_list(tokens, multiline),
        Token::Void => {
            tokens.consume();
            Ok(Expr::Void(span))
        }
        Token::True | Token::False => parse_bool(tokens),
        Token::Snake | Token::Pascal => parse_path(tokens).map(Expr::Item),
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

fn parse_paren(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    let span = tokens.take(Token::LParen).unwrap();

    let expr = parse_expr(tokens, multiline)?;

    let close = tokens.take(Token::RParen).unwrap();

    Ok(Expr::Paren(Box::new(expr), span.join(close)))
}

fn parse_integer(tokens: &mut TokenStream) -> miette::Result<Expr> {
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

    let base = Base::Dec;
    let mut digits = Vec::new();

    for c in span.as_str().chars() {
        digits.push(c.to_digit(base.radix()).unwrap() as u8);
    }

    Ok(Expr::Int(false, base, digits, span))
}

fn parse_string(tokens: &mut TokenStream) -> miette::Result<Expr> {
    let (token, span) = tokens.consume();

    match token {
        Token::String => Ok(Expr::String(span.as_str(), span)),
        _ => unreachable!(),
    }
}

fn parse_list(tokens: &mut TokenStream, multiline: bool) -> miette::Result<Expr> {
    let start = tokens.expect(Token::LBracket)?;

    let mut items = Vec::new();
    let mut rest = None;

    while !tokens.is(Token::RBracket) {
        if tokens.take(Token::DotDot).is_some() {
            rest = Some(Box::new(parse_expr(tokens, multiline)?));
            break;
        }

        items.push(parse_binary(tokens, multiline)?);

        if tokens.take(Token::Comma).is_none() {
            break;
        }
    }

    let end = tokens.expect(Token::RBracket)?;

    Ok(Expr::List(items, rest, start.join(end)))
}

fn parse_closure(tokens: &mut TokenStream) -> miette::Result<Expr> {
    let _start = tokens.expect(Token::Pipe)?;

    let mut args = Vec::new();

    while !tokens.is(Token::Pipe) {
        args.push(parse_argument(tokens)?);
    }

    let _end = tokens.expect(Token::Pipe)?;

    let body = match is_block(tokens) {
        true => parse_block(tokens)?,
        false => parse_binary(tokens, false)?,
    };

    Ok(Expr::Closure(args, Box::new(body)))
}

fn parse_empty_closure(tokens: &mut TokenStream) -> miette::Result<Expr> {
    let _start = tokens.expect(Token::PipePipe)?;

    let body = match is_block(tokens) {
        true => parse_block(tokens)?,
        false => parse_binary(tokens, false)?,
    };

    Ok(Expr::Closure(Vec::new(), Box::new(body)))
}

fn parse_bool(tokens: &mut TokenStream) -> miette::Result<Expr> {
    let (token, span) = tokens.consume();

    match token {
        Token::True => Ok(Expr::Bool(true, span)),
        Token::False => Ok(Expr::Bool(false, span)),
        _ => unreachable!(),
    }
}

fn parse_path(tokens: &mut TokenStream) -> miette::Result<Path> {
    let (name, mut span) = parse_segment(tokens)?;
    let mut segments = vec![name];

    while tokens.take(Token::Colon).is_some() {
        let (name, new_span) = parse_segment(tokens)?;
        segments.push(name);
        span = span.join(new_span);
    }

    Ok(Path { segments, span })
}

fn parse_segment(tokens: &mut TokenStream) -> miette::Result<(&'static str, Span)> {
    match tokens.consume() {
        (Token::Snake, span) => Ok((span.as_str(), span)),
        (Token::Pascal, span) => Ok((span.as_str(), span)),
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
