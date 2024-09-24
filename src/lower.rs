use std::{collections::HashMap, mem};

use miette::Severity;

use crate::{
    ast, hir,
    number::{Base, IntKind},
    span::Span,
};

pub fn type_register_ast(
    unit: &mut hir::Unit,
    module: usize,
    ast: &ast::Module,
) -> miette::Result<()> {
    for decl in &ast.decls {
        if let ast::Decl::Type(ast::Type::Adt(adt)) = decl {
            let vis = match adt.vis {
                ast::Vis::Public => hir::Vis::Public,
                ast::Vis::Private => hir::Vis::Private,
            };

            let id = unit.push_adt(hir::Adt {
                vis,
                name: adt.name,
                generics: Vec::new(),
                variants: Vec::new(),
                span: adt.span,
            });

            unit.modules[module].adts.insert(adt.name, id);
        }
    }

    Ok(())
}

pub fn type_resolve_ast(
    unit: &mut hir::Unit,
    module: usize,
    ast: &ast::Module,
) -> miette::Result<()> {
    for decl in &ast.decls {
        if let ast::Decl::Import(import) = decl {
            let imported = find_module(unit, module, &import.path)?;
            let name = unit.modules[imported].name;

            unit.modules[module].modules.insert(name, imported);
        }
    }

    Ok(())
}

pub fn type_construct_ast(
    unit: &mut hir::Unit,
    module: usize,
    ast: &ast::Module,
) -> miette::Result<()> {
    for decl in &ast.decls {
        if let ast::Decl::Type(ast::Type::Adt(adt)) = decl {
            let id = unit.modules[module].adts[adt.name];
            let mut generics = mem::take(&mut unit.adts[id].generics);

            for variant in &adt.variants {
                let mut cx = TyCx {
                    unit,
                    module,
                    generics: &mut generics,
                    new_generics: true,
                    func: None,
                };

                let fields = lower_arguments(&mut cx, &variant.fields)?;
                let variant = hir::Variant {
                    name: variant.name,
                    fields,
                };

                unit.adts[id].variants.push(variant);
            }

            unit.adts[id].generics = generics;
        }
    }

    Ok(())
}

pub fn func_register_ast(
    unit: &mut hir::Unit,
    module: usize,
    ast: &ast::Module,
) -> miette::Result<()> {
    for decl in &ast.decls {
        if let ast::Decl::Func(func) = decl {
            let mut generics = Vec::new();

            let id = unit.funcs.len();

            let mut cx = TyCx {
                unit,
                module,
                generics: &mut generics,
                new_generics: true,
                func: Some(id),
            };
            let input = lower_arguments(&mut cx, &func.input)?;
            let output = lower_output(&mut cx, &func.output)?;

            let vis = match func.vis {
                ast::Vis::Public => hir::Vis::Public,
                ast::Vis::Private => hir::Vis::Private,
            };

            unit.push_func(hir::Func {
                vis,
                name: func.name,
                generics,
                input,
                output,
                locals: Vec::new(),
                captures: Vec::new(),
                body: hir::Expr {
                    kind: hir::ExprKind::Void,
                    ty: hir::Ty::void(),
                },
            });

            unit.modules[module].funcs.insert(func.name, id);
        }
    }

    Ok(())
}

pub fn func_construct_ast(
    unit: &mut hir::Unit,
    module: usize,
    ast: &ast::Module,
) -> miette::Result<()> {
    for decl in &ast.decls {
        if let ast::Decl::Func(func) = decl {
            let id = unit.modules[module].funcs[func.name];

            // Lower the body of the function.
            if let Some(body) = &func.body {
                let arguments = &unit.funcs[id].input.clone();

                let mut cx = BodyCx {
                    unit,
                    arguments,
                    module,
                    locals: Vec::new(),
                    scope: Vec::new(),
                };

                let body = lower_expr(&mut cx, body)?;

                unit.funcs[id].locals = cx.locals;
                unit.funcs[id].body = body;

                unit.unify(
                    unit.funcs[id].body.ty.clone(),
                    unit.funcs[id].output.clone(),
                );
            } else {
                unit.unify(hir::Ty::void(), unit.funcs[id].output.clone());
            }
        }
    }

    Ok(())
}

struct TyCx<'a> {
    unit: &'a hir::Unit,
    module: usize,
    generics: &'a mut Vec<hir::Generic>,
    new_generics: bool,
    func: Option<usize>,
}

fn lower_arguments(cx: &mut TyCx, ast: &[ast::Argument]) -> miette::Result<Vec<hir::Argument>> {
    let mut arguments = Vec::new();

    for arg in ast {
        arguments.push(lower_argument(cx, arg)?);
    }

    Ok(arguments)
}

fn lower_argument(cx: &mut TyCx, ast: &ast::Argument) -> miette::Result<hir::Argument> {
    let ty = match ast.ty {
        Some(ref ty) => lower_ty(cx, ty)?,
        None => hir::Ty::Inferred(hir::Tid::new(), hir::Inferred::Any, cx.func),
    };

    Ok(hir::Argument {
        name: ast.name,
        ty,
        span: ast.span,
    })
}

fn lower_output(cx: &mut TyCx, output: &Option<ast::Ty>) -> miette::Result<hir::Ty> {
    match output {
        Some(ty) => Ok(lower_ty(cx, ty)?),
        None => Ok(hir::Ty::Inferred(
            hir::Tid::new(),
            hir::Inferred::Any,
            cx.func,
        )),
    }
}

fn lower_ty(cx: &mut TyCx, ty: &ast::Ty) -> miette::Result<hir::Ty> {
    match ty {
        ast::Ty::Void => Ok(hir::Ty::void()),
        ast::Ty::Bool => Ok(hir::Ty::bool()),
        ast::Ty::Str => Ok(hir::Ty::string()),
        ast::Ty::Int(kind) => Ok(hir::Ty::Partial(hir::Part::Int(*kind), Vec::new())),
        ast::Ty::Tuple(tys) => {
            let mut args = Vec::new();

            for ty in tys {
                args.push(lower_ty(cx, ty)?);
            }

            Ok(hir::Ty::Partial(hir::Part::Tuple, args))
        }
        ast::Ty::Item(path) => {
            let index = find_adt(cx.unit, cx.module, path)?;

            let generics = cx.unit.adts[index]
                .generics
                .iter()
                .map(|_| hir::Ty::Inferred(hir::Tid::new(), hir::Inferred::Any, cx.func))
                .collect();

            Ok(hir::Ty::Partial(hir::Part::Adt(index), generics))
        }
        ast::Ty::List(ty) => {
            let ty = lower_ty(cx, ty)?;
            Ok(hir::Ty::Partial(hir::Part::List, vec![ty]))
        }
        ast::Ty::Generic(generic) => {
            if let Some(index) = cx.generics.iter().position(|g| g.name == generic.name) {
                return Ok(hir::Ty::Partial(hir::Part::Generic(index), Vec::new()));
            }

            if !cx.new_generics {
                return Err(miette::miette!(
                    severity = Severity::Error,
                    code = "invalid::generic",
                    "generic `{}` not found",
                    generic.name
                ));
            }

            let index = cx.generics.len();
            cx.generics.push(hir::Generic { name: generic.name });
            Ok(hir::Ty::Partial(hir::Part::Generic(index), Vec::new()))
        }
    }
}

#[derive(Debug)]
enum Item {
    Func(usize),
    Variant(usize, usize),
}

fn find_module(unit: &hir::Unit, module: usize, path: &ast::Path) -> miette::Result<usize> {
    let mut current = module;

    for segment in path.segments.iter() {
        match unit.modules[current].modules.get(segment) {
            Some(&next) => current = next,
            None => {
                return Err(miette::miette!(
                    severity = Severity::Error,
                    code = "invalid::path",
                    labels = vec![path.span.label("here")],
                    "module not found `{}`",
                    segment
                )
                .with_source_code(path.span));
            }
        }
    }

    Ok(current)
}

fn find_adt(unit: &hir::Unit, module: usize, path: &ast::Path) -> miette::Result<usize> {
    let mut current = module;

    for segment in path.segments.iter().take(path.segments.len() - 1) {
        match unit.modules[current].modules.get(segment) {
            Some(&next) => current = next,
            None => {
                return Err(miette::miette!(
                    severity = Severity::Error,
                    code = "invalid::path",
                    labels = vec![path.span.label("here")],
                    "module not found `{}`",
                    segment
                )
                .with_source_code(path.span));
            }
        }
    }

    let name = path
        .segments
        .last()
        .expect("path should have at least one segment");

    if let Some(&id) = unit.modules[current].adts.get(name) {
        if unit.adts[id].vis == hir::Vis::Public || current == module {
            return Ok(id);
        }
    }

    Err(miette::miette!(
        severity = Severity::Error,
        code = "invalid::path",
        labels = vec![path.span.label("here")],
        "invalid item `{}`",
        name
    ))
}

fn find_variant(
    unit: &hir::Unit,
    module: usize,
    path: &ast::Path,
) -> miette::Result<(usize, usize)> {
    match resolve_item(unit, module, path)? {
        Item::Variant(adt, variant) => Ok((adt, variant)),
        _ => Err(miette::miette!(
            severity = Severity::Error,
            code = "invalid::path",
            labels = vec![path.span.label("here")],
            "expected variant"
        )
        .with_source_code(path.span)),
    }
}

fn resolve_item(unit: &hir::Unit, module: usize, path: &ast::Path) -> miette::Result<Item> {
    let mut current = module;

    for segment in path.segments.iter().take(path.segments.len() - 1) {
        match unit.modules[current].modules.get(segment) {
            Some(&next) => current = next,
            None => {
                return Err(miette::miette!(
                    severity = Severity::Error,
                    code = "invalid::path",
                    labels = vec![path.span.label("here")],
                    "module not found `{}`",
                    segment
                )
                .with_source_code(path.span));
            }
        }
    }

    let name = path
        .segments
        .last()
        .expect("path should have at least one segment");

    for (i, adt) in unit.adts.iter().enumerate() {
        if adt.vis == hir::Vis::Private && current != module {
            continue;
        }

        for (j, variant) in adt.variants.iter().enumerate() {
            if variant.name == *name {
                return Ok(Item::Variant(i, j));
            }
        }
    }

    if let Some(&id) = unit.modules[current].funcs.get(name) {
        if unit.funcs[id].vis == hir::Vis::Public || current == module {
            return Ok(Item::Func(id));
        }
    }

    Err(miette::miette!(
        severity = Severity::Error,
        code = "invalid::path",
        labels = vec![path.span.label("here")],
        "invalid item `{}`",
        name
    )
    .with_source_code(path.span))
}

struct BodyCx<'a> {
    unit: &'a mut hir::Unit,
    arguments: &'a [hir::Argument],
    module: usize,
    locals: Vec<hir::Local>,
    scope: Vec<(String, usize)>,
}

fn lower_expr(cx: &mut BodyCx, ast: &ast::Expr) -> miette::Result<hir::Expr> {
    match ast {
        ast::Expr::Int(negative, base, digits, span) => {
            lower_int(cx, *negative, *base, digits, *span)
        }
        ast::Expr::Bool(value, span) => lower_bool(cx, *value, *span),
        ast::Expr::String(value, span) => lower_string(cx, value, *span),
        ast::Expr::Paren(expr, _) => lower_expr(cx, expr),
        ast::Expr::Item(path) => lower_item(cx, path),
        ast::Expr::Tuple(exprs) => lower_tuple(cx, exprs),
        ast::Expr::List(exprs, rest, _) => lower_list(cx, exprs, rest),
        ast::Expr::Block(block) => lower_block(cx, block),
        ast::Expr::Field(expr, name) => lower_field(cx, expr, name),
        ast::Expr::Call(func, args) => lower_call(cx, func, args),
        ast::Expr::Pipe(expr, exprs) => lower_pipe(cx, expr, exprs),
        ast::Expr::Binary(op, lhs, rhs) => lower_binary(cx, *op, lhs, rhs),
        ast::Expr::Let(name, expr) => lower_let(cx, name, expr),
        ast::Expr::Match(input, arms) => lower_match(cx, input, arms),
        ast::Expr::Closure(_, _) => todo!(),
    }
}

fn lower_int(
    _cx: &mut BodyCx,
    negative: bool,
    base: Base,
    digits: &[u8],
    _span: Span,
) -> miette::Result<hir::Expr> {
    let ty = hir::Ty::inferred(hir::Inferred::Int(IntKind::Int));

    let kind = hir::ExprKind::Int(negative, base, digits.to_vec());
    Ok(hir::Expr { kind, ty })
}

fn lower_bool(_cx: &mut BodyCx, value: bool, _span: Span) -> miette::Result<hir::Expr> {
    let ty = hir::Ty::bool();
    let kind = hir::ExprKind::Bool(value);
    Ok(hir::Expr { kind, ty })
}

fn lower_string(_cx: &mut BodyCx, value: &'static str, _span: Span) -> miette::Result<hir::Expr> {
    let ty = hir::Ty::string();
    let kind = hir::ExprKind::String(value);
    Ok(hir::Expr { kind, ty })
}

fn lower_item(cx: &mut BodyCx, path: &ast::Path) -> miette::Result<hir::Expr> {
    if path.segments.len() == 1 {
        let name = path
            .segments
            .first()
            .expect("path should have at least one segment");

        for (scope_name, id) in cx.scope.iter().rev() {
            if scope_name == name {
                let ty = cx.locals[*id].ty.clone();
                let kind = hir::ExprKind::Local(*id);
                return Ok(hir::Expr { kind, ty });
            }
        }

        for (i, argument) in cx.arguments.iter().enumerate() {
            if argument.name == *name {
                let ty = argument.ty.clone();
                let kind = hir::ExprKind::Argument(i);
                return Ok(hir::Expr { kind, ty });
            }
        }
    }

    let item = resolve_item(cx.unit, cx.module, path)?;

    match item {
        Item::Func(id) => {
            let mut parts = Vec::new();

            let mut generics = HashMap::new();

            for argument in &cx.unit.funcs[id].input {
                parts.push(cx.unit.env.use_ty(&mut generics, &argument.ty));
            }

            parts.push(cx.unit.env.use_ty(&mut generics, &cx.unit.funcs[id].output));

            let kind = hir::ExprKind::Func(id);
            let ty = hir::Ty::Partial(hir::Part::Func, parts);
            Ok(hir::Expr { kind, ty })
        }
        Item::Variant(adt, index) => {
            let variant = &cx.unit.adts[adt].variants[index];

            let generics = cx.unit.adts[adt]
                .generics
                .iter()
                .map(|_| hir::Ty::inferred(hir::Inferred::Any))
                .collect();

            if variant.fields.is_empty() {
                let ty = hir::Ty::Partial(hir::Part::Adt(adt), generics);
                let kind = hir::ExprKind::Variant(adt, index);
                return Ok(hir::Expr { kind, ty });
            }

            let output = hir::Ty::Partial(hir::Part::Adt(adt), generics.clone());

            let mut parts = Vec::new();

            for field in variant.fields.iter() {
                parts.push(field.ty.specialize(&generics));
            }

            parts.push(output);

            let ty = hir::Ty::Partial(hir::Part::Func, parts);
            let kind = hir::ExprKind::Variant(adt, index);
            Ok(hir::Expr { kind, ty })
        }
    }
}

fn lower_tuple(cx: &mut BodyCx, exprs: &[ast::Expr]) -> miette::Result<hir::Expr> {
    let mut args = Vec::new();
    let mut tys = Vec::new();

    for expr in exprs {
        let arg = lower_expr(cx, expr)?;
        tys.push(arg.ty.clone());
        args.push(arg);
    }

    let ty = hir::Ty::Partial(hir::Part::Tuple, tys);
    let kind = hir::ExprKind::Tuple(args);
    Ok(hir::Expr { kind, ty })
}

fn lower_list(
    cx: &mut BodyCx,
    exprs: &[ast::Expr],
    rest: &Option<Box<ast::Expr>>,
) -> miette::Result<hir::Expr> {
    let mut args = Vec::new();
    let ty = hir::Ty::any();

    for expr in exprs {
        let arg = lower_expr(cx, expr)?;
        cx.unit.unify(arg.ty.clone(), ty.clone());

        args.push(arg);
    }

    let ty = hir::Ty::Partial(hir::Part::List, vec![ty]);

    let rest = match rest {
        Some(rest) => {
            let rest = lower_expr(cx, rest)?;
            cx.unit.unify(rest.ty.clone(), ty.clone());
            Some(Box::new(rest))
        }
        None => None,
    };

    let kind = hir::ExprKind::List(args, rest);
    Ok(hir::Expr { kind, ty })
}

fn lower_block(cx: &mut BodyCx, block: &[ast::Expr]) -> miette::Result<hir::Expr> {
    let scope = cx.scope.len();

    let mut exprs = Vec::new();
    let mut ty = None;

    for expr in block {
        let expr = lower_expr(cx, expr)?;
        ty = Some(expr.ty.clone());

        exprs.push(expr);
    }

    cx.scope.truncate(scope);

    let kind = hir::ExprKind::Block(exprs);
    let ty = ty.expect("block should have at least one expression");

    Ok(hir::Expr { kind, ty })
}

fn lower_field(cx: &mut BodyCx, expr: &ast::Expr, name: &'static str) -> miette::Result<hir::Expr> {
    let expr = lower_expr(cx, expr)?;
    let ty = hir::Ty::Field(Box::new(expr.ty.clone()), name);
    let kind = hir::ExprKind::Field(Box::new(expr), name);
    cx.unit.normalize(ty.clone());
    Ok(hir::Expr { kind, ty })
}

fn lower_call(
    cx: &mut BodyCx,
    func: &ast::Expr,
    arguments: &[Option<ast::Expr>],
) -> miette::Result<hir::Expr> {
    let func = lower_expr(cx, func)?;

    let mut args = Vec::new();
    let mut tys = Vec::new();

    for argument in arguments {
        match argument {
            Some(argument) => {
                let arg = lower_expr(cx, argument)?;
                tys.push(Some(arg.ty.clone()));
                args.push(Some(arg));
            }
            None => {
                args.push(None);
                tys.push(None);
            }
        }
    }

    let ty = hir::Ty::Call(Box::new(func.ty.clone()), tys);
    let kind = hir::ExprKind::Call(Box::new(func), args);
    cx.unit.normalize(ty.clone());
    Ok(hir::Expr { kind, ty })
}

fn lower_pipe(
    cx: &mut BodyCx,
    pipee: &ast::Expr,
    exprs: &[ast::Expr],
) -> miette::Result<hir::Expr> {
    let mut pipee = lower_expr(cx, pipee)?;
    let mut ty = pipee.ty.clone();

    for expr in exprs {
        let expr = lower_expr(cx, expr)?;

        ty = hir::Ty::Pipe(Box::new(ty), Box::new(expr.ty.clone()));
        pipee = hir::Expr {
            kind: hir::ExprKind::Pipe(Box::new(pipee), Box::new(expr)),
            ty: ty.clone(),
        };
    }

    cx.unit.normalize(ty.clone());
    Ok(pipee)
}

fn lower_binary(
    cx: &mut BodyCx,
    op: ast::BinOp,
    lhs: &ast::Expr,
    rhs: &ast::Expr,
) -> miette::Result<hir::Expr> {
    let lhs = lower_expr(cx, lhs)?;
    let rhs = lower_expr(cx, rhs)?;

    match op {
        ast::BinOp::Add
        | ast::BinOp::Sub
        | ast::BinOp::Mul
        | ast::BinOp::Div
        | ast::BinOp::Rem
        | ast::BinOp::Lt
        | ast::BinOp::Le
        | ast::BinOp::Gt
        | ast::BinOp::Ge => {
            let ty = hir::Ty::inferred(hir::Inferred::Int(IntKind::Int));

            cx.unit.unify(lhs.ty.clone(), ty.clone());
            cx.unit.unify(rhs.ty.clone(), ty.clone());

            let kind = hir::ExprKind::Binary(op, Box::new(lhs), Box::new(rhs));
            Ok(hir::Expr { kind, ty })
        }
        ast::BinOp::Eq | ast::BinOp::Ne => {
            cx.unit.unify(lhs.ty.clone(), rhs.ty.clone());

            let ty = hir::Ty::bool();
            let kind = hir::ExprKind::Binary(op, Box::new(lhs), Box::new(rhs));
            Ok(hir::Expr { kind, ty })
        }
        ast::BinOp::And | ast::BinOp::Or => {
            let ty = hir::Ty::bool();

            cx.unit.unify(lhs.ty.clone(), ty.clone());
            cx.unit.unify(rhs.ty.clone(), ty.clone());

            let kind = hir::ExprKind::Binary(op, Box::new(lhs), Box::new(rhs));
            Ok(hir::Expr { kind, ty })
        }
    }
}

fn lower_let(cx: &mut BodyCx, name: &'static str, expr: &ast::Expr) -> miette::Result<hir::Expr> {
    let value = lower_expr(cx, expr)?;

    let id = cx.locals.len();
    let ty = value.ty.clone();

    cx.locals.push(hir::Local { name, ty });
    cx.scope.push((name.to_owned(), id));

    let kind = hir::ExprKind::Let(id, Box::new(value));
    let ty = hir::Ty::void();
    Ok(hir::Expr { kind, ty })
}

fn lower_match(cx: &mut BodyCx, input: &ast::Expr, arms: &[ast::Arm]) -> miette::Result<hir::Expr> {
    let input = lower_expr(cx, input)?;

    let mut pats = Vec::new();

    for arm in arms {
        pats.push(lower_pat(cx, &arm.pat, &input.ty)?);
    }

    let mut tree = Match::None;
    let ty = hir::Ty::any();

    for (pat, arm) in pats.into_iter().zip(arms.iter()) {
        let scope = cx.scope.len();

        let arm_ty = build_match_tree(
            cx,
            input.clone(),
            pat,
            &mut std::iter::empty(),
            &mut tree,
            &arm.expr,
            Vec::new(),
        )?;

        cx.scope.truncate(scope);
        cx.unit.unify(ty.clone(), arm_ty);
    }

    Ok(build_match_expr(cx, tree)?.unwrap())
}

fn lower_pat(cx: &mut BodyCx, pat: &ast::Pat, ty: &hir::Ty) -> miette::Result<Pat> {
    match pat.kind {
        ast::PatKind::Bind(name) => Ok(Pat::Bind(name)),
        ast::PatKind::Bool(b) => {
            cx.unit.unify(ty.clone(), hir::Ty::bool());
            Ok(Pat::Bool(b))
        }
        ast::PatKind::Tuple(ref pats) => pats
            .iter()
            .map(|pat| lower_pat(cx, pat, &hir::Ty::any()))
            .collect::<miette::Result<Vec<_>>>()
            .map(Pat::Tuple),
        ast::PatKind::Variant(ref path, ref pats) => {
            let (adt, index) = find_variant(cx.unit, cx.module, path)?;

            let generics: Vec<_> = cx.unit.adts[adt]
                .generics
                .iter()
                .map(|_| hir::Ty::any())
                .collect();

            let variant = &cx.unit.adts[adt].variants[index];

            if pats.len() != variant.fields.len() {
                return Err(miette::miette!(
                    severity = Severity::Error,
                    code = "invalid::pattern",
                    "expected `{}` fields, found `{}`",
                    variant.fields.len(),
                    pats.len()
                ));
            }

            let mut hir = Vec::new();

            for (pat, field) in pats.iter().zip(variant.fields.clone()) {
                let ty = field.ty.specialize(&generics);
                let pat = lower_pat(cx, pat, &ty)?;
                hir.push((ty, pat));
            }

            let adt_ty = hir::Ty::Partial(hir::Part::Adt(adt), generics);
            cx.unit.unify(ty.clone(), adt_ty);

            Ok(Pat::Variant(adt, index, hir))
        }
        ast::PatKind::List(ref pats, ref rest) => {
            let mut rest = match rest {
                Some(Some(rest)) => lower_pat(cx, rest, &hir::Ty::any())?,
                Some(None) => Pat::Bind(None),
                None => Pat::List(None),
            };

            for pat in pats.iter().rev() {
                let pat = lower_pat(cx, pat, &hir::Ty::any())?;
                rest = Pat::List(Some(Box::new((pat, rest))));
            }

            let ty = hir::Ty::Partial(hir::Part::List, vec![hir::Ty::any()]);
            cx.unit.unify(ty.clone(), ty);

            Ok(rest)
        }
    }
}

#[derive(Debug)]
enum Pat {
    Bind(Option<&'static str>),
    Bool(bool),
    Tuple(Vec<Pat>),
    Variant(usize, usize, Vec<(hir::Ty, Pat)>),
    List(Option<Box<(Pat, Pat)>>),
}

fn build_match_tree(
    cx: &mut BodyCx,
    input: hir::Expr,
    pat: Pat,
    pats: &mut dyn Iterator<Item = (hir::Expr, Pat)>,
    tree: &mut Match,
    body: &ast::Expr,
    mut locals: Vec<(usize, hir::Expr)>,
) -> miette::Result<hir::Ty> {
    match pat {
        Pat::Bool(value) => {
            let Match::Bool {
                r#true,
                r#false,
                default,
                ..
            } = tree.as_bool(&input)?
            else {
                unreachable!()
            };

            let subtree = match value {
                true => match r#false {
                    Some(_) => default,
                    None => r#true.get_or_insert_with(|| Box::new(Match::None)),
                },
                false => match r#true {
                    Some(_) => default,
                    None => r#false.get_or_insert_with(|| Box::new(Match::None)),
                },
            };

            match pats.next() {
                Some((input, pat)) => build_match_tree(cx, input, pat, pats, subtree, body, locals),
                None => {
                    let expr = lower_expr(cx, body)?;
                    let ty = expr.ty.clone();

                    if matches!(subtree.as_ref(), Match::None) {
                        *subtree = Box::new(Match::Leaf(locals, expr));
                    }

                    Ok(ty)
                }
            }
        }
        Pat::Tuple(items) => {
            let items = items
                .into_iter()
                .enumerate()
                .map(|(i, pat)| {
                    let kind = hir::ExprKind::TupleField(Box::new(input.clone()), i);
                    let ty = hir::Ty::Tuple(Box::new(input.ty.clone()), i);
                    cx.unit.normalize(ty.clone());

                    let input = hir::Expr { kind, ty };
                    (input, pat)
                })
                .collect::<Vec<_>>();

            let mut pats = items.into_iter().chain(pats);

            let (input, pat) = pats.next().unwrap();
            build_match_tree(cx, input, pat, &mut pats, tree, body, locals)
        }
        Pat::Variant(adt, variant, fields) => {
            let variants = cx.unit.adts[adt].variants.len();
            let Match::Adt {
                variants, default, ..
            } = tree.as_adt(&input, adt, variants)?
            else {
                unreachable!()
            };

            let subtree = match variants[variant] {
                Some(ref mut pair) => pair,
                None => {
                    let use_default = variants
                        .iter()
                        .enumerate()
                        .all(|(i, v)| v.is_some() && i != variant);

                    match use_default {
                        true => default,
                        false => variants[variant].get_or_insert_with(|| Match::None),
                    }
                }
            };

            let fields = fields.into_iter().enumerate().map(move |(i, (ty, pat))| {
                let kind = hir::ExprKind::VariantField(Box::new(input.clone()), variant, i);
                let ty = ty.clone();

                let input = hir::Expr { kind, ty };
                (input, pat)
            });

            let mut pats = fields.chain(pats);

            match pats.next() {
                Some((input, pat)) => {
                    build_match_tree(cx, input, pat, &mut pats, subtree, body, locals)
                }
                None => {
                    let expr = lower_expr(cx, body)?;
                    let ty = expr.ty.clone();

                    if matches!(subtree, Match::None) {
                        *subtree = Match::Leaf(locals, expr);
                    }

                    Ok(ty)
                }
            }
        }

        Pat::List(pat) => {
            let Match::List {
                some,
                none,
                default,
                ..
            } = tree.as_list(&input)?
            else {
                unreachable!()
            };

            match pat.map(|p| *p) {
                Some((head, tail)) => {
                    let some = match none {
                        Some(_) => default,
                        None => some.get_or_insert_with(|| Box::new(Match::None)),
                    };

                    let head_ty = hir::Ty::any();
                    let list_ty = hir::Ty::Partial(hir::Part::List, vec![head_ty.clone()]);

                    cx.unit.unify(input.ty.clone(), list_ty);

                    let head_kind = hir::ExprKind::ListHead(Box::new(input.clone()));
                    let head_expr = hir::Expr {
                        kind: head_kind,
                        ty: head_ty,
                    };

                    let tail_kind = hir::ExprKind::ListTail(Box::new(input.clone()));
                    let tail_expr = hir::Expr {
                        kind: tail_kind,
                        ty: input.ty.clone(),
                    };

                    let mut pats = [(head_expr, head), (tail_expr, tail)]
                        .into_iter()
                        .chain(pats);

                    let (input, pat) = pats.next().unwrap();
                    build_match_tree(cx, input, pat, &mut pats, some, body, locals)
                }
                None => {
                    let none = match some {
                        Some(_) => default,
                        None => none.get_or_insert_with(|| Box::new(Match::None)),
                    };

                    match pats.next() {
                        Some((input, pat)) => {
                            build_match_tree(cx, input, pat, pats, none, body, locals)
                        }
                        None => {
                            let expr = lower_expr(cx, body)?;
                            let ty = expr.ty.clone();

                            if matches!(none.as_ref(), Match::None) {
                                *none = Box::new(Match::Leaf(locals, expr));
                            }

                            Ok(ty)
                        }
                    }
                }
            }
        }
        Pat::Bind(name) => {
            if let Some(name) = name {
                let id = cx.locals.len();
                let ty = input.ty.clone();

                cx.locals.push(hir::Local {
                    name,
                    ty: ty.clone(),
                });
                cx.scope.push((name.to_owned(), id));

                locals.push((id, input));
            }

            let tree = match tree {
                Match::None => {
                    *tree = Match::Bind(Box::new(Match::None));

                    match tree {
                        Match::Bind(tree) => tree,
                        _ => unreachable!(),
                    }
                }
                Match::Leaf(_, _) => unreachable!(),
                Match::Bind(tree) => tree,
                Match::Bool { default, .. }
                | Match::Adt { default, .. }
                | Match::List { default, .. } => default,
            };

            match pats.next() {
                Some((input, pat)) => build_match_tree(cx, input, pat, pats, tree, body, locals),
                None => {
                    let expr = lower_expr(cx, body)?;
                    let ty = expr.ty.clone();

                    tree.visit(&mut move |tree| {
                        if matches!(tree, Match::None) {
                            *tree = Match::Leaf(locals.clone(), expr.clone());
                        }
                    });

                    Ok(ty)
                }
            }
        }
    }
}

fn build_match_expr(cx: &mut BodyCx, tree: Match) -> miette::Result<Option<hir::Expr>> {
    match tree {
        Match::None => Ok(None),
        Match::Leaf(locals, expr) => {
            let mut exprs = locals
                .into_iter()
                .map(|(id, expr)| {
                    let ty = expr.ty.clone();
                    let kind = hir::ExprKind::Let(id, Box::new(expr));
                    hir::Expr { kind, ty }
                })
                .collect::<Vec<_>>();

            let ty = expr.ty.clone();
            exprs.push(expr);

            let kind = hir::ExprKind::Block(exprs);
            Ok(Some(hir::Expr { kind, ty }))
        }
        Match::Bind(tree) => build_match_expr(cx, *tree),
        Match::Bool {
            input,
            r#true,
            r#false,
            default,
        } => {
            let (r#true, r#false) = match (r#true, r#false) {
                (Some(r#true), Some(r#false)) => (r#true, r#false),
                (Some(r#true), None) => (r#true, default),
                (None, Some(r#false)) => (default, r#false),
                (None, None) => unreachable!(),
            };

            let ty = hir::Ty::any();

            let r#true = build_match_expr(cx, *r#true)?.map(Box::new).unwrap();
            let r#false = build_match_expr(cx, *r#false)?.map(Box::new).unwrap();

            cx.unit.unify(ty.clone(), r#true.ty.clone());
            cx.unit.unify(ty.clone(), r#false.ty.clone());

            let r#match = hir::Match::Bool(r#true, r#false);
            let kind = hir::ExprKind::Match(Box::new(input), r#match);

            Ok(Some(hir::Expr { kind, ty }))
        }
        Match::Adt {
            input,
            adt,
            variants,
            default,
        } => {
            let mut exprs = Vec::new();
            let ty = hir::Ty::any();

            for variant in variants {
                match variant {
                    Some(subtree) => {
                        let expr = build_match_expr(cx, subtree)?.unwrap();
                        cx.unit.unify(ty.clone(), expr.ty.clone());
                        exprs.push(Some(expr));
                    }
                    None => exprs.push(None),
                }
            }

            let default = build_match_expr(cx, *default)?.map(Box::new);

            if let Some(ref default) = default {
                cx.unit.unify(ty.clone(), default.ty.clone());
            }

            let r#match = hir::Match::Adt(adt, exprs, default);
            let kind = hir::ExprKind::Match(Box::new(input), r#match);
            Ok(Some(hir::Expr { kind, ty }))
        }
        Match::List {
            input,
            some,
            none,
            default,
        } => {
            let (some, none) = match (some, none) {
                (Some(some), Some(none)) => (some, none),
                (Some(some), None) => (some, default),
                (None, Some(none)) => (default, none),
                (None, None) => unreachable!(),
            };

            let ty = hir::Ty::any();

            let some = build_match_expr(cx, *some)?.map(Box::new).unwrap();
            let none = build_match_expr(cx, *none)?.map(Box::new).unwrap();

            cx.unit.unify(ty.clone(), some.ty.clone());
            cx.unit.unify(ty.clone(), none.ty.clone());

            let r#match = hir::Match::List(some, none);
            let kind = hir::ExprKind::Match(Box::new(input), r#match);
            Ok(Some(hir::Expr { kind, ty }))
        }
    }
}

#[derive(Clone)]
enum Match {
    None,
    Bind(Box<Match>),
    Leaf(Vec<(usize, hir::Expr)>, hir::Expr),
    Bool {
        input: hir::Expr,
        r#true: Option<Box<Match>>,
        r#false: Option<Box<Match>>,
        default: Box<Match>,
    },
    Adt {
        input: hir::Expr,
        adt: usize,
        variants: Vec<Option<Match>>,
        default: Box<Match>,
    },
    List {
        input: hir::Expr,
        some: Option<Box<Match>>,
        none: Option<Box<Match>>,
        default: Box<Match>,
    },
}

impl Match {
    fn as_bool(&mut self, input: &hir::Expr) -> miette::Result<&mut Match> {
        match self {
            Match::None | Match::Leaf(_, _) => {
                *self = Self::Bool {
                    input: input.clone(),
                    r#true: None,
                    r#false: None,
                    default: Box::new(self.clone()),
                };

                Ok(self)
            }
            Match::Bind(tree) => {
                *self = Self::Bool {
                    input: input.clone(),
                    r#true: None,
                    r#false: None,
                    default: tree.clone(),
                };

                Ok(self)
            }
            Match::Bool { .. } => Ok(self),
            _ => Err(miette::miette!(
                severity = Severity::Error,
                code = "invalid::match",
                "expected bool"
            )),
        }
    }

    fn as_adt(
        &mut self,
        input: &hir::Expr,
        index: usize,
        variants: usize,
    ) -> miette::Result<&mut Match> {
        match self {
            Match::None | Match::Leaf(_, _) => {
                *self = Self::Adt {
                    input: input.clone(),
                    adt: index,
                    variants: vec![None; variants],
                    default: Box::new(self.clone()),
                };

                Ok(self)
            }
            Match::Bind(tree) => tree.as_adt(input, index, variants),
            Match::Adt { adt, .. } => {
                if *adt != index {
                    return Err(miette::miette!(
                        severity = Severity::Error,
                        code = "invalid::match",
                        "expected ADT `{}`, found `{}`",
                        index,
                        adt
                    ));
                }

                Ok(self)
            }
            _ => Err(miette::miette!(
                severity = Severity::Error,
                code = "invalid::match",
                "expected ADT"
            )),
        }
    }

    fn as_list(&mut self, input: &hir::Expr) -> miette::Result<&mut Match> {
        match self {
            Match::None | Match::Leaf(_, _) => {
                *self = Self::List {
                    input: input.clone(),
                    some: None,
                    none: None,
                    default: Box::new(self.clone()),
                };

                Ok(self)
            }
            Match::Bind(tree) => tree.as_list(input),
            Match::List { .. } => Ok(self),
            _ => Err(miette::miette!(
                severity = Severity::Error,
                code = "invalid::match",
                "expected list"
            )),
        }
    }

    fn visit(&mut self, f: &mut impl FnMut(&mut Match)) {
        match self {
            Match::None | Match::Leaf(_, _) => {}
            Match::Bind(tree) => tree.visit(f),
            Match::Bool {
                r#true,
                r#false,
                default,
                ..
            } => {
                if let Some(r#true) = r#true {
                    r#true.visit(f);
                }

                if let Some(r#false) = r#false {
                    r#false.visit(f);
                }

                default.visit(f);
            }
            Match::Adt {
                variants, default, ..
            } => {
                for variant in variants.iter_mut().flatten() {
                    variant.visit(f);
                }

                default.visit(f);
            }
            Match::List {
                some,
                none,
                default,
                ..
            } => {
                if let Some(some) = some {
                    some.visit(f);
                }

                if let Some(none) = none {
                    none.visit(f);
                }

                default.visit(f);
            }
        }

        f(self);
    }
}

impl std::fmt::Debug for Match {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Match::None => write!(f, "None"),
            Match::Leaf(_, body) => write!(f, "Leaf({:?})", body),
            Match::Bind(tree) => f.debug_tuple("Bind").field(tree).finish(),
            Match::Bool {
                r#true,
                r#false,
                default,
                ..
            } => f
                .debug_struct("Bool")
                .field("r#true", r#true)
                .field("r#false", r#false)
                .field("default", default)
                .finish(),
            Match::Adt {
                adt,
                variants,
                default,
                ..
            } => f
                .debug_struct("Adt")
                .field("adt", adt)
                .field("variants", variants)
                .field("default", default)
                .finish(),
            Match::List {
                some,
                none,
                default,
                ..
            } => f
                .debug_struct("List")
                .field("some", some)
                .field("none", none)
                .field("default", default)
                .finish(),
        }
    }
}
