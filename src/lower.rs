use std::{collections::HashMap, mem};

use miette::Severity;

use crate::{
    ast, hir,
    number::{Base, IntKind},
    span::Span,
};

pub fn lower_ast(unit: &mut hir::Unit, module: usize, ast: &ast::Module) -> miette::Result<()> {
    for decl in &ast.decls {
        if let ast::Decl::Type(ast::Type::Adt(adt)) = decl {
            let id = unit.push_adt(hir::Adt {
                name: adt.name,
                generics: Vec::new(),
                variants: Vec::new(),
                span: adt.span,
            });

            unit.modules[module].adts.insert(adt.name, id);
        }
    }

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

            unit.push_func(hir::Func {
                name: func.name,
                generics,
                input,
                output,
                locals: Vec::new(),
                body: hir::Expr {
                    kind: hir::ExprKind::Void,
                    ty: hir::Ty::void(),
                },
            });

            unit.modules[module].funcs.insert(func.name, id);
        }
    }

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
                unit.unify(body.ty.clone(), unit.funcs[id].output.clone());

                unit.funcs[id].body = body;
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

    match unit.modules[current].adts.get(name) {
        Some(&id) => Ok(id),
        None => Err(miette::miette!(
            severity = Severity::Error,
            code = "invalid::path",
            labels = vec![path.span.label("here")],
            "invalid item `{}`",
            name
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
        for (j, variant) in adt.variants.iter().enumerate() {
            if variant.name == *name {
                return Ok(Item::Variant(i, j));
            }
        }
    }

    match unit.modules[current].funcs.get(name) {
        Some(&id) => Ok(Item::Func(id)),
        None => Err(miette::miette!(
            severity = Severity::Error,
            code = "invalid::path",
            labels = vec![path.span.label("here")],
            "invalid item `{}`",
            name
        )
        .with_source_code(path.span)),
    }
}

struct BodyCx<'a> {
    unit: &'a mut hir::Unit,
    arguments: &'a [hir::Argument],
    module: usize,
    locals: Vec<hir::Local>,
    scope: Vec<usize>,
}

fn lower_expr(cx: &mut BodyCx, ast: &ast::Expr) -> miette::Result<hir::Expr> {
    match ast {
        ast::Expr::Int(negative, base, digits, span) => {
            lower_int(cx, *negative, *base, digits, *span)
        }
        ast::Expr::Paren(expr, _) => lower_expr(cx, expr),
        ast::Expr::Item(path) => lower_item(cx, path),
        ast::Expr::Tuple(exprs) => lower_tuple(cx, exprs),
        ast::Expr::List(_) => todo!(),
        ast::Expr::Block(block) => lower_block(cx, block),
        ast::Expr::Field(expr, name) => lower_field(cx, expr, name),
        ast::Expr::Call(func, args) => lower_call(cx, func, args),
        ast::Expr::Pipe(expr, exprs) => lower_pipe(cx, expr, exprs),
        ast::Expr::Let(name, expr) => lower_let(cx, name, expr),
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

fn lower_item(cx: &mut BodyCx, path: &ast::Path) -> miette::Result<hir::Expr> {
    if path.segments.len() == 1 {
        let name = path
            .segments
            .first()
            .expect("path should have at least one segment");

        for &id in &cx.scope {
            if cx.locals[id].name == *name {
                let ty = cx.locals[id].ty.clone();
                let kind = hir::ExprKind::Local(id);
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
            let mut generics = generics.into_iter().enumerate().collect();

            for field in variant.fields.iter() {
                let ty = cx.unit.env.use_ty(&mut generics, &field.ty);
                parts.push(ty);
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
        let argument = argument.as_ref().unwrap();

        let arg = lower_expr(cx, argument)?;
        tys.push(arg.ty.clone());
        args.push(arg);
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

fn lower_let(cx: &mut BodyCx, name: &'static str, expr: &ast::Expr) -> miette::Result<hir::Expr> {
    let value = lower_expr(cx, expr)?;

    let id = cx.locals.len();
    let ty = value.ty.clone();

    cx.locals.push(hir::Local { name, ty });
    cx.scope.push(id);

    let kind = hir::ExprKind::Let(id, Box::new(value));
    let ty = hir::Ty::void();
    Ok(hir::Expr { kind, ty })
}
