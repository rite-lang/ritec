use std::mem;

use miette::Severity;

use crate::{ast, hir, number::Base, span::Span};

fn is_language_result(adt: &ast::Adt) -> bool {
    for decorator in &adt.decorators {
        let is_language = decorator.name == "language";
        let is_result = decorator.args.first().map_or(false, |a| a == "Result");

        if is_language && is_result {
            return true;
        }
    }

    false
}

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

            let (generics, _) = lower_generics(adt.generics.as_ref());

            let id = unit.push_adt(hir::Adt {
                decorators: adt.decorators.clone(),
                name: adt.name,
                generics,
                variants: Vec::new(),
                span: adt.span,
            });

            let import = hir::Import {
                vis,
                kind: hir::ImportKind::Adt(id),
                span: adt.span,
            };

            if is_language_result(adt) {
                unit.result = Some(id);
            }

            unit.modules[module].imports.insert(adt.name, import);
        }

        if let ast::Decl::Type(ast::Type::Single(single)) = decl {
            let vis = match single.vis {
                ast::Vis::Public => hir::Vis::Public,
                ast::Vis::Private => hir::Vis::Private,
            };

            let (generics, _) = lower_generics(single.generics.as_ref());

            let id = unit.push_adt(hir::Adt {
                decorators: single.decorators.clone(),
                name: single.name,
                generics,
                variants: Vec::new(),
                span: single.span,
            });

            let import = hir::Import {
                vis,
                kind: hir::ImportKind::Adt(id),
                span: single.span,
            };

            unit.modules[module].imports.insert(single.name, import);
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
            let vis = match import.vis {
                ast::Vis::Public => hir::Vis::Public,
                ast::Vis::Private => hir::Vis::Private,
            };

            let name = import.path.item();
            let imported = find_import(unit, module, &import.path)?;

            let import = hir::Import {
                vis,
                kind: imported.kind.clone(),
                span: import.span,
            };

            unit.modules[module].imports.insert(name, import);
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
            let import = &unit.modules[module].imports[adt.name];
            let hir::ImportKind::Adt(id) = import.kind else {
                unreachable!("expected adt");
            };

            let (mut generics, new_generics) = lower_generics(adt.generics.as_ref());

            for variant in &adt.variants {
                let mut cx = TyCx {
                    unit,
                    module,
                    generics: &mut generics,
                    new_generics,
                    inferring: false,
                    func: None,
                };

                let fields = lower_fields(&mut cx, variant.fields.iter())?;

                let variant = hir::Variant {
                    name: variant.name,
                    fields,
                };

                unit.adts[id].variants.push(variant);
            }

            unit.adts[id].generics = generics;
        }

        if let ast::Decl::Type(ast::Type::Single(single)) = decl {
            let import = &unit.modules[module].imports[single.name];
            let hir::ImportKind::Adt(id) = import.kind else {
                unreachable!("expected adt");
            };

            let (mut generics, new_generics) = lower_generics(single.generics.as_ref());

            let mut cx = TyCx {
                unit,
                module,
                generics: &mut generics,
                new_generics,
                inferring: false,
                func: None,
            };

            let fields = lower_fields(&mut cx, single.fields.iter())?;

            let variant = hir::Variant {
                name: single.name,
                fields,
            };

            unit.adts[id].variants.push(variant);
            unit.adts[id].generics = generics;
        }
    }

    Ok(())
}

fn lower_generics(generics: Option<&Vec<ast::Generic>>) -> (Vec<hir::Generic>, bool) {
    match generics {
        Some(generics) => {
            let generics = generics
                .iter()
                .map(|generic| hir::Generic {
                    name: generic.name,
                    span: generic.span,
                })
                .collect();

            (generics, false)
        }
        None => (Vec::new(), true),
    }
}

fn lower_fields<'a>(
    cx: &mut TyCx,
    iter: impl Iterator<Item = &'a ast::Field>,
) -> miette::Result<Vec<hir::Argument>> {
    let mut fields: Vec<hir::Argument> = Vec::new();

    for field in iter {
        if let Some(other) = fields.iter().find(|f| f.name == field.name) {
            return Err(miette::miette!(
                severity = Severity::Error,
                code = "invalid::field",
                labels = vec![other.span.label("here"), field.span.label("and here")],
                "field `{}` already exists",
                field.name
            )
            .with_source_code(field.span));
        }

        let ty = match field.ty {
            Some(ref ty) => lower_ty(cx, ty, field.span)?,
            None => {
                let index = cx.add_generic(field.name, field.span)?;
                hir::Ty::Partial(hir::Part::Generic(index, None), Vec::new(), field.span)
            }
        };

        fields.push(hir::Argument {
            name: field.name,
            ty,
            span: field.span,
        });
    }

    Ok(fields)
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
                inferring: true,
                func: Some(id),
            };

            let mut input: Vec<hir::Argument> = Vec::new();

            for arg in &func.input {
                if let Some(other) = input.iter().find(|a| a.name == arg.name) {
                    return Err(miette::miette!(
                        severity = Severity::Error,
                        code = "invalid::argument",
                        labels = vec![other.span.label("here"), arg.span.label("and here")],
                        "argument `{}` already exists",
                        arg.name
                    )
                    .with_source_code(arg.span));
                }

                input.push(lower_argument(&mut cx, arg)?);
            }

            let output = lower_output(&mut cx, &func.output, func.span)?;

            let vis = match func.vis {
                ast::Vis::Public => hir::Vis::Public,
                ast::Vis::Private => hir::Vis::Private,
            };

            unit.push_func(hir::Func {
                decorators: func.decorators.clone(),
                name: func.name,
                generics,
                input,
                output,
                locals: Vec::new(),
                captures: Vec::new(),
                body: hir::Expr {
                    kind: hir::ExprKind::Void,
                    ty: hir::Ty::void(func.span),
                },
            });

            let import = hir::Import {
                vis,
                kind: hir::ImportKind::Func(id),
                span: func.span,
            };

            unit.modules[module].imports.insert(func.name, import);
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
            let import = &unit.modules[module].imports[func.name];
            let hir::ImportKind::Func(id) = import.kind else {
                unreachable!("expected func");
            };

            // Lower the body of the function.
            if let Some(body) = &func.body {
                let mut generics = mem::take(&mut unit.funcs[id].generics);

                let arguments = &unit.funcs[id].input.clone();
                let output = unit.funcs[id].output.clone();

                let mut cx = BodyCx {
                    unit,
                    arguments,
                    generics: &mut generics,
                    output: &output,
                    module,
                    locals: Vec::new(),
                    scope: Vec::new(),
                    capture: None,
                };

                let body = lower_expr(&mut cx, body)?;

                unit.funcs[id].locals = cx.locals;
                unit.funcs[id].body = body;
                unit.funcs[id].generics = generics;

                unit.unify(
                    unit.funcs[id].body.ty.clone(),
                    unit.funcs[id].output.clone(),
                    func.span,
                );
            } else {
                unit.unify(
                    hir::Ty::void(func.span),
                    unit.funcs[id].output.clone(),
                    func.span,
                );
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
    inferring: bool,
    func: Option<usize>,
}

impl<'a> TyCx<'a> {
    pub fn add_generic(&mut self, name: &'static str, span: Span) -> miette::Result<usize> {
        if let Some(generic) = self.generics.iter().find(|g| g.name == name) {
            return Err(miette::miette!(
                severity = Severity::Error,
                code = "invalid::generic",
                labels = vec![generic.span.label("here"), span.label("and here")],
                "generic `{}` already exists",
                name
            )
            .with_source_code(span));
        }

        let index = self.generics.len();
        self.generics.push(hir::Generic { name, span });
        Ok(index)
    }
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
        Some(ref ty) => lower_ty(cx, ty, ast.span)?,
        None => hir::Ty::Inferred(hir::Tid::new(), hir::Inferred::Any, cx.func, ast.span),
    };

    Ok(hir::Argument {
        name: ast.name,
        ty,
        span: ast.span,
    })
}

fn lower_output(cx: &mut TyCx, output: &Option<ast::Ty>, span: Span) -> miette::Result<hir::Ty> {
    match output {
        Some(ty) => Ok(lower_ty(cx, ty, span)?),
        None => Ok(hir::Ty::Inferred(
            hir::Tid::new(),
            hir::Inferred::Any,
            cx.func,
            span,
        )),
    }
}

fn lower_ty(cx: &mut TyCx, ty: &ast::Ty, span: Span) -> miette::Result<hir::Ty> {
    match ty {
        ast::Ty::Void => Ok(hir::Ty::void(span)),
        ast::Ty::Bool => Ok(hir::Ty::bool(span)),
        ast::Ty::Str => Ok(hir::Ty::string(span)),
        ast::Ty::Inferred => {
            if !cx.inferring {
                return Err(miette::miette!(
                    severity = Severity::Error,
                    code = "invalid::infer",
                    "cannot infer type"
                ));
            }

            Ok(hir::Ty::Inferred(
                hir::Tid::new(),
                hir::Inferred::Any,
                cx.func,
                span,
            ))
        }
        ast::Ty::Int(kind) => Ok(hir::Ty::Partial(hir::Part::Int(*kind), Vec::new(), span)),
        ast::Ty::Ref(ty) => {
            let ty = lower_ty(cx, ty, span)?;
            Ok(hir::Ty::Partial(hir::Part::Ref, vec![ty], span))
        }
        ast::Ty::Tuple(tys) => {
            let mut args = Vec::new();

            for ty in tys {
                args.push(lower_ty(cx, ty, span)?);
            }

            Ok(hir::Ty::Partial(hir::Part::Tuple, args, span))
        }
        ast::Ty::Item(path, generics) => {
            let index = find_adt(cx.unit, cx.module, path)?;

            let generics = match generics {
                Some(generics) => {
                    if generics.len() != cx.unit.adts[index].generics.len() {
                        return Err(miette::miette!(
                            severity = Severity::Error,
                            code = "invalid::generic",
                            labels = [path.span.label("here")],
                            "expected {} generics, found {}",
                            cx.unit.adts[index].generics.len(),
                            generics.len()
                        )
                        .with_source_code(path.span));
                    }

                    generics
                        .iter()
                        .map(|ty| lower_ty(cx, ty, span))
                        .collect::<miette::Result<Vec<_>>>()?
                }
                None => {
                    if !cx.inferring && !cx.unit.adts[index].generics.is_empty() {
                        return Err(miette::miette!(
                            severity = Severity::Error,
                            code = "invalid::generic",
                            labels = [path.span.label("here")],
                            "missing generics"
                        )
                        .with_source_code(path.span));
                    }

                    cx.unit.adts[index]
                        .generics
                        .iter()
                        .map(|_| {
                            hir::Ty::Inferred(hir::Tid::new(), hir::Inferred::Any, cx.func, span)
                        })
                        .collect()
                }
            };

            Ok(hir::Ty::Partial(hir::Part::Adt(index), generics, span))
        }
        ast::Ty::List(ty) => {
            let ty = lower_ty(cx, ty, span)?;
            Ok(hir::Ty::Partial(hir::Part::List, vec![ty], span))
        }
        ast::Ty::Func(input, output) => {
            let mut args = Vec::new();

            for ty in input {
                args.push(lower_ty(cx, ty, span)?);
            }

            match output {
                Some(output) => args.push(lower_ty(cx, output, span)?),
                None => args.push(hir::Ty::Inferred(
                    hir::Tid::new(),
                    hir::Inferred::Any,
                    cx.func,
                    span,
                )),
            }

            Ok(hir::Ty::Partial(hir::Part::Func, args, span))
        }
        ast::Ty::Generic(generic) => {
            if let Some(index) = cx.generics.iter().position(|g| g.name == generic.name) {
                return Ok(hir::Ty::Partial(
                    hir::Part::Generic(index, cx.func),
                    Vec::new(),
                    span,
                ));
            }

            if !cx.new_generics {
                return Err(miette::miette!(
                    severity = Severity::Error,
                    code = "invalid::generic",
                    "generic `{}` not found",
                    generic.name
                ));
            }

            let index = cx.add_generic(generic.name, generic.span)?;
            Ok(hir::Ty::Partial(
                hir::Part::Generic(index, cx.func),
                Vec::new(),
                span,
            ))
        }
    }
}

#[derive(Debug)]
enum Item {
    Func(usize),
    Variant(usize, usize),
}

fn find_import<'a>(
    unit: &'a hir::Unit,
    module: usize,
    path: &ast::Path,
) -> miette::Result<&'a hir::Import> {
    let mut current = module;

    for segment in path.segments[0..path.segments.len() - 1].iter() {
        match unit.modules[current].get_import(segment, current == module) {
            Some(hir::Import {
                kind: hir::ImportKind::Module(next),
                ..
            }) => current = *next,
            Some(_) | None => {
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

    match unit.modules[current].get_import(name, current == module) {
        Some(import) => Ok(import),
        None => Err(miette::miette!(
            severity = Severity::Error,
            code = "invalid::path",
            labels = vec![path.span.label("here")],
            "module not found `{}`",
            name
        )
        .with_source_code(path.span)),
    }
}

fn find_adt(unit: &hir::Unit, module: usize, path: &ast::Path) -> miette::Result<usize> {
    match find_import(unit, module, path)? {
        hir::Import {
            kind: hir::ImportKind::Adt(adt),
            ..
        } => Ok(*adt),
        _ => Err(miette::miette!(
            severity = Severity::Error,
            code = "invalid::path",
            labels = vec![path.span.label("here")],
            "expected adt"
        )
        .with_source_code(path.span)),
    }
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

    let name = path
        .segments
        .last()
        .expect("path should have at least one segment");

    for (i, segment) in path
        .segments
        .iter()
        .take(path.segments.len() - 1)
        .enumerate()
    {
        match unit.modules[current].get_import(segment, current == module) {
            Some(hir::Import {
                kind: hir::ImportKind::Module(next),
                ..
            }) => current = *next,
            Some(hir::Import {
                kind: hir::ImportKind::Adt(adt),
                ..
            }) if i == path.segments.len() - 2 => {
                let variant = unit.adts[*adt].find_variant(name)?;
                return Ok(Item::Variant(*adt, variant));
            }
            Some(_) | None => {
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

    for import in unit.modules[current].imports.values() {
        if current != module && import.vis == hir::Vis::Private {
            continue;
        }

        if let hir::ImportKind::Adt(adt) = import.kind {
            for (j, variant) in unit.adts[adt].variants.iter().enumerate() {
                if variant.name == *name {
                    return Ok(Item::Variant(adt, j));
                }
            }
        }
    }

    if let Some(hir::Import {
        kind: hir::ImportKind::Func(func),
        ..
    }) = unit.modules[current].get_import(name, current == module)
    {
        return Ok(Item::Func(*func));
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
    generics: &'a mut Vec<hir::Generic>,
    output: &'a hir::Ty,
    module: usize,
    locals: Vec<hir::Local>,
    scope: Vec<(String, usize)>,
    capture: Option<CaptureCx>,
}

impl<'a> BodyCx<'a> {
    fn as_ty_cx(&mut self) -> TyCx {
        TyCx {
            unit: self.unit,
            module: self.module,
            generics: self.generics,
            new_generics: false,
            inferring: true,
            func: None,
        }
    }
}

struct CaptureCx {
    arguments: Vec<hir::Argument>,
    locals: Vec<hir::Local>,
    scope: Vec<(String, usize)>,
    captures: Vec<(Capture, hir::Ty)>,
    parent: Option<Box<CaptureCx>>,
}

impl CaptureCx {
    fn add_capture(&mut self, capture: Capture, ty: hir::Ty) -> usize {
        if let Some(index) = self.captures.iter().position(|(c, _)| *c == capture) {
            return index;
        }

        let index = self.captures.len();
        self.captures.push((capture, ty));
        index
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Capture {
    Argument(usize),
    Captured(usize),
    Local(usize),
}

fn lower_expr(cx: &mut BodyCx, ast: &ast::Expr) -> miette::Result<hir::Expr> {
    match ast {
        ast::Expr::Void(span) => lower_void(cx, *span),
        ast::Expr::Int(negative, base, digits, span) => {
            lower_int(cx, *negative, *base, digits, *span)
        }
        ast::Expr::Bool(value, span) => lower_bool(cx, *value, *span),
        ast::Expr::String(value, span) => lower_string(cx, value, *span),
        ast::Expr::Paren(expr, _) => lower_expr(cx, expr),
        ast::Expr::Item(path) => lower_item(cx, path),
        ast::Expr::Tuple(exprs) => lower_tuple(cx, exprs),
        ast::Expr::List(exprs, rest, span) => lower_list(cx, exprs, rest, *span),
        ast::Expr::Block(block) => lower_block(cx, block),
        ast::Expr::As(expr, ty) => lower_as(cx, expr, ty),
        ast::Expr::Field(expr, name) => lower_field(cx, expr, name),
        ast::Expr::Call(func, args, spread) => lower_call(cx, func, args, spread.as_deref()),
        ast::Expr::Pipe(expr, exprs) => lower_pipe(cx, expr, exprs),
        ast::Expr::Binary(op, lhs, rhs, span) => lower_binary(cx, *op, lhs, rhs, *span),
        ast::Expr::Unary(op, expr, span) => lower_unary(cx, *op, expr, *span),
        ast::Expr::Let(pat, ty, expr) => lower_let(cx, pat, ty.as_ref(), expr, expr.span()),
        ast::Expr::Mut(name, ty, expr) => lower_mut(cx, name, ty.as_ref(), expr),
        ast::Expr::LetAssert(pat, ty, expr) => lower_let_assert(cx, pat, ty.as_ref(), expr),
        ast::Expr::Assign(lhs, rhs) => lower_assign(cx, lhs, rhs),
        ast::Expr::Match(input, arms, span) => lower_match(cx, input, arms, *span),
        ast::Expr::Return(expr, span) => lower_return(cx, expr.as_deref(), *span),
        ast::Expr::Closure(args, body) => lower_closure(cx, args, body),
        ast::Expr::Panic(message, span) => lower_panic(cx, message, *span),
        ast::Expr::Assert(expr, message, span) => lower_assert(cx, expr, *message, *span),
        ast::Expr::Try(expr, span) => lower_try(cx, expr, *span),
    }
}

fn lower_void(_cx: &mut BodyCx, span: Span) -> miette::Result<hir::Expr> {
    let ty = hir::Ty::void(span);
    let kind = hir::ExprKind::Void;
    Ok(hir::Expr { kind, ty })
}

fn lower_int(
    _cx: &mut BodyCx,
    negative: bool,
    base: Base,
    digits: &[u8],
    span: Span,
) -> miette::Result<hir::Expr> {
    let kind = match negative {
        true => hir::Inferred::Signed,
        false => hir::Inferred::Unsigned,
    };

    let ty = hir::Ty::inferred(kind, span);

    let kind = hir::ExprKind::Int(negative, base, digits.to_vec());
    Ok(hir::Expr { kind, ty })
}

fn lower_bool(_cx: &mut BodyCx, value: bool, span: Span) -> miette::Result<hir::Expr> {
    let ty = hir::Ty::bool(span);
    let kind = hir::ExprKind::Bool(value);
    Ok(hir::Expr { kind, ty })
}

fn lower_string(_cx: &mut BodyCx, value: &'static str, span: Span) -> miette::Result<hir::Expr> {
    let ty = hir::Ty::string(span);
    let kind = hir::ExprKind::String(value);
    Ok(hir::Expr { kind, ty })
}

fn lower_panic(_cx: &mut BodyCx, message: &'static str, span: Span) -> miette::Result<hir::Expr> {
    let ty = hir::Ty::inferred(hir::Inferred::Any, span);
    let kind = hir::ExprKind::Panic(message);
    Ok(hir::Expr { kind, ty })
}

fn lower_assert(
    cx: &mut BodyCx,
    expr: &ast::Expr,
    message: Option<&'static str>,
    span: Span,
) -> miette::Result<hir::Expr> {
    let expr = lower_expr(cx, expr)?;

    cx.unit.unify(expr.ty.clone(), hir::Ty::bool(span), span);

    let success = hir::Expr {
        kind: hir::ExprKind::Void,
        ty: hir::Ty::void(span),
    };

    let failure = hir::Expr {
        kind: hir::ExprKind::Panic(message.unwrap_or("assertion failed")),
        ty: hir::Ty::void(span),
    };

    let r#match = hir::Match::Bool(Box::new(success), Box::new(failure));
    let kind = hir::ExprKind::Match(Box::new(expr), r#match);
    let ty = hir::Ty::void(span);

    Ok(hir::Expr { kind, ty })
}

fn lower_try(cx: &mut BodyCx, expr: &ast::Expr, span: Span) -> miette::Result<hir::Expr> {
    let result = cx.unit.result.expect("result type should be set");

    let input = lower_expr(cx, expr)?;

    let local = cx.locals.len();
    cx.locals.push(hir::Local {
        name: "",
        ty: input.ty.clone(),
        mutable: false,
    });

    let value = hir::Expr {
        kind: hir::ExprKind::Local(local),
        ty: input.ty.clone(),
    };

    let ok_ty = hir::Ty::any(span);
    let err_ty = hir::Ty::any(span);

    let result_ty = hir::Ty::Partial(
        hir::Part::Adt(result),
        vec![ok_ty.clone(), err_ty.clone()],
        span,
    );

    let ok_value = hir::Expr {
        kind: hir::ExprKind::VariantField(Box::new(value.clone()), 0, 0),
        ty: ok_ty.clone(),
    };

    let err_value = hir::Expr {
        kind: hir::ExprKind::VariantField(Box::new(value.clone()), 1, 0),
        ty: err_ty.clone(),
    };

    let err_value = hir::Expr {
        kind: hir::ExprKind::VariantNew(result, 1, vec![err_value]),
        ty: hir::Ty::Partial(
            hir::Part::Adt(result),
            vec![hir::Ty::any(span), err_ty.clone()],
            span,
        ),
    };

    cx.unit.unify(input.ty.clone(), result_ty, span);
    cx.unit.unify(cx.output.clone(), err_value.ty.clone(), span);

    let err_expr = hir::Expr {
        kind: hir::ExprKind::Return(Box::new(err_value)),
        ty: hir::Ty::void(span),
    };

    let r#match = hir::Match::Adt(result, vec![Some(ok_value), Some(err_expr)], None);
    let expr = hir::Expr {
        kind: hir::ExprKind::Match(Box::new(value), r#match),
        ty: ok_ty.clone(),
    };

    let exprs = vec![
        hir::Expr {
            kind: hir::ExprKind::Let(local, Box::new(input)),
            ty: hir::Ty::void(span),
        },
        expr,
    ];

    Ok(hir::Expr {
        kind: hir::ExprKind::Block(exprs),
        ty: ok_ty,
    })
}

fn lower_item(cx: &mut BodyCx, path: &ast::Path) -> miette::Result<hir::Expr> {
    if path.segments.len() == 1 {
        let name = path
            .segments
            .first()
            .expect("path should have at least one segment");

        for (scope_name, id) in cx.scope.iter().rev() {
            if scope_name == name {
                if !cx.locals[*id].mutable {
                    let ty = cx.locals[*id].ty.clone();
                    let kind = hir::ExprKind::Local(*id);
                    return Ok(hir::Expr { kind, ty });
                }

                let ty = cx.locals[*id].ty.clone();
                let kind = hir::ExprKind::Local(*id);
                let expr = hir::Expr { kind, ty };

                let ty = hir::Ty::any(path.span);
                let outer = cx.locals[*id].ty.clone();
                (cx.unit).unify(hir::Ty::new_ref(ty.clone(), path.span), outer, path.span);

                let kind = hir::ExprKind::Deref(Box::new(expr));
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

        if let Some(ref mut capture_cx) = cx.capture {
            if let Some((mutable, capture, ty)) = find_item_capture(capture_cx, name) {
                if !mutable {
                    let kind = hir::ExprKind::Capture(capture);
                    return Ok(hir::Expr { kind, ty });
                }

                let outer = ty.clone();
                let kind = hir::ExprKind::Capture(capture);
                let expr = hir::Expr { kind, ty };

                let ty = hir::Ty::any(path.span);
                (cx.unit).unify(hir::Ty::new_ref(ty.clone(), path.span), outer, path.span);

                let kind = hir::ExprKind::Deref(Box::new(expr));
                return Ok(hir::Expr { kind, ty });
            }
        }
    }

    let item = resolve_item(cx.unit, cx.module, path)?;

    match item {
        Item::Func(id) => {
            let mut parts = Vec::new();

            let column = cx.unit.env.next_column(id);

            for argument in &cx.unit.funcs[id].input {
                parts.push(cx.unit.env.use_ty(column, &argument.ty, path.span));
            }

            let output = &cx.unit.funcs[id].output;
            parts.push(cx.unit.env.use_ty(column, output, path.span));

            let kind = hir::ExprKind::Func(id);
            let ty = hir::Ty::Partial(hir::Part::Func, parts, path.span);

            Ok(hir::Expr { kind, ty })
        }
        Item::Variant(adt, index) => {
            let variant = &cx.unit.adts[adt].variants[index];

            let generics = cx.unit.adts[adt]
                .generics
                .iter()
                .map(|_| hir::Ty::inferred(hir::Inferred::Any, path.span))
                .collect();

            if variant.fields.is_empty() {
                let ty = hir::Ty::Partial(hir::Part::Adt(adt), generics, path.span);
                let kind = hir::ExprKind::Variant(adt, index);
                return Ok(hir::Expr { kind, ty });
            }

            let output = hir::Ty::Partial(hir::Part::Adt(adt), generics.clone(), path.span);

            let mut parts = Vec::new();

            for field in variant.fields.iter() {
                parts.push(field.ty.specialize(&generics));
            }

            parts.push(output);

            let ty = hir::Ty::Partial(hir::Part::Func, parts, path.span);
            let kind = hir::ExprKind::Variant(adt, index);
            Ok(hir::Expr { kind, ty })
        }
    }
}

fn find_item_capture(cx: &mut CaptureCx, name: &str) -> Option<(bool, usize, hir::Ty)> {
    for (i, local) in cx.locals.clone().into_iter().enumerate() {
        if local.name == name {
            let ty = local.ty.clone();
            let index = cx.add_capture(Capture::Local(i), ty.clone());
            return Some((local.mutable, index, ty));
        }
    }

    for (i, argument) in cx.arguments.clone().into_iter().enumerate() {
        if argument.name == name {
            let ty = argument.ty.clone();
            let index = cx.add_capture(Capture::Argument(i), ty.clone());
            return Some((false, index, ty));
        }
    }

    if let Some(ref mut parent) = cx.parent {
        if let Some((mutable, capture, ty)) = find_item_capture(parent, name) {
            let index = cx.add_capture(Capture::Captured(capture), ty.clone());
            return Some((mutable, index, ty));
        }
    }

    None
}

fn lower_tuple(cx: &mut BodyCx, exprs: &[ast::Expr]) -> miette::Result<hir::Expr> {
    let mut args = Vec::new();
    let mut tys = Vec::new();

    let start_span = exprs.first().unwrap().span();
    let end_span = exprs.last().unwrap().span();
    let span = start_span.join(end_span);

    for expr in exprs {
        let arg = lower_expr(cx, expr)?;
        tys.push(arg.ty.clone());
        args.push(arg);
    }

    let ty = hir::Ty::Partial(hir::Part::Tuple, tys, span);
    let kind = hir::ExprKind::Tuple(args);
    Ok(hir::Expr { kind, ty })
}

fn lower_list(
    cx: &mut BodyCx,
    exprs: &[ast::Expr],
    rest: &Option<Box<ast::Expr>>,
    span: Span,
) -> miette::Result<hir::Expr> {
    let mut args = Vec::new();
    let ty = hir::Ty::any(span);

    for expr in exprs {
        let arg = lower_expr(cx, expr)?;
        cx.unit.unify(arg.ty.clone(), ty.clone(), span);

        args.push(arg);
    }

    let ty = hir::Ty::Partial(hir::Part::List, vec![ty], span);

    let rest = match rest {
        Some(rest) => {
            let rest = lower_expr(cx, rest)?;
            cx.unit.unify(rest.ty.clone(), ty.clone(), span);
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

fn lower_as(cx: &mut BodyCx, expr: &ast::Expr, ty: &ast::Ty) -> miette::Result<hir::Expr> {
    let span = expr.span();
    let expr = lower_expr(cx, expr)?;
    let ty = lower_ty(&mut cx.as_ty_cx(), ty, span)?;

    let kind = hir::ExprKind::As(Box::new(expr), ty.clone());
    Ok(hir::Expr { kind, ty })
}

fn lower_field(cx: &mut BodyCx, expr: &ast::Expr, name: &'static str) -> miette::Result<hir::Expr> {
    let span = expr.span();
    let expr = lower_expr(cx, expr)?;
    let ty = hir::Ty::Field(Box::new(expr.ty.clone()), name, span);
    let kind = hir::ExprKind::Field(Box::new(expr), name);
    cx.unit.normalize(ty.clone(), span);
    Ok(hir::Expr { kind, ty })
}

fn lower_call(
    cx: &mut BodyCx,
    func: &ast::Expr,
    arguments: &[ast::CallArgument],
    spread: Option<&ast::Expr>,
) -> miette::Result<hir::Expr> {
    let span = func.span();
    let func = lower_expr(cx, func)?;

    let (args, tys) = match func.kind {
        hir::ExprKind::Func(index) => {
            if spread.is_some() {
                return Err(miette::miette!(
                    severity = Severity::Error,
                    code = "invalid::call",
                    labels = vec![span.label("here")],
                    "spread arguments are only supported when calling type constructors"
                )
                .with_source_code(span));
            }

            let (args, tys) = lower_named_arguments(
                cx,
                |cx, name| cx.unit.funcs[index].find_argument(name),
                arguments,
                cx.unit.funcs[index].input.len(),
                span,
            )?;

            (
                args.into_iter().flatten().collect(),
                tys.into_iter().flatten().collect(),
            )
        }
        hir::ExprKind::Variant(adt, index) => {
            let (mut args, mut tys) = lower_named_arguments(
                cx,
                |cx, name| cx.unit.adts[adt].variants[index].find_field(name),
                arguments,
                cx.unit.adts[adt].variants[index].fields.len(),
                span,
            )?;

            if let Some(spread) = spread {
                let spread = lower_expr(cx, spread)?;

                for (i, arg) in args.iter_mut().enumerate() {
                    if arg.is_none() {
                        let field_name = cx.unit.adts[adt].variants[index].fields[i].name;

                        if !cx.unit.adts[adt].is_field_assessible(i) {
                            return Err(miette::miette!(
                                severity = Severity::Error,
                                code = "invalid::call",
                                labels = vec![span.label("here")],
                                "field `{}` is not accessible",
                                field_name
                            )
                            .with_source_code(span));
                        }

                        let kind = hir::ExprKind::Field(Box::new(spread.clone()), field_name);
                        let ty = hir::Ty::Field(Box::new(spread.ty.clone()), field_name, span);

                        tys[i] = Some(Some(ty.clone()));
                        *arg = Some(Some(hir::Expr { kind, ty }));
                    }
                }
            }

            (
                args.into_iter().flatten().collect(),
                tys.into_iter().flatten().collect(),
            )
        }
        _ => {
            if spread.is_some() {
                return Err(miette::miette!(
                    severity = Severity::Error,
                    code = "invalid::call",
                    labels = vec![span.label("here")],
                    "spread arguments are only supported when calling type constructors"
                )
                .with_source_code(span));
            }

            let mut args = Vec::new();
            let mut tys = Vec::new();

            for argument in arguments {
                match argument {
                    ast::CallArgument::Positional(Some(argument)) => {
                        let arg = lower_expr(cx, argument)?;
                        tys.push(Some(arg.ty.clone()));
                        args.push(Some(arg));
                    }
                    ast::CallArgument::Positional(None) => {
                        args.push(None);
                        tys.push(None);
                    }
                    ast::CallArgument::Named(_, _) => {
                        return Err(miette::miette!(
                            severity = Severity::Error,
                            code = "invalid::call",
                            labels = vec![span.label("here")],
                            "named arguments are not supported when calling dynamic functions"
                        )
                        .with_source_code(span));
                    }
                }
            }

            (args, tys)
        }
    };

    let ty = hir::Ty::Call(Box::new(func.ty.clone()), tys, span);
    let kind = hir::ExprKind::Call(Box::new(func), args);
    cx.unit.normalize(ty.clone(), span);
    Ok(hir::Expr { kind, ty })
}

type Args = Vec<Option<Option<hir::Expr>>>;
type Tys = Vec<Option<Option<hir::Ty>>>;

fn lower_named_arguments(
    cx: &mut BodyCx,
    find_argument: impl Fn(&BodyCx, &str) -> Option<usize>,
    arguments: &[ast::CallArgument],
    argument_count: usize,
    span: Span,
) -> miette::Result<(Args, Tys)> {
    if arguments.len() > argument_count {
        return Err(miette::miette!(
            severity = Severity::Error,
            code = "invalid::call",
            labels = vec![span.label("here")],
            "too many arguments"
        )
        .with_source_code(span));
    }

    let mut tys = vec![None; argument_count];
    let mut args = vec![None; argument_count];

    let mut has_named = false;

    for (i, argument) in arguments.iter().enumerate() {
        match argument {
            ast::CallArgument::Positional(Some(argument)) => {
                if has_named {
                    return Err(miette::miette!(
                        severity = Severity::Error,
                        code = "invalid::call",
                        labels = vec![span.label("here")],
                        "positional arguments must come before named arguments"
                    )
                    .with_source_code(span));
                }

                let arg = lower_expr(cx, argument)?;
                tys[i] = Some(Some(arg.ty.clone()));
                args[i] = Some(Some(arg));
            }
            ast::CallArgument::Positional(None) => {
                tys[i] = Some(None);
                args[i] = Some(None);
            }
            ast::CallArgument::Named(name, argument) => {
                let Some(index) = find_argument(cx, name) else {
                    return Err(miette::miette!(
                        severity = Severity::Error,
                        code = "invalid::call",
                        labels = vec![span.label("here")],
                        "argument `{}` not found",
                        name
                    )
                    .with_source_code(span));
                };

                if args[index].is_some() {
                    return Err(miette::miette!(
                        severity = Severity::Error,
                        code = "invalid::call",
                        labels = vec![span.label("here")],
                        "argument `{}` specified more than once",
                        name
                    )
                    .with_source_code(span));
                }

                has_named = true;
                let arg = lower_expr(cx, argument)?;
                tys[index] = Some(Some(arg.ty.clone()));
                args[index] = Some(Some(arg));
            }
        }
    }

    Ok((args, tys))
}

fn lower_pipe(
    cx: &mut BodyCx,
    pipee: &ast::Expr,
    exprs: &[ast::Expr],
) -> miette::Result<hir::Expr> {
    let span = pipee.span();
    let mut pipee = lower_expr(cx, pipee)?;
    let mut ty = pipee.ty.clone();

    for expr in exprs {
        let (expr, tys, args) = match expr {
            ast::Expr::Call(func, arguments, spread) => {
                if spread.is_some() {
                    return Err(miette::miette!(
                        severity = Severity::Error,
                        code = "invalid::call",
                        labels = vec![span.label("here")],
                        "spread arguments are not supported when piping"
                    )
                    .with_source_code(span));
                }

                let expr = lower_expr(cx, func)?;
                let mut tys = Vec::new();
                let mut args = Vec::new();

                for arg in arguments {
                    match arg {
                        ast::CallArgument::Positional(Some(arg)) => {
                            let arg = lower_expr(cx, arg)?;
                            tys.push(Some(arg.ty.clone()));
                            args.push(Some(arg));
                        }
                        ast::CallArgument::Positional(None) => {
                            tys.push(None);
                            args.push(None);
                        }
                        ast::CallArgument::Named(_, _) => {
                            return Err(miette::miette!(
                                severity = Severity::Error,
                                code = "invalid::call",
                                labels = vec![span.label("here")],
                                "named arguments are not supported when piping"
                            )
                            .with_source_code(span));
                        }
                    }
                }

                (Box::new(expr), tys, args)
            }
            expr => {
                let expr = lower_expr(cx, expr)?;
                (Box::new(expr), Vec::new(), Vec::new())
            }
        };

        ty = hir::Ty::Pipe(Box::new(ty), Box::new(expr.ty.clone()), tys, span);
        pipee = hir::Expr {
            kind: hir::ExprKind::Pipe(Box::new(pipee), expr, args),
            ty: ty.clone(),
        };
    }

    cx.unit.normalize(ty.clone(), span);
    Ok(pipee)
}

fn lower_binary(
    cx: &mut BodyCx,
    op: ast::BinOp,
    lhs: &ast::Expr,
    rhs: &ast::Expr,
    span: Span,
) -> miette::Result<hir::Expr> {
    let lhs = lower_expr(cx, lhs)?;
    let rhs = lower_expr(cx, rhs)?;

    match op {
        ast::BinOp::Add | ast::BinOp::Sub | ast::BinOp::Mul | ast::BinOp::Div | ast::BinOp::Rem => {
            let ty = hir::Ty::inferred(hir::Inferred::Unsigned, span);

            cx.unit.unify(lhs.ty.clone(), ty.clone(), span);
            cx.unit.unify(rhs.ty.clone(), ty.clone(), span);

            let kind = hir::ExprKind::Binary(op, Box::new(lhs), Box::new(rhs));
            Ok(hir::Expr { kind, ty })
        }

        ast::BinOp::Lt | ast::BinOp::Le | ast::BinOp::Gt | ast::BinOp::Ge => {
            let ty = hir::Ty::inferred(hir::Inferred::Unsigned, span);

            cx.unit.unify(lhs.ty.clone(), ty.clone(), span);
            cx.unit.unify(rhs.ty.clone(), ty.clone(), span);

            let kind = hir::ExprKind::Binary(op, Box::new(lhs), Box::new(rhs));
            let ty = hir::Ty::bool(span);

            Ok(hir::Expr { kind, ty })
        }

        ast::BinOp::Eq | ast::BinOp::Ne => {
            cx.unit.unify(lhs.ty.clone(), rhs.ty.clone(), span);

            let ty = hir::Ty::bool(span);
            let kind = hir::ExprKind::Binary(op, Box::new(lhs), Box::new(rhs));
            Ok(hir::Expr { kind, ty })
        }
        ast::BinOp::And | ast::BinOp::Or => {
            let ty = hir::Ty::bool(span);

            cx.unit.unify(lhs.ty.clone(), ty.clone(), span);
            cx.unit.unify(rhs.ty.clone(), ty.clone(), span);

            let kind = hir::ExprKind::Binary(op, Box::new(lhs), Box::new(rhs));
            Ok(hir::Expr { kind, ty })
        }
    }
}

fn lower_unary(
    cx: &mut BodyCx,
    op: ast::UnOp,
    expr: &ast::Expr,
    span: Span,
) -> miette::Result<hir::Expr> {
    let expr = lower_expr(cx, expr)?;

    match op {
        ast::UnOp::Ref => {
            let ty = hir::Ty::Partial(hir::Part::Ref, vec![expr.ty.clone()], span);
            let kind = hir::ExprKind::Ref(Box::new(expr));
            Ok(hir::Expr { kind, ty })
        }
        ast::UnOp::Deref => {
            let ty = hir::Ty::any(span);
            let mut_ty = hir::Ty::Partial(hir::Part::Ref, vec![ty.clone()], span);
            cx.unit.unify(expr.ty.clone(), mut_ty.clone(), span);

            let kind = hir::ExprKind::Deref(Box::new(expr));
            Ok(hir::Expr { kind, ty })
        }
        ast::UnOp::Neg => {
            let ty = hir::Ty::inferred(hir::Inferred::Signed, span);
            cx.unit.unify(expr.ty.clone(), ty.clone(), span);

            let kind = hir::ExprKind::Unary(hir::UnOp::Neg, Box::new(expr));
            Ok(hir::Expr { kind, ty })
        }
        ast::UnOp::Not => {
            let ty = hir::Ty::bool(span);
            cx.unit.unify(expr.ty.clone(), ty.clone(), span);

            let kind = hir::ExprKind::Unary(hir::UnOp::Not, Box::new(expr));
            Ok(hir::Expr { kind, ty })
        }
    }
}

fn lower_return(
    cx: &mut BodyCx,
    expr: Option<&ast::Expr>,
    span: Span,
) -> miette::Result<hir::Expr> {
    let ty = cx.output.clone();

    match expr {
        Some(expr) => {
            let expr = lower_expr(cx, expr)?;
            cx.unit.unify(expr.ty.clone(), ty.clone(), span);

            let kind = hir::ExprKind::Return(Box::new(expr));
            Ok(hir::Expr {
                kind,
                ty: hir::Ty::void(span),
            })
        }
        None => {
            let kind = hir::ExprKind::Return(Box::new(hir::Expr {
                kind: hir::ExprKind::Void,
                ty: hir::Ty::void(span),
            }));
            cx.unit.unify(hir::Ty::void(span), ty.clone(), span);

            Ok(hir::Expr {
                kind,
                ty: hir::Ty::void(span),
            })
        }
    }
}

fn lower_let(
    cx: &mut BodyCx,
    pat: &ast::Pat,
    ty: Option<&ast::Ty>,
    expr: &ast::Expr,
    span: Span,
) -> miette::Result<hir::Expr> {
    let value = lower_expr(cx, expr)?;

    if let Some(ty) = ty {
        let ty = lower_ty(&mut cx.as_ty_cx(), ty, expr.span())?;
        cx.unit.unify(value.ty.clone(), ty, span);
    }

    let span = pat.span;
    let pat = lower_pat(cx, pat, &value.ty)?;

    let local = cx.locals.len();
    cx.locals.push(hir::Local {
        mutable: false,
        name: "",
        ty: value.ty.clone(),
    });

    let input = hir::Expr {
        kind: hir::ExprKind::Local(local),
        ty: value.ty.clone(),
    };

    if !is_pat_irrefutable(cx, &pat) {
        return Err(miette::miette!(
            severity = Severity::Error,
            code = "invalid::pat",
            labels = [span.label("here")],
            "irrefutable pattern required"
        )
        .with_source_code(span));
    }

    let mut locals = Vec::new();

    build_pat_destructure(cx, input, &pat, &mut locals)?;

    let mut exprs = vec![hir::Expr {
        kind: hir::ExprKind::Let(local, Box::new(value)),
        ty: hir::Ty::void(span),
    }];

    for (name, expr) in locals {
        let id = cx.locals.len();

        cx.locals.push(hir::Local {
            mutable: false,
            name,
            ty: expr.ty.clone(),
        });
        cx.scope.push((String::from(name), id));

        let kind = hir::ExprKind::Let(id, Box::new(expr));
        exprs.push(hir::Expr {
            kind,
            ty: hir::Ty::void(span),
        });
    }

    let kind = hir::ExprKind::Block(exprs);
    let ty = hir::Ty::void(span);
    Ok(hir::Expr { kind, ty })
}

fn lower_mut(
    cx: &mut BodyCx,
    name: &'static str,
    ty: Option<&ast::Ty>,
    expr: &ast::Expr,
) -> miette::Result<hir::Expr> {
    let span = expr.span();
    let value = lower_expr(cx, expr)?;

    if let Some(ty) = ty {
        let ty = lower_ty(&mut cx.as_ty_cx(), ty, expr.span())?;
        cx.unit.unify(value.ty.clone(), ty, span);
    }

    let id = cx.locals.len();
    let ty = hir::Ty::Partial(hir::Part::Ref, vec![value.ty.clone()], expr.span());

    cx.locals.push(hir::Local {
        mutable: true,
        name,
        ty,
    });
    cx.scope.push((name.to_owned(), id));

    let value_ty = value.ty.clone();
    let kind = hir::ExprKind::Ref(Box::new(value));
    let expr = hir::Expr { kind, ty: value_ty };

    let kind = hir::ExprKind::Let(id, Box::new(expr));
    let ty = hir::Ty::void(span);
    Ok(hir::Expr { kind, ty })
}

fn lower_let_assert(
    cx: &mut BodyCx,
    ast_pat: &ast::Pat,
    ty: Option<&ast::Ty>,
    expr: &ast::Expr,
) -> miette::Result<hir::Expr> {
    let span = expr.span();
    let expr = lower_expr(cx, expr)?;
    let pat = lower_pat(cx, ast_pat, &expr.ty)?;

    if let Some(ty) = ty {
        let ty = lower_ty(&mut cx.as_ty_cx(), ty, span)?;
        cx.unit.unify(expr.ty.clone(), ty, span);
    }

    let local = cx.locals.len();
    cx.locals.push(hir::Local {
        mutable: false,
        name: "",
        ty: expr.ty.clone(),
    });

    let input = hir::Expr {
        kind: hir::ExprKind::Local(local),
        ty: expr.ty.clone(),
    };

    let mut locals = Vec::new();
    build_pat_destructure(cx, input.clone(), &pat, &mut locals)?;
    let check = build_pat_check(cx, input, pat)?;

    let mut exprs = Vec::new();

    for (name, expr) in locals {
        let id = cx.locals.len();

        cx.locals.push(hir::Local {
            mutable: false,
            name,
            ty: expr.ty.clone(),
        });
        cx.scope.push((String::from(name), id));

        let kind = hir::ExprKind::Let(id, Box::new(expr));
        exprs.push(hir::Expr {
            kind,
            ty: hir::Ty::void(ast_pat.span),
        });
    }

    let kind = hir::ExprKind::Block(exprs);
    let ty = hir::Ty::void(ast_pat.span);

    let output = hir::Expr { kind, ty };

    let output = match check {
        Some(check) => {
            let panic = hir::Expr {
                kind: hir::ExprKind::Panic("assertion failed"),
                ty: hir::Ty::void(ast_pat.span),
            };

            let r#match = hir::Match::Bool(Box::new(output), Box::new(panic));
            let kind = hir::ExprKind::Match(Box::new(check), r#match);
            let ty = hir::Ty::void(ast_pat.span);
            hir::Expr { kind, ty }
        }
        None => output,
    };

    let exprs = vec![
        hir::Expr {
            kind: hir::ExprKind::Let(local, Box::new(expr)),
            ty: hir::Ty::void(span),
        },
        output,
    ];

    let kind = hir::ExprKind::Block(exprs);
    let ty = hir::Ty::void(span);
    Ok(hir::Expr { kind, ty })
}

fn lower_assign(cx: &mut BodyCx, lhs: &ast::Expr, rhs: &ast::Expr) -> miette::Result<hir::Expr> {
    let span = lhs.span().join(rhs.span());

    let lhs = lower_expr(cx, lhs)?;
    let rhs = lower_expr(cx, rhs)?;

    let kind = hir::ExprKind::Assign(Box::new(lhs), Box::new(rhs));
    let ty = hir::Ty::void(span);
    Ok(hir::Expr { kind, ty })
}

fn lower_closure(
    cx: &mut BodyCx,
    args: &[ast::Argument],
    body: &ast::Expr,
) -> miette::Result<hir::Expr> {
    let mut input = Vec::new();

    for arg in args {
        let ty = match arg.ty {
            Some(ref ty) => lower_ty(&mut cx.as_ty_cx(), ty, arg.span)?,
            None => hir::Ty::any(arg.span),
        };

        input.push(hir::Argument {
            name: arg.name,
            ty,
            span: arg.span,
        });
    }

    let span = body.span();
    let output = hir::Ty::any(span);

    let mut body_cx = BodyCx {
        unit: cx.unit,
        arguments: &input.clone(),
        generics: cx.generics,
        output: &output,
        module: cx.module,
        locals: Vec::new(),
        scope: Vec::new(),
        capture: Some(CaptureCx {
            arguments: cx.arguments.to_vec(),
            locals: cx.locals.to_vec(),
            scope: cx.scope.to_vec(),
            captures: Vec::new(),
            parent: cx.capture.take().map(Box::new),
        }),
    };

    let body = Box::new(lower_expr(&mut body_cx, body)?);
    (body_cx.unit).unify(body_cx.output.clone(), body.ty.clone(), span);

    let capture = body_cx.capture.take().unwrap();
    cx.capture = capture.parent.map(|parent| *parent);

    let mut captured = Vec::new();

    for (capture, ty) in capture.captures {
        match capture {
            Capture::Argument(index) => {
                let kind = hir::ExprKind::Argument(index);
                captured.push(hir::Expr { kind, ty });
            }
            Capture::Local(index) => {
                let kind = hir::ExprKind::Local(index);
                captured.push(hir::Expr { kind, ty });
            }
            Capture::Captured(index) => {
                let kind = hir::ExprKind::Capture(index);
                captured.push(hir::Expr { kind, ty });
            }
        }
    }

    let output = body.ty.clone();
    let locals = body_cx.locals;

    let parts = input
        .iter()
        .map(|arg| arg.ty.clone())
        .chain([output.clone()])
        .collect();

    let kind = hir::ExprKind::Closure(locals, captured, body);
    let ty = hir::Ty::Partial(hir::Part::Func, parts, span);
    Ok(hir::Expr { kind, ty })
}

fn lower_match(
    cx: &mut BodyCx,
    input: &ast::Expr,
    arms: &[ast::Arm],
    span: Span,
) -> miette::Result<hir::Expr> {
    let expr = lower_expr(cx, input)?;

    let local = cx.locals.len();
    cx.locals.push(hir::Local {
        mutable: false,
        name: "",
        ty: expr.ty.clone(),
    });

    let input = hir::Expr {
        kind: hir::ExprKind::Local(local),
        ty: expr.ty.clone(),
    };

    let mut tree = Match::None;
    let ty = hir::Ty::any(span);

    for arm in arms {
        let pat = lower_pat(cx, &arm.pat, &input.ty)?;

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
        cx.unit.unify(ty.clone(), arm_ty, arm.span);
    }

    if tree.missing_patterns() {
        return Err(miette::miette!(
            severity = Severity::Error,
            code = "invalid::pattern",
            labels = vec![span.label("here")],
            help = "consider adding a catch-all pattern `_ -> panic`",
            "match expression doesn't cover all possible patterns"
        )
        .with_source_code(span));
    }

    let exprs = vec![
        hir::Expr {
            kind: hir::ExprKind::Let(local, Box::new(expr)),
            ty: hir::Ty::void(span),
        },
        build_match_expr(cx, tree, span)?.unwrap(),
    ];

    let kind = hir::ExprKind::Block(exprs);
    Ok(hir::Expr { kind, ty })
}

fn lower_pat(cx: &mut BodyCx, pat: &ast::Pat, ty: &hir::Ty) -> miette::Result<Pat> {
    match pat.kind {
        ast::PatKind::Bind(name) => Ok(Pat::Bind(name, pat.span)),
        ast::PatKind::Bool(b) => {
            cx.unit.unify(ty.clone(), hir::Ty::bool(pat.span), pat.span);
            Ok(Pat::Bool(b, pat.span))
        }
        ast::PatKind::Tuple(ref ast) => {
            let mut pats = Vec::new();
            let mut tys = Vec::new();

            for pat in ast {
                let ty = hir::Ty::any(pat.span);
                let pat = lower_pat(cx, pat, &ty)?;
                pats.push(pat);
                tys.push(ty);
            }

            let tuple = hir::Ty::Partial(hir::Part::Tuple, tys, pat.span);
            cx.unit.unify(ty.clone(), tuple, pat.span);

            Ok(Pat::Tuple(pats, pat.span))
        }
        ast::PatKind::Variant(ref path, ref pats) => {
            let (adt, index) = find_variant(cx.unit, cx.module, path)?;

            let generics: Vec<_> = cx.unit.adts[adt]
                .generics
                .iter()
                .map(|_| hir::Ty::any(pat.span))
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

            let adt_ty = hir::Ty::Partial(hir::Part::Adt(adt), generics, path.span);
            cx.unit.unify(ty.clone(), adt_ty, pat.span);

            Ok(Pat::Variant(adt, index, hir, pat.span))
        }
        ast::PatKind::List(ref pats, ref rest) => {
            let span = pat.span;

            let mut rest = match rest {
                Some(Some(rest)) => lower_pat(cx, rest, &hir::Ty::any(span))?,
                Some(None) => Pat::Bind(None, span),
                None => Pat::List(None, span),
            };

            for pat in pats.iter().rev() {
                let span = pat.span;
                let pat = lower_pat(cx, pat, &hir::Ty::any(pat.span))?;
                rest = Pat::List(Some(Box::new((pat, rest))), span);
            }

            let ty = hir::Ty::Partial(hir::Part::List, vec![hir::Ty::any(span)], span);
            cx.unit.unify(ty.clone(), ty, span);

            Ok(rest)
        }
    }
}

#[derive(Clone, Debug)]
enum Pat {
    Bind(Option<&'static str>, Span),
    Bool(bool, Span),
    Tuple(Vec<Pat>, Span),
    Variant(usize, usize, Vec<(hir::Ty, Pat)>, Span),
    List(Option<Box<(Pat, Pat)>>, Span),
}

fn build_pat_destructure(
    cx: &mut BodyCx,
    input: hir::Expr,
    pat: &Pat,
    locals: &mut Vec<(&'static str, hir::Expr)>,
) -> miette::Result<()> {
    match pat {
        Pat::Bind(name, _) => match name {
            Some(name) => {
                locals.push((name, input));
                Ok(())
            }
            None => Ok(()),
        },
        Pat::Bool(_, _) => Ok(()),
        Pat::Tuple(pats, span) => {
            for (i, pat) in pats.iter().enumerate() {
                let input = hir::Expr {
                    kind: hir::ExprKind::TupleField(Box::new(input.clone()), i),
                    ty: hir::Ty::Tuple(Box::new(input.ty.clone()), i, *span),
                };

                cx.unit.normalize(input.ty.clone(), *span);
                build_pat_destructure(cx, input, pat, locals)?;
            }

            Ok(())
        }
        Pat::Variant(_, variant, fields, _) => {
            for (i, (ty, pat)) in fields.iter().enumerate() {
                let input = hir::Expr {
                    kind: hir::ExprKind::VariantField(Box::new(input.clone()), *variant, i),
                    ty: ty.clone(),
                };

                build_pat_destructure(cx, input, pat, locals)?;
            }

            Ok(())
        }
        Pat::List(pats, span) => {
            if let Some((head, tail)) = pats.as_deref() {
                let ty = hir::Ty::any(*span);

                let expr = hir::Expr {
                    kind: hir::ExprKind::ListHead(Box::new(input.clone())),
                    ty: ty.clone(),
                };

                build_pat_destructure(cx, expr, head, locals)?;

                let ty = hir::Ty::Partial(hir::Part::List, vec![ty], *span);
                cx.unit.unify(input.ty.clone(), ty.clone(), *span);

                let expr = hir::Expr {
                    kind: hir::ExprKind::ListTail(Box::new(input.clone())),
                    ty,
                };

                build_pat_destructure(cx, expr, tail, locals)?;
            }

            Ok(())
        }
    }
}

fn is_pat_irrefutable(cx: &mut BodyCx, pat: &Pat) -> bool {
    match pat {
        Pat::Bind(_, _) => true,
        Pat::Bool(_, _) => false,
        Pat::Tuple(pats, _) => pats.iter().all(|pat| is_pat_irrefutable(cx, pat)),
        Pat::Variant(adt, _, fields, _) => {
            if cx.unit.adts[*adt].variants.len() > 1 {
                return false;
            }

            fields.iter().all(|(_, pat)| is_pat_irrefutable(cx, pat))
        }
        Pat::List(_, _) => false,
    }
}

fn build_pat_check(
    cx: &mut BodyCx,
    input: hir::Expr,
    pat: Pat,
) -> miette::Result<Option<hir::Expr>> {
    match pat {
        Pat::Bind(_, _) => Ok(None),
        Pat::Bool(value, span) => match value {
            true => Ok(Some(input)),
            false => Ok(Some(hir::Expr {
                kind: hir::ExprKind::Unary(hir::UnOp::Not, Box::new(input)),
                ty: hir::Ty::bool(span),
            })),
        },
        Pat::Tuple(pats, span) => {
            let mut items = Vec::new();

            for (i, pat) in pats.into_iter().enumerate() {
                let input = hir::Expr {
                    kind: hir::ExprKind::TupleField(Box::new(input.clone()), items.len()),
                    ty: hir::Ty::Tuple(Box::new(input.ty.clone()), i, span),
                };

                if let Some(expr) = build_pat_check(cx, input, pat)? {
                    items.push(expr);
                }
            }

            if items.is_empty() {
                return Ok(None);
            }

            let mut expr = items.pop().unwrap();

            for item in items.into_iter().rev() {
                expr = hir::Expr {
                    kind: hir::ExprKind::Binary(ast::BinOp::And, Box::new(expr), Box::new(item)),
                    ty: hir::Ty::bool(span),
                };
            }

            Ok(Some(expr))
        }
        Pat::Variant(_, variant, fields, span) => {
            let check = hir::Expr {
                kind: hir::ExprKind::IsVariant(Box::new(input.clone()), variant),
                ty: hir::Ty::bool(span),
            };

            let mut items = Vec::new();

            for (i, (ty, pat)) in fields.into_iter().enumerate() {
                let input = hir::Expr {
                    kind: hir::ExprKind::VariantField(Box::new(input.clone()), variant, i),
                    ty: ty.clone(),
                };

                if let Some(expr) = build_pat_check(cx, input, pat)? {
                    items.push(expr);
                }
            }

            if items.is_empty() {
                return Ok(Some(check));
            }

            let mut expr = hir::Expr {
                kind: hir::ExprKind::Binary(
                    ast::BinOp::And,
                    Box::new(check),
                    Box::new(items.pop().unwrap()),
                ),
                ty: hir::Ty::bool(span),
            };

            for item in items.into_iter().rev() {
                expr = hir::Expr {
                    kind: hir::ExprKind::Binary(ast::BinOp::And, Box::new(expr), Box::new(item)),
                    ty: hir::Ty::bool(span),
                };
            }

            Ok(Some(expr))
        }
        Pat::List(pats, span) => match pats.map(|p| *p) {
            Some((head_pat, tail_pat)) => {
                let ty = hir::Ty::any(span);

                let head_kind = hir::ExprKind::ListHead(Box::new(input.clone()));
                let head_expr = hir::Expr {
                    kind: head_kind,
                    ty: ty.clone(),
                };

                let ty = hir::Ty::Partial(hir::Part::List, vec![ty.clone()], span);
                cx.unit.unify(input.ty.clone(), ty.clone(), span);

                let tail_kind = hir::ExprKind::ListTail(Box::new(input.clone()));
                let tail_expr = hir::Expr {
                    kind: tail_kind,
                    ty,
                };

                let empty = hir::Expr {
                    kind: hir::ExprKind::ListEmpty(Box::new(input)),
                    ty: hir::Ty::bool(span),
                };

                let not_empty = hir::Expr {
                    kind: hir::ExprKind::Unary(hir::UnOp::Not, Box::new(empty.clone())),
                    ty: hir::Ty::bool(span),
                };

                let head = build_pat_check(cx, head_expr, head_pat)?;
                let tail = build_pat_check(cx, tail_expr, tail_pat)?;

                let check = match (head, tail) {
                    (Some(head), Some(tail)) => {
                        let head = Box::new(head);
                        let tail = Box::new(tail);
                        Some(hir::Expr {
                            kind: hir::ExprKind::Binary(ast::BinOp::And, head, tail),
                            ty: hir::Ty::bool(span),
                        })
                    }
                    (Some(expr), None) => Some(expr),
                    (None, Some(expr)) => Some(expr),
                    (None, None) => None,
                };

                match check {
                    Some(check) => Ok(Some(hir::Expr {
                        kind: hir::ExprKind::Binary(
                            ast::BinOp::And,
                            Box::new(not_empty),
                            Box::new(check),
                        ),
                        ty: hir::Ty::bool(span),
                    })),
                    None => Ok(Some(not_empty)),
                }
            }
            None => {
                let kind = hir::ExprKind::ListEmpty(Box::new(input));
                let ty = hir::Ty::bool(span);
                Ok(Some(hir::Expr { kind, ty }))
            }
        },
    }
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
        Pat::Bool(value, _) => {
            let next_pat = pats.next();

            if matches!(tree, Match::Bind(_)) && next_pat.is_none() {
                let expr = lower_expr(cx, body)?;
                let ty = expr.ty.clone();

                return Ok(ty);
            }

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

            match next_pat {
                Some((input, pat)) => build_match_tree(cx, input, pat, pats, subtree, body, locals),
                None => {
                    let expr = lower_expr(cx, body)?;
                    let ty = expr.ty.clone();

                    if matches!(subtree.as_ref(), Match::None) {
                        **subtree = Match::Leaf(locals, expr);
                    }

                    Ok(ty)
                }
            }
        }
        Pat::Tuple(items, span) => {
            let items = items
                .into_iter()
                .enumerate()
                .map(|(i, pat)| {
                    let kind = hir::ExprKind::TupleField(Box::new(input.clone()), i);
                    let ty = hir::Ty::Tuple(Box::new(input.ty.clone()), i, span);
                    cx.unit.normalize(ty.clone(), span);

                    let input = hir::Expr { kind, ty };
                    (input, pat)
                })
                .collect::<Vec<_>>();

            let mut pats = items.into_iter().chain(pats);

            let (input, pat) = pats.next().unwrap();
            build_match_tree(cx, input, pat, &mut pats, tree, body, locals)
        }
        Pat::Variant(adt, variant, fields, _) => {
            let fields = fields.into_iter().enumerate().map(|(i, (ty, pat))| {
                let kind = hir::ExprKind::VariantField(Box::new(input.clone()), variant, i);
                let ty = ty.clone();

                let input = hir::Expr { kind, ty };
                (input, pat)
            });

            let mut pats = fields.chain(pats);
            let next_pat = pats.next();

            if matches!(tree, Match::Bind(_)) && next_pat.is_none() {
                let expr = lower_expr(cx, body)?;
                let ty = expr.ty.clone();
                return Ok(ty);
            }

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

            match next_pat {
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
        Pat::List(pat, span) => match pat.map(|p| *p) {
            Some((head, tail)) => {
                let Match::List {
                    some,
                    none,
                    default,
                    ..
                } = tree.as_list(&input)?
                else {
                    unreachable!()
                };

                let some = match none {
                    Some(_) => default,
                    None => some.get_or_insert_with(|| Box::new(Match::None)),
                };

                let head_ty = hir::Ty::any(span);
                let list_ty = hir::Ty::Partial(hir::Part::List, vec![head_ty.clone()], span);

                cx.unit.unify(input.ty.clone(), list_ty.clone(), span);

                let head_kind = hir::ExprKind::ListHead(Box::new(input.clone()));
                let head_expr = hir::Expr {
                    kind: head_kind,
                    ty: head_ty,
                };

                let tail_kind = hir::ExprKind::ListTail(Box::new(input.clone()));
                let tail_expr = hir::Expr {
                    kind: tail_kind,
                    ty: list_ty,
                };

                let mut pats = [(head_expr, head), (tail_expr, tail)]
                    .into_iter()
                    .chain(pats);

                let (input, pat) = pats.next().unwrap();
                build_match_tree(cx, input, pat, &mut pats, some, body, locals)
            }
            None => {
                let next_pat = pats.next();

                if matches!(tree, Match::Bind(_)) && next_pat.is_none() {
                    let expr = lower_expr(cx, body)?;
                    let ty = expr.ty.clone();
                    return Ok(ty);
                }

                let Match::List {
                    some,
                    none,
                    default,
                    ..
                } = tree.as_list(&input)?
                else {
                    unreachable!()
                };

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
                            **none = Match::Leaf(locals, expr);
                        }

                        Ok(ty)
                    }
                }
            }
        },
        Pat::Bind(name, span) => {
            if let Some(name) = name {
                let id = cx.locals.len();
                let ty = input.ty.clone();

                cx.locals.push(hir::Local {
                    mutable: false,
                    name,
                    ty: ty.clone(),
                });
                cx.scope.push((name.to_owned(), id));

                locals.push((id, input));
            }

            if matches!(tree, Match::None) {
                *tree = Match::Bind(Box::new(Match::None));
            }

            let Some((input, pat)) = pats.next() else {
                let expr = lower_expr(cx, body)?;
                let ty = expr.ty.clone();

                tree.visit(&mut move |tree| {
                    if matches!(tree, Match::None) {
                        *tree = Match::Leaf(locals.clone(), expr.clone());
                    }
                });

                return Ok(ty);
            };

            let pats = pats.collect::<Vec<_>>();

            match tree {
                Match::None => unreachable!(),
                Match::Bind(subtree) => {
                    build_match_tree(cx, input, pat, &mut pats.into_iter(), subtree, body, locals)
                }
                Match::Leaf(_, _) => unreachable!(),
                Match::Bool {
                    r#true,
                    r#false,
                    default,
                    ..
                } => {
                    let (r#true, r#false) = match (r#true, r#false) {
                        (Some(r#true), Some(r#false)) => (r#true, r#false),
                        (Some(r#true), None) => (r#true, default),
                        (None, Some(r#false)) => (default, r#false),
                        (None, None) => unreachable!(),
                    };

                    let r#true = build_match_tree(
                        cx,
                        input.clone(),
                        pat.clone(),
                        &mut pats.clone().into_iter(),
                        r#true,
                        body,
                        locals.clone(),
                    )?;
                    let r#false = build_match_tree(
                        cx,
                        input,
                        pat,
                        &mut pats.into_iter(),
                        r#false,
                        body,
                        locals,
                    )?;

                    cx.unit.unify(r#true.clone(), r#false.clone(), span);

                    Ok(r#true)
                }
                Match::Adt {
                    variants, default, ..
                } => {
                    let ty = build_match_tree(
                        cx,
                        input.clone(),
                        pat.clone(),
                        &mut pats.clone().into_iter(),
                        default,
                        body,
                        locals.clone(),
                    )?;

                    for variant in variants.iter_mut().flatten() {
                        let variant_ty = build_match_tree(
                            cx,
                            input.clone(),
                            pat.clone(),
                            &mut pats.clone().into_iter(),
                            variant,
                            body,
                            locals.clone(),
                        )?;

                        cx.unit.unify(ty.clone(), variant_ty, span);
                    }

                    Ok(ty)
                }
                Match::List {
                    some,
                    none,
                    default,
                    ..
                } => {
                    let (some, none) = match (some, none) {
                        (Some(some), Some(none)) => (some, none),
                        (Some(some), None) => (some, default),
                        (None, Some(none)) => (default, none),
                        (None, None) => unreachable!(),
                    };

                    let some = build_match_tree(
                        cx,
                        input.clone(),
                        pat.clone(),
                        &mut pats.clone().into_iter(),
                        some,
                        body,
                        locals.clone(),
                    )?;
                    let none = build_match_tree(
                        cx,
                        input,
                        pat,
                        &mut pats.into_iter(),
                        none,
                        body,
                        locals,
                    )?;

                    cx.unit.unify(some.clone(), none.clone(), span);

                    Ok(some)
                }
            }
        }
    }
}

fn build_match_expr(cx: &mut BodyCx, tree: Match, span: Span) -> miette::Result<Option<hir::Expr>> {
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
        Match::Bind(tree) => build_match_expr(cx, *tree, span),
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

            let ty = hir::Ty::any(span);

            let r#true = build_match_expr(cx, *r#true, span)?.map(Box::new).unwrap();
            let r#false = build_match_expr(cx, *r#false, span)?.map(Box::new).unwrap();

            cx.unit.unify(ty.clone(), r#true.ty.clone(), span);
            cx.unit.unify(ty.clone(), r#false.ty.clone(), span);

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
            let ty = hir::Ty::any(span);

            for variant in variants {
                match variant {
                    Some(subtree) => {
                        let expr = build_match_expr(cx, subtree, span)?.unwrap();
                        cx.unit.unify(ty.clone(), expr.ty.clone(), span);
                        exprs.push(Some(expr));
                    }
                    None => exprs.push(None),
                }
            }

            let default = build_match_expr(cx, *default, span)?.map(Box::new);

            if let Some(ref default) = default {
                cx.unit.unify(ty.clone(), default.ty.clone(), span);
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

            let ty = hir::Ty::any(span);

            let some = build_match_expr(cx, *some, span)?.map(Box::new).unwrap();
            let none = build_match_expr(cx, *none, span)?.map(Box::new).unwrap();

            cx.unit.unify(ty.clone(), some.ty.clone(), span);
            cx.unit.unify(ty.clone(), none.ty.clone(), span);

            let empty = hir::Expr {
                kind: hir::ExprKind::ListEmpty(Box::new(input.clone())),
                ty: hir::Ty::bool(span),
            };

            let r#match = hir::Match::Bool(none, some);
            let kind = hir::ExprKind::Match(Box::new(empty), r#match);
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

    fn missing_patterns(&self) -> bool {
        match self {
            Match::None => true,
            Match::Bind(tree) => tree.missing_patterns(),
            Match::Leaf(_, _) => false,
            Match::Bool {
                r#true,
                r#false,
                default,
                ..
            } => {
                if let Some(r#true) = r#true {
                    if r#true.missing_patterns() {
                        return true;
                    }
                }

                if let Some(r#false) = r#false {
                    if r#false.missing_patterns() {
                        return true;
                    }
                }

                (r#true.is_none() || r#false.is_none()) && default.missing_patterns()
            }
            Match::Adt {
                variants, default, ..
            } => {
                for variant in variants.iter().flatten() {
                    if variant.missing_patterns() {
                        return true;
                    }
                }

                variants.iter().any(Option::is_none) && default.missing_patterns()
            }
            Match::List {
                some,
                none,
                default,
                ..
            } => {
                if let Some(some) = some {
                    if some.missing_patterns() {
                        return true;
                    }
                }

                if let Some(none) = none {
                    if none.missing_patterns() {
                        return true;
                    }
                }

                (some.is_none() || none.is_none()) && default.missing_patterns()
            }
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
