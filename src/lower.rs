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
    scope: Vec<(String, usize)>,
}

fn lower_expr(cx: &mut BodyCx, ast: &ast::Expr) -> miette::Result<hir::Expr> {
    match ast {
        ast::Expr::Int(negative, base, digits, span) => {
            lower_int(cx, *negative, *base, digits, *span)
        }
        ast::Expr::Bool(value, span) => lower_bool(cx, *value, *span),
        ast::Expr::Paren(expr, _) => lower_expr(cx, expr),
        ast::Expr::Item(path) => lower_item(cx, path),
        ast::Expr::Tuple(exprs) => lower_tuple(cx, exprs),
        ast::Expr::List(_) => todo!(),
        ast::Expr::Block(block) => lower_block(cx, block),
        ast::Expr::Field(expr, name) => lower_field(cx, expr, name),
        ast::Expr::Call(func, args) => lower_call(cx, func, args),
        ast::Expr::Pipe(expr, exprs) => lower_pipe(cx, expr, exprs),
        ast::Expr::Binary(op, lhs, rhs) => lower_binary(cx, *op, lhs, rhs),
        ast::Expr::Let(name, expr) => lower_let(cx, name, expr),
        ast::Expr::Match(input, arms) => lower_match(cx, input, arms),
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

fn lower_item(cx: &mut BodyCx, path: &ast::Path) -> miette::Result<hir::Expr> {
    if path.segments.len() == 1 {
        let name = path
            .segments
            .first()
            .expect("path should have at least one segment");

        for (scope_name, id) in &cx.scope {
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

fn lower_binary(
    cx: &mut BodyCx,
    op: ast::BinOp,
    lhs: &ast::Expr,
    rhs: &ast::Expr,
) -> miette::Result<hir::Expr> {
    let lhs = lower_expr(cx, lhs)?;
    let rhs = lower_expr(cx, rhs)?;

    match op {
        ast::BinOp::Add | ast::BinOp::Sub | ast::BinOp::Mul | ast::BinOp::Div | ast::BinOp::Rem => {
            let ty = hir::Ty::inferred(hir::Inferred::Int(IntKind::Int));

            cx.unit.unify(lhs.ty.clone(), ty.clone());
            cx.unit.unify(rhs.ty.clone(), ty.clone());

            let kind = hir::ExprKind::Binary(op, Box::new(lhs), Box::new(rhs));
            Ok(hir::Expr { kind, ty })
        }
        _ => todo!(),
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

    let local = cx.locals.len();
    cx.locals.push(hir::Local {
        name: "",
        ty: input.ty.clone(),
    });

    let mut tree = Match::None;
    let ty = hir::Ty::any();

    for (pat, arm) in pats.into_iter().zip(arms.iter()) {
        let scope = cx.scope.len();

        let arm_ty = build_match_tree(
            cx,
            local,
            pat,
            &mut std::iter::empty(),
            &mut tree,
            &arm.expr,
        )?;

        cx.scope.truncate(scope);
        cx.unit.unify(ty.clone(), arm_ty);
    }

    let exprs = vec![
        hir::Expr {
            kind: hir::ExprKind::Let(local, Box::new(input)),
            ty: hir::Ty::void(),
        },
        build_match_expr(cx, tree)?.unwrap(),
    ];

    let kind = hir::ExprKind::Block(exprs);
    Ok(hir::Expr { kind, ty })
}

fn lower_pat(cx: &mut BodyCx, pat: &ast::Pat, ty: &hir::Ty) -> miette::Result<Pat> {
    match pat.kind {
        ast::PatKind::Bind(name) => Ok(Pat::Bind(name)),
        ast::PatKind::Bool(b) => {
            cx.unit.unify(ty.clone(), hir::Ty::bool());
            Ok(Pat::Bool(b))
        }
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
    }
}

#[derive(Debug)]
enum Pat {
    Bind(Option<&'static str>),
    Bool(bool),
    Variant(usize, usize, Vec<(hir::Ty, Pat)>),
}

fn build_match_tree(
    cx: &mut BodyCx,
    input: usize,
    pat: Pat,
    pats: &mut dyn Iterator<Item = (usize, Pat)>,
    tree: &mut Match,
    body: &ast::Expr,
) -> miette::Result<hir::Ty> {
    match pat {
        Pat::Bool(value) => {
            todo!()
        }
        Pat::Variant(adt, variant, fields) => {
            let variants = cx.unit.adts[adt].variants.len();
            let Match::Adt { variants, .. } = tree.as_adt(input, adt, variants)? else {
                unreachable!()
            };

            let (locals, subtree) = match variants[variant] {
                Some(ref mut pair) => pair,
                None => {
                    let mut locals = Vec::new();

                    for (ty, _) in &fields {
                        let id = cx.locals.len();
                        let ty = ty.clone();
                        cx.locals.push(hir::Local { name: "", ty });
                        locals.push(id);
                    }

                    variants[variant] = Some((locals, Match::None));
                    variants[variant].as_mut().unwrap()
                }
            };

            let fields = fields
                .into_iter()
                .enumerate()
                .map(move |(i, (_ty, pat))| (locals[i], pat));

            let mut pats = fields.chain(pats);

            match pats.next() {
                Some((input, pat)) => build_match_tree(cx, input, pat, &mut pats, subtree, body),
                None => {
                    let expr = lower_expr(cx, body)?;
                    let ty = expr.ty.clone();
                    *subtree = Match::Leaf(expr);
                    Ok(ty)
                }
            }
        }
        Pat::Bind(name) => {
            if let Some(name) = name {
                cx.scope.push((name.to_owned(), input));
            }

            match pats.next() {
                Some((input, pat)) => build_match_tree(cx, input, pat, pats, tree, body),
                None => {
                    assert!(matches!(tree.default(), Match::None));
                    let expr = lower_expr(cx, body)?;
                    let ty = expr.ty.clone();
                    *tree.default() = Match::Leaf(expr);
                    Ok(ty)
                }
            }
        }
    }
}

fn build_match_expr(cx: &mut BodyCx, tree: Match) -> miette::Result<Option<hir::Expr>> {
    match tree {
        Match::None => Ok(None),
        Match::Leaf(expr) => Ok(Some(expr)),
        Match::Bool {
            input,
            r#true,
            r#false,
            default,
        } => {
            todo!()
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
                    Some((locals, subtree)) => {
                        let expr = build_match_expr(cx, subtree)?.unwrap();
                        cx.unit.unify(ty.clone(), expr.ty.clone());
                        exprs.push(Some((locals, expr)));
                    }
                    None => exprs.push(None),
                }
            }

            let default = build_match_expr(cx, *default)?.map(Box::new);

            if let Some(ref default) = default {
                cx.unit.unify(ty.clone(), default.ty.clone());
            }

            let r#match = hir::Match::Adt(adt, exprs, default);
            let kind = hir::ExprKind::Match(input, r#match);
            Ok(Some(hir::Expr { kind, ty }))
        }
    }
}

#[derive(Clone, Debug)]
enum Match {
    None,
    Leaf(hir::Expr),
    Bool {
        input: usize,
        r#true: Option<(usize, hir::Expr)>,
        r#false: Option<(usize, hir::Expr)>,
        default: Box<Match>,
    },
    Adt {
        input: usize,
        adt: usize,
        variants: Vec<Option<(Vec<usize>, Match)>>,
        default: Box<Match>,
    },
}

impl Match {
    fn as_bool(&mut self, input: usize) -> miette::Result<&mut Match> {
        match self {
            Match::None | Match::Leaf(_) => {
                *self = Self::Bool {
                    input,
                    r#true: None,
                    r#false: None,
                    default: Box::new(self.clone()),
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
        input: usize,
        index: usize,
        variants: usize,
    ) -> miette::Result<&mut Match> {
        match self {
            Match::None | Match::Leaf(_) => {
                *self = Self::Adt {
                    input,
                    adt: index,
                    variants: vec![None; variants],
                    default: Box::new(self.clone()),
                };

                Ok(self)
            }
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

    fn default(&mut self) -> &mut Match {
        match self {
            Match::None | Match::Leaf(_) => self,
            Match::Bool { default, .. } | Match::Adt { default, .. } => default,
        }
    }
}
