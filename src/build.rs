use std::collections::HashMap;

use crate::{
    ast::BinOp,
    mir,
    number::{Base, IntKind},
    rir,
};

pub fn build(unit: &rir::Unit, func: usize) -> miette::Result<mir::Mir> {
    let mut mir = mir::Mir { funcs: Vec::new() };
    let mut funcs = HashMap::new();

    let generics = vec![mir::Ty::Int(IntKind::Int); unit.funcs[func].generics.len()];
    build_func(&mut mir, &mut funcs, unit, func, &generics)?;

    Ok(mir)
}

fn build_func(
    mir: &mut mir::Mir,
    funcs: &mut Funcs,
    unit: &rir::Unit,
    index: usize,
    generics: &[mir::Ty],
) -> miette::Result<usize> {
    let key = (index, generics.to_vec());
    if let Some(&index) = funcs.get(&key) {
        return Ok(index);
    }

    let func = &unit.funcs[index];
    assert_eq!(func.generics.len(), generics.len());

    let mut builder = Builder {
        mir,
        funcs,
        unit,
        generics,
    };

    let input = func
        .input
        .iter()
        .map(|a| build_ty(&mut builder, &a.ty))
        .collect();

    let output = build_ty(&mut builder, &func.output);

    let locals = func
        .locals
        .iter()
        .map(|ty| build_ty(&mut builder, ty))
        .collect();

    let index = builder.mir.funcs.len();
    builder.mir.funcs.push(mir::Func {
        input,
        output,
        locals,
        ..Default::default()
    });

    builder.funcs.insert(key, index);

    let body = build_expr(&mut builder, &func.body)?;

    mir.funcs[index].body = body;

    Ok(index)
}

type Funcs = HashMap<(usize, Vec<mir::Ty>), usize>;

struct Builder<'a> {
    mir: &'a mut mir::Mir,
    funcs: &'a mut Funcs,
    unit: &'a rir::Unit,
    generics: &'a [mir::Ty],
}

fn build_ty(builder: &mut Builder, ty: &rir::Ty) -> mir::Ty {
    match ty {
        rir::Ty::Void => mir::Ty::Void,
        rir::Ty::Bool => mir::Ty::Bool,
        rir::Ty::Int(kind) => mir::Ty::Int(*kind),
        rir::Ty::List(item) => {
            let item = Box::new(build_ty(builder, item));
            mir::Ty::List(item)
        }
        rir::Ty::Tuple(items) => {
            let items = build_ty_vec(builder, items);

            mir::Ty::Tuple(items)
        }
        rir::Ty::Func(input, output) => {
            let input = build_ty_vec(builder, input);
            let output = Box::new(build_ty(builder, output));
            mir::Ty::Func(input, output)
        }
        rir::Ty::Adt(index, generics) => {
            let generics: Vec<_> = generics.iter().map(|ty| build_ty(builder, ty)).collect();

            let mut builder = Builder {
                mir: builder.mir,
                funcs: builder.funcs,
                unit: builder.unit,
                generics: &generics,
            };

            let mut variants = Vec::new();

            for variant in &builder.unit.adts[*index].variants {
                let fields = variant
                    .fields
                    .iter()
                    .map(|field| build_ty(&mut builder, &field.ty))
                    .collect();

                variants.push(mir::Variant { fields });
            }

            mir::Ty::Adt(variants)
        }
        rir::Ty::Generic(index) => builder.generics[*index].clone(),
    }
}

fn build_ty_vec(builder: &mut Builder, tys: &[rir::Ty]) -> Vec<mir::Ty> {
    tys.iter().map(|ty| build_ty(builder, ty)).collect()
}

fn extract_generics(unit: &rir::Unit, ty: &rir::Ty, expected: &mir::Ty) -> Vec<mir::Ty> {
    fn recurse(unit: &rir::Unit, generics: &mut Vec<Option<mir::Ty>>, ty: &rir::Ty, ex: &mir::Ty) {
        match (ty, ex) {
            (rir::Ty::Void, mir::Ty::Void) => {}
            (rir::Ty::Int(kind), mir::Ty::Int(ex)) => {
                assert_eq!(kind, ex);
            }
            (rir::Ty::List(item), mir::Ty::List(ex)) => {
                recurse(unit, generics, item, ex);
            }
            (rir::Ty::Tuple(items), mir::Ty::Tuple(ex)) => {
                for (ty, ex) in items.iter().zip(ex) {
                    recurse(unit, generics, ty, ex);
                }
            }
            (rir::Ty::Func(input, output), mir::Ty::Func(ex_input, ex_output)) => {
                for (ty, ex) in input.iter().zip(ex_input) {
                    recurse(unit, generics, ty, ex);
                }

                recurse(unit, generics, output, ex_output);
            }
            (rir::Ty::Adt(index, args), mir::Ty::Adt(ex)) => {
                let mut adt_generics = Vec::new();

                for (variant, ex) in unit.adts[*index].variants.iter().zip(ex) {
                    for (field, ex) in variant.fields.iter().zip(ex.fields.iter()) {
                        recurse(unit, &mut adt_generics, &field.ty, ex);
                    }
                }

                for (ty, ex) in args.iter().zip(adt_generics) {
                    let ex = ex.expect("expected generic");
                    recurse(unit, generics, ty, &ex);
                }
            }
            (rir::Ty::Generic(index), ex) => {
                if generics.len() <= *index {
                    generics.resize_with(*index + 1, || None);
                }

                match generics[*index] {
                    Some(ref generic) => assert_eq!(generic, ex),
                    None => generics[*index] = Some(ex.clone()),
                }
            }
            _ => todo!(),
        }
    }

    let mut generics = Vec::new();
    recurse(unit, &mut generics, ty, expected);
    generics.into_iter().map(Option::unwrap).collect()
}

fn build_expr(builder: &mut Builder, expr: &rir::Expr) -> miette::Result<mir::Expr> {
    match expr.kind {
        rir::ExprKind::Void => build_void_expr(builder, &expr.ty),
        rir::ExprKind::Int(negative, base, ref value) => {
            build_int_expr(builder, &expr.ty, negative, base, value)
        }
        rir::ExprKind::Bool(value) => build_bool_expr(builder, &expr.ty, value),
        rir::ExprKind::Func(index) => build_func_expr(builder, &expr.ty, index),
        rir::ExprKind::Variant(adt, variant) => build_variant_expr(builder, &expr.ty, adt, variant),
        rir::ExprKind::Local(index) => build_local_expr(builder, &expr.ty, index),
        rir::ExprKind::Argument(index) => build_argument_expr(builder, &expr.ty, index),
        rir::ExprKind::Tuple(ref exprs) => build_tuple_expr(builder, &expr.ty, exprs),
        rir::ExprKind::List(_) => todo!(),
        rir::ExprKind::Block(ref exprs) => build_block_expr(builder, &expr.ty, exprs),
        rir::ExprKind::Field(ref expr, index) => build_field_expr(builder, &expr.ty, expr, index),
        rir::ExprKind::VariantField(ref expr, variant, index) => {
            build_variant_field_expr(builder, &expr.ty, expr, variant, index)
        }
        rir::ExprKind::Call(ref func, ref args) => build_call_expr(builder, &expr.ty, func, args),
        rir::ExprKind::Pipe(ref lhs, ref rhs) => build_pipe_expr(builder, &expr.ty, lhs, rhs),
        rir::ExprKind::Binary(op, ref lhs, ref rhs) => {
            build_binary_expr(builder, &expr.ty, op, lhs, rhs)
        }
        rir::ExprKind::Let(index, ref expr) => build_let_expr(builder, &expr.ty, index, expr),
        rir::ExprKind::Match(ref input, ref r#match) => {
            build_match_expr(builder, &expr.ty, input, r#match)
        }
    }
}

fn build_void_expr(_builder: &mut Builder, _ty: &rir::Ty) -> miette::Result<mir::Expr> {
    let kind = mir::ExprKind::Const(mir::Constant::Void);
    let ty = mir::Ty::Void;

    Ok(mir::Expr { kind, ty })
}

fn build_int_expr(
    builder: &mut Builder,
    ty: &rir::Ty,
    signed: bool,
    base: Base,
    value: &[u8],
) -> miette::Result<mir::Expr> {
    let kind = mir::ExprKind::Const(mir::Constant::Int(signed, base, value.into()));
    let ty = build_ty(builder, ty);
    Ok(mir::Expr { kind, ty })
}

fn build_bool_expr(builder: &mut Builder, ty: &rir::Ty, value: bool) -> miette::Result<mir::Expr> {
    let kind = mir::ExprKind::Const(mir::Constant::Bool(value));
    let ty = build_ty(builder, ty);
    Ok(mir::Expr { kind, ty })
}

fn build_func_expr(builder: &mut Builder, ty: &rir::Ty, index: usize) -> miette::Result<mir::Expr> {
    let ty = build_ty(builder, ty);
    let generics = extract_generics(builder.unit, &builder.unit.funcs[index].ty(), &ty);

    let index = build_func(builder.mir, builder.funcs, builder.unit, index, &generics)?;

    let kind = mir::ExprKind::Const(mir::Constant::Func(index));
    Ok(mir::Expr { kind, ty })
}

fn build_variant_expr(
    builder: &mut Builder,
    ty: &rir::Ty,
    adt: usize,
    index: usize,
) -> miette::Result<mir::Expr> {
    let variant = &builder.unit.adts[adt].variants[index];

    if variant.fields.is_empty() {
        let kind = mir::ExprKind::Adt(index, Vec::new());
        let ty = build_ty(builder, ty);
        return Ok(mir::Expr { kind, ty });
    }

    let ty = build_ty(builder, ty);

    let mir::Ty::Func(ref input, ref output) = ty else {
        panic!("expected function");
    };

    assert_eq!(input.len(), variant.fields.len());

    let mut items = Vec::new();

    for (i, ty) in input.iter().cloned().enumerate() {
        let kind = mir::ExprKind::Argument(i);
        items.push(mir::Expr { kind, ty });
    }

    let func = mir::Func {
        input: input.clone(),
        output: output.as_ref().clone(),
        locals: Vec::new(),
        body: mir::Expr {
            kind: mir::ExprKind::Adt(index, items),
            ty: output.as_ref().clone(),
        },
    };

    let index = builder.mir.funcs.len();
    builder.mir.funcs.push(func);

    let kind = mir::ExprKind::Const(mir::Constant::Func(index));
    Ok(mir::Expr { kind, ty })
}

fn build_local_expr(
    builder: &mut Builder,
    ty: &rir::Ty,
    index: usize,
) -> miette::Result<mir::Expr> {
    let kind = mir::ExprKind::Local(index);
    let ty = build_ty(builder, ty);
    Ok(mir::Expr { kind, ty })
}

fn build_argument_expr(
    builder: &mut Builder,
    ty: &rir::Ty,
    index: usize,
) -> miette::Result<mir::Expr> {
    let kind = mir::ExprKind::Argument(index);
    let ty = build_ty(builder, ty);
    Ok(mir::Expr { kind, ty })
}

fn build_tuple_expr(
    builder: &mut Builder,
    ty: &rir::Ty,
    exprs: &[rir::Expr],
) -> miette::Result<mir::Expr> {
    let exprs = exprs
        .iter()
        .map(|expr| build_expr(builder, expr))
        .collect::<miette::Result<_>>()?;

    let kind = mir::ExprKind::Adt(0, exprs);
    let ty = build_ty(builder, ty);
    Ok(mir::Expr { kind, ty })
}

fn build_block_expr(
    builder: &mut Builder,
    ty: &rir::Ty,
    exprs: &[rir::Expr],
) -> miette::Result<mir::Expr> {
    let exprs = exprs
        .iter()
        .map(|expr| build_expr(builder, expr))
        .collect::<miette::Result<_>>()?;

    let kind = mir::ExprKind::Block(exprs);
    let ty = build_ty(builder, ty);
    Ok(mir::Expr { kind, ty })
}

fn build_field_expr(
    builder: &mut Builder,
    ty: &rir::Ty,
    expr: &rir::Expr,
    index: usize,
) -> miette::Result<mir::Expr> {
    let expr = build_expr(builder, expr)?;

    let kind = mir::ExprKind::Field(Box::new(expr), index);
    let ty = build_ty(builder, ty);
    Ok(mir::Expr { kind, ty })
}

fn build_variant_field_expr(
    builder: &mut Builder,
    ty: &rir::Ty,
    expr: &rir::Expr,
    variant: usize,
    index: usize,
) -> miette::Result<mir::Expr> {
    let expr = build_expr(builder, expr)?;

    let kind = mir::ExprKind::VariantField(Box::new(expr), variant, index);
    let ty = build_ty(builder, ty);
    Ok(mir::Expr { kind, ty })
}

fn build_call_expr(
    builder: &mut Builder,
    ty: &rir::Ty,
    func: &rir::Expr,
    args: &[rir::Expr],
) -> miette::Result<mir::Expr> {
    let func = build_expr(builder, func)?;
    let args = args
        .iter()
        .map(|arg| build_expr(builder, arg))
        .collect::<miette::Result<_>>()?;

    let kind = mir::ExprKind::Call(Box::new(func), args);
    let ty = build_ty(builder, ty);
    Ok(mir::Expr { kind, ty })
}

fn build_pipe_expr(
    builder: &mut Builder,
    ty: &rir::Ty,
    lhs: &rir::Expr,
    rhs: &rir::Expr,
) -> miette::Result<mir::Expr> {
    let rhs = build_expr(builder, rhs)?;

    let mut arguments = Vec::new();

    let mir::Ty::Func(ref args, _) = rhs.ty else {
        panic!("expected function");
    };

    match args.len() {
        1 => arguments.push(build_expr(builder, lhs)?),
        n => {
            for i in 0..n {
                let expr = build_expr(builder, lhs)?;

                let ty = match expr.ty {
                    mir::Ty::Tuple(ref items) => items[i].clone(),
                    _ => panic!("expected tuple"),
                };

                let kind = mir::ExprKind::Field(Box::new(expr), i);

                arguments.push(mir::Expr { kind, ty });
            }
        }
    }

    let kind = mir::ExprKind::Call(Box::new(rhs), arguments);
    let ty = build_ty(builder, ty);
    Ok(mir::Expr { kind, ty })
}

fn build_binary_expr(
    builder: &mut Builder,
    ty: &rir::Ty,
    op: BinOp,
    lhs: &rir::Expr,
    rhs: &rir::Expr,
) -> miette::Result<mir::Expr> {
    let lhs = build_expr(builder, lhs)?;
    let rhs = build_expr(builder, rhs)?;

    let kind = mir::ExprKind::Binary(op, Box::new(lhs), Box::new(rhs));
    let ty = build_ty(builder, ty);
    Ok(mir::Expr { kind, ty })
}

fn build_let_expr(
    builder: &mut Builder,
    ty: &rir::Ty,
    index: usize,
    expr: &rir::Expr,
) -> miette::Result<mir::Expr> {
    let expr = build_expr(builder, expr)?;

    let kind = mir::ExprKind::Let(index, Box::new(expr));
    let ty = build_ty(builder, ty);
    Ok(mir::Expr { kind, ty })
}

fn build_match_expr(
    builder: &mut Builder,
    ty: &rir::Ty,
    input: &rir::Expr,
    r#match: &rir::Match,
) -> miette::Result<mir::Expr> {
    let input = build_expr(builder, input)?;

    match r#match {
        rir::Match::Bool(r#true, r#false) => {
            let r#true = build_expr(builder, r#true)?;
            let r#false = build_expr(builder, r#false)?;

            let r#match = mir::Match::Bool(Box::new(r#true), Box::new(r#false));
            let kind = mir::ExprKind::Match(Box::new(input), r#match);
            let ty = build_ty(builder, ty);
            Ok(mir::Expr { kind, ty })
        }
        rir::Match::Adt(variants, default) => {
            let mut items = Vec::new();

            for variant in variants {
                let Some(expr) = variant else {
                    items.push(None);
                    continue;
                };

                let expr = build_expr(builder, expr)?;
                items.push(Some(expr));
            }

            let default = default
                .as_ref()
                .map(|expr| build_expr(builder, expr))
                .transpose()?;

            let r#match = mir::Match::Adt(items, default.map(Box::new));
            let kind = mir::ExprKind::Match(Box::new(input), r#match);
            let ty = build_ty(builder, ty);
            Ok(mir::Expr { kind, ty })
        }
    }
}
