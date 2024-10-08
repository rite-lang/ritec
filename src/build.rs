use miette::Severity;

use crate::{
    ast::BinOp,
    hir,
    number::{FloatKind, IntKind},
    rir,
    span::Span,
};

pub fn build(hir: &hir::Unit) -> miette::Result<rir::Unit> {
    let mut rir = rir::Unit::default();

    for adt in &hir.adts {
        let mut generics = adt
            .generics
            .iter()
            .map(|_generic| (None, rir::Generic {}))
            .collect::<Vec<_>>();

        let mut variants = Vec::new();

        for variant in &adt.variants {
            let mut fields = Vec::new();

            for field in &variant.fields {
                let ty = build_ty(hir, &mut generics, false, &field.ty)?;

                fields.push(rir::Argument {
                    ty,
                    span: Some(field.span),
                });
            }

            variants.push(rir::Variant { fields });
        }

        let generics = generics
            .into_iter()
            .map(|(_, generic)| generic)
            .collect::<Vec<_>>();

        rir.adts.push(rir::Adt {
            decorators: adt.decorators.clone(),
            name: adt.name.to_string(),
            generics,
            variants,
        });
    }

    let mut func_generics = Vec::new();

    for func in &hir.funcs {
        let mut generics = func
            .generics
            .iter()
            .map(|_generic| (None, rir::Generic {}))
            .collect::<Vec<_>>();

        let mut input = Vec::new();

        for argument in &func.input {
            let ty = build_ty(hir, &mut generics, true, &argument.ty)?;

            input.push(rir::Argument {
                ty,
                span: Some(argument.span),
            });
        }

        let output = build_ty(hir, &mut generics, true, &func.output)?;

        let mut locals = Vec::new();

        for local in &func.locals {
            let ty = build_ty(hir, &mut generics, false, &local.ty)?;

            locals.push(rir::Local { ty });
        }

        let mut captures = Vec::new();

        for capture in &func.captures {
            let ty = build_ty(hir, &mut generics, false, capture)?;

            captures.push(rir::Capture { ty });
        }

        func_generics.push(generics.clone());

        let generics = generics
            .into_iter()
            .map(|(_, generic)| generic)
            .collect::<Vec<_>>();

        rir.funcs.push(rir::Func {
            decorators: func.decorators.clone(),
            name: func.name.to_string(),
            generics,
            input,
            output,
            locals,
            captures,
            body: rir::Block::new(),
        });
    }

    for (i, func) in hir.funcs.iter().enumerate() {
        let mut builder = Builder {
            hir,
            rir: &mut rir,
            generics: &mut func_generics[i],
            func: i,
        };

        let mut block = rir::Block::new();

        let value = build_value(&mut builder, &mut block, &func.body)?;
        (block.statements).push(rir::Statement::Return { value: Some(value) });

        rir.funcs[i].body = block;
    }

    Ok(rir)
}

struct Builder<'a> {
    hir: &'a hir::Unit,
    rir: &'a mut rir::Unit,
    generics: &'a mut Generics,
    func: usize,
}

impl<'a> Builder<'a> {
    fn build_ty(&mut self, ty: &hir::Ty) -> miette::Result<rir::Ty> {
        build_ty(self.hir, self.generics, false, ty)
    }

    fn locals(&self) -> &[rir::Local] {
        &self.rir.funcs[self.func].locals
    }

    fn locals_mut(&mut self) -> &mut Vec<rir::Local> {
        &mut self.rir.funcs[self.func].locals
    }

    fn make_temp(&mut self, ty: rir::Ty) -> rir::Place {
        let index = self.locals().len();
        let local = rir::Local { ty: ty.clone() };
        self.locals_mut().push(local);

        rir::Place {
            location: rir::Location::Local(index),
            projection: Vec::new(),
            ty,
        }
    }
}

type Generics = Vec<(Option<hir::Tid>, rir::Generic)>;

fn build_place(
    builder: &mut Builder,
    block: &mut rir::Block,
    expr: &hir::Expr,
) -> miette::Result<rir::Place> {
    match expr.kind {
        hir::ExprKind::Local(index) => Ok(rir::Place {
            location: rir::Location::Local(index),
            projection: Vec::new(),
            ty: builder.build_ty(&expr.ty)?,
        }),

        hir::ExprKind::Argument(index) => Ok(rir::Place {
            location: rir::Location::Argument(index),
            projection: Vec::new(),
            ty: builder.build_ty(&expr.ty)?,
        }),

        hir::ExprKind::Capture(index) => Ok(rir::Place {
            location: rir::Location::Capture(index),
            projection: Vec::new(),
            ty: builder.build_ty(&expr.ty)?,
        }),

        hir::ExprKind::Deref(ref expr) => {
            let mut place = build_place(builder, block, expr)?;

            let rir::Ty::Ref(ty) = place.ty() else {
                unreachable!("unexpected deref: {:?}", place.ty)
            };

            place.projection.push(rir::Projection {
                kind: rir::ProjectionKind::Deref,
                ty: ty.as_ref().clone(),
                span: None,
            });

            Ok(place)
        }

        hir::ExprKind::Field(ref base, field) => {
            let mut place = build_place(builder, block, base)?;

            while let rir::Ty::Ref(ty) = place.ty() {
                place.projection.push(rir::Projection {
                    kind: rir::ProjectionKind::Deref,
                    ty: ty.as_ref().clone(),
                    span: None,
                });
            }

            let rir::Ty::Adt(index, _) = place.ty() else {
                unreachable!("unexpected field: {:?}", place.ty)
            };

            let Some((field, _)) = builder.hir.adts[*index].find_field(field) else {
                return Err(miette::miette!("field not found: {:?}", field));
            };

            place.projection.push(rir::Projection {
                kind: rir::ProjectionKind::Field {
                    variant: None,
                    field,
                },
                ty: builder.build_ty(&expr.ty)?,
                span: None,
            });

            Ok(place)
        }

        hir::ExprKind::VariantField(ref base, variant, field) => {
            let mut place = build_place(builder, block, base)?;

            assert!(matches!(place.ty(), rir::Ty::Adt(_, _)));

            place.projection.push(rir::Projection {
                kind: rir::ProjectionKind::Field {
                    variant: Some(variant),
                    field,
                },
                ty: builder.build_ty(&expr.ty)?,
                span: None,
            });

            Ok(place)
        }

        hir::ExprKind::TupleField(ref base, index) => {
            let mut place = build_place(builder, block, base)?;

            assert!(matches!(place.ty(), rir::Ty::Tuple(_)));

            place.projection.push(rir::Projection {
                kind: rir::ProjectionKind::Field {
                    variant: None,
                    field: index,
                },
                ty: builder.build_ty(&expr.ty)?,
                span: None,
            });

            Ok(place)
        }

        hir::ExprKind::Void
        | hir::ExprKind::String(_)
        | hir::ExprKind::Int(_, _, _)
        | hir::ExprKind::Bool(_)
        | hir::ExprKind::Func(_)
        | hir::ExprKind::Variant(_, _)
        | hir::ExprKind::Tuple(_)
        | hir::ExprKind::List(_, _)
        | hir::ExprKind::ListHead(_)
        | hir::ExprKind::ListTail(_)
        | hir::ExprKind::ListEmpty(_)
        | hir::ExprKind::Block(_)
        | hir::ExprKind::As(_, _)
        | hir::ExprKind::IsVariant(_, _)
        | hir::ExprKind::VariantNew(_, _, _)
        | hir::ExprKind::Call(_, _)
        | hir::ExprKind::Pipe(_, _, _)
        | hir::ExprKind::Binary(_, _, _)
        | hir::ExprKind::Unary(_, _)
        | hir::ExprKind::Ref(_)
        | hir::ExprKind::Let(_, _)
        | hir::ExprKind::Assign(_, _)
        | hir::ExprKind::Closure(_, _, _)
        | hir::ExprKind::Match(_, _)
        | hir::ExprKind::Panic(_)
        | hir::ExprKind::Return(_) => {
            let value = build_value(builder, block, expr)?;
            let ty = builder.build_ty(&expr.ty)?;
            let place = builder.make_temp(ty);

            block.statements.push(rir::Statement::Assign {
                place: place.clone(),
                value,
            });

            Ok(place)
        }
    }
}

fn build_operand(
    builder: &mut Builder,
    block: &mut rir::Block,
    expr: &hir::Expr,
) -> miette::Result<rir::Operand> {
    match expr.kind {
        hir::ExprKind::Void => Ok(rir::Operand::Constant(rir::Constant::Void)),

        hir::ExprKind::String(s) => Ok(rir::Operand::Constant(rir::Constant::String(s))),

        hir::ExprKind::Int(negative, base, ref value) => {
            // don't one-line this rustfmt
            Ok(rir::Operand::Constant(rir::Constant::Int(
                negative,
                base,
                value.clone(),
                builder.build_ty(&expr.ty)?,
            )))
        }

        hir::ExprKind::Bool(b) => Ok(rir::Operand::Constant(rir::Constant::Bool(b))),

        hir::ExprKind::Block(ref exprs) => {
            for (i, expr) in exprs.iter().enumerate() {
                if i < exprs.len() - 1 {
                    let value = build_value(builder, block, expr)?;

                    block.statements.push(rir::Statement::Use { value });
                    continue;
                }

                return build_operand(builder, block, expr);
            }

            Ok(rir::Operand::Constant(rir::Constant::Void))
        }

        hir::ExprKind::Let(index, ref value) => {
            let place = rir::Place {
                location: rir::Location::Local(index),
                projection: Vec::new(),
                ty: builder.build_ty(&value.ty)?,
            };

            let value = build_value(builder, block, value)?;

            block.statements.push(rir::Statement::Assign {
                place: place.clone(),
                value,
            });

            Ok(rir::Operand::Constant(rir::Constant::Void))
        }

        hir::ExprKind::Match(ref input, ref r#match) => {
            let index = builder.locals().len();
            let local = rir::Local {
                ty: builder.build_ty(&input.ty)?,
            };
            builder.locals_mut().push(local);

            let place = rir::Place {
                location: rir::Location::Local(index),
                projection: Vec::new(),
                ty: builder.build_ty(&expr.ty)?,
            };

            let input = build_operand(builder, block, input)?;

            match r#match {
                hir::Match::Bool(r#true, r#false) => {
                    let mut true_block = rir::Block::new();
                    let mut false_block = rir::Block::new();

                    let true_value = build_value(builder, &mut true_block, r#true)?;
                    let false_value = build_value(builder, &mut false_block, r#false)?;

                    true_block.statements.push(rir::Statement::Assign {
                        place: place.clone(),
                        value: true_value,
                    });
                    false_block.statements.push(rir::Statement::Assign {
                        place: place.clone(),
                        value: false_value,
                    });

                    block.statements.push(rir::Statement::MatchBool {
                        input,
                        r#true: true_block,
                        r#false: false_block,
                    });
                }
                hir::Match::Adt(_, variants, default) => {
                    let mut blocks = Vec::new();

                    for variant in variants {
                        let Some(variant) = variant else {
                            blocks.push(None);
                            continue;
                        };

                        let mut block = rir::Block::new();

                        let value = build_value(builder, &mut block, variant)?;
                        block.statements.push(rir::Statement::Assign {
                            place: place.clone(),
                            value,
                        });

                        blocks.push(Some(block));
                    }

                    let default_block = match default {
                        Some(default) => {
                            let mut default_block = rir::Block::new();

                            let value = build_value(builder, &mut default_block, default)?;

                            default_block.statements.push(rir::Statement::Assign {
                                place: place.clone(),
                                value,
                            });

                            Some(default_block)
                        }
                        None => None,
                    };

                    block.statements.push(rir::Statement::MatchAdt {
                        input,
                        variants: blocks,
                        default: default_block,
                    });
                }
            }

            Ok(rir::Operand::Copy(place))
        }

        hir::ExprKind::Assign(ref lhs, ref rhs) => {
            let place = build_place(builder, block, lhs)?;
            let value = build_value(builder, block, rhs)?;

            let len = place.projection.len().saturating_sub(1);
            if !(place.ty.is_mut() || place.projection[..len].iter().any(|p| p.ty.is_mut()))
                || place.projection.is_empty()
            {
                return Err(miette::miette!("cannot assign to immutable place"));
            }

            block.statements.push(rir::Statement::Assign {
                place: place.clone(),
                value,
            });

            Ok(rir::Operand::Copy(place))
        }

        hir::ExprKind::Panic(message) => {
            block.statements.push(rir::Statement::Panic { message });

            Ok(rir::Operand::Constant(rir::Constant::Void))
        }

        hir::ExprKind::Return(ref expr) => {
            let value = build_value(builder, block, expr)?;

            (block.statements).push(rir::Statement::Return { value: Some(value) });

            Ok(rir::Operand::Constant(rir::Constant::Void))
        }

        hir::ExprKind::Func(_)
        | hir::ExprKind::Variant(_, _)
        | hir::ExprKind::Tuple(_)
        | hir::ExprKind::List(_, _)
        | hir::ExprKind::ListHead(_)
        | hir::ExprKind::ListTail(_)
        | hir::ExprKind::ListEmpty(_)
        | hir::ExprKind::As(_, _)
        | hir::ExprKind::IsVariant(_, _)
        | hir::ExprKind::VariantNew(_, _, _)
        | hir::ExprKind::Call(_, _)
        | hir::ExprKind::Pipe(_, _, _)
        | hir::ExprKind::Binary(_, _, _)
        | hir::ExprKind::Unary(_, _)
        | hir::ExprKind::Ref(_)
        | hir::ExprKind::Closure(_, _, _) => {
            let value = build_value(builder, block, expr)?;
            let ty = builder.build_ty(&expr.ty)?;
            let place = builder.make_temp(ty);

            block.statements.push(rir::Statement::Assign {
                place: place.clone(),
                value,
            });

            Ok(rir::Operand::Copy(place))
        }

        hir::ExprKind::Local(_)
        | hir::ExprKind::Argument(_)
        | hir::ExprKind::Capture(_)
        | hir::ExprKind::Field(_, _)
        | hir::ExprKind::VariantField(_, _, _)
        | hir::ExprKind::TupleField(_, _)
        | hir::ExprKind::Deref(_) => {
            let place = build_place(builder, block, expr)?;

            Ok(rir::Operand::Copy(place))
        }
    }
}

fn build_value(
    builder: &mut Builder,
    block: &mut rir::Block,
    expr: &hir::Expr,
) -> miette::Result<rir::Value> {
    match expr.kind {
        hir::ExprKind::Func(index) => {
            let mut generics = Vec::new();

            let func = &builder.rir.funcs[index].ty();
            let ty = builder.build_ty(&expr.ty)?;

            extract_generics(&ty, func, &mut generics);

            let generics = generics.into_iter().map(Option::unwrap).collect();

            Ok(rir::Value::Func(index, Vec::new(), generics))
        }

        hir::ExprKind::List(ref exprs, ref rest) => {
            let exprs = exprs
                .iter()
                .map(|expr| build_operand(builder, block, expr))
                .collect::<miette::Result<_>>()?;

            let rest = rest
                .as_ref()
                .map(|expr| build_operand(builder, block, expr))
                .transpose()?;

            Ok(rir::Value::List(exprs, rest))
        }

        hir::ExprKind::ListHead(ref list) => {
            let list = build_operand(builder, block, list)?;

            Ok(rir::Value::ListHead(list))
        }

        hir::ExprKind::ListTail(ref list) => {
            let list = build_operand(builder, block, list)?;

            Ok(rir::Value::ListTail(list))
        }

        hir::ExprKind::ListEmpty(ref list) => {
            let list = build_operand(builder, block, list)?;
            Ok(rir::Value::ListEmpty(list))
        }

        hir::ExprKind::Binary(op, ref lhs, ref rhs) => match op {
            BinOp::And => {
                let lhs = build_operand(builder, block, lhs)?;

                let mut rhs_block = rir::Block::new();

                let rhs = build_value(builder, &mut rhs_block, rhs)?;

                let result = builder.make_temp(rir::Ty::Bool);
                rhs_block.statements.push(rir::Statement::Assign {
                    place: result.clone(),
                    value: rhs,
                });

                let mut false_block = rir::Block::new();
                false_block.statements.push(rir::Statement::Assign {
                    place: result.clone(),
                    value: rir::Value::Use(rir::Operand::Constant(rir::Constant::Bool(false))),
                });

                block.statements.push(rir::Statement::MatchBool {
                    input: lhs,
                    r#true: rhs_block,
                    r#false: false_block,
                });

                Ok(rir::Value::Use(rir::Operand::Move(result)))
            }
            _ => {
                let lhs = build_operand(builder, block, lhs)?;
                let rhs = build_operand(builder, block, rhs)?;

                Ok(rir::Value::Binary(op, lhs, rhs))
            }
        },

        hir::ExprKind::Unary(op, ref expr) => {
            let expr = build_operand(builder, block, expr)?;

            Ok(rir::Value::Unary(op, expr))
        }

        hir::ExprKind::As(ref expr, ref ty) => {
            let from_ty = builder.build_ty(&expr.ty)?;
            let to_ty = builder.build_ty(ty)?;
            let expr = build_operand(builder, block, expr)?;

            match to_ty {
                rir::Ty::Int(kind) => {
                    if !matches!(from_ty, rir::Ty::Int(_)) {
                        return Err(miette::miette!(
                            "invalid cast: {:?} as {:?}",
                            from_ty,
                            to_ty
                        ));
                    }

                    Ok(rir::Value::Cast(rir::Cast::Int(kind), expr))
                }
                _ => Err(miette::miette!(
                    "invalid cast: {:?} as {:?}",
                    from_ty,
                    to_ty
                )),
            }
        }

        hir::ExprKind::Tuple(ref exprs) => {
            let exprs = exprs
                .iter()
                .map(|expr| build_operand(builder, block, expr))
                .collect::<miette::Result<_>>()?;

            Ok(rir::Value::Tuple(exprs))
        }

        hir::ExprKind::Variant(adt_index, variant_index) => {
            let ty = builder.build_ty(&expr.ty)?;
            let adt = &builder.rir.adts[adt_index];
            let variant = &adt.variants[variant_index];

            if variant.fields.is_empty() {
                return Ok(rir::Value::Adt(variant_index, Vec::new()));
            }

            let rir::Ty::Func(ref input, ref output) = ty else {
                unreachable!("unexpected variant: {:?}", ty)
            };

            assert_eq!(input.len(), variant.fields.len());

            let mut items = Vec::new();

            for (i, ty) in input.iter().cloned().enumerate() {
                let place = rir::Place {
                    location: rir::Location::Argument(i),
                    projection: Vec::new(),
                    ty,
                };
                items.push(rir::Operand::Move(place));
            }

            let body = rir::Block {
                statements: vec![rir::Statement::Return {
                    value: Some(rir::Value::Adt(variant_index, items)),
                }],
            };

            let input = input
                .iter()
                .cloned()
                .map(|ty| rir::Argument { ty, span: None })
                .collect();

            let output = output.as_ref().clone();

            let func = rir::Func {
                decorators: Vec::new(),
                name: String::new(),
                generics: builder.generics.iter().map(|(_, g)| g.clone()).collect(),
                input,
                output,
                locals: Vec::new(),
                captures: Vec::new(),
                body,
            };

            let index = builder.rir.funcs.len();
            builder.rir.funcs.push(func);

            let generics = (0..builder.generics.len()).map(rir::Ty::Generic).collect();

            Ok(rir::Value::Func(index, Vec::new(), generics))
        }

        hir::ExprKind::IsVariant(ref expr, variant) => {
            let expr = build_operand(builder, block, expr)?;
            Ok(rir::Value::IsVariant(expr, variant))
        }

        hir::ExprKind::VariantNew(_, variant, ref exprs) => {
            let mut items = Vec::new();

            for expr in exprs {
                items.push(build_operand(builder, block, expr)?);
            }

            Ok(rir::Value::Adt(variant, items))
        }

        hir::ExprKind::Call(ref func, ref args) => {
            let func_ty = builder.build_ty(&func.ty)?;

            let func = build_operand(builder, block, func)?;

            let rir::Ty::Func(ref input, ref output) = func_ty else {
                unreachable!("unexpected call: {:?}", func_ty)
            };

            let output = output.as_ref().clone();

            if !args.iter().any(Option::is_none) && args.len() == input.len() {
                let args = args
                    .iter()
                    .map(Option::as_ref)
                    .map(Option::unwrap)
                    .map(|arg| build_operand(builder, block, arg))
                    .collect::<miette::Result<_>>()?;

                let temp = builder.make_temp(output);

                block.statements.push(rir::Statement::Call {
                    place: temp.clone(),
                    func,
                    args,
                });

                return Ok(rir::Value::Use(rir::Operand::Move(temp)));
            }

            let mut arguments = Vec::new();
            let mut captured = Vec::new();
            let mut captures = Vec::new();

            captured.push(func);
            captures.push(rir::Capture {
                ty: func_ty.clone(),
            });

            let mut operands = Vec::new();

            for (provided, arg) in args.iter().zip(input.iter()) {
                match provided {
                    Some(arg) => {
                        let operand = build_operand(builder, block, arg)?;

                        let ty = builder.build_ty(&arg.ty)?;

                        let index = captured.len();
                        captured.push(operand);
                        captures.push(rir::Capture { ty: ty.clone() });

                        let place = rir::Place {
                            location: rir::Location::Capture(index),
                            projection: Vec::new(),
                            ty,
                        };

                        operands.push(rir::Operand::Copy(place));
                    }
                    None => {
                        let index = arguments.len();

                        let place = rir::Place {
                            location: rir::Location::Argument(index),
                            projection: Vec::new(),
                            ty: arg.clone(),
                        };

                        operands.push(rir::Operand::Move(place));

                        arguments.push(rir::Argument {
                            ty: arg.clone(),
                            span: None,
                        });
                    }
                }
            }

            for arg in input.iter().skip(args.len()) {
                let index = arguments.len();

                let place = rir::Place {
                    location: rir::Location::Argument(index),
                    projection: Vec::new(),
                    ty: arg.clone(),
                };

                operands.push(rir::Operand::Move(place));

                arguments.push(rir::Argument {
                    ty: arg.clone(),
                    span: None,
                });
            }

            let func = rir::Place {
                location: rir::Location::Capture(0),
                projection: Vec::new(),
                ty: func_ty,
            };

            let temp = rir::Place {
                location: rir::Location::Local(0),
                projection: Vec::new(),
                ty: output.clone(),
            };

            let body = rir::Block {
                statements: vec![
                    rir::Statement::Call {
                        place: temp.clone(),
                        func: rir::Operand::Copy(func),
                        args: operands,
                    },
                    rir::Statement::Return {
                        value: Some(rir::Value::Use(rir::Operand::Move(temp))),
                    },
                ],
            };

            let func = rir::Func {
                decorators: Vec::new(),
                name: String::new(),
                generics: builder.generics.iter().map(|(_, g)| g.clone()).collect(),
                input: arguments,
                locals: vec![rir::Local { ty: output.clone() }],
                output,
                captures,
                body,
            };

            let index = builder.rir.funcs.len();
            builder.rir.funcs.push(func);

            let generics = (0..builder.generics.len()).map(rir::Ty::Generic).collect();

            Ok(rir::Value::Func(index, captured, generics))
        }

        hir::ExprKind::Pipe(ref lhs, ref rhs, ref arguments) => {
            let rhs_ty = builder.build_ty(&rhs.ty)?;
            let rhs = build_operand(builder, block, rhs)?;

            let rir::Ty::Func(ref input, ref output) = rhs_ty else {
                unreachable!("unexpected pipe: {:?}", rhs_ty)
            };

            let missing = input.len() - arguments.len();
            let slots = arguments
                .iter()
                .fold(0, |slots, arg| slots + arg.is_none() as usize);

            let is_tuple = missing + slots > 1;

            if !is_tuple {
                let mut operands = Vec::new();

                if missing > 0 {
                    assert_eq!(missing, 1);

                    operands.push(build_operand(builder, block, lhs)?);
                }

                for arg in arguments {
                    match arg {
                        Some(arg) => operands.push(build_operand(builder, block, arg)?),
                        None => operands.push(build_operand(builder, block, lhs)?),
                    }
                }

                // Emit a temporary local to safe the results of the call
                // Return the value of the temporary local
                let temp = builder.make_temp(output.as_ref().clone());

                block.statements.push(rir::Statement::Call {
                    place: temp.clone(),
                    func: rhs,
                    args: operands,
                });

                return Ok(rir::Value::Use(rir::Operand::Move(temp)));
            }

            let lhs = build_place(builder, block, lhs)?;
            let rir::Ty::Tuple(ref input) = lhs.ty() else {
                unreachable!("unexpected pipe: {:?}", lhs.ty)
            };

            let mut operands = Vec::new();

            for (i, ty) in input.iter().enumerate() {
                let mut place = lhs.clone();
                place.projection.push(rir::Projection {
                    kind: rir::ProjectionKind::Field {
                        variant: None,
                        field: i,
                    },
                    ty: ty.clone(),
                    span: None,
                });

                operands.push(rir::Operand::Copy(place));
            }

            let mut index = missing;

            for arg in arguments {
                match arg {
                    Some(arg) => operands.push(build_operand(builder, block, arg)?),
                    None => {
                        let mut place = lhs.clone();
                        place.projection.push(rir::Projection {
                            kind: rir::ProjectionKind::Field {
                                variant: None,
                                field: index,
                            },
                            ty: input[index].clone(),
                            span: None,
                        });

                        operands.push(rir::Operand::Copy(place));
                        index += 1;
                    }
                }
            }

            // Make function call by assigning to temp variable and returning that.
            let temp = builder.make_temp(output.as_ref().clone());

            block.statements.push(rir::Statement::Call {
                place: temp.clone(),
                func: rhs,
                args: operands,
            });

            Ok(rir::Value::Use(rir::Operand::Move(temp)))
        }

        hir::ExprKind::Ref(ref expr) => {
            let place = build_place(builder, block, expr)?;
            Ok(rir::Value::Ref(place))
        }

        hir::ExprKind::Closure(ref locals, ref captured, ref body) => {
            let ty = builder.build_ty(&expr.ty)?;

            let rir::Ty::Func(ref input, ref output) = ty else {
                unreachable!("unexpected closure: {:?}", ty)
            };

            let generics = builder.generics.iter().map(|(_, g)| g.clone()).collect();

            let input = input
                .iter()
                .map(|arg| rir::Argument {
                    ty: arg.clone(),
                    span: None,
                })
                .collect();

            let output = output.as_ref().clone();

            let locals = locals
                .iter()
                .map(|local| {
                    Ok(rir::Local {
                        ty: builder.build_ty(&local.ty)?,
                    })
                })
                .collect::<miette::Result<_>>()?;

            let captures = captured
                .iter()
                .map(|capture| {
                    Ok(rir::Capture {
                        ty: builder.build_ty(&capture.ty)?,
                    })
                })
                .collect::<miette::Result<_>>()?;

            let func = rir::Func {
                decorators: Vec::new(),
                name: String::from("closure"),
                generics,
                input,
                output,
                locals,
                captures,
                body: rir::Block::new(),
            };

            let index = builder.rir.funcs.len();
            builder.rir.funcs.push(func);

            {
                let mut builder = Builder {
                    hir: builder.hir,
                    rir: builder.rir,
                    generics: builder.generics,
                    func: index,
                };

                let mut block = rir::Block::new();

                let value = build_value(&mut builder, &mut block, body)?;
                (block.statements).push(rir::Statement::Return { value: Some(value) });

                builder.rir.funcs[index].body = block;
            }

            let captured = captured
                .iter()
                .map(|capture| build_operand(builder, block, capture))
                .collect::<miette::Result<_>>()?;

            let generics = (0..builder.generics.len()).map(rir::Ty::Generic).collect();

            Ok(rir::Value::Func(index, captured, generics))
        }

        hir::ExprKind::Void
        | hir::ExprKind::String(_)
        | hir::ExprKind::Int(_, _, _)
        | hir::ExprKind::Bool(_)
        | hir::ExprKind::Local(_)
        | hir::ExprKind::Argument(_)
        | hir::ExprKind::Capture(_)
        | hir::ExprKind::Block(_)
        | hir::ExprKind::Field(_, _)
        | hir::ExprKind::VariantField(_, _, _)
        | hir::ExprKind::TupleField(_, _)
        | hir::ExprKind::Deref(_)
        | hir::ExprKind::Let(_, _)
        | hir::ExprKind::Assign(_, _)
        | hir::ExprKind::Match(_, _)
        | hir::ExprKind::Panic(_)
        | hir::ExprKind::Return(_) => {
            let operand = build_operand(builder, block, expr)?;

            Ok(rir::Value::Use(operand))
        }
    }
}

fn build_ty(
    unit: &hir::Unit,
    generics: &mut Generics,
    new_generics: bool,
    ty: &hir::Ty,
) -> miette::Result<rir::Ty> {
    fn build_inferred(
        generics: &mut Generics,
        new_generics: bool,
        ty: &hir::Ty,
    ) -> Result<rir::Ty, miette::Result<Span>> {
        let hir::Ty::Inferred(tid, kind, func, span) = *ty else {
            unreachable!()
        };

        if let Some(index) = generics.iter().position(|(t, _)| *t == Some(tid)) {
            return Ok(rir::Ty::Generic(index));
        }

        if func.is_some() && new_generics {
            let index = generics.len();
            generics.push((Some(tid), rir::Generic {}));
            return Ok(rir::Ty::Generic(index));
        }

        match kind {
            hir::Inferred::Any => Err(Ok(span)),
            hir::Inferred::Unsigned => Ok(rir::Ty::Int(IntKind::Int)),
            hir::Inferred::Signed => Ok(rir::Ty::Int(IntKind::Int)),
            hir::Inferred::Float => Ok(rir::Ty::Float(FloatKind::F64)),
        }
    }

    fn build_partial(
        unit: &hir::Unit,
        generics: &mut Generics,
        new_generics: bool,
        ty: &hir::Ty,
    ) -> Result<rir::Ty, miette::Result<Span>> {
        let hir::Ty::Partial(part, ref arguments, _) = *ty else {
            unreachable!()
        };

        match part {
            hir::Part::Void => {
                assert!(arguments.is_empty());
                Ok(rir::Ty::Void)
            }
            hir::Part::Bool => {
                assert!(arguments.is_empty());
                Ok(rir::Ty::Bool)
            }
            hir::Part::Str => {
                assert!(arguments.is_empty());
                Ok(rir::Ty::Str)
            }
            hir::Part::List => {
                assert_eq!(arguments.len(), 1);
                Ok(rir::Ty::List(Box::new(recurse(
                    unit,
                    generics,
                    new_generics,
                    &arguments[0],
                )?)))
            }
            hir::Part::Ref => {
                assert_eq!(arguments.len(), 1);
                Ok(rir::Ty::Ref(Box::new(recurse(
                    unit,
                    generics,
                    new_generics,
                    &arguments[0],
                )?)))
            }
            hir::Part::Tuple => {
                let tuple = recurse_vec(unit, generics, new_generics, arguments)?;
                Ok(rir::Ty::Tuple(tuple))
            }
            hir::Part::Func => {
                let input = arguments
                    .iter()
                    .take(arguments.len() - 1)
                    .map(|ty| recurse(unit, generics, new_generics, ty))
                    .collect::<Result<_, _>>()?;

                let output = arguments.last().unwrap();
                let output = recurse(unit, generics, new_generics, output)?;

                Ok(rir::Ty::Func(input, Box::new(output)))
            }
            hir::Part::Int(kind) => {
                assert!(arguments.is_empty());
                Ok(rir::Ty::Int(kind))
            }
            hir::Part::Adt(index) => {
                let generics = recurse_vec(unit, generics, new_generics, arguments)?;
                Ok(rir::Ty::Adt(index, generics))
            }
            hir::Part::Generic(index, _) => {
                assert!(arguments.is_empty());
                Ok(rir::Ty::Generic(index))
            }
        }
    }

    fn recurse(
        unit: &hir::Unit,
        generics: &mut Generics,
        new_generics: bool,
        ty: &hir::Ty,
    ) -> Result<rir::Ty, miette::Result<Span>> {
        let ty = unit.env.get(ty);

        match ty {
            hir::Ty::Inferred(..) => build_inferred(generics, new_generics, &ty),
            hir::Ty::Partial(..) => match build_partial(unit, generics, new_generics, &ty) {
                Ok(ty) => Ok(ty),
                Err(err) => Err(wrap_err(unit, err, &ty)),
            },
            _ => unreachable!(),
        }
    }

    fn recurse_vec(
        unit: &hir::Unit,
        generics: &mut Generics,
        new_generics: bool,
        tys: &[hir::Ty],
    ) -> Result<Vec<rir::Ty>, miette::Result<Span>> {
        tys.iter()
            .map(|ty| recurse(unit, generics, new_generics, ty))
            .collect::<Result<_, _>>()
    }

    fn wrap_err<T>(
        unit: &hir::Unit,
        result: miette::Result<Span>,
        ty: &hir::Ty,
    ) -> miette::Result<T> {
        let span = result?;

        Err(miette::miette!(
            severity = Severity::Error,
            code = "unspecified::type",
            help = "consider adding type annotations",
            labels = [span.label("here")],
            "could not infer type {}",
            ty.format(unit),
        )
        .with_source_code(span))
    }

    match recurse(unit, generics, new_generics, ty) {
        Ok(ty) => Ok(ty),
        Err(err) => wrap_err(unit, err, ty),
    }
}

fn build_ty_vec(
    unit: &hir::Unit,
    generics: &mut Generics,
    new_generics: bool,
    tys: &[hir::Ty],
) -> miette::Result<Vec<rir::Ty>> {
    tys.iter()
        .map(|ty| build_ty(unit, generics, new_generics, ty))
        .collect::<miette::Result<_>>()
}

fn extract_generics(ty: &rir::Ty, from: &rir::Ty, generics: &mut Vec<Option<rir::Ty>>) {
    match (ty, from) {
        (rir::Ty::Void, rir::Ty::Void) => {}
        (rir::Ty::Bool, rir::Ty::Bool) => {}
        (rir::Ty::Str, rir::Ty::Str) => {}
        (rir::Ty::Int(a), rir::Ty::Int(b)) => {
            assert_eq!(a, b);
        }
        (rir::Ty::Ref(a), rir::Ty::Ref(b)) => extract_generics(a, b, generics),
        (rir::Ty::List(a), rir::Ty::List(b)) => extract_generics(a, b, generics),
        (rir::Ty::Tuple(a), rir::Ty::Tuple(b)) => {
            assert_eq!(a.len(), b.len());

            for (a, b) in a.iter().zip(b.iter()) {
                extract_generics(a, b, generics);
            }
        }
        (rir::Ty::Func(a, b), rir::Ty::Func(c, d)) => {
            assert_eq!(a.len(), c.len());

            for (a, c) in a.iter().zip(c.iter()) {
                extract_generics(a, c, generics);
            }

            extract_generics(b, d, generics);
        }
        (rir::Ty::Adt(a, b), rir::Ty::Adt(c, d)) => {
            assert_eq!(a, c);

            for (a, b) in b.iter().zip(d.iter()) {
                extract_generics(a, b, generics);
            }
        }
        (ty, rir::Ty::Generic(index)) => {
            if generics.len() <= *index {
                generics.resize_with(*index + 1, || None);
            }

            match &generics[*index] {
                Some(generic) => assert_eq!(ty, generic, "generics {:?}", generics),
                None => generics[*index] = Some(ty.clone()),
            }
        }
        (_, _) => unreachable!("unexpected type: {:?} != {:?}", ty, from),
    }
}
