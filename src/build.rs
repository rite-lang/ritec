use crate::{hir, rir};

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

        hir::ExprKind::Deref(ref expr) => {
            let mut place = build_place(builder, block, expr)?;

            assert!(matches!(place.ty, rir::Ty::Mut(_)));

            place.projection.push(rir::Projection {
                kind: rir::ProjectionKind::Deref,
                ty: builder.build_ty(&expr.ty)?,
                span: None,
            });

            Ok(place)
        }

        hir::ExprKind::Field(ref expr, field) => {
            let mut place = build_place(builder, block, expr)?;

            let rir::Ty::Adt(index, _) = place.ty else {
                unreachable!("unexpected field: {:?}", place.ty)
            };

            let (field, _) = builder.hir.adts[index].find_field(field)?;

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

        hir::ExprKind::VariantField(ref expr, variant, field) => {
            let mut place = build_place(builder, block, expr)?;

            assert!(matches!(place.ty, rir::Ty::Adt(_, _)));

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

        hir::ExprKind::TupleField(ref expr, index) => {
            let mut place = build_place(builder, block, expr)?;

            assert!(matches!(place.ty, rir::Ty::Tuple(_)));

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
        | hir::ExprKind::Block(_)
        | hir::ExprKind::Call(_, _)
        | hir::ExprKind::Pipe(_, _)
        | hir::ExprKind::Binary(_, _, _)
        | hir::ExprKind::Mut(_)
        | hir::ExprKind::Let(_, _)
        | hir::ExprKind::Assign(_, _)
        | hir::ExprKind::Closure(_, _)
        | hir::ExprKind::Match(_, _) => {
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

        hir::ExprKind::Int(negative, base, ref value) => Ok(rir::Operand::Constant(
            rir::Constant::Int(negative, base, value.clone()),
        )),

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

            Ok(rir::Operand::Copy(place))
        }

        hir::ExprKind::Match(ref expr, ref r#match) => {
            let input = build_operand(builder, block, expr)?;

            let index = builder.locals().len();
            let local = rir::Local {
                ty: builder.build_ty(&expr.ty)?,
            };
            builder.locals_mut().push(local);

            let place = rir::Place {
                location: rir::Location::Local(index),
                projection: Vec::new(),
                ty: builder.build_ty(&expr.ty)?,
            };

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
                hir::Match::List(some, none) => {
                    let mut some_block = rir::Block::new();
                    let mut none_block = rir::Block::new();

                    let some_value = build_value(builder, &mut some_block, some)?;
                    let none_value = build_value(builder, &mut none_block, none)?;

                    some_block.statements.push(rir::Statement::Assign {
                        place: place.clone(),
                        value: some_value,
                    });
                    none_block.statements.push(rir::Statement::Assign {
                        place: place.clone(),
                        value: none_value,
                    });

                    block.statements.push(rir::Statement::MatchList {
                        input,
                        some: some_block,
                        none: none_block,
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

        hir::ExprKind::Func(_)
        | hir::ExprKind::Variant(_, _)
        | hir::ExprKind::Tuple(_)
        | hir::ExprKind::List(_, _)
        | hir::ExprKind::ListHead(_)
        | hir::ExprKind::ListTail(_)
        | hir::ExprKind::Call(_, _)
        | hir::ExprKind::Pipe(_, _)
        | hir::ExprKind::Binary(_, _, _)
        | hir::ExprKind::Mut(_)
        | hir::ExprKind::Closure(_, _) => {
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

        hir::ExprKind::Binary(op, ref lhs, ref rhs) => {
            let lhs = build_operand(builder, block, lhs)?;
            let rhs = build_operand(builder, block, rhs)?;

            Ok(rir::Value::Binary(op, lhs, rhs))
        }

        hir::ExprKind::Tuple(ref exprs) => {
            let exprs = exprs
                .iter()
                .map(|expr| build_operand(builder, block, expr))
                .collect::<miette::Result<_>>()?;

            Ok(rir::Value::Tuple(exprs))
        }

        hir::ExprKind::Variant(adt, index) => {
            let ty = builder.build_ty(&expr.ty)?;
            let adt = &builder.rir.adts[adt];
            let variant = &adt.variants[index];

            if variant.fields.is_empty() {
                return Ok(rir::Value::Adt(index, Vec::new()));
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
                    value: Some(rir::Value::Adt(index, items)),
                }],
            };

            let input = input
                .iter()
                .cloned()
                .map(|ty| rir::Argument { ty, span: None })
                .collect();

            let output = output.as_ref().clone();

            let func = rir::Func {
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

        hir::ExprKind::Call(ref func, ref args) => {
            let func_ty = builder.build_ty(&func.ty)?;

            let func = build_operand(builder, block, func)?;

            let rir::Ty::Func(ref input, ref output) = func_ty else {
                unreachable!("unexpected call: {:?}", func_ty)
            };

            if !args.iter().any(Option::is_none) && args.len() == input.len() {
                let args = args
                    .iter()
                    .map(Option::as_ref)
                    .map(Option::unwrap)
                    .map(|arg| build_operand(builder, block, arg))
                    .collect::<miette::Result<_>>()?;

                return Ok(rir::Value::Call(func, args));
            }

            let output = output.as_ref().clone();

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

            let body = rir::Block {
                statements: vec![rir::Statement::Return {
                    value: Some(rir::Value::Call(rir::Operand::Copy(func), operands)),
                }],
            };

            let func = rir::Func {
                name: String::new(),
                generics: builder.generics.iter().map(|(_, g)| g.clone()).collect(),
                input: arguments,
                output,
                locals: Vec::new(),
                captures,
                body,
            };

            let index = builder.rir.funcs.len();
            builder.rir.funcs.push(func);

            let generics = (0..builder.generics.len()).map(rir::Ty::Generic).collect();

            Ok(rir::Value::Func(index, captured, generics))
        }

        hir::ExprKind::Pipe(ref lhs, ref rhs) => {
            let rhs_ty = builder.build_ty(&rhs.ty)?;
            let rhs = build_operand(builder, block, rhs)?;

            let rir::Ty::Func(ref input, _) = rhs_ty else {
                unreachable!("unexpected pipe: {:?}", rhs_ty)
            };

            let mut arguments = Vec::new();

            match input.len() {
                1 => arguments.push(build_operand(builder, block, lhs)?),
                _ => {
                    let place = build_place(builder, block, lhs)?;

                    for (i, ty) in input.iter().enumerate() {
                        let mut place = place.clone();

                        place.projection.push(rir::Projection {
                            kind: rir::ProjectionKind::Field {
                                variant: None,
                                field: i,
                            },
                            ty: ty.clone(),
                            span: None,
                        });

                        arguments.push(rir::Operand::Copy(place));
                    }
                }
            }

            Ok(rir::Value::Call(rhs, arguments))
        }

        hir::ExprKind::Mut(ref expr) => {
            let place = build_place(builder, block, expr)?;
            Ok(rir::Value::Mut(place))
        }

        hir::ExprKind::Closure(index, ref captures) => {
            let ty = builder.build_ty(&expr.ty)?;
            let func = &builder.rir.funcs[index];

            let mut generics = Vec::new();

            extract_generics(&ty, &func.ty(), &mut generics);
            let generics = generics.into_iter().map(Option::unwrap).collect();

            let mut operands = Vec::new();

            for capture in captures {
                let operand = build_operand(builder, block, capture)?;
                operands.push(operand);
            }

            Ok(rir::Value::Func(index, operands, generics))
        }

        hir::ExprKind::Void
        | hir::ExprKind::String(_)
        | hir::ExprKind::Int(_, _, _)
        | hir::ExprKind::Bool(_)
        | hir::ExprKind::Local(_)
        | hir::ExprKind::Argument(_)
        | hir::ExprKind::Block(_)
        | hir::ExprKind::Field(_, _)
        | hir::ExprKind::VariantField(_, _, _)
        | hir::ExprKind::TupleField(_, _)
        | hir::ExprKind::Deref(_)
        | hir::ExprKind::Let(_, _)
        | hir::ExprKind::Assign(_, _)
        | hir::ExprKind::Match(_, _) => {
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
    match unit.env.get(ty) {
        hir::Ty::Inferred(tid, kind, func) => {
            if let Some(index) = generics.iter().position(|(t, _)| *t == Some(tid)) {
                return Ok(rir::Ty::Generic(index));
            }

            if func.is_some() && new_generics {
                let index = generics.len();
                generics.push((Some(tid), rir::Generic {}));
                return Ok(rir::Ty::Generic(index));
            }

            match kind {
                hir::Inferred::Any => todo!("{:?}", ty),
                hir::Inferred::Int(kind) => Ok(rir::Ty::Int(kind)),
                hir::Inferred::Float(_) => todo!(),
            }
        }
        hir::Ty::Partial(part, arguments) => match part {
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
                Ok(rir::Ty::List(Box::new(build_ty(
                    unit,
                    generics,
                    new_generics,
                    &arguments[0],
                )?)))
            }
            hir::Part::Mut => {
                assert_eq!(arguments.len(), 1);
                Ok(rir::Ty::Mut(Box::new(build_ty(
                    unit,
                    generics,
                    new_generics,
                    &arguments[0],
                )?)))
            }
            hir::Part::Tuple => {
                let tuple = build_ty_vec(unit, generics, new_generics, &arguments)?;
                Ok(rir::Ty::Tuple(tuple))
            }
            hir::Part::Func => {
                let input = arguments
                    .iter()
                    .take(arguments.len() - 1)
                    .map(|ty| build_ty(unit, generics, new_generics, ty))
                    .collect::<miette::Result<_>>()?;

                let output = arguments.last().unwrap();
                let output = build_ty(unit, generics, new_generics, output)?;

                Ok(rir::Ty::Func(input, Box::new(output)))
            }
            hir::Part::Int(kind) => {
                assert!(arguments.is_empty());
                Ok(rir::Ty::Int(kind))
            }
            hir::Part::Adt(index) => {
                let generics = build_ty_vec(unit, generics, new_generics, &arguments)?;
                Ok(rir::Ty::Adt(index, generics))
            }
            hir::Part::Generic(index) => {
                assert!(arguments.is_empty());
                Ok(rir::Ty::Generic(index))
            }
        },
        hir::Ty::Field(_, _) | hir::Ty::Tuple(_, _) | hir::Ty::Call(_, _) | hir::Ty::Pipe(_, _) => {
            unreachable!("unexpected field, call or pipe: {}", ty)
        }
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
        (rir::Ty::Mut(a), rir::Ty::Mut(b)) => extract_generics(a, b, generics),
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
                Some(generic) => assert_eq!(ty, generic),
                None => generics[*index] = Some(ty.clone()),
            }
        }
        (_, _) => unreachable!("unexpected type: {:?} != {:?}", ty, from),
    }
}
