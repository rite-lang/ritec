use std::{
    collections::HashMap,
    ops::{Index, IndexMut},
};

use mir::{Blocks, Locals};
use ritec_diagnostic::Diagnostic;
use ritec_hir as hir;
use ritec_mir as mir;

type BlockAnd<T> = (mir::BlockId, T);

macro_rules! unpack {
    ($block:ident = $block_and:expr) => {{
        let (block, value) = $block_and;
        $block = block;
        value
    }};
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct BodyKey {
    body: hir::BodyId,
    generics: Vec<mir::Type>,
}

struct BuildContext<'hir> {
    bodies: HashMap<BodyKey, mir::BodyId>,
    hir: &'hir hir::Unit,
    mir: mir::Unit,
}

struct Builder<'a, 'bcx> {
    bcx: &'a mut BuildContext<'bcx>,
    locals: &'a mut Locals,
    blocks: &'a mut Blocks,
    body: &'a hir::Body,
    specialization: hir::Spec,
}

impl Index<mir::BlockId> for Builder<'_, '_> {
    type Output = mir::Block;

    fn index(&self, index: mir::BlockId) -> &Self::Output {
        &self.blocks[index]
    }
}

impl IndexMut<mir::BlockId> for Builder<'_, '_> {
    fn index_mut(&mut self, index: mir::BlockId) -> &mut Self::Output {
        &mut self.blocks[index]
    }
}

impl<'bcx, 'hir> Builder<'bcx, 'hir> {
    fn create_block(&mut self) -> mir::BlockId {
        self.blocks.push(mir::Block::new())
    }

    fn build_type(&mut self, hir: &hir::Type) -> Result<mir::Type, Diagnostic> {
        let known = self.bcx.hir.types.know(hir, &self.specialization)?;

        self.build_known(known)
    }

    fn build_known(&mut self, known: hir::Known) -> Result<mir::Type, Diagnostic> {
        Ok(match known.item {
            hir::Item::Void => mir::Type::VOID,
            hir::Item::Bool => mir::Type::Bool,
            hir::Item::Int { signed, width } => mir::Type::Int { signed, width },
            hir::Item::Float { width } => mir::Type::Float { width },
            hir::Item::Pointer { mutable } => {
                let pointee = self.build_known(known.params[0].clone())?;

                mir::Type::Pointer {
                    mutable,
                    pointee: Box::new(pointee),
                }
            }
            hir::Item::Tuple => {
                let fields = known
                    .params
                    .into_iter()
                    .map(|ty| self.build_known(ty))
                    .collect::<Result<_, _>>()?;

                mir::Type::Struct { fields }
            }
            hir::Item::Slice => {
                let element = self.build_known(known.params[0].clone())?;

                mir::Type::Array {
                    element: Box::new(element),
                    length: 0,
                }
            }
            hir::Item::Function => {
                let mut params = known.params.into_iter();

                let output = self.build_known(params.next().unwrap())?;

                let arguments = params
                    .map(|ty| self.build_known(ty))
                    .collect::<Result<_, _>>()?;

                mir::Type::Function {
                    arguments,
                    output: Box::new(output),
                }
            }
            hir::Item::Struct(id) => {
                let hir_struct = &self.bcx.hir.types[id];

                let mut specialization = hir::Spec::new();

                for (&generic, ty) in hir_struct.generics.iter().zip(&known.params) {
                    specialization.insert(generic, ty.to_type());
                }

                let mut builder = Builder {
                    bcx: self.bcx,
                    locals: self.locals,
                    blocks: self.blocks,
                    body: self.body,
                    specialization,
                };

                let mut fields = Vec::new();

                for field in &hir_struct.fields {
                    let ty = builder.build_type(&field.ty)?;
                    fields.push(ty);
                }

                mir::Type::Struct { fields }
            }
            hir::Item::Enum(id) => {
                let hir_enum = &self.bcx.hir.types[id];

                let mut specialization = hir::Spec::new();

                for (&generic, ty) in hir_enum.generics.iter().zip(&known.params) {
                    specialization.insert(generic, ty.to_type());
                }

                let mut builder = Builder {
                    bcx: self.bcx,
                    locals: self.locals,
                    blocks: self.blocks,
                    body: self.body,
                    specialization,
                };

                let mut variants = Vec::new();

                for variant in &hir_enum.variants {
                    let mut fields = Vec::new();

                    for field in &variant.fields {
                        let ty = builder.build_type(field)?;
                        fields.push(ty);
                    }

                    let variant = mir::Type::Struct { fields };
                    variants.push(variant);
                }

                let discriminant = mir::Type::Int {
                    signed: false,
                    width: Some(32),
                };

                let union = mir::Type::Union { variants };

                mir::Type::Struct {
                    fields: vec![discriminant, union],
                }
            }
        })
    }

    fn build_temp_local(&mut self, block: mir::BlockId, value: mir::Value) -> mir::Place {
        let ty = value.ty();
        let decl = mir::LocalDecl {
            mutable: true,
            ty: ty.clone(),
        };

        let base = self.locals.push(decl);

        let place = mir::Place {
            base,
            ty,
            projections: Vec::new(),
        };

        let statement = mir::Statement::Assign(place.clone(), value);
        self[block].statements.push(statement);

        place
    }

    fn build_pat(
        &mut self,
        block: mir::BlockId,
        pat: &hir::Pat,
        matched: mir::Place,
    ) -> (mir::BlockId, mir::BlockId) {
        match pat {
            hir::Pat::Variant(id, index, fields) => {
                let hir_enum = &self.bcx.hir.types[*id];
                let discriminant = hir_enum.variants[*index].discriminant;

                let mir::Type::Struct {
                    fields: ref enum_fields,
                } = matched.ty
                else {
                    unreachable!("expected struct type {:?}", matched.ty)
                };

                let mut dicriminant_value = matched.clone();
                dicriminant_value.projections.push(mir::Projection {
                    kind: mir::ProjectionKind::Field(0),
                    ty: enum_fields[0].clone(),
                });

                let mut union_value = matched.clone();
                union_value.projections.push(mir::Projection {
                    kind: mir::ProjectionKind::Field(1),
                    ty: enum_fields[1].clone(),
                });

                let mut success = self.create_block();
                let failure = self.create_block();

                self[block].terminator = mir::Terminator::Switch {
                    discriminant: mir::Operand::Copy(dicriminant_value),
                    default: failure,
                    cases: vec![(discriminant, success)],
                };

                let mir::Type::Union { ref variants } = enum_fields[1] else {
                    unreachable!()
                };

                let mut variant_value = union_value.clone();
                variant_value.projections.push(mir::Projection {
                    kind: mir::ProjectionKind::Field(*index),
                    ty: variants[*index].clone(),
                });

                let mir::Type::Struct {
                    fields: ref field_types,
                } = variants[*index]
                else {
                    unreachable!()
                };

                for (i, field) in fields.iter().enumerate() {
                    let mut field_value = variant_value.clone();

                    field_value.projections.push(mir::Projection {
                        kind: mir::ProjectionKind::Field(i),
                        ty: field_types[i].clone(),
                    });

                    let (s, f) = self.build_pat(success, field, field_value);
                    self[f].terminator = mir::Terminator::Goto(failure);
                    success = s;
                }

                (success, failure)
            }
            hir::Pat::Binding(local_id) => {
                let base = mir::Local {
                    index: local_id.index(),
                };

                let place = mir::Place {
                    base,
                    ty: self.locals[base].ty.clone(),
                    projections: Vec::new(),
                };

                let value = mir::Value::Use(mir::Operand::Copy(matched));
                let statement = mir::Statement::Assign(place.clone(), value);
                self[block].statements.push(statement);

                (block, block)
            }
        }
    }

    fn build_place(
        &mut self,
        mut block: mir::BlockId,
        expr: &hir::Expr,
    ) -> Result<BlockAnd<mir::Place>, Diagnostic> {
        match expr.kind {
            hir::ExprKind::Local(local_id) => {
                let local = &self.body.locals[local_id];
                let ty = self.build_type(&local.ty)?;

                let base = mir::Local {
                    index: local_id.index(),
                };

                let place = mir::Place {
                    base,
                    ty,
                    projections: Vec::new(),
                };

                Ok((block, place))
            }

            hir::ExprKind::Field(ref base, ref name) => {
                let (_, _, index, deref_count) =
                    (self.bcx.hir.types).fetch_field_index(&base.ty, name, &self.specialization)?;

                let mut place = unpack!(block = self.build_place(block, base)?);
                let ty = self.build_type(&expr.ty)?;

                let kind = mir::ProjectionKind::Field(index);

                for _ in 0..deref_count {
                    place.projections.push(mir::Projection {
                        kind: mir::ProjectionKind::Deref,
                        ty: mir::Type::VOID,
                    });
                }

                place.projections.push(mir::Projection { kind, ty });

                Ok((block, place))
            }

            hir::ExprKind::Deref(ref base) => {
                let mut place = unpack!(block = self.build_place(block, base)?);
                let ty = self.build_type(&expr.ty)?;

                let kind = mir::ProjectionKind::Deref;
                place.projections.push(mir::Projection { kind, ty });

                Ok((block, place))
            }

            hir::ExprKind::Call(ref callee, ref args) => {
                let callee = unpack!(block = self.build_operand(block, callee)?);

                let mut arguments = Vec::new();

                for arg in args {
                    let argument = unpack!(block = self.build_operand(block, arg)?);
                    arguments.push(argument);
                }

                let ty = self.build_type(&expr.ty)?;
                let decl = mir::LocalDecl {
                    mutable: true,
                    ty: ty.clone(),
                };

                let base = self.locals.push(decl);
                let place = mir::Place {
                    base,
                    ty,
                    projections: Vec::new(),
                };

                let new_block = self.create_block();

                self[block].terminator = mir::Terminator::Call {
                    callee,
                    arguments,
                    destination: place.clone(),
                    target: Some(new_block),
                };

                Ok((new_block, place))
            }

            hir::ExprKind::Match(ref discriminant, ref arms) => {
                let value = unpack!(block = self.build_value(block, discriminant)?);
                let value = self.build_temp_local(block, value);
                let ty = self.build_type(&expr.ty)?;

                let local = mir::LocalDecl {
                    mutable: true,
                    ty: ty.clone(),
                };

                let base = self.locals.push(local);

                let place = mir::Place {
                    base,
                    ty,
                    projections: Vec::new(),
                };

                let end_block = self.create_block();

                for arm in arms {
                    let (mut success, failure) = self.build_pat(block, &arm.pat, value.clone());
                    block = failure;

                    let value = unpack!(success = self.build_value(success, &arm.expr)?);

                    let statemtent = mir::Statement::Assign(place.clone(), value);
                    self[success].statements.push(statemtent);
                    self[success].terminator = mir::Terminator::Goto(end_block);
                }

                self[block].terminator = mir::Terminator::Goto(end_block);

                Ok((end_block, place))
            }

            hir::ExprKind::Const(_)
            | hir::ExprKind::Let(_, _)
            | hir::ExprKind::Cast(_)
            | hir::ExprKind::Assign(_, _)
            | hir::ExprKind::Binary(_, _, _)
            | hir::ExprKind::Struct(_, _, _)
            | hir::ExprKind::Variant(_, _, _, _)
            | hir::ExprKind::Sizeof(_)
            | hir::ExprKind::Ref(_)
            | hir::ExprKind::If(_, _, _)
            | hir::ExprKind::Block(_)
            | hir::ExprKind::Intrinsic(_, _) => {
                let value = unpack!(block = self.build_value(block, expr)?);
                let place = self.build_temp_local(block, value);
                Ok((block, place))
            }
        }
    }

    fn build_operand(
        &mut self,
        mut block: mir::BlockId,
        expr: &hir::Expr,
    ) -> Result<BlockAnd<mir::Operand>, Diagnostic> {
        match expr.kind {
            hir::ExprKind::Assign(ref lhs, ref rhs) => {
                let place = unpack!(block = self.build_place(block, lhs)?);
                let value = unpack!(block = self.build_value(block, rhs)?);

                let statement = mir::Statement::Assign(place, value);

                self[block].statements.push(statement);

                Ok((block, mir::Operand::VOID))
            }

            hir::ExprKind::Let(local_id, ref value) => {
                let local = &self.body.locals[local_id];
                let ty = self.build_type(&local.ty)?;

                let base = mir::Local {
                    index: local_id.index(),
                };

                let place = mir::Place {
                    base,
                    ty,
                    projections: Vec::new(),
                };

                let value = unpack!(block = self.build_value(block, value)?);

                let statement = mir::Statement::Assign(place, value);
                self[block].statements.push(statement);

                Ok((block, mir::Operand::VOID))
            }

            hir::ExprKind::If(ref cond, ref then, ref otherwise) => {
                let cond = unpack!(block = self.build_operand(block, cond)?);

                let then_start = self.create_block();
                let (then_end, then) = self.build_value(then_start, then)?;

                match otherwise {
                    Some(otherwise) => {
                        let else_block = self.create_block();
                        let (else_end, otherwise) = self.build_value(else_block, otherwise)?;

                        let ty = self.build_type(&expr.ty)?;
                        let decl = mir::LocalDecl {
                            mutable: false,
                            ty: ty.clone(),
                        };

                        let base = self.locals.push(decl);

                        let place = mir::Place {
                            base,
                            ty,
                            projections: Vec::new(),
                        };

                        let statement = mir::Statement::Assign(place.clone(), then);
                        self[then_end].statements.push(statement);

                        let statement = mir::Statement::Assign(place.clone(), otherwise);
                        self[else_end].statements.push(statement);

                        let end_block = self.create_block();

                        self[then_end].terminator = mir::Terminator::Goto(end_block);
                        self[else_end].terminator = mir::Terminator::Goto(end_block);

                        self[block].terminator = mir::Terminator::Switch {
                            discriminant: cond,
                            default: then_start,
                            cases: vec![(0, else_block)],
                        };

                        Ok((end_block, mir::Operand::Move(place)))
                    }
                    None => {
                        let end_block = self.create_block();

                        self[then_end].terminator = mir::Terminator::Goto(end_block);

                        self[block].terminator = mir::Terminator::Switch {
                            discriminant: cond,
                            default: then_start,
                            cases: vec![(0, end_block)],
                        };

                        Ok((end_block, mir::Operand::VOID))
                    }
                }
            }

            hir::ExprKind::Block(ref exprs) => {
                let mut operand = mir::Operand::VOID;

                for expr in exprs {
                    operand = unpack!(block = self.build_operand(block, expr)?);
                }

                Ok((block, operand))
            }

            hir::ExprKind::Const(ref constant) => match constant {
                hir::Const::Void => Ok((block, mir::Operand::VOID)),
                hir::Const::Int(v) => {
                    let kind = mir::ConstKind::Int(*v as i64);
                    let constant = mir::Const {
                        kind,
                        ty: self.build_type(&expr.ty)?,
                    };

                    Ok((block, mir::Operand::Const(constant)))
                }
                hir::Const::Float(v) => {
                    let kind = mir::ConstKind::Float(*v);
                    let constant = mir::Const {
                        kind,
                        ty: self.build_type(&expr.ty)?,
                    };

                    Ok((block, mir::Operand::Const(constant)))
                }
                hir::Const::Null => {
                    let ty = self.build_type(&expr.ty)?;
                    let kind = mir::ConstKind::Int(0);
                    let constant = mir::Const { kind, ty };

                    Ok((block, mir::Operand::Const(constant)))
                }
                hir::Const::Func(body_id, generics) => {
                    let generics = generics
                        .iter()
                        .map(|ty| self.specialization.specialize(ty))
                        .collect::<Vec<_>>();

                    let body = &self.bcx.hir.bodies[*body_id];
                    let id = build_body(
                        self.bcx,
                        *body_id,
                        body,
                        &generics,
                        self.specialization.clone(),
                    )?;
                    let ty = self.build_type(&expr.ty)?;

                    let kind = mir::ConstKind::Body(id);
                    let constant = mir::Const { kind, ty };

                    Ok((block, mir::Operand::Const(constant)))
                }
                hir::Const::Method {
                    implementor,
                    trait_id,
                    trait_generics,
                    method_generics,
                    index,
                } => {
                    let (trait_impl, specialization) = self.bcx.hir.types.fetch_trait_impl(
                        *trait_id,
                        trait_generics,
                        implementor,
                        &self.specialization,
                    )?;

                    let body_id = trait_impl.methods[*index].body;
                    let body = &self.bcx.hir.bodies[body_id];

                    let body =
                        build_body(self.bcx, body_id, body, method_generics, specialization)?;

                    let ty = self.build_type(&expr.ty)?;

                    let kind = mir::ConstKind::Body(body);
                    let constant = mir::Const { kind, ty };

                    Ok((block, mir::Operand::Const(constant)))
                }
                hir::Const::AssocMethod {
                    ref implementor,
                    ref name,
                    ref generics,
                    ref arguments,
                } => {
                    let (method, specialization) = self.bcx.hir.types.fetch_assoc_method(
                        implementor,
                        name,
                        generics,
                        arguments.as_deref(),
                        &self.specialization,
                    )?;

                    let body_id = method.body;
                    let body = &self.bcx.hir.bodies[body_id];

                    let body = build_body(self.bcx, body_id, body, generics, specialization)?;

                    let ty = self.build_type(&expr.ty)?;

                    let kind = mir::ConstKind::Body(body);
                    let constant = mir::Const { kind, ty };

                    Ok((block, mir::Operand::Const(constant)))
                }
            },

            hir::ExprKind::Local(_)
            | hir::ExprKind::Cast(_)
            | hir::ExprKind::Call(_, _)
            | hir::ExprKind::Binary(_, _, _)
            | hir::ExprKind::Struct(_, _, _)
            | hir::ExprKind::Variant(_, _, _, _)
            | hir::ExprKind::Field(_, _)
            | hir::ExprKind::Ref(_)
            | hir::ExprKind::Deref(_)
            | hir::ExprKind::Sizeof(_)
            | hir::ExprKind::Match(_, _)
            | hir::ExprKind::Intrinsic(_, _) => {
                let place = unpack!(block = self.build_place(block, expr)?);
                Ok((block, mir::Operand::Copy(place)))
            }
        }
    }

    fn build_value(
        &mut self,
        mut block: mir::BlockId,
        expr: &hir::Expr,
    ) -> Result<BlockAnd<mir::Value>, Diagnostic> {
        match expr.kind {
            hir::ExprKind::Cast(ref base) => {
                let base = unpack!(block = self.build_operand(block, base)?);
                let ty = self.build_type(&expr.ty)?;

                Ok((block, mir::Value::Cast(base, ty)))
            }

            hir::ExprKind::Binary(op, ref lhs, ref rhs) => {
                let lhs = unpack!(block = self.build_operand(block, lhs)?);
                let rhs = unpack!(block = self.build_operand(block, rhs)?);

                let op = match op {
                    hir::BinaryOp::Add => mir::BinaryOp::Add,
                    hir::BinaryOp::Sub => mir::BinaryOp::Sub,
                    hir::BinaryOp::Mul => mir::BinaryOp::Mul,
                    hir::BinaryOp::Div => mir::BinaryOp::Div,
                    hir::BinaryOp::Rem => mir::BinaryOp::Rem,
                    hir::BinaryOp::Eq => mir::BinaryOp::Eq,
                    _ => todo!(),
                };

                Ok((block, mir::Value::Binary(op, lhs, rhs)))
            }

            hir::ExprKind::Ref(ref base) => {
                let place = unpack!(block = self.build_place(block, base)?);
                Ok((block, mir::Value::AddressOf(true, place)))
            }

            hir::ExprKind::Struct(id, ref _generics, ref fields) => {
                let hir_struct = &self.bcx.hir.types[id];

                let mut values = Vec::new();

                for (_field, value) in hir_struct.fields.iter().zip(fields) {
                    let value = unpack!(block = self.build_operand(block, value)?);
                    values.push(value);
                }

                Ok((block, mir::Value::Struct(values)))
            }

            hir::ExprKind::Variant(id, ref _generics, index, ref fields) => {
                let hir_enum = &self.bcx.hir.types[id];

                let mut values = Vec::new();

                for field in fields {
                    let field = unpack!(block = self.build_operand(block, field)?);
                    values.push(field);
                }

                let variant = mir::Value::Struct(values);
                let variant = self.build_temp_local(block, variant);

                let ty = self.build_type(&expr.ty)?;

                let mir::Type::Struct { fields } = ty else {
                    unreachable!()
                };

                let mir::Type::Union { variants } = fields[1].clone() else {
                    unreachable!()
                };

                let union = mir::Value::Union(mir::Operand::Move(variant), variants);
                let union = self.build_temp_local(block, union);

                let discriminant = hir_enum.variants[index].discriminant;
                let discriminant = mir::Operand::Const(mir::Const {
                    kind: mir::ConstKind::Int(discriminant as i64),
                    ty: mir::Type::Int {
                        signed: false,
                        width: Some(32),
                    },
                });

                let fields = vec![discriminant, mir::Operand::Move(union)];

                Ok((block, mir::Value::Struct(fields)))
            }

            hir::ExprKind::Sizeof(ref ty) => {
                let ty = self.build_type(ty)?;
                Ok((block, mir::Value::Sizeof(ty)))
            }

            hir::ExprKind::Intrinsic(name, ref args) => {
                let mut operands = Vec::new();

                for arg in args {
                    let operand = unpack!(block = self.build_operand(block, arg)?);
                    operands.push(operand);
                }

                let ty = self.build_type(&expr.ty)?;
                Ok((block, mir::Value::Intrinsic(name, operands, ty)))
            }

            hir::ExprKind::Const(_)
            | hir::ExprKind::Local(_)
            | hir::ExprKind::Let(_, _)
            | hir::ExprKind::Assign(_, _)
            | hir::ExprKind::Call(_, _)
            | hir::ExprKind::If(_, _, _)
            | hir::ExprKind::Field(_, _)
            | hir::ExprKind::Deref(_)
            | hir::ExprKind::Match(_, _)
            | hir::ExprKind::Block(_) => {
                let operand = unpack!(block = self.build_operand(block, expr)?);
                Ok((block, mir::Value::Use(operand)))
            }
        }
    }
}

fn build_body(
    bcx: &mut BuildContext,
    body_id: hir::BodyId,
    hir_body: &hir::Body,
    generics: &[hir::Type],
    mut specialization: hir::Spec,
) -> Result<mir::BodyId, Diagnostic> {
    for (&generic, ty) in hir_body.generics.iter().zip(generics.iter()) {
        specialization.insert(generic, ty.clone());
    }

    let mut blocks = Blocks::new();
    let mut locals = Locals::new();

    let mut builder = Builder {
        bcx,
        locals: &mut locals,
        blocks: &mut blocks,
        body: hir_body,
        specialization,
    };

    let generics = generics
        .iter()
        .map(|ty| builder.build_type(ty))
        .collect::<Result<_, _>>()?;

    let key = BodyKey {
        body: body_id,
        generics,
    };

    if let Some(&id) = builder.bcx.bodies.get(&key) {
        return Ok(id);
    }

    let id = builder.bcx.mir.bodies.alloc();
    builder.bcx.bodies.insert(key, id);

    for (local_id, local) in hir_body.locals.iter() {
        let ty = builder.build_type(&local.ty)?;

        let decl = mir::LocalDecl {
            mutable: local.mutable,
            ty,
        };

        let local = mir::Local {
            index: local_id.index(),
        };

        builder.locals.insert(local, decl);
    }

    let mut arguments = Vec::new();

    for argument in &hir_body.arguments {
        let local = mir::Local {
            index: argument.index(),
        };

        arguments.push(local);
    }

    let output = builder.build_type(&hir_body.output)?;

    let entry = builder.create_block();
    let (block, value) = builder.build_value(entry, &hir_body.expr)?;
    builder[block].terminator = mir::Terminator::Return(value);

    let mir_body = mir::Body {
        arguments,
        locals,
        output,
        blocks,
        entry,
    };

    bcx.mir.bodies.insert(id, mir_body);

    Ok(id)
}

pub fn build(unit: &hir::Unit) -> Result<mir::Unit, Diagnostic> {
    let mut bcx = BuildContext {
        bodies: HashMap::new(),
        hir: unit,
        mir: Default::default(),
    };

    for (id, body) in unit.bodies.iter() {
        if body.is_generic() {
            continue;
        }

        let body_id = build_body(&mut bcx, id, body, &[], Default::default())?;

        if body.name.as_deref() == Some("main") {
            bcx.mir.entry = Some(body_id);
        }
    }

    Ok(bcx.mir)
}
