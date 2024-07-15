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
    specialization: hir::Specialization,
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
        let known = self.bcx.hir.types.query(hir)?;

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

                let fields = hir_struct
                    .fields
                    .iter()
                    .map(|field| self.build_type(&field.type_))
                    .collect::<Result<_, _>>()?;

                mir::Type::Struct { fields }
            }
            hir::Item::Enum(_) => todo!(),
        })
    }

    fn build_place(
        &mut self,
        mut block: mir::BlockId,
        expr: &hir::Expr,
    ) -> Result<BlockAnd<mir::Place>, Diagnostic> {
        match expr.kind {
            hir::ExprKind::Local(local_id) => {
                let local = &self.body.locals[local_id];
                let ty = self.build_type(&local.type_)?;

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

            hir::ExprKind::Const(_)
            | hir::ExprKind::Let(_, _)
            | hir::ExprKind::Assign(_, _)
            | hir::ExprKind::Block(_)
            | hir::ExprKind::Intrinsic(_, _) => {
                todo!()
            }
        }
    }

    fn build_operand(
        &mut self,
        mut block: mir::BlockId,
        expr: &hir::Expr,
    ) -> Result<BlockAnd<mir::Operand>, Diagnostic> {
        match expr.kind {
            hir::ExprKind::Let(local_id, ref value) => {
                let local = &self.body.locals[local_id];
                let ty = self.build_type(&local.type_)?;

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

            hir::ExprKind::Block(ref exprs) => {
                let mut operand = mir::Operand::VOID;

                for expr in exprs {
                    operand = unpack!(block = self.build_operand(block, expr)?);
                }

                Ok((block, operand))
            }

            hir::ExprKind::Const(ref constant) => match constant {
                hir::Constant::Void => Ok((block, mir::Operand::VOID)),
                hir::Constant::Int(v) => {
                    let kind = mir::ConstKind::Int(*v as i64);
                    let constant = mir::Const {
                        kind,
                        ty: self.build_type(&expr.ty)?,
                    };

                    Ok((block, mir::Operand::Const(constant)))
                }
                hir::Constant::Float(v) => {
                    let kind = mir::ConstKind::Float(*v);
                    let constant = mir::Const {
                        kind,
                        ty: self.build_type(&expr.ty)?,
                    };

                    Ok((block, mir::Operand::Const(constant)))
                }
                hir::Constant::Func(body_id, generics) => {
                    let body = &self.bcx.hir.bodies[*body_id];
                    let id = build_body(self.bcx, *body_id, body, generics)?;
                    let ty = self.build_type(&expr.ty)?;

                    let kind = mir::ConstKind::Body(id);
                    let constant = mir::Const { kind, ty };

                    Ok((block, mir::Operand::Const(constant)))
                }
                hir::Constant::Method { .. } => todo!(),
            },

            hir::ExprKind::Local(_)
            | hir::ExprKind::Assign(_, _)
            | hir::ExprKind::Call(_, _)
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
            hir::ExprKind::Const(_)
            | hir::ExprKind::Local(_)
            | hir::ExprKind::Let(_, _)
            | hir::ExprKind::Assign(_, _)
            | hir::ExprKind::Call(_, _)
            | hir::ExprKind::Block(_)
            | hir::ExprKind::Intrinsic(_, _) => {
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
) -> Result<mir::BodyId, Diagnostic> {
    let mut specialization = hir::Specialization::new();

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

    for (local_id, local) in hir_body.locals.iter() {
        let ty = builder.build_type(&local.type_)?;

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

    let id = bcx.mir.bodies.push(mir_body);
    bcx.bodies.insert(key, id);

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

        build_body(&mut bcx, id, body, &[])?;
    }

    Ok(bcx.mir)
}
