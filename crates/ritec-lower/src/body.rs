use std::ops::{Deref, DerefMut};

use ritec_ast as ast;
use ritec_diagnostic::{Diagnostic, Span};
use ritec_hir as hir;

use crate::{
    r#type::{Resolved, TypeContext},
    Lowerer,
};

struct BodyLowerer<'a, 'b> {
    lowerer: &'a mut Lowerer,
    tcx: &'a mut TypeContext<'b>,
    output: &'a hir::Type,
    contract: hir::ContractId,
    self_argument: Option<hir::LocalId>,
    locals: &'a mut hir::Locals,
    scope: Vec<hir::LocalId>,
}

impl Deref for BodyLowerer<'_, '_> {
    type Target = Lowerer;

    fn deref(&self) -> &Self::Target {
        self.lowerer
    }
}

impl DerefMut for BodyLowerer<'_, '_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.lowerer
    }
}

impl BodyLowerer<'_, '_> {
    fn lower_type(&mut self, ast: &ast::Type) -> Result<hir::Type, Diagnostic> {
        self.tcx.lower_type(&self.lowerer.unit, ast)
    }

    fn resolve_path(&mut self, ast: &ast::Path) -> Result<Resolved, Diagnostic> {
        self.tcx.resolve_path(&self.lowerer.unit, ast)
    }

    fn get_local(&self, name: &str) -> Option<hir::LocalId> {
        self.scope
            .iter()
            .rev()
            .find(|&&local_id| self.locals[local_id].name.as_deref() == Some(name))
            .copied()
    }

    fn lower_void_expr(&mut self, ast: &ast::VoidExpr) -> Result<hir::Expr, Diagnostic> {
        Ok(hir::Expr::void(ast.span))
    }

    fn lower_func_expr(
        &mut self,
        id: hir::BodyId,
        generics: Vec<hir::Type>,
        span: Span,
    ) -> Result<hir::Expr, Diagnostic> {
        let body = &self.unit.bodies[id];

        let mut specialization = hir::Spec::new();

        for (&generic, type_) in body.generics.iter().zip(&generics) {
            specialization.insert(generic, type_.clone());
        }

        let mut params = Vec::new();

        params.push(specialization.specialize(&body.output));

        for argument in &body.arguments {
            let local = &body.locals[*argument];
            params.push(specialization.specialize(&local.ty));
        }

        let kind = hir::ExprKind::Const(hir::Const::Func(id, generics));
        let span = Some(span);
        let ty = hir::Type::Partial(hir::Partial {
            item: hir::Item::Function,
            params,
        });

        let contract = body.contract;
        self.unit.types.satisfy(contract, &specialization);

        Ok(hir::Expr { kind, span, ty })
    }

    fn lower_item_expr(&mut self, ast: &ast::ItemExpr) -> Result<hir::Expr, Diagnostic> {
        if let Some(ident) = ast.path.ident() {
            if let Some(local) = self.get_local(ident) {
                return Ok(hir::Expr {
                    kind: hir::ExprKind::Local(local),
                    span: Some(ast.span),
                    ty: self.locals[local].ty.clone(),
                });
            }
        }

        match self.resolve_path(&ast.path)? {
            Resolved::Func(id, generics) => self.lower_func_expr(id, generics, ast.span),
            Resolved::SelfArgument => match self.self_argument {
                Some(local) => Ok(hir::Expr {
                    kind: hir::ExprKind::Local(local),
                    span: Some(ast.span),
                    ty: self.locals[local].ty.clone(),
                }),
                None => {
                    let message = "self argument not found".to_string();
                    Err(Diagnostic::new(message).with_span(ast.span))
                }
            },
            Resolved::EnumVariant(id, ref generics, index) => {
                if self.unit.types[id].variants[index].fields.is_empty() {
                    let kind = hir::ExprKind::Variant(id, generics.clone(), index, Vec::new());
                    let span = Some(ast.span);
                    let ty = hir::Type::Partial(hir::Partial {
                        item: hir::Item::Enum(id),
                        params: generics.clone(),
                    });

                    Ok(hir::Expr { kind, span, ty })
                } else {
                    let body_id = self.unit.types[id].variants[index].builder.unwrap();
                    self.lower_func_expr(body_id, generics.clone(), ast.span)
                }
            }
            Resolved::Assoc(implementor, name, ref generics) => {
                let ty = hir::Type::Projected(hir::Projected {
                    contract: self.contract,
                    base: Box::new(implementor.clone()),
                    projection: hir::Projection::AssocMethod {
                        name: name.clone(),
                        generics: generics.clone(),
                        arguments: None,
                    },
                });

                Ok(hir::Expr {
                    kind: hir::ExprKind::Const(hir::Const::AssocMethod {
                        implementor,
                        name,
                        generics: generics.clone(),
                        arguments: None,
                    }),
                    span: Some(ast.span),
                    ty,
                })
            }
            Resolved::TraitMethod(trait_id, trait_generics, method_index, method_generics) => {
                let hir_trait = &self.unit.types[trait_id];

                let trait_spec =
                    hir::Spec::specify(&hir_trait.generics, &trait_generics, ast.span)?;

                let hir_method = &hir_trait.methods[method_index];

                let method_spec =
                    hir::Spec::specify(&hir_method.generics, &method_generics, ast.span)?;

                let implementor = hir::Type::unknown(ast.span);

                let constant = hir::Const::Method {
                    implementor: implementor.clone(),
                    trait_id,
                    trait_generics,
                    method_generics,
                    index: method_index,
                };

                let mut spec = hir::Spec::new();
                spec.insert(hir_trait.self_generic, implementor);
                spec.extend(&trait_spec);
                spec.extend(&method_spec);

                let mut params = Vec::new();
                params.push(spec.specialize(&hir_method.output));

                for argument in &hir_method.arguments {
                    params.push(spec.specialize(argument));
                }

                let kind = hir::ExprKind::Const(constant);
                let span = Some(ast.span);
                let ty = hir::Type::Partial(hir::Partial {
                    item: hir::Item::Function,
                    params,
                });

                Ok(hir::Expr { kind, span, ty })
            }
            resolved => {
                let message = format!(
                    "unexpected item, found {:?} with path {:?}",
                    resolved, ast.path
                );
                Err(Diagnostic::new(message).with_span(ast.span))
            }
        }
    }

    fn lower_int_expr(&mut self, ast: &ast::LitIntExpr) -> Result<hir::Expr, Diagnostic> {
        let kind = hir::ExprKind::Const(hir::Const::Int(ast.value));
        let span = Some(ast.span);
        let ty = hir::Type::Unknown(hir::Unknown {
            kind: hir::UnknownKind::Number { float: false },
            uid: hir::Uid::new(),
            span: ast.span,
        });

        Ok(hir::Expr { kind, span, ty })
    }

    fn lower_float_expr(&mut self, ast: &ast::LitFloatExpr) -> Result<hir::Expr, Diagnostic> {
        let kind = hir::ExprKind::Const(hir::Const::Float(ast.value));
        let span = Some(ast.span);
        let ty = hir::Type::Unknown(hir::Unknown {
            kind: hir::UnknownKind::Number { float: true },
            uid: hir::Uid::new(),
            span: ast.span,
        });

        Ok(hir::Expr { kind, span, ty })
    }

    fn lower_null_expr(&mut self, ast: &ast::NullExpr) -> Result<hir::Expr, Diagnostic> {
        let kind = hir::ExprKind::Const(hir::Const::Null);
        let span = Some(ast.span);
        let ty = hir::Type::Partial(hir::Partial {
            item: hir::Item::Pointer { mutable: true },
            params: vec![hir::Type::unknown(ast.span)],
        });

        Ok(hir::Expr { kind, span, ty })
    }

    fn lower_as_expr(&mut self, ast: &ast::AsExpr) -> Result<hir::Expr, Diagnostic> {
        let value = self.lower_expr(&ast.expr)?;
        let ty = self.lower_type(&ast.type_)?;

        let kind = hir::ExprKind::Cast(Box::new(value));
        let span = Some(ast.span);

        Ok(hir::Expr { kind, span, ty })
    }

    fn lower_struct_expr(&mut self, ast: &ast::StructExpr) -> Result<hir::Expr, Diagnostic> {
        let ty = self.lower_type(&ast::Type::Path(ast.item.clone()))?;

        let hir::Type::Partial(hir::Partial {
            item: hir::Item::Struct(id),
            params: mut generics,
        }) = ty
        else {
            let message = format!("expected struct, found {:?}", ast.item);
            return Err(Diagnostic::new(message).with_span(ast.span));
        };

        let struct_ = self.unit.types[id].clone();

        if generics.len() > struct_.generics.len() {
            let message = format!("too many generics for struct `{:?}`", id);
            return Err(Diagnostic::new(message).with_span(ast.span));
        }

        while generics.len() < struct_.generics.len() {
            generics.push(hir::Type::unknown(ast.span));
        }

        let mut specialization = hir::Spec::new();

        for (&generic, ty) in struct_.generics.iter().zip(&generics) {
            specialization.insert(generic, ty.clone());
        }

        let mut fields = vec![None; struct_.fields.len()];

        for field in &ast.fields {
            let Some(index) = struct_.field_index(&field.name) else {
                let message = format!("field `{:?}` not found in struct `{:?}`", field.name, id);
                return Err(Diagnostic::new(message).with_span(field.span));
            };

            if fields[index].is_some() {
                let message = format!("field `{:?}` already initialized", field.name);
                return Err(Diagnostic::new(message).with_span(field.span));
            }

            let field = self.lower_expr(&field.value)?;
            let ty = field.ty.clone();
            fields[index] = Some(field);

            let field_ty = struct_.fields[index].ty.clone();
            let field_ty = specialization.specialize(&field_ty);

            self.unit.types.unify(ty, field_ty);
        }

        let fields = fields.into_iter().map(|field| field.unwrap()).collect();

        let kind = hir::ExprKind::Struct(id, generics.clone(), fields);
        let span = Some(ast.span);
        let ty = hir::Type::Partial(hir::Partial {
            item: hir::Item::Struct(id),
            params: generics,
        });

        Ok(hir::Expr { kind, span, ty })
    }

    fn lower_field_expr(&mut self, ast: &ast::FieldExpr) -> Result<hir::Expr, Diagnostic> {
        let base = self.lower_expr(&ast.base)?;
        let base_ty = base.ty.clone();

        let kind = hir::ExprKind::Field(Box::new(base), ast.name.clone());
        let span = Some(ast.span);
        let ty = hir::Type::Projected(hir::Projected {
            contract: self.contract,
            base: Box::new(base_ty),
            projection: hir::Projection::Field {
                name: ast.name.clone(),
            },
        });

        Ok(hir::Expr { kind, span, ty })
    }

    fn lower_call_expr(&mut self, ast: &ast::CallExpr) -> Result<hir::Expr, Diagnostic> {
        let mut callee = self.lower_expr(&ast.callee)?;
        let ty = hir::Type::unknown(ast.span);

        let mut arguments = Vec::new();
        let mut params = Vec::new();

        params.push(ty.clone());

        for argumnet in &ast.arguments {
            let argument = self.lower_expr(argumnet)?;
            params.push(argument.ty.clone());
            arguments.push(argument);
        }

        if let hir::ExprKind::Const(hir::Const::AssocMethod {
            ref mut arguments, ..
        }) = callee.kind
        {
            *arguments = Some(params.clone());

            if let hir::Type::Projected(hir::Projected {
                projection:
                    hir::Projection::AssocMethod {
                        ref mut arguments, ..
                    },
                ..
            }) = callee.ty
            {
                *arguments = Some(params.clone());
            }
        }

        let func = hir::Type::Partial(hir::Partial {
            item: hir::Item::Function,
            params,
        });

        self.unit.types.unify(callee.ty.clone(), func);

        let kind = hir::ExprKind::Call(Box::new(callee), arguments);
        let span = Some(ast.span);

        Ok(hir::Expr { kind, span, ty })
    }

    fn lower_unary_expr(&mut self, ast: &ast::UnaryExpr) -> Result<hir::Expr, Diagnostic> {
        let value = self.lower_expr(&ast.expr)?;

        match ast.op {
            ast::UnaryOp::Neg => todo!(),
            ast::UnaryOp::Not => todo!(),
            ast::UnaryOp::Ref { mutable } => {
                let ty = hir::Type::Partial(hir::Partial {
                    item: hir::Item::Pointer { mutable },
                    params: vec![value.ty.clone()],
                });

                let kind = hir::ExprKind::Ref(Box::new(value));
                let span = Some(ast.span);

                Ok(hir::Expr { kind, span, ty })
            }
            ast::UnaryOp::Deref => {
                let ty = hir::Type::unknown(ast.span);
                let pointer_ty = hir::Type::Partial(hir::Partial {
                    item: hir::Item::Pointer { mutable: false },
                    params: vec![ty.clone()],
                });

                self.unit.types.unify(value.ty.clone(), pointer_ty);

                let kind = hir::ExprKind::Deref(Box::new(value));
                let span = Some(ast.span);

                Ok(hir::Expr { kind, span, ty })
            }
        }
    }

    fn lower_binary_math(
        &mut self,
        lhs: hir::Expr,
        rhs: hir::Expr,
        trait_id: hir::TraitId,
    ) -> hir::Expr {
        let constant = hir::Const::Method {
            implementor: lhs.ty.clone(),
            trait_id,
            trait_generics: vec![rhs.ty.clone()],
            method_generics: Vec::new(),
            index: 0,
        };

        let ty = hir::Type::Partial(hir::Partial {
            item: hir::Item::Function,
            params: vec![lhs.ty.clone(), rhs.ty.clone()],
        });

        let kind = hir::ExprKind::Const(constant);
        let span = None;

        let method = hir::Expr { kind, span, ty };

        let ty = hir::Type::Projected(hir::Projected {
            contract: self.contract,
            base: Box::new(lhs.ty.clone()),
            projection: hir::Projection::TraitType {
                trait_id,
                generics: vec![rhs.ty.clone()],
                index: 0,
            },
        });

        let kind = hir::ExprKind::Call(Box::new(method), vec![lhs, rhs]);
        let span = None;

        hir::Expr { kind, span, ty }
    }

    fn lower_binary_eq(&mut self, lhs: hir::Expr, rhs: hir::Expr) -> hir::Expr {
        let trait_id = self.unit.builtins.eq_trait;

        let constant = hir::Const::Method {
            implementor: lhs.ty.clone(),
            trait_id,
            trait_generics: vec![rhs.ty.clone()],
            method_generics: Vec::new(),
            index: 0,
        };

        let ty = hir::Type::Partial(hir::Partial {
            item: hir::Item::Function,
            params: vec![lhs.ty.clone(), rhs.ty.clone()],
        });

        let kind = hir::ExprKind::Const(constant);
        let span = None;

        let method = hir::Expr { kind, span, ty };

        let ty = hir::Type::BOOL;
        let kind = hir::ExprKind::Call(Box::new(method), vec![lhs, rhs]);
        let span = None;

        hir::Expr { kind, span, ty }
    }

    fn lower_binary_expr(&mut self, ast: &ast::BinaryExpr) -> Result<hir::Expr, Diagnostic> {
        let lhs = self.lower_expr(&ast.lhs)?;
        let rhs = self.lower_expr(&ast.rhs)?;

        self.unit.types.unify(lhs.ty.clone(), rhs.ty.clone());

        match ast.op {
            ast::BinaryOp::Add => {
                Ok(self.lower_binary_math(lhs, rhs, self.unit.builtins.add_trait))
            }
            ast::BinaryOp::Sub => {
                Ok(self.lower_binary_math(lhs, rhs, self.unit.builtins.sub_trait))
            }
            ast::BinaryOp::Mul => {
                Ok(self.lower_binary_math(lhs, rhs, self.unit.builtins.mul_trait))
            }
            ast::BinaryOp::Div => {
                Ok(self.lower_binary_math(lhs, rhs, self.unit.builtins.div_trait))
            }
            ast::BinaryOp::Eq => Ok(self.lower_binary_eq(lhs, rhs)),
            _ => unimplemented!(),
        }
    }

    fn lower_assign_expr(&mut self, ast: &ast::AssignExpr) -> Result<hir::Expr, Diagnostic> {
        let lhs = self.lower_expr(&ast.lhs)?;
        let rhs = self.lower_expr(&ast.rhs)?;

        let kind = hir::ExprKind::Assign(Box::new(lhs), Box::new(rhs));
        let span = Some(ast.span);
        let ty = hir::Type::VOID;

        Ok(hir::Expr { kind, span, ty })
    }

    fn lower_let_expr(&mut self, ast: &ast::LetExpr) -> Result<hir::Expr, Diagnostic> {
        // get the type if specified
        let ty = match ast.type_ {
            Some(ref type_) => self.lower_type(type_)?,
            None => hir::Type::unknown(ast.span),
        };

        // lower the value
        let value = match ast.value {
            Some(ref value) => self.lower_expr(value)?,
            None => unimplemented!(),
        };

        // create a new local
        let local = self.locals.push(hir::Local {
            mutable: ast.mutable,
            name: Some(ast.name.clone()),
            ty: value.ty.clone(),
        });

        // push the local to the scope
        self.scope.push(local);

        // unify the type of the value with the type of the local
        self.unit.types.unify(value.ty.clone(), ty.clone());

        let kind = hir::ExprKind::Let(local, Box::new(value));
        let span = Some(ast.span);

        Ok(hir::Expr {
            kind,
            span,
            ty: hir::Type::VOID,
        })
    }

    fn lower_if_expr(&mut self, ast: &ast::IfExpr) -> Result<hir::Expr, Diagnostic> {
        let cond = self.lower_expr(&ast.cond)?;
        self.unit.types.unify(cond.ty.clone(), hir::Type::BOOL);

        let then = self.lower_expr(&ast.then)?;
        let ty = then.ty.clone();

        let otherwise = match ast.otherwise {
            Some(ref expr) => {
                let expr = self.lower_expr(expr)?;
                self.unit.types.unify(expr.ty.clone(), ty.clone());
                Some(Box::new(expr))
            }
            None => {
                self.unit.types.unify(ty.clone(), hir::Type::VOID);
                None
            }
        };

        let kind = hir::ExprKind::If(Box::new(cond), Box::new(then), otherwise);
        let span = Some(ast.span);

        Ok(hir::Expr { kind, span, ty })
    }

    fn lower_item_pat(
        &mut self,
        ast: &ast::ItemPat,
        ty: &hir::Type,
    ) -> Result<hir::Pat, Diagnostic> {
        if let Some(ident) = ast.item.ident() {
            let local = hir::Local {
                mutable: false,
                name: Some(String::from(ident)),
                ty: ty.clone(),
            };

            let local = self.locals.push(local);
            self.scope.push(local);

            return Ok(hir::Pat::Binding(local));
        }

        match self.resolve_path(&ast.item)? {
            Resolved::EnumVariant(id, _, index) => Ok(hir::Pat::Variant(id, index, Vec::new())),
            _ => {
                let message = format!("expected enum variant, found {:?}", ast.item);
                Err(Diagnostic::new(message).with_span(ast.span))
            }
        }
    }

    fn lower_tuple_pat(
        &mut self,
        ast: &ast::TuplePat,
        ty: &hir::Type,
    ) -> Result<hir::Pat, Diagnostic> {
        if let Some(ref path) = ast.path {
            match self.resolve_path(path)? {
                Resolved::EnumVariant(id, generics, index) => {
                    let hir_enum = &self.unit.types[id];

                    let mut specialization = hir::Spec::new();

                    for (&generic, ty) in hir_enum.generics.iter().zip(&generics) {
                        specialization.insert(generic, ty.clone());
                    }

                    let enum_ty = hir::Type::Partial(hir::Partial {
                        item: hir::Item::Enum(id),
                        params: generics.clone(),
                    });

                    self.unit.types.unify(ty.clone(), enum_ty.clone());

                    let mut fields = Vec::new();

                    for (i, pat) in ast.pats.iter().enumerate() {
                        let ty = self.unit.types[id].variants[index].fields[i].clone();
                        let ty = specialization.specialize(&ty);
                        let pat = self.lower_pat(pat, &ty)?;
                        fields.push(pat);
                    }

                    return Ok(hir::Pat::Variant(id, index, fields));
                }
                _ => todo!(),
            }
        }

        todo!()
    }

    fn lower_pat(&mut self, ast: &ast::Pat, ty: &hir::Type) -> Result<hir::Pat, Diagnostic> {
        match ast {
            ast::Pat::Item(ast) => self.lower_item_pat(ast, ty),
            ast::Pat::Tuple(ast) => self.lower_tuple_pat(ast, ty),
        }
    }

    fn lower_match_expr(&mut self, ast: &ast::MatchExpr) -> Result<hir::Expr, Diagnostic> {
        let value = self.lower_expr(&ast.value)?;
        let ty = hir::Type::unknown(ast.span);

        let mut arms = Vec::new();

        for arm in ast.arms.iter() {
            let pat = self.lower_pat(&arm.pat, &value.ty)?;
            let expr = self.lower_expr(&arm.body)?;

            self.unit.types.unify(expr.ty.clone(), ty.clone());

            arms.push(hir::Arm { pat, expr });
        }

        let kind = hir::ExprKind::Match(Box::new(value), arms);
        let span = Some(ast.span);

        Ok(hir::Expr { kind, span, ty })
    }

    fn lower_block_expr(&mut self, ast: &ast::BlockExpr) -> Result<hir::Expr, Diagnostic> {
        let mut lowerer = BodyLowerer {
            lowerer: self.lowerer,
            tcx: self.tcx,
            output: self.output,
            contract: self.contract,
            self_argument: self.self_argument,
            locals: self.locals,
            scope: self.scope.clone(),
        };

        let mut exprs = Vec::new();
        let mut ty = hir::Type::VOID;

        for expr in &ast.exprs {
            let expr = lowerer.lower_expr(expr)?;
            ty = expr.ty.clone();
            exprs.push(expr);
        }

        let kind = hir::ExprKind::Block(exprs);
        let span = Some(ast.span);

        Ok(hir::Expr { kind, span, ty })
    }

    pub fn lower_expr(&mut self, ast: &ast::Expr) -> Result<hir::Expr, Diagnostic> {
        match ast {
            ast::Expr::Void(expr) => self.lower_void_expr(expr),
            ast::Expr::Item(expr) => self.lower_item_expr(expr),
            ast::Expr::LitInt(expr) => self.lower_int_expr(expr),
            ast::Expr::LitFloat(expr) => self.lower_float_expr(expr),
            ast::Expr::Null(expr) => self.lower_null_expr(expr),
            ast::Expr::As(expr) => self.lower_as_expr(expr),
            ast::Expr::Struct(expr) => self.lower_struct_expr(expr),
            ast::Expr::Paren(expr) => self.lower_expr(&expr.expr),
            ast::Expr::Field(expr) => self.lower_field_expr(expr),
            ast::Expr::Call(expr) => self.lower_call_expr(expr),
            ast::Expr::Unary(expr) => self.lower_unary_expr(expr),
            ast::Expr::Binary(expr) => self.lower_binary_expr(expr),
            ast::Expr::Assign(expr) => self.lower_assign_expr(expr),
            ast::Expr::Let(expr) => self.lower_let_expr(expr),
            ast::Expr::Loop(_) => todo!(),
            ast::Expr::If(expr) => self.lower_if_expr(expr),
            ast::Expr::Match(expr) => self.lower_match_expr(expr),
            ast::Expr::Block(expr) => self.lower_block_expr(expr),
        }
    }
}

impl Lowerer {
    pub fn lower_body(
        &mut self,
        tcx: &mut TypeContext,
        output: &hir::Type,
        contract: hir::ContractId,
        self_argument: Option<hir::LocalId>,
        locals: &mut hir::Locals,
        ast: &ast::Expr,
    ) -> Result<hir::Expr, Diagnostic> {
        let scope = locals.keys().collect();

        let mut lowerer = BodyLowerer {
            lowerer: self,
            tcx,
            output,
            contract,
            self_argument,
            locals,
            scope,
        };

        lowerer.lower_expr(ast)
    }
}
