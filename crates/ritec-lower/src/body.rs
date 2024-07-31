use std::ops::{Deref, DerefMut};

use hir::FromPartial;
use ritec_ast as ast;
use ritec_diagnostic::{Diagnostic, Span};
use ritec_hir as hir;

use crate::{
    r#type::{Resolved, TyCx},
    Lowerer,
};

struct BodyLowerer<'a, 'b> {
    lowerer: &'a mut Lowerer,
    tcx: &'a mut TyCx<'b>,
    output: &'a hir::Ty,
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
    fn lower_type(&mut self, ast: &ast::Type) -> Result<hir::Ty, Diagnostic> {
        Ok(self.tcx.lower_type(&self.lowerer.unit, ast)?.to_ty())
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
        generics: Vec<hir::KnownTy>,
        span: Span,
    ) -> Result<hir::Expr, Diagnostic> {
        let generics: Vec<_> = generics.iter().map(hir::KnownTy::to_ty).collect();
        let constant = hir::Const::Func(id, generics.clone());
        let kind = hir::ExprKind::Const(constant);
        let span = Some(span);
        let ty = self.unit[id].func_ty(&generics)?;

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
                if self.unit[id].variants[index].fields.is_empty() {
                    let generics: Vec<_> = generics.iter().map(hir::KnownTy::to_ty).collect();

                    let kind = hir::ExprKind::Variant(id, generics.clone(), index, Vec::new());
                    let span = Some(ast.span);
                    let ty = hir::Ty::new_enum(id, generics);

                    Ok(hir::Expr { kind, span, ty })
                } else {
                    let body_id = self.unit[id].variants[index].builder.unwrap();
                    self.lower_func_expr(body_id, generics.clone(), ast.span)
                }
            }
            Resolved::Assoc(implementor, name, ref generics) => {
                let generics: Vec<_> = generics.iter().map(hir::KnownTy::to_ty).collect();

                let ty = hir::Ty::Proj(hir::ProjTy {
                    base: Box::new(implementor.to_ty()),
                    proj: hir::Projection::Method {
                        name: name.clone(),
                        generics: generics.clone(),
                    },
                });

                Ok(hir::Expr {
                    kind: hir::ExprKind::Const(hir::Const::AssocMethod {
                        implementor: implementor.to_ty(),
                        name,
                        generics,
                        arguments: None,
                    }),
                    span: Some(ast.span),
                    ty,
                })
            }
            Resolved::TraitMethod(trait_id, trait_generics, method_index, method_generics) => {
                let hir_trait = &self.unit[trait_id];

                let trait_spec = hir::Spec::specified(&hir_trait.generics, &trait_generics)?;

                let hir_method = &hir_trait.methods[method_index];

                let method_spec = hir::Spec::specified(&hir_method.generics, &method_generics)?;

                let implementor = hir::Ty::new_unknown(None);

                let trait_generics = trait_generics.iter().map(hir::KnownTy::to_ty).collect();
                let method_generics = method_generics.iter().map(hir::KnownTy::to_ty).collect();

                let constant = hir::Const::Method {
                    implementor: implementor.clone(),
                    trait_id,
                    trait_generics,
                    method_generics,
                    index: method_index,
                };

                let mut spec = hir::Spec::new();
                spec.insert(hir_trait.self_generic, implementor);
                spec.extend(trait_spec.to_ty());
                spec.extend(method_spec.to_ty());

                let mut params = Vec::new();

                for argument in &hir_method.arguments {
                    params.push(spec.specialize(argument));
                }

                let kind = hir::ExprKind::Const(constant);
                let span = Some(ast.span);
                let ty = hir::Ty::new_func(params, hir::Ty::VOID);

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
        let ty = hir::Ty::new_unknown(Some(hir::UnknownKind::Number(false)));
        Ok(hir::Expr { kind, span, ty })
    }

    fn lower_float_expr(&mut self, ast: &ast::LitFloatExpr) -> Result<hir::Expr, Diagnostic> {
        let kind = hir::ExprKind::Const(hir::Const::Float(ast.value));
        let span = Some(ast.span);
        let ty = hir::Ty::new_unknown(Some(hir::UnknownKind::Number(true)));
        Ok(hir::Expr { kind, span, ty })
    }

    fn lower_null_expr(&mut self, ast: &ast::NullExpr) -> Result<hir::Expr, Diagnostic> {
        let kind = hir::ExprKind::Const(hir::Const::Null);
        let span = Some(ast.span);
        let ty = hir::Ty::new_pointer(true, hir::Ty::VOID);

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

        let hir::Ty::Partial(hir::TyPart::Struct(id), generics) = ty else {
            let message = format!("expected struct, found {:?}", ast.item);
            return Err(Diagnostic::new(message).with_span(ast.span));
        };

        let struct_def = self.unit[id].clone();

        let mut fields = vec![None; struct_def.fields.len()];

        for field in &ast.fields {
            let Some(index) = struct_def.field_index(&field.name) else {
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

            let field_ty = struct_def.field_ty(&generics, index)?;

            self.unit.env.assign(ty, field_ty);
        }

        let fields = fields.into_iter().map(|field| field.unwrap()).collect();

        let kind = hir::ExprKind::Struct(id, generics.clone(), fields);
        let span = Some(ast.span);
        let ty = hir::Ty::new_struct(id, generics);

        Ok(hir::Expr { kind, span, ty })
    }

    fn lower_field_expr(&mut self, ast: &ast::FieldExpr) -> Result<hir::Expr, Diagnostic> {
        let base = self.lower_expr(&ast.base)?;
        let base_ty = base.ty.clone();

        let kind = hir::ExprKind::Field(Box::new(base), ast.name.clone());
        let span = Some(ast.span);
        let ty = hir::Ty::Proj(hir::ProjTy {
            base: Box::new(base_ty),
            proj: hir::Projection::Field {
                name: ast.name.clone(),
            },
        });

        Ok(hir::Expr { kind, span, ty })
    }

    fn lower_call_expr(&mut self, ast: &ast::CallExpr) -> Result<hir::Expr, Diagnostic> {
        let callee = self.lower_expr(&ast.callee)?;
        let ty = hir::Ty::new_unknown(None);

        let mut arguments = Vec::new();
        let mut params = Vec::new();

        for argumnet in &ast.arguments {
            let argument = self.lower_expr(argumnet)?;
            params.push(argument.ty.clone());
            arguments.push(argument);
        }

        let func = hir::Ty::new_func(params, ty.clone());

        self.unit.env.assign(callee.ty.clone(), func);

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
                let ty = hir::Ty::new_pointer(mutable, value.ty.clone());

                let kind = hir::ExprKind::Ref(Box::new(value));
                let span = Some(ast.span);

                Ok(hir::Expr { kind, span, ty })
            }
            ast::UnaryOp::Deref => {
                let kind = hir::ExprKind::Deref(Box::new(value.clone()));
                let span = Some(ast.span);
                let ty = hir::Ty::Proj(hir::ProjTy {
                    base: Box::new(value.ty),
                    proj: hir::Projection::Deref,
                });

                Ok(hir::Expr { kind, span, ty })
            }
        }
    }

    fn lower_binary_math(
        &mut self,
        lhs: hir::Expr,
        rhs: hir::Expr,
        lang_trait: hir::LangTrait,
    ) -> hir::Expr {
        let trait_id = self.unit.get_lang_trait(lang_trait).unwrap();

        let constant = hir::Const::Method {
            implementor: lhs.ty.clone(),
            trait_id,
            trait_generics: vec![rhs.ty.clone()],
            method_generics: Vec::new(),
            index: 0,
        };

        let kind = hir::ExprKind::Const(constant);
        let span = None;

        let method = hir::Expr {
            kind,
            span,
            ty: todo!(),
        };

        let ty = hir::Ty::Proj(hir::ProjTy {
            base: Box::new(lhs.ty.clone()),
            proj: hir::Projection::Assoc {
                trait_id,
                trait_generics: vec![rhs.ty.clone()],
                assoc_index: 0,
            },
        });

        let kind = hir::ExprKind::Call(Box::new(method), vec![lhs, rhs]);
        let span = None;

        hir::Expr { kind, span, ty }
    }

    fn lower_binary_eq(&mut self, lhs: hir::Expr, rhs: hir::Expr) -> hir::Expr {
        todo!()
    }

    fn lower_binary_expr(&mut self, ast: &ast::BinaryExpr) -> Result<hir::Expr, Diagnostic> {
        let lhs = self.lower_expr(&ast.lhs)?;
        let rhs = self.lower_expr(&ast.rhs)?;

        match ast.op {
            ast::BinaryOp::Add => Ok(self.lower_binary_math(lhs, rhs, hir::LangTrait::Add)),
            ast::BinaryOp::Eq => Ok(self.lower_binary_eq(lhs, rhs)),
            _ => unimplemented!(),
        }
    }

    fn lower_assign_expr(&mut self, ast: &ast::AssignExpr) -> Result<hir::Expr, Diagnostic> {
        let lhs = self.lower_expr(&ast.lhs)?;
        let rhs = self.lower_expr(&ast.rhs)?;

        let kind = hir::ExprKind::Assign(Box::new(lhs), Box::new(rhs));
        let span = Some(ast.span);
        let ty = hir::Ty::VOID;

        Ok(hir::Expr { kind, span, ty })
    }

    fn lower_let_expr(&mut self, ast: &ast::LetExpr) -> Result<hir::Expr, Diagnostic> {
        // get the type if specified
        let ty = match ast.type_ {
            Some(ref type_) => self.lower_type(type_)?,
            None => hir::Ty::new_unknown(None),
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
        self.unit.env.assign(value.ty.clone(), ty.clone());

        let kind = hir::ExprKind::Let(local, Box::new(value));
        let span = Some(ast.span);

        Ok(hir::Expr {
            kind,
            span,
            ty: hir::Ty::VOID,
        })
    }

    fn lower_if_expr(&mut self, ast: &ast::IfExpr) -> Result<hir::Expr, Diagnostic> {
        let cond = self.lower_expr(&ast.cond)?;
        self.unit.env.assign(cond.ty.clone(), hir::Ty::BOOL);

        let then = self.lower_expr(&ast.then)?;
        let ty = then.ty.clone();

        let otherwise = match ast.otherwise {
            Some(ref expr) => {
                let expr = self.lower_expr(expr)?;
                self.unit.env.assign(expr.ty.clone(), ty.clone());
                Some(Box::new(expr))
            }
            None => {
                self.unit.env.assign(ty.clone(), hir::Ty::VOID);
                None
            }
        };

        let kind = hir::ExprKind::If(Box::new(cond), Box::new(then), otherwise);
        let span = Some(ast.span);

        Ok(hir::Expr { kind, span, ty })
    }

    fn lower_item_pat(&mut self, ast: &ast::ItemPat, ty: &hir::Ty) -> Result<hir::Pat, Diagnostic> {
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
        ty: &hir::Ty,
    ) -> Result<hir::Pat, Diagnostic> {
        if let Some(ref path) = ast.path {
            match self.resolve_path(path)? {
                Resolved::EnumVariant(id, generics, index) => {
                    let enum_def = &self.unit[id];

                    let mut specialization = hir::Spec::new();

                    for (&generic, ty) in enum_def.generics.iter().zip(&generics) {
                        specialization.insert(generic, ty.clone());
                    }

                    let enum_ty = hir::KnownTy::new_enum(id, generics.to_vec()).to_ty();

                    self.unit.env.assign(ty.clone(), enum_ty.clone());

                    let mut fields = Vec::new();

                    for (i, pat) in ast.pats.iter().enumerate() {
                        let ty = self.unit[id].field_ty(&generics, index, i)?;
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

    fn lower_pat(&mut self, ast: &ast::Pat, ty: &hir::Ty) -> Result<hir::Pat, Diagnostic> {
        match ast {
            ast::Pat::Item(ast) => self.lower_item_pat(ast, ty),
            ast::Pat::Tuple(ast) => self.lower_tuple_pat(ast, ty),
        }
    }

    fn lower_match_expr(&mut self, ast: &ast::MatchExpr) -> Result<hir::Expr, Diagnostic> {
        let value = self.lower_expr(&ast.value)?;
        let ty = hir::Ty::new_unknown(None);

        let mut arms = Vec::new();

        for arm in ast.arms.iter() {
            let pat = self.lower_pat(&arm.pat, &value.ty)?;
            let expr = self.lower_expr(&arm.body)?;

            self.unit.env.assign(expr.ty.clone(), ty.clone());

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
        let mut ty = hir::Ty::VOID;

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
        tcx: &mut TyCx,
        output: &hir::Ty,
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
