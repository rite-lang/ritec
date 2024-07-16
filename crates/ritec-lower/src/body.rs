use std::ops::{Deref, DerefMut};

use ritec_ast as ast;
use ritec_diagnostic::{Diagnostic, Span};
use ritec_hir as hir;

use crate::{
    r#type::{ItemQuery, TypeContext},
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

    fn query_item(&mut self, ast: &ast::Item) -> Result<ItemQuery, Diagnostic> {
        self.tcx.query_item(&self.lowerer.unit, ast)
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
        mut generics: Vec<hir::Type>,
        span: Span,
    ) -> Result<hir::Expr, Diagnostic> {
        let body = &self.unit.bodies[id];

        if generics.is_empty() {
            generics = (0..body.generics.len())
                .map(|_| hir::Type::unknown(span))
                .collect();
        }

        let mut specialization = hir::Specialization::new();

        for (&generic, type_) in body.generics.iter().zip(&generics) {
            specialization.insert(generic, type_.clone());
        }

        let mut params = Vec::new();

        params.push(specialization.specialize(&body.output));

        for argument in &body.arguments {
            let local = &body.locals[*argument];
            params.push(specialization.specialize(&local.ty));
        }

        let kind = hir::ExprKind::Const(hir::Constant::Func(id, generics));
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
        if let Some(ident) = ast.item.ident() {
            if let Some(local) = self.get_local(ident) {
                return Ok(hir::Expr {
                    kind: hir::ExprKind::Local(local),
                    span: Some(ast.span),
                    ty: self.locals[local].ty.clone(),
                });
            }
        }

        match self.query_item(&ast.item)? {
            ItemQuery::Func(id, generics) => self.lower_func_expr(id, generics, ast.span),
            ItemQuery::SelfArgument => match self.self_argument {
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
            _ => {
                let message = format!("expected function, found {:?}", ast.item);
                Err(Diagnostic::new(message).with_span(ast.span))
            }
        }
    }

    fn lower_int_expr(&mut self, ast: &ast::LitIntExpr) -> Result<hir::Expr, Diagnostic> {
        let kind = hir::ExprKind::Const(hir::Constant::Int(ast.value));
        let span = Some(ast.span);
        let ty = hir::Type::Unknown(hir::Unknown {
            kind: hir::UnknownKind::Number { float: false },
            uid: hir::Uid::new(),
            span: ast.span,
        });

        Ok(hir::Expr { kind, span, ty })
    }

    fn lower_float_expr(&mut self, ast: &ast::LitFloatExpr) -> Result<hir::Expr, Diagnostic> {
        let kind = hir::ExprKind::Const(hir::Constant::Float(ast.value));
        let span = Some(ast.span);
        let ty = hir::Type::Unknown(hir::Unknown {
            kind: hir::UnknownKind::Number { float: true },
            uid: hir::Uid::new(),
            span: ast.span,
        });

        Ok(hir::Expr { kind, span, ty })
    }

    fn lower_null_expr(&mut self, ast: &ast::NullExpr) -> Result<hir::Expr, Diagnostic> {
        let kind = hir::ExprKind::Const(hir::Constant::Null);
        let span = Some(ast.span);
        let ty = hir::Type::Partial(hir::Partial {
            item: hir::Item::Pointer { mutable: true },
            params: vec![hir::Type::unknown(ast.span)],
        });

        Ok(hir::Expr { kind, span, ty })
    }

    fn lower_struct_expr(&mut self, ast: &ast::StructExpr) -> Result<hir::Expr, Diagnostic> {
        let ItemQuery::Struct(id, mut generics) = self.query_item(&ast.item)? else {
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

        let mut specialization = hir::Specialization::new();

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
        let callee = self.lower_expr(&ast.callee)?;
        let ty = hir::Type::unknown(ast.span);

        let mut arguments = Vec::new();
        let mut params = Vec::new();

        params.push(ty.clone());

        for argumnet in &ast.arguments {
            let argument = self.lower_expr(argumnet)?;
            params.push(argument.ty.clone());
            arguments.push(argument);
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
            ast::UnaryOp::Ref => {
                let ty = hir::Type::Partial(hir::Partial {
                    item: hir::Item::Pointer { mutable: true },
                    params: vec![value.ty.clone()],
                });

                let kind = hir::ExprKind::Ref(Box::new(value));
                let span = Some(ast.span);

                Ok(hir::Expr { kind, span, ty })
            }
            ast::UnaryOp::Deref => {
                let ty = hir::Type::unknown(ast.span);
                let pointer_ty = hir::Type::Partial(hir::Partial {
                    item: hir::Item::Pointer { mutable: true },
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
        let constant = hir::Constant::Method {
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
            projection: hir::Projection::Associated {
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

        let constant = hir::Constant::Method {
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

        self.unit.types.unify(lhs.ty.clone(), rhs.ty.clone());

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

            return Ok(hir::Pat::Binding(local));
        }

        let message = format!("expected local, found {:?}", ast);
        Err(Diagnostic::new(message).with_span(ast.span))
    }

    fn lower_pat(&mut self, ast: &ast::Pat, ty: &hir::Type) -> Result<hir::Pat, Diagnostic> {
        match ast {
            ast::Pat::Item(item) => self.lower_item_pat(item, ty),
            ast::Pat::Tuple(_) => todo!(),
        }
    }

    fn lower_match_expr(&mut self, ast: &ast::MatchExpr) -> Result<hir::Expr, Diagnostic> {
        let value = self.lower_expr(&ast.value)?;
        let ty = hir::Type::unknown(ast.span);

        let mut arms = Vec::new();

        for arm in ast.arms.iter() {
            let pat = self.lower_pat(&arm.pat, &value.ty)?;
            let expr = self.lower_expr(&arm.expr)?;

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
        let mut type_ = hir::Type::VOID;

        for expr in &ast.exprs {
            let expr = lowerer.lower_expr(expr)?;
            type_ = expr.ty.clone();
            exprs.push(expr);
        }

        let kind = hir::ExprKind::Block(exprs);
        let span = Some(ast.span);

        Ok(hir::Expr {
            kind,
            span,
            ty: type_,
        })
    }

    pub fn lower_expr(&mut self, ast: &ast::Expr) -> Result<hir::Expr, Diagnostic> {
        match ast {
            ast::Expr::Void(expr) => self.lower_void_expr(expr),
            ast::Expr::Item(expr) => self.lower_item_expr(expr),
            ast::Expr::LitInt(expr) => self.lower_int_expr(expr),
            ast::Expr::LitFloat(expr) => self.lower_float_expr(expr),
            ast::Expr::Null(expr) => self.lower_null_expr(expr),
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
