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
        generics: Vec<hir::Type>,
        span: Span,
    ) -> Result<hir::Expr, Diagnostic> {
        let body = &self.unit.bodies[id];

        let mut specialization = hir::Specialization::new();

        for (&generic, type_) in body.generics.iter().zip(&generics) {
            specialization.insert(generic, type_.clone());
        }

        let mut params = Vec::new();

        params.push(specialization.specialize(&body.output));

        for argument in &body.arguments {
            let local = &body.locals[*argument];
            params.push(specialization.specialize(&local.type_));
        }

        let kind = hir::ExprKind::Const(hir::Constant::Func(id, generics));
        let span = Some(span);
        let ty = hir::Type::Partial(hir::Partial {
            item: hir::Item::Function,
            params,
        });

        Ok(hir::Expr { kind, span, ty })
    }

    fn lower_item_expr(&mut self, ast: &ast::ItemExpr) -> Result<hir::Expr, Diagnostic> {
        if let Some(ident) = ast.item.ident() {
            if let Some(local) = self.get_local(ident) {
                return Ok(hir::Expr {
                    kind: hir::ExprKind::Local(local),
                    span: Some(ast.span),
                    ty: self.locals[local].type_.clone(),
                });
            }
        }

        match self.query_item(&ast.item)? {
            ItemQuery::Func(id, generics) => self.lower_func_expr(id, generics, ast.span),
            _ => {
                let message = format!("expected function, found {:?}", ast.item);
                Err(Diagnostic::new(message).with_span(ast.span))
            }
        }
    }

    fn lower_int_expr(&mut self, ast: &ast::LitIntExpr) -> Result<hir::Expr, Diagnostic> {
        let kind = hir::ExprKind::Const(hir::Constant::Int(ast.value));
        let span = Some(ast.span);
        let type_ = hir::Type::Partial(hir::Partial {
            item: hir::Item::Int {
                signed: true,
                width: Some(32),
            },
            params: Vec::new(),
        });

        Ok(hir::Expr {
            kind,
            span,
            ty: type_,
        })
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

    fn lower_binary_expr(&mut self, ast: &ast::BinaryExpr) -> Result<hir::Expr, Diagnostic> {
        let lhs = self.lower_expr(&ast.lhs)?;
        let rhs = self.lower_expr(&ast.rhs)?;

        let (trait_id, trait_) = self
            .unit
            .types
            .traits
            .iter()
            .find(|(_, trait_)| trait_.name.as_deref() == Some("Add"))
            .unwrap();

        let mut specialization = hir::Specialization::new();
        specialization.insert(trait_.generics[0], rhs.ty.clone());

        let constant = hir::Constant::Method {
            implementor: lhs.ty.clone(),
            trait_id,
            generics: vec![rhs.ty.clone()],
            index: 0,
        };

        let kind = hir::ExprKind::Const(constant);
        let span = Some(ast.span);

        let type_ = hir::Type::Projected(hir::Projected {
            contract: trait_.contract,
            base: Box::new(lhs.ty.clone()),
            projection: hir::Projection::Associated {
                trait_id,
                generics: vec![rhs.ty.clone()],
                index: 0,
            },
        });

        Ok(hir::Expr {
            kind,
            span,
            ty: type_,
        })
    }

    fn lower_let_expr(&mut self, ast: &ast::LetExpr) -> Result<hir::Expr, Diagnostic> {
        // get the type if specified
        let type_ = match ast.type_ {
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
            type_: type_.clone(),
        });

        // push the local to the scope
        self.scope.push(local);

        // unify the type of the value with the type of the local
        self.unit.types.unify(value.ty.clone(), type_.clone());

        let kind = hir::ExprKind::Let(local, Box::new(value));
        let span = Some(ast.span);

        Ok(hir::Expr {
            kind,
            span,
            ty: hir::Type::VOID,
        })
    }

    fn lower_block_expr(&mut self, ast: &ast::BlockExpr) -> Result<hir::Expr, Diagnostic> {
        let mut lowerer = BodyLowerer {
            lowerer: self.lowerer,
            tcx: self.tcx,
            output: self.output,
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
            ast::Expr::LitFloat(_) => todo!(),
            ast::Expr::Struct(_) => todo!(),
            ast::Expr::Paren(expr) => self.lower_expr(&expr.expr),
            ast::Expr::Field(_) => todo!(),
            ast::Expr::Call(expr) => self.lower_call_expr(expr),
            ast::Expr::Unary(_) => todo!(),
            ast::Expr::Binary(expr) => self.lower_binary_expr(expr),
            ast::Expr::Let(expr) => self.lower_let_expr(expr),
            ast::Expr::Loop(_) => todo!(),
            ast::Expr::Match(_) => todo!(),
            ast::Expr::Block(expr) => self.lower_block_expr(expr),
        }
    }
}

impl Lowerer {
    pub fn lower_body(
        &mut self,
        tcx: &mut TypeContext,
        output: &hir::Type,
        locals: &mut hir::Locals,
        ast: &ast::Expr,
    ) -> Result<hir::Expr, Diagnostic> {
        let scope = locals.keys().collect();

        let mut lowerer = BodyLowerer {
            lowerer: self,
            tcx,
            output,
            locals,
            scope,
        };

        lowerer.lower_expr(ast)
    }
}
