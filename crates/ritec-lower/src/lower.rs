use hir::FromPartial;
use ritec_ast as ast;
use ritec_diagnostic::Diagnostic;
use ritec_hir as hir;

use crate::{
    r#type::{Resolved, TyCx},
    Lowerer,
};

impl Lowerer {
    fn lower_trait_bound(
        &self,
        tcx: &mut TyCx,
        base: hir::KnownTy,
        ast: &ast::TraitBound,
    ) -> Result<hir::Bound, Diagnostic> {
        let Resolved::Trait(trait_id, generics) = tcx.resolve_path(&self.unit, &ast.item)? else {
            todo!()
        };

        assert!(generics.is_empty());

        let trait_ = &self.unit[trait_id];

        let mut generics = Vec::new();

        for generic in ast.generics.iter() {
            let generic = tcx.lower_type(&self.unit, generic)?;
            generics.push(generic);
        }

        if generics.len() != trait_.generics.len() {
            let message = format!(
                "expected {} generics, found {}",
                trait_.generics.len(),
                generics.len()
            );
            return Err(Diagnostic::new(message).with_span(ast.span));
        }

        let mut assocs = vec![None; trait_.assocs.len()];

        for ast in ast.types.iter() {
            if let Some(i) = trait_.assoc_index(&ast.name) {
                assert!(assocs[i].is_none());
                assocs[i] = Some(tcx.lower_type(&self.unit, &ast.type_)?);
            } else {
                let message = format!("trait has no associated type `{}`", ast.name);
                return Err(Diagnostic::new(message));
            }
        }

        Ok(hir::Bound {
            implementor: base,
            trait_id,
            generics,
            assocs,
            span: Some(ast.span),
        })
    }

    fn lower_contract(
        &self,
        tcx: &mut TyCx,
        contract: &mut hir::Contract,
        ast: &ast::Contract,
    ) -> Result<(), Diagnostic> {
        for clause in ast.clauses.iter() {
            let base = tcx.lower_type(&self.unit, &clause.bindee)?;

            for bound in clause.bounds.iter() {
                let bound = self.lower_trait_bound(tcx, base.clone(), bound)?;
                contract.bounds.push(bound);
            }
        }

        Ok(())
    }

    fn lower_enum(&mut self, module: hir::ModuleId, ast: &ast::Enum) -> Result<(), Diagnostic> {
        let id = self.unit.modules[module].enums[&ast.name];

        let mut generics = Vec::new();

        for generic in ast.generics.iter() {
            generics.push((generic.name.clone(), hir::Generic::new()));
        }

        let mut tcx = TyCx {
            module,
            generics: &mut generics,
            allow_new_generics: true,
            trait_id: None,
            self_ty: None,
            spec: None,
        };

        let mut variants = Vec::new();

        for (i, ast) in ast.variants.iter().enumerate() {
            let name = ast.name.clone();
            let mut fields = Vec::new();

            for field in ast.fields.iter() {
                fields.push(tcx.lower_type(&self.unit, field)?);
            }

            variants.push(hir::VariantDef {
                name,
                fields,
                builder: None,
                span: Some(ast.span),
            });
        }

        tcx.allow_new_generics = false;

        let mut contract = hir::Contract::default();
        self.lower_contract(&mut tcx, &mut contract, &ast.contract)?;
        let contract = self.unit.contracts.push(contract);

        let generics: Vec<_> = tcx.generics.iter().map(|(_, g)| *g).collect();
        let params = generics
            .iter()
            .copied()
            .map(hir::KnownTy::Generic)
            .collect();

        let self_type = hir::KnownTy::new_enum(id, params);

        for (i, variant) in variants.iter_mut().enumerate() {
            if variant.fields.is_empty() {
                continue;
            }

            let mut arguments = Vec::new();
            let mut locals = hir::Locals::new();
            let mut fields = Vec::new();

            for field in variant.fields.iter() {
                let local = hir::Local {
                    mutable: false,
                    name: None,
                    ty: field.to_ty(),
                };

                let local_id = locals.push(local);
                let argument = hir::Argument {
                    name: None,
                    local: local_id,
                    ty: field.clone(),
                };

                arguments.push(argument);

                let field = hir::Expr {
                    kind: hir::ExprKind::Local(local_id),
                    span: None,
                    ty: field.to_ty(),
                };

                fields.push(field);
            }

            let params = generics.iter().copied().map(hir::Ty::Generic).collect();

            let expr = hir::Expr {
                kind: hir::ExprKind::Variant(id, params, i, fields),
                span: None,
                ty: self_type.to_ty(),
            };

            let body = hir::Body {
                name: None,
                arguments,
                output: self_type.clone(),
                generics: generics.clone(),
                contract,
                locals,
                expr,
            };

            let body_id = self.unit.bodies.push(body);
            variant.builder = Some(body_id);
        }

        let hir = hir::EnumDef {
            name: Some(ast.name.clone()),
            contract,
            generics,
            variants,
            span: Some(ast.span),
        };

        self.unit.enums.insert(id, hir);

        Ok(())
    }

    fn lower_struct(&mut self, module: hir::ModuleId, ast: &ast::Struct) -> Result<(), Diagnostic> {
        let id = self.unit.modules[module].structs[&ast.name];

        let mut generics = Vec::new();

        for generic in ast.generics.iter() {
            generics.push((generic.name.clone(), hir::Generic::new()));
        }

        let mut tcx = TyCx {
            module,
            generics: &mut generics,
            allow_new_generics: true,
            trait_id: None,
            self_ty: None,
            spec: None,
        };

        let mut fields = Vec::new();

        for field in ast.fields.iter() {
            let name = field.name.clone();
            let ty = tcx.lower_type(&self.unit, &field.type_)?;
            let span = Some(field.span);
            fields.push(hir::FieldDef { name, ty, span });
        }

        let mut contract = hir::Contract::default();
        self.lower_contract(&mut tcx, &mut contract, &ast.contract)?;
        let contract = self.unit.contracts.push(contract);

        let generics = generics.into_iter().map(|(_, g)| g).collect();

        let hir = hir::StructDef {
            name: Some(ast.name.clone()),
            contract,
            generics,
            fields,
            span: Some(ast.span),
        };

        self.unit.structs.insert(id, hir);

        Ok(())
    }

    fn lower_function(
        &mut self,
        module: hir::ModuleId,
        ast: &ast::Function,
    ) -> Result<(), Diagnostic> {
        let id = self.unit.modules[module].funcs[&ast.name];

        let mut generics = Vec::new();

        for generic in ast.generics.iter() {
            generics.push((generic.name.clone(), hir::Generic::new()));
        }

        let mut tcx = TyCx {
            module,
            generics: &mut generics,
            allow_new_generics: true,
            trait_id: None,
            self_ty: None,
            spec: None,
        };

        let mut arguments = Vec::new();
        let mut locals = hir::Locals::new();

        for argument in ast.arguments.iter() {
            let ty = tcx.lower_type(&self.unit, &argument.type_)?;

            let local = hir::Local {
                mutable: argument.mutable,
                name: Some(argument.name.clone()),
                ty: ty.to_ty(),
            };

            let argument = hir::Argument {
                name: Some(argument.name.clone()),
                local: locals.push(local),
                ty,
            };

            arguments.push(argument);
        }

        let output = match &ast.output {
            Some(output) => tcx.lower_type(&self.unit, output)?,
            None => hir::KnownTy::VOID,
        };

        tcx.allow_new_generics = false;

        let mut contract = hir::Contract::default();
        self.lower_contract(&mut tcx, &mut contract, &ast.contract)?;
        let contract = self.unit.contracts.push(contract);

        let expr = self.lower_body(
            &mut tcx,
            &output.to_ty(),
            contract,
            None,
            &mut locals,
            &ast.body,
        )?;

        self.unit.env.assign(expr.ty.clone(), output.to_ty());

        let generics = generics.into_iter().map(|(_, g)| g).collect();

        let hir = hir::Body {
            name: Some(ast.name.clone()),
            arguments,
            output,
            generics,
            contract,
            locals,
            expr,
        };

        self.unit.bodies.insert(id, hir);

        Ok(())
    }

    fn lower_trait(&mut self, module: hir::ModuleId, ast: &ast::Trait) -> Result<(), Diagnostic> {
        let id = self.unit.modules[module].traits[&ast.name];
        let trait_def = &self.unit.traits[id];

        let mut generics = Vec::new();

        for (ast, &generic) in ast.generics.iter().zip(&trait_def.generics) {
            generics.push((ast.name.clone(), generic));
        }

        let self_type = hir::KnownTy::Generic(trait_def.self_generic);

        let mut tcx = TyCx {
            module,
            generics: &mut generics,
            allow_new_generics: false,
            trait_id: Some(id),
            self_ty: Some(self_type.clone()),
            spec: None,
        };

        let mut contract = hir::Contract::default();
        self.lower_contract(&mut tcx, &mut contract, &ast.contract)?;

        let trait_generics: Vec<_> = trait_def
            .generics
            .iter()
            .cloned()
            .map(hir::KnownTy::Generic)
            .collect();

        for (index, assoc) in ast.types.iter().enumerate() {
            let base = hir::KnownTy::Assoc(
                Box::new(self_type.clone()),
                id,
                trait_generics.clone(),
                index,
            );

            for bound in assoc.bounds.iter() {
                let bound = self.lower_trait_bound(&mut tcx, base.clone(), bound)?;
                contract.bounds.push(bound);
            }
        }

        for method in ast.methods.iter() {
            let name = method.name.clone();
            let mut generics = generics.clone();
            let offset = generics.len();

            for generic in method.generics.iter() {
                generics.push((generic.name.clone(), hir::Generic::new()));
            }

            let mut tcx = TyCx {
                module,
                generics: &mut generics,
                allow_new_generics: true,
                trait_id: Some(id),
                self_ty: Some(self_type.clone()),
                spec: None,
            };

            let mut arguments = Vec::new();

            if let Some(self_argument) = method.self_argument {
                let type_ = match self_argument {
                    ast::SelfArgument::Value => self_type.clone(),
                    ast::SelfArgument::Ref => hir::KnownTy::new_pointer(false, self_type.clone()),
                    ast::SelfArgument::MutRef => hir::KnownTy::new_pointer(true, self_type.clone()),
                };

                arguments.push(type_);
            }

            for argument in method.arguments.iter() {
                let type_ = tcx.lower_type(&self.unit, &argument.type_)?;
                arguments.push(type_);
            }

            let output = match method.output {
                Some(ref output) => tcx.lower_type(&self.unit, output)?,
                None => hir::KnownTy::VOID,
            };

            let mut contract = hir::Contract::default();
            self.lower_contract(&mut tcx, &mut contract, &method.contract)?;
            let contract = self.unit.contracts.push(contract);

            let generics = generics.into_iter().skip(offset).map(|(_, g)| g).collect();

            let method = hir::MethodDef {
                name,
                generics,
                arguments,
                output,
                contract,
                span: Some(method.span),
            };

            self.unit[id].methods.push(method);
        }

        let trait_ = &self.unit[id];
        self.unit.contracts.insert(trait_.contract, contract);

        Ok(())
    }

    fn lower_trait_impl(
        &mut self,
        module: hir::ModuleId,
        ast: &ast::TraitImpl,
    ) -> Result<(), Diagnostic> {
        // the type context for lowering the trait
        let mut tcx = TyCx {
            module,
            generics: &mut Vec::new(),
            allow_new_generics: true,
            trait_id: None,
            self_ty: None,
            spec: None,
        };

        let query = tcx.resolve_path(&self.unit, &ast.trait_)?;
        let Resolved::Trait(trait_id, generics) = query else {
            let message = format!("trait `{:?}` not found", ast.trait_);
            return Err(Diagnostic::new(message).with_span(ast.span));
        };

        let implementor = tcx.lower_type(&self.unit, &ast.implementor)?;

        tcx.trait_id = Some(trait_id);
        tcx.self_ty = Some(implementor.clone());

        let trait_ = self.unit[trait_id].clone();

        let mut specialization = hir::Spec::new();
        specialization.insert(trait_.self_generic, implementor.clone());

        for (&generic, type_) in trait_.generics.iter().zip(&generics) {
            specialization.insert(generic, type_.clone());
        }

        tcx.spec = Some(&specialization);
        tcx.allow_new_generics = false;

        let mut contract = hir::Contract::default();
        self.lower_contract(&mut tcx, &mut contract, &ast.contract)?;
        let contract = self.unit.contracts.push(contract);

        let mut assocs = vec![None; trait_.assocs.len()];

        for assoc in ast.types.iter() {
            let Some(index) = trait_.assoc_index(&assoc.name) else {
                let message = format!("trait has no associated type `{}`", assoc.name);
                return Err(Diagnostic::new(message).with_span(assoc.span));
            };

            if assocs[index].is_some() {
                let message = format!("duplicate associated type `{}`", assoc.name);
                return Err(Diagnostic::new(message).with_span(assoc.span));
            }

            let assoc_impl = hir::AssocImpl {
                ty: tcx.lower_type(&self.unit, &assoc.type_)?,
                span: Some(assoc.span),
            };

            assocs[index] = Some(assoc_impl);
        }

        let assocs = assocs.into_iter().map(Option::unwrap).collect();

        let mut methods = vec![None; trait_.methods.len()];

        for method in ast.methods.iter() {
            let Some(index) = trait_.method_index(&method.name) else {
                let message = format!("trait has no method `{}`", method.name);
                return Err(Diagnostic::new(message).with_span(method.span));
            };

            if methods[index].is_some() {
                let message = format!("duplicate method `{}`", method.name);
                return Err(Diagnostic::new(message).with_span(method.span));
            }

            let mut tcx = TyCx {
                module,
                generics: &mut tcx.generics.clone(),
                allow_new_generics: true,
                trait_id: Some(trait_id),
                self_ty: Some(implementor.clone()),
                spec: Some(&specialization),
            };

            for generic in method.generics.iter() {
                tcx.generics
                    .push((generic.name.clone(), hir::Generic::new()));
            }

            let mut arguments = Vec::new();
            let mut locals = hir::Locals::new();

            let self_argument = match method.self_argument {
                Some(self_argument) => {
                    let ty = match self_argument {
                        ast::SelfArgument::Value => implementor.clone(),
                        ast::SelfArgument::Ref => {
                            hir::KnownTy::new_pointer(false, implementor.clone())
                        }
                        ast::SelfArgument::MutRef => {
                            hir::KnownTy::new_pointer(true, implementor.clone())
                        }
                    };

                    let local = hir::Local {
                        mutable: false,
                        name: Some(String::from("self")),
                        ty: ty.to_ty(),
                    };

                    let id = locals.push(local);

                    let argument = hir::Argument {
                        name: Some(String::from("self")),
                        local: id,
                        ty,
                    };

                    arguments.push(argument);

                    Some(id)
                }
                None => None,
            };

            for argument in method.arguments.iter() {
                let ty = tcx.lower_type(&self.unit, &argument.type_)?;

                let local = hir::Local {
                    mutable: argument.mutable,
                    name: Some(argument.name.clone()),
                    ty: ty.to_ty(),
                };

                let argument = hir::Argument {
                    name: Some(argument.name.clone()),
                    local: locals.push(local),
                    ty,
                };

                arguments.push(argument);
            }

            let output = match &method.output {
                Some(output) => tcx.lower_type(&self.unit, output)?,
                None => hir::KnownTy::VOID,
            };

            tcx.allow_new_generics = false;

            let mut contract = hir::Contract::default();
            self.lower_contract(&mut tcx, &mut contract, &method.contract)?;
            let contract = self.unit.contracts.push(contract);

            let expr = self.lower_body(
                &mut tcx,
                &output.to_ty(),
                contract,
                self_argument,
                &mut locals,
                &method.body,
            )?;

            self.unit.env.assign(expr.ty.clone(), output.to_ty());

            let generics: Vec<_> = tcx.generics.iter().map(|(_, g)| *g).collect();

            let body = hir::Body {
                name: Some(method.name.clone()),
                arguments: arguments.clone(),
                output,
                generics: generics.clone(),
                contract,
                locals,
                expr,
            };

            let method = hir::MethodImpl {
                name: method.name.clone(),
                body: self.unit.bodies.push(body),
                span: Some(method.span),
            };

            methods[index] = Some(method);
        }

        let methods = methods.into_iter().map(Option::unwrap).collect();

        let trait_impl = hir::TraitImpl {
            trait_id,
            generics,
            implementor,
            contract,
            assocs,
            methods,
            span: Some(ast.span),
        };

        self.unit.trait_impls.push(trait_impl);

        Ok(())
    }

    fn lower_impl(&mut self, module: hir::ModuleId, ast: &ast::Impl) -> Result<(), Diagnostic> {
        let mut tcx = TyCx {
            module,
            generics: &mut Vec::new(),
            allow_new_generics: true,
            trait_id: None,
            self_ty: None,
            spec: None,
        };

        let implementor = tcx.lower_type(&self.unit, &ast.implementor)?;
        tcx.self_ty = Some(implementor.clone());

        let mut contract = hir::Contract::default();
        self.lower_contract(&mut tcx, &mut contract, &ast.contract)?;
        let contract = self.unit.contracts.push(contract);

        let mut methods = Vec::new();

        for method in ast.methods.iter() {
            let mut tcx = TyCx {
                module,
                generics: &mut tcx.generics.clone(),
                allow_new_generics: true,
                trait_id: None,
                self_ty: tcx.self_ty.clone(),
                spec: None,
            };

            for generic in method.generics.iter() {
                tcx.generics
                    .push((generic.name.clone(), hir::Generic::new()));
            }

            let mut arguments = Vec::new();
            let mut locals = hir::Locals::new();

            let self_argument = match method.self_argument {
                Some(self_argument) => {
                    let ty = match self_argument {
                        ast::SelfArgument::Value => implementor.clone(),
                        ast::SelfArgument::Ref => {
                            hir::KnownTy::new_pointer(false, implementor.clone())
                        }
                        ast::SelfArgument::MutRef => {
                            hir::KnownTy::new_pointer(true, implementor.clone())
                        }
                    };

                    let local = hir::Local {
                        mutable: false,
                        name: Some(String::from("self")),
                        ty: ty.to_ty(),
                    };

                    let id = locals.push(local);
                    let argument = hir::Argument {
                        name: Some(String::from("self")),
                        local: id,
                        ty,
                    };

                    arguments.push(argument);
                    Some(id)
                }
                None => None,
            };

            for argument in method.arguments.iter() {
                let ty = tcx.lower_type(&self.unit, &argument.type_)?;

                let local = hir::Local {
                    mutable: argument.mutable,
                    name: Some(argument.name.clone()),
                    ty: ty.to_ty(),
                };

                let argument = hir::Argument {
                    name: Some(argument.name.clone()),
                    local: locals.push(local),
                    ty,
                };

                arguments.push(argument);
            }

            let output = match &method.output {
                Some(output) => tcx.lower_type(&self.unit, output)?,
                None => hir::KnownTy::VOID,
            };

            tcx.allow_new_generics = false;

            let mut contract = hir::Contract::default();
            self.lower_contract(&mut tcx, &mut contract, &method.contract)?;
            let contract = self.unit.contracts.push(contract);

            let expr = self.lower_body(
                &mut tcx,
                &output.to_ty(),
                contract,
                self_argument,
                &mut locals,
                &method.body,
            )?;

            self.unit.env.assign(expr.ty.clone(), output.to_ty());

            let generics: Vec<_> = tcx.generics.iter().map(|(_, g)| *g).collect();

            let body = hir::Body {
                name: Some(method.name.clone()),
                arguments,
                output,
                generics: generics.clone(),
                contract,
                locals,
                expr,
            };

            let method = hir::MethodImpl {
                name: method.name.clone(),
                body: self.unit.bodies.push(body),
                span: Some(method.span),
            };

            methods.push(method)
        }

        let generics = tcx.generics.iter().map(|(_, g)| *g).collect();

        let impl_ = hir::Impl {
            generics,
            implementor,
            contract,
            methods,
        };

        self.unit.impls.push(impl_);

        Ok(())
    }

    fn lower_module_decl(
        &mut self,
        module: hir::ModuleId,
        ast: &ast::ModuleDecl,
    ) -> Result<(), Diagnostic> {
        let module = self.unit.modules[module].modules[&ast.name];

        match ast.module {
            Some(ref ast) => self.lower_module(module, ast),
            None => todo!(),
        }
    }

    pub fn lower_module(
        &mut self,
        module: hir::ModuleId,
        ast: &ast::Module,
    ) -> Result<(), Diagnostic> {
        for decl in ast.decls.iter() {
            match decl {
                ast::Decl::Enum(ast) => self.lower_enum(module, ast)?,
                ast::Decl::Struct(ast) => self.lower_struct(module, ast)?,
                ast::Decl::Function(ast) => self.lower_function(module, ast)?,
                ast::Decl::Trait(ast) => self.lower_trait(module, ast)?,
                ast::Decl::TraitImpl(ast) => self.lower_trait_impl(module, ast)?,
                ast::Decl::Impl(ast) => self.lower_impl(module, ast)?,
                ast::Decl::Module(ast) => self.lower_module_decl(module, ast)?,
            }
        }

        Ok(())
    }
}
