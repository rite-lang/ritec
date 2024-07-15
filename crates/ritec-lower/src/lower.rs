use ritec_ast as ast;
use ritec_diagnostic::Diagnostic;
use ritec_hir as hir;

use crate::{
    r#type::{ItemQuery, TypeContext},
    Lowerer,
};

impl Lowerer {
    fn lower_trait_bound(
        &self,
        tcx: &mut TypeContext,
        base: hir::Type,
        ast: &ast::TraitBound,
    ) -> Result<hir::TraitBound, Diagnostic> {
        let ItemQuery::Trait(trait_id, generics) = tcx.query_item(&self.unit, &ast.item)? else {
            todo!()
        };

        assert!(generics.is_empty());

        let trait_ = &self.unit.types[trait_id];

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

        let mut types = vec![None; trait_.assocs.len()];

        for assoc in ast.types.iter() {
            if let Some(i) = trait_
                .assocs
                .iter()
                .position(|assoc_| assoc_.name == assoc.name)
            {
                assert!(types[i].is_none());
                types[i] = Some(tcx.lower_type(&self.unit, &assoc.type_)?);
            } else {
                let message = format!("trait has no associated type `{}`", assoc.name);
                return Err(Diagnostic::new(message));
            }
        }

        Ok(hir::TraitBound {
            base,
            trait_id,
            generics,
            types,
        })
    }

    fn lower_contract(
        &self,
        tcx: &mut TypeContext,
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
        let mut generics = Vec::new();

        for generic in ast.generics.iter() {
            generics.push((generic.name.clone(), hir::Generic::new()));
        }

        let mut tcx = TypeContext {
            module,
            generics: &mut generics,
            allow_new_generics: true,
            trait_id: None,
        };

        let mut variants = Vec::new();

        for variant in ast.variants.iter() {
            let name = variant.name.clone();
            let mut fields = Vec::new();

            for field in variant.fields.iter() {
                fields.push(tcx.lower_type(&self.unit, field)?);
            }

            variants.push(hir::Variant { name, fields });
        }

        let mut contract = hir::Contract::default();
        self.lower_contract(&mut tcx, &mut contract, &ast.contract)?;
        let contract = self.unit.types.contracts.push(contract);

        let generics = generics.into_iter().map(|(_, g)| g).collect();

        let hir = hir::Enum {
            contract,
            generics,
            variants,
        };

        let id = self.unit.modules[module].enums[&ast.name];
        self.unit.types.enums.insert(id, hir);

        Ok(())
    }

    fn lower_struct(&mut self, module: hir::ModuleId, ast: &ast::Struct) -> Result<(), Diagnostic> {
        let mut generics = Vec::new();

        for generic in ast.generics.iter() {
            generics.push((generic.name.clone(), hir::Generic::new()));
        }

        let mut tcx = TypeContext {
            module,
            generics: &mut generics,
            allow_new_generics: true,
            trait_id: None,
        };

        let mut fields = Vec::new();

        for field in ast.fields.iter() {
            let name = field.name.clone();
            let type_ = tcx.lower_type(&self.unit, &field.type_)?;
            fields.push(hir::Field { name, type_ });
        }

        let mut contract = hir::Contract::default();
        self.lower_contract(&mut tcx, &mut contract, &ast.contract)?;
        let contract = self.unit.types.contracts.push(contract);

        let generics = generics.into_iter().map(|(_, g)| g).collect();

        let hir = hir::Struct {
            contract,
            generics,
            fields,
        };

        let id = self.unit.modules[module].structs[&ast.name];
        self.unit.types.structs.insert(id, hir);

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

        let mut tcx = TypeContext {
            module,
            generics: &mut generics,
            allow_new_generics: true,
            trait_id: None,
        };

        let mut arguments = Vec::new();
        let mut locals = hir::Locals::new();

        for argument in ast.arguments.iter() {
            let local = hir::Local {
                mutable: argument.mutable,
                name: Some(argument.name.clone()),
                type_: tcx.lower_type(&self.unit, &argument.type_)?,
            };

            let id = locals.push(local);
            arguments.push(id);
        }

        let output = match &ast.output {
            Some(output) => tcx.lower_type(&self.unit, output)?,
            None => hir::Type::VOID,
        };

        let mut contract = hir::Contract::default();
        self.lower_contract(&mut tcx, &mut contract, &ast.contract)?;
        let contract = self.unit.types.contracts.push(contract);

        let expr = self.lower_body(&mut tcx, &output, &mut locals, &ast.body)?;

        self.unit.types.unify(expr.ty.clone(), output.clone());

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
        let trait_ = &self.unit.types.traits[id];

        let mut generics = Vec::new();

        for (ast, &generic) in ast.generics.iter().zip(&trait_.generics) {
            generics.push((ast.name.clone(), generic));
        }

        let mut tcx = TypeContext {
            module,
            generics: &mut generics,
            allow_new_generics: false,
            trait_id: Some(id),
        };

        let mut contract = hir::Contract::default();
        self.lower_contract(&mut tcx, &mut contract, &ast.contract)?;

        let trait_generics: Vec<_> = trait_
            .generics
            .iter()
            .cloned()
            .map(hir::Type::Generic)
            .collect();

        for (index, assoc) in ast.types.iter().enumerate() {
            let base = hir::Type::Projected(hir::Projected {
                contract: trait_.contract,
                base: Box::new(hir::Type::SelfType),
                projection: hir::Projection::Associated {
                    trait_id: id,
                    generics: trait_generics.clone(),
                    index,
                },
            });

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

            let mut tcx = TypeContext {
                module,
                generics: &mut generics,
                allow_new_generics: true,
                trait_id: Some(id),
            };

            let mut arguments = Vec::new();

            if let Some(self_argument) = method.self_argument {
                let type_ = match self_argument {
                    ast::SelfArgument::Value => hir::Type::SelfType,
                    ast::SelfArgument::Ref => hir::Type::Partial(hir::Partial {
                        item: hir::Item::Pointer { mutable: false },
                        params: vec![hir::Type::SelfType],
                    }),
                    ast::SelfArgument::MutRef => hir::Type::Partial(hir::Partial {
                        item: hir::Item::Pointer { mutable: true },
                        params: vec![hir::Type::SelfType],
                    }),
                };

                arguments.push(type_);
            }

            for argument in method.arguments.iter() {
                let type_ = tcx.lower_type(&self.unit, &argument.type_)?;
                arguments.push(type_);
            }

            let output = match method.output {
                Some(ref output) => tcx.lower_type(&self.unit, output)?,
                None => hir::Type::VOID,
            };

            let mut contract = hir::Contract::default();
            self.lower_contract(&mut tcx, &mut contract, &method.contract)?;
            let contract = self.unit.types.contracts.push(contract);

            let generics = generics.into_iter().skip(offset).map(|(_, g)| g).collect();

            let method = hir::TraitMethod {
                name,
                generics,
                arguments,
                output,
                contract,
            };

            self.unit.types[id].methods.push(method);
        }

        let trait_ = &self.unit.types[id];
        self.unit.types.contracts.insert(trait_.contract, contract);

        Ok(())
    }

    fn lower_trait_impl(
        &mut self,
        module: hir::ModuleId,
        ast: &ast::TraitImpl,
    ) -> Result<(), Diagnostic> {
        let mut tcx = TypeContext {
            module,
            generics: &mut Vec::new(),
            allow_new_generics: false,
            trait_id: None,
        };

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
                ast::Decl::Module(ast) => self.lower_module_decl(module, ast)?,
            }
        }

        Ok(())
    }
}
