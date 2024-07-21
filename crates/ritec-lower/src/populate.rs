use ritec_ast as ast;
use ritec_ast::Parser;
use ritec_diagnostic::Diagnostic;
use ritec_hir as hir;

use crate::Lowerer;

impl Lowerer {
    fn populate_enum(
        &mut self,
        module: &mut hir::Module,
        ast: &ast::Enum,
    ) -> Result<(), Diagnostic> {
        let name = ast.name.clone();
        let id = self.unit.types.enums.alloc();

        module.enums.insert(name, id);

        Ok(())
    }

    fn populate_struct(
        &mut self,
        module: &mut hir::Module,
        ast: &ast::Struct,
    ) -> Result<(), Diagnostic> {
        let name = ast.name.clone();
        let id = self.unit.types.structs.alloc();

        module.structs.insert(name, id);

        Ok(())
    }

    fn populate_function(
        &mut self,
        module: &mut hir::Module,
        ast: &ast::Function,
    ) -> Result<(), Diagnostic> {
        let name = ast.name.clone();
        let id = self.unit.bodies.alloc();

        module.funcs.insert(name, id);

        Ok(())
    }

    fn populate_trait(
        &mut self,
        module: &mut hir::Module,
        ast: &ast::Trait,
    ) -> Result<(), Diagnostic> {
        let name = ast.name.clone();

        let mut generics = Vec::new();

        for _ in ast.generics.iter() {
            generics.push(hir::Generic::new());
        }

        let mut assocs = Vec::new();

        for ty in ast.types.iter() {
            assocs.push(hir::Assoc {
                name: ty.name.clone(),
            });
        }

        let contract = self.unit.types.contracts.alloc();

        let id = self.unit.types.traits.push(hir::Trait {
            self_generic: hir::Generic::new(),
            name: Some(ast.name.clone()),
            generics,
            contract,
            assocs,
            methods: Vec::new(),
        });

        module.traits.insert(name, id);

        Ok(())
    }

    fn populate_sub_module(
        &mut self,
        parser_state: &Parser,
        module: &mut hir::Module,
        ast: &ast::ModuleDecl,
    ) -> Result<(), Diagnostic> {
        match parser_state.get_module(ast.path.clone()) {
            Some(ast_module) => {
                let name = ast.name.clone();
                let path = ast.path.clone();

                // If the globally unique name already is register
                // we can just use that instead.
                if let Some(id) = self.unit.module_map.get(&path) {
                    module.modules.insert(name, *id);

                    return Ok(()); // already populated
                }

                // Create new hir module
                let mut sub_module = hir::Module::default();
                self.populate_module(parser_state, &mut sub_module, ast_module)?;
                let id = self.unit.modules.push(sub_module);

                // Add globally unique name to global module map
                self.unit.module_map.insert(path, id);
                // Add local name to module list.
                module.modules.insert(name, id);

                Ok(())
            }
            None => todo!(),
        }
    }

    pub fn populate_use(
        &mut self,
        module: &mut hir::Module,
        ast: &ast::UseDecl,
    ) -> Result<(), Diagnostic> {
        Ok(())
    }

    pub fn populate_module(
        &mut self,
        parser_state: &Parser,
        module: &mut hir::Module,
        ast: &ast::Module,
    ) -> Result<(), Diagnostic> {
        module.use_builtins(&self.unit.builtins);

        for decl in ast.decls.iter() {
            match decl {
                ast::Decl::Enum(ast) => self.populate_enum(module, ast)?,
                ast::Decl::Struct(ast) => self.populate_struct(module, ast)?,
                ast::Decl::Function(ast) => self.populate_function(module, ast)?,
                ast::Decl::Trait(ast) => self.populate_trait(module, ast)?,
                ast::Decl::TraitImpl(_) | ast::Decl::Impl(_) => {
                    // there is nothing to do here
                }
                ast::Decl::Module(ast) => self.populate_sub_module(parser_state, module, ast)?,
                ast::Decl::Use(ast) => self.populate_use(module, ast)?,
            }
        }

        Ok(())
    }
}
