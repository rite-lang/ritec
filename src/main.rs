#![allow(dead_code)]

mod ast;
mod build;
mod hir;
mod infer;
mod interner;
mod interpret;
mod lex;
mod lower;
mod mir;
mod number;
mod parse;
mod rir;
mod span;
mod token;

use std::{fs, path::Path};

use interner::Interner;
use interpret::Interpreter;

fn main() -> miette::Result<()> {
    miette::set_panic_hook();

    let mut compiler = Compiler::new();

    let std = compiler.add_dir("std", Path::new("std"))?;
    let test = compiler.add_dir("test", Path::new("test"))?;

    for module in compiler.unit.modules[test].modules.clone().into_values() {
        compiler.unit.modules[module].modules.insert("std", std);
    }

    compiler.lower()?;

    let module = compiler.unit.modules[test].modules["test"];
    let main = compiler.unit.modules[module].funcs["main"];

    let mir = compiler.compile(main)?;

    let interpreter = Interpreter::new(&mir);
    let value = interpreter.run(0);

    println!("Output: {}", value);

    Ok(())
}

struct Compiler {
    interner: Interner,
    unit: hir::Unit,
    modules: Vec<(usize, ast::Module)>,
}

impl Compiler {
    fn new() -> Self {
        Self {
            interner: Interner::new(),
            unit: hir::Unit::new(),
            modules: Vec::new(),
        }
    }

    fn add_dir(&mut self, name: &str, path: &Path) -> miette::Result<usize> {
        let name = self.interner.intern(name);
        let mut module = hir::Module::new(name);

        for entry in fs::read_dir(path).map_err(|err| miette::miette!("{}", err.to_string()))? {
            let entry = entry.map_err(|err| miette::miette!("{}", err.to_string()))?;
            let file_type = entry
                .file_type()
                .map_err(|err| miette::miette!("{}", err.to_string()))?;

            if file_type.is_file() {
                if !entry.path().extension().map_or(false, |ext| ext == "ri") {
                    continue;
                };

                let source = fs::read_to_string(entry.path())
                    .map_err(|err| miette::miette!("{}", err.to_string()))?;
                let source = self.interner.intern(source);

                let name = entry.path();
                let name = name.file_stem().unwrap();
                let name = self.interner.intern(name.to_str().unwrap());

                let mut tokens = lex::lex(name, source)?;
                let ast = parse::parse(&mut tokens)?;

                let submodule = self.unit.push_module(hir::Module::new(name));
                self.modules.push((submodule, ast));

                module.modules.insert(name, submodule);
            }
        }

        let modules = module.modules.clone();

        for &module in module.modules.values() {
            self.unit.modules[module].modules = modules.clone();
        }

        Ok(self.unit.push_module(module))
    }

    fn lower(&mut self) -> miette::Result<()> {
        for (module, ast) in self.modules.iter() {
            lower::type_register_ast(&mut self.unit, *module, ast)?;
        }

        for (module, ast) in self.modules.iter() {
            lower::type_resolve_ast(&mut self.unit, *module, ast)?;
        }

        for (module, ast) in self.modules.iter() {
            lower::type_construct_ast(&mut self.unit, *module, ast)?;
        }

        for (module, ast) in self.modules.iter() {
            lower::func_register_ast(&mut self.unit, *module, ast)?;
        }

        for (module, ast) in self.modules.iter() {
            lower::func_construct_ast(&mut self.unit, *module, ast)?;
        }

        Ok(())
    }

    fn compile(mut self, main: usize) -> miette::Result<mir::Mir> {
        infer::infer(&mut self.unit)?;

        let rir = rir::Unit::from_hir(self.unit)?;
        build::build(&rir, main)
    }
}
