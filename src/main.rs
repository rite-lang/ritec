#![allow(dead_code)]

mod ast;
mod build;
mod hir;
mod infer;
mod interner;
mod interpret;
mod lex;
mod lower;
mod number;
mod parse;
mod rir;
mod span;
mod specialize;
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

    let (main, rir) = compiler.compile(main)?;

    let interpreter = Interpreter::new(&rir);
    let output = interpreter.interpret(main);

    println!("{}", output);

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
        let index = self.unit.push_module(hir::Module::new(name));

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

                let subpath = entry.path();
                let subname = subpath.file_stem().unwrap();
                let subpath = self.interner.intern(subpath.to_str().unwrap());
                let subname = self.interner.intern(subname.to_str().unwrap());

                let mut tokens = lex::lex(subpath, source)?;
                let ast = parse::parse(&mut tokens)?;

                let mut submodule = hir::Module::new(subname);
                submodule.modules.insert(name, index);

                let submodule = self.unit.push_module(submodule);
                self.modules.push((submodule, ast));

                self.unit.modules[index].modules.insert(subname, submodule);
            }
        }

        let module = &self.unit.modules[index];
        let modules = module.modules.clone();

        for &module in modules.values() {
            self.unit.modules[module].modules.extend(modules.clone());
        }

        Ok(index)
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

    fn compile(mut self, main: usize) -> miette::Result<(usize, rir::Unit<rir::Specific>)> {
        infer::infer(&mut self.unit)?;

        let rir = build::build(&self.unit)?;
        Ok(specialize::specialize(rir, main))
    }
}
