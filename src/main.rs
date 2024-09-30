#![allow(dead_code)]

mod ast;
mod build;
mod decorator;
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

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use clap::Parser;
use hir::{Import, ImportKind, Vis};
use interner::Interner;
use interpret::interpret::Interpreter;
use span::Span;

#[derive(Parser)]
struct Options {
    #[clap(default_value = "ritec")]
    project: PathBuf,
}

fn main() -> miette::Result<()> {
    let options = Options::parse();

    miette::set_panic_hook();

    let mut compiler = Compiler::new();

    let name = options.project.file_stem().unwrap();
    let name = name.to_str().unwrap();
    let name = compiler.interner.intern(name);

    let std = compiler.add_dir("std", Path::new("std"))?;
    let module = compiler.add_dir(name, &options.project)?;

    let std_import = Import {
        vis: Vis::Private,
        kind: ImportKind::Module(std),
        span: Span {
            lo: 0,
            hi: 0,
            file: "",
            source: "",
        },
    };

    let module_import = Import {
        vis: Vis::Private,
        kind: ImportKind::Module(module),
        span: Span {
            lo: 0,
            hi: 0,
            file: "",
            source: "",
        },
    };

    for module in &mut compiler.unit.modules {
        module.imports.insert("std", std_import.clone());
        module.imports.insert(name, module_import.clone());
    }

    compiler.lower()?;

    let module = &compiler.unit.modules[module].imports["main"];
    let ImportKind::Func(main) = module.kind else {
        panic!();
    };

    let (main, rir) = compiler.compile(main)?;

    let mut interpreter = Interpreter::new(&rir);
    interpreter.interpret(main);

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
        self.add_dir_internal(name, index, path, true)?;
        Ok(index)
    }

    fn add_dir_internal(
        &mut self,
        name: &'static str,
        index: usize,
        path: &Path,
        is_root: bool,
    ) -> miette::Result<()> {
        let mut modules = HashMap::new();

        for entry in fs::read_dir(path).map_err(|err| miette::miette!("{}", err.to_string()))? {
            let entry = entry.map_err(|err| miette::miette!("{}", err.to_string()))?;
            let file_type = entry
                .file_type()
                .map_err(|err| miette::miette!("{}", err.to_string()))?;

            let subpath = entry.path();
            let subname = subpath.file_stem().unwrap();
            let subpath = self.interner.intern(subpath.to_str().unwrap());
            let subname = self.interner.intern(subname.to_str().unwrap());

            let submodule = match modules.get(subname) {
                Some(module) => *module,
                None => {
                    if is_root && subname == name {
                        index
                    } else {
                        let mut submodule = hir::Module::new(subname);

                        let import = Import {
                            vis: Vis::Public,
                            kind: ImportKind::Module(index),
                            span: Span {
                                lo: 0,
                                hi: 0,
                                file: "",
                                source: "",
                            },
                        };
                        submodule.imports.insert(name, import);

                        let submodule = self.unit.push_module(submodule);
                        self.unit.modules[index].imports.insert(
                            subname,
                            Import {
                                vis: Vis::Public,
                                kind: ImportKind::Module(submodule),
                                span: Span {
                                    lo: 0,
                                    hi: 0,
                                    file: "",
                                    source: "",
                                },
                            },
                        );

                        modules.insert(subname, submodule);
                        submodule
                    }
                }
            };

            if file_type.is_dir() {
                self.add_dir_internal(subname, submodule, &entry.path(), false)?;
            }

            if file_type.is_file() {
                if !entry.path().extension().map_or(false, |ext| ext == "ri") {
                    continue;
                };

                let source = fs::read_to_string(entry.path())
                    .map_err(|err| miette::miette!("{}", err.to_string()))?;
                let source = self.interner.intern(source);

                let mut tokens = lex::lex(subpath, source)?;
                let ast = parse::parse(&mut tokens)?;

                self.modules.push((submodule, ast));
            }
        }

        for module in modules.values() {
            for (name, submodule) in modules.iter() {
                let import = Import {
                    vis: Vis::Private,
                    kind: ImportKind::Module(*submodule),
                    span: Span {
                        lo: 0,
                        hi: 0,
                        file: "",
                        source: "",
                    },
                };

                self.unit.modules[*module].imports.insert(name, import);
            }
        }

        Ok(())
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
