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

use std::fs;

use interner::Interner;
use interpret::Interpreter;

fn main() -> miette::Result<()> {
    miette::set_panic_hook();

    let file = "test.ri";
    let mut interner = Interner::new();
    let source = interner.intern(fs::read_to_string(file).unwrap());
    let mut tokens = lex::lex(file, source)?;
    let ast = parse::parse(&mut tokens)?;

    let mut unit = hir::Unit::new();
    let module = unit.push_module(hir::Module::new("test"));
    lower::lower_ast(&mut unit, module, &ast)?;

    infer::infer(&mut unit)?;

    let main = unit.modules[module].funcs["main"];

    let rir = rir::Unit::from_hir(unit)?;
    let mir = build::build(&rir, main)?;

    let interpreter = Interpreter::new(&mir);
    let value = interpreter.run(0);

    println!("{:?}", value);

    Ok(())
}
