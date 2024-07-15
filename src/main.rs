use std::{fs, path::PathBuf};

use clap::Parser;
use ritec_type_repl::TypeReplOptions;

#[derive(Debug, Parser)]
struct AstOptions {
    /// The path to the file to parse.
    path: PathBuf,
}

impl AstOptions {
    pub fn run(&self) -> eyre::Result<()> {
        let source = fs::read_to_string(&self.path)?;
        let mut tokenizer = ritec_parse::Tokenizer::new();
        let mut stream = tokenizer.tokenize(&source).unwrap();

        let ast = ritec_ast::parse_module(&mut stream).unwrap();

        println!("{:#?}", ast);

        Ok(())
    }
}

#[derive(Debug, Parser)]
struct HirOptions {
    /// The path to the file to parse.
    path: PathBuf,
}

impl HirOptions {
    pub fn run(&self) -> eyre::Result<()> {
        let source = fs::read_to_string(&self.path)?;
        let mut tokenizer = ritec_parse::Tokenizer::new();
        let mut stream = tokenizer.tokenize(&source).unwrap();

        let ast = ritec_ast::parse_module(&mut stream).unwrap();

        let mut lowerer = ritec_lower::Lowerer::default();
        let mut module = ritec_hir::Module::default();
        lowerer.populate_module(&mut module, &ast).unwrap();
        let root = lowerer.unit.modules.push(module);
        lowerer.lower_module(root, &ast).unwrap();
        lowerer.unit.types.solve().unwrap();

        println!("{:#?}", lowerer.unit);

        Ok(())
    }
}

#[derive(Debug, Parser)]
struct MirOptions {
    /// The path to the file to parse.
    path: PathBuf,
}

impl MirOptions {
    pub fn run(&self) -> eyre::Result<()> {
        let source = fs::read_to_string(&self.path)?;
        let mut tokenizer = ritec_parse::Tokenizer::new();
        let mut stream = tokenizer.tokenize(&source).unwrap();

        let ast = ritec_ast::parse_module(&mut stream).unwrap();

        let mut lowerer = ritec_lower::Lowerer::default();
        let mut module = ritec_hir::Module::default();
        lowerer.populate_module(&mut module, &ast).unwrap();
        let root = lowerer.unit.modules.push(module);
        lowerer.lower_module(root, &ast).unwrap();
        lowerer.unit.types.solve().unwrap();

        let mir = ritec_build::build(&lowerer.unit).unwrap();

        println!("{:#?}", mir);

        Ok(())
    }
}

#[derive(Debug, Parser)]
struct COptions {
    /// The path to the file to parse.
    path: PathBuf,
}

impl COptions {
    pub fn run(&self) -> eyre::Result<()> {
        let source = fs::read_to_string(&self.path)?;
        let mut tokenizer = ritec_parse::Tokenizer::new();
        let mut stream = tokenizer.tokenize(&source).unwrap();

        let ast = ritec_ast::parse_module(&mut stream).unwrap();

        let mut lowerer = ritec_lower::Lowerer::default();
        let mut module = ritec_hir::Module::default();
        lowerer.populate_module(&mut module, &ast).unwrap();
        let root = lowerer.unit.modules.push(module);
        lowerer.lower_module(root, &ast).unwrap();
        lowerer.unit.types.solve().unwrap();

        let mir = ritec_build::build(&lowerer.unit).unwrap();
        let c = ritec_codegen_c::codegen(&mir);

        println!("{}", c);
        std::fs::write("out.c", c)?;

        Ok(())
    }
}

#[derive(Debug, Parser)]
struct RunOptions {
    /// The path to the file to parse.
    path: PathBuf,
}

impl RunOptions {}

#[derive(Debug, Parser)]
enum Subcommand {
    /// Parse a file and print the AST.
    Ast(AstOptions),

    /// Parse a file and print the HIR.
    Hir(HirOptions),

    /// Parse a file and print the MIR.
    Mir(MirOptions),

    /// Parse a file and print the C code.
    C(COptions),

    /// Run the type REPL.
    Type(TypeReplOptions),
}

#[derive(Debug, Parser)]
struct Options {
    #[clap(subcommand)]
    subcommand: Subcommand,
}

fn main() -> eyre::Result<()> {
    color_eyre::install()?;

    let options = Options::parse();

    match options.subcommand {
        Subcommand::Ast(ast) => ast.run(),
        Subcommand::Hir(hir) => hir.run(),
        Subcommand::Mir(mir) => mir.run(),
        Subcommand::C(c) => c.run(),
        Subcommand::Type(type_repl) => type_repl.run(),
    }
}
