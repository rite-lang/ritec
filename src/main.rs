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
enum Subcommand {
    /// Parse a file and print the AST.
    Ast(AstOptions),
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
        Subcommand::Type(type_repl) => type_repl.run(),
    }
}
