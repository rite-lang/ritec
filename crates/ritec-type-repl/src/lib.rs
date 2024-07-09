mod repl;
mod variable;

use std::{io::prelude::*, path::PathBuf};

use clap::Parser;
use eyre::Context;

#[derive(Parser, Debug)]
pub struct TypeReplOptions {
    /// The path to the file to run.
    file: Option<PathBuf>,
}

impl TypeReplOptions {
    pub fn run(self) -> eyre::Result<()> {
        let mut repl = repl::Repl::new();

        if let Some(file) = self.file {
            return repl.run_file(&file).wrap_err("Failed to run file");
        }

        let mut stdin = std::io::stdin().lock();
        let mut stdout = std::io::stdout().lock();

        println!("Ritec Type REPL");

        loop {
            write!(stdout, "> ")?;
            stdout.flush()?;

            let mut input = String::new();
            stdin.read_line(&mut input)?;

            if input.trim().is_empty() {
                continue;
            }

            if let Err(error) = repl.run_line(&input) {
                writeln!(stdout, "{}", error)?;
            }
        }
    }
}
