use std::error::Error;

mod repl;

fn main() -> Result<(), Box<dyn Error>> {
    let mut repl = repl::Repl::new();
    let stdin = std::io::stdin();

    for line in stdin.lines() {
        let line = line?;

        if let Err(err) = repl.run_line(&line) {
            eprintln!("{:?}", err);
        }
    }

    Ok(())
}
