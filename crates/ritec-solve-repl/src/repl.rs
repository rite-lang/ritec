use std::collections::HashMap;

use ritec_diagnostic::Diagnostic;
use ritec_parse::{ParseError, Token, TokenStream, Tokenizer};
use ritec_solve::{Solver, Uid, Unknown, Variable};
use ritec_span::Span;

#[derive(Debug)]
pub enum Error {
    Diagnostic(Diagnostic),
    ParseError(ParseError),
}

impl From<Diagnostic> for Error {
    fn from(diagnostic: Diagnostic) -> Error {
        Error::Diagnostic(diagnostic)
    }
}

impl From<ParseError> for Error {
    fn from(parse_error: ParseError) -> Error {
        Error::ParseError(parse_error)
    }
}

pub struct Repl {
    tokenizer: Tokenizer,
    unknowns: HashMap<String, Unknown>,
    solver: Solver,
}

impl Repl {
    pub fn new() -> Repl {
        Repl {
            tokenizer: Tokenizer::new(),
            unknowns: HashMap::new(),
            solver: Solver::new(),
        }
    }

    fn unknown(&mut self, name: String, span: Span) -> Unknown {
        if let Some(unknown) = self.unknowns.get(&name) {
            return unknown.clone();
        }

        let unknown = Unknown {
            uid: Uid::new(),
            span,
        };

        self.unknowns.insert(name.to_string(), unknown.clone());

        unknown
    }

    fn parse_variable(&mut self, stream: &mut TokenStream) -> Result<Variable, Error> {
        let (token, span) = stream.peek();

        match token {
            Token::Ident(ident) => {
                stream.consume();
                Ok(Variable::Unknown(self.unknown(ident, span)))
            }
            _ => {
                let diagnostic = Diagnostic::new("expected variable").with_span(span);
                Err(Error::from(diagnostic))
            }
        }
    }

    fn parse_statement(&mut self, stream: &mut TokenStream) -> Result<(), Error> {
        let (token, span) = stream.peek();

        match token {
            Token::Ident(_) => {
                let lhs = self.parse_variable(stream)?;
                stream.expect(Token::Eq)?;
                let rhs = self.parse_variable(stream)?;

                println!("{} = {}", lhs, rhs);

                Ok(())
            }
            Token::Trait => todo!(),
            _ => {
                let diagnostic = Diagnostic::new("expected statement").with_span(span);
                Err(Error::from(diagnostic))
            }
        }
    }

    pub fn run_line(&mut self, line: &str) -> Result<(), Error> {
        self.tokenizer.tokenize_line(line)?;

        let mut stream = self.tokenizer.take_stream();

        self.parse_statement(&mut stream)
    }
}
