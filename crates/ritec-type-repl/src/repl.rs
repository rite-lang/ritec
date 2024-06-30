use std::collections::HashMap;

use ritec_diagnostic::{Diagnostic, Span};
use ritec_parse::{Delim, ParseError, Token, TokenStream, Tokenizer};
use ritec_type::{Item, Partial, Solver, TypeError, Uid, Unknown, Variable};

#[derive(Debug)]
pub enum Error {
    Diagnostic(Diagnostic),
    Parse(ParseError),
    Type(TypeError),
}

impl From<Diagnostic> for Error {
    fn from(diagnostic: Diagnostic) -> Error {
        Error::Diagnostic(diagnostic)
    }
}

impl From<ParseError> for Error {
    fn from(parse_error: ParseError) -> Error {
        Error::Parse(parse_error)
    }
}

impl From<TypeError> for Error {
    fn from(type_error: TypeError) -> Error {
        Error::Type(type_error)
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

    fn parse_ident_variable(
        &mut self,
        stream: &mut TokenStream,
        ident: String,
        span: Span,
    ) -> Result<Variable, Error> {
        match ident.as_str() {
            "void" => {
                stream.consume();
                self.parse_partial_variable(stream, Item::Void)
            }
            "bool" => {
                stream.consume();
                self.parse_partial_variable(stream, Item::Bool)
            }
            "tuple" => {
                stream.consume();
                self.parse_partial_variable(stream, Item::Tuple)
            }
            "slice" => {
                stream.consume();
                self.parse_partial_variable(stream, Item::Slice)
            }
            _ => {
                stream.consume();
                let unknown = self.unknown(ident.to_string(), span);
                Ok(Variable::Unknown(unknown))
            }
        }
    }

    fn parse_partial_variable(
        &mut self,
        stream: &mut TokenStream,
        item: Item,
    ) -> Result<Variable, Error> {
        let mut params = Vec::new();

        if stream.take(Token::Bracket(Delim::Open)) {
            loop {
                if stream.take(Token::Bracket(Delim::Close)) {
                    break;
                }

                params.push(self.parse_variable(stream)?);
                stream.take(Token::Comma);
            }
        }

        let partial = Partial { item, params };
        Ok(Variable::Partial(partial))
    }

    fn parse_pointer_variable(&mut self, stream: &mut TokenStream) -> Result<Variable, Error> {
        stream.expect(Token::Star)?;

        let mutable = stream.take(Token::Mut);

        let item = Item::Pointer { mutable };

        self.parse_partial_variable(stream, item)
    }

    fn parse_variable(&mut self, stream: &mut TokenStream) -> Result<Variable, Error> {
        let (token, span) = stream.peek();

        match token {
            Token::Ident(ident) => self.parse_ident_variable(stream, ident, span),
            Token::Fn => {
                stream.consume();
                self.parse_partial_variable(stream, Item::Function)
            }
            Token::Star => self.parse_pointer_variable(stream),
            _ => {
                let diagnostic = Diagnostic::new("expected variable").with_span(span);
                Err(Error::from(diagnostic))
            }
        }
    }

    fn parse_ident_statement(
        &mut self,
        stream: &mut TokenStream,
        ident: &str,
        _span: Span,
    ) -> Result<(), Error> {
        match ident {
            "print" => {
                stream.consume();
                let variable = self.parse_variable(stream)?;
                println!("{}", self.solver.world().substitute(&variable));

                Ok(())
            }
            _ => self.parse_unify_statement(stream),
        }
    }

    fn parse_unify_statement(&mut self, stream: &mut TokenStream) -> Result<(), Error> {
        let lhs = self.parse_variable(stream)?;
        stream.expect(Token::Eq)?;
        let rhs = self.parse_variable(stream)?;

        self.solver.unify(lhs, rhs);
        self.solver.solve()?;

        Ok(())
    }

    fn parse_statement(&mut self, stream: &mut TokenStream) -> Result<(), Error> {
        let (token, span) = stream.peek();

        match token {
            Token::Ident(ident) => self.parse_ident_statement(stream, &ident, span),
            Token::Star | Token::Fn => self.parse_unify_statement(stream),
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
