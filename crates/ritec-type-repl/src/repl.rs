use std::{collections::HashMap, fs, path::Path};

use ritec_diagnostic::{Diagnostic, Span};
use ritec_hir::{
    Assoc, Contract, ContractId, Generic, Trait, TraitBound, TraitId, TraitImpl, Types, Unknown,
};
use ritec_parse::{Token, TokenStream, Tokenizer};
use ritec_source::SourceId;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0:?}")]
    Diagnostic(Diagnostic),

    #[error("{0}")]
    Io(#[from] std::io::Error),
}

impl From<Diagnostic> for Error {
    fn from(diagnostic: Diagnostic) -> Error {
        Error::Diagnostic(diagnostic)
    }
}

pub(crate) struct TraitDef {
    pub(crate) name: String,
    pub(crate) trait_: TraitId,
    pub(crate) types: Vec<String>,
}

pub(crate) struct FnDef {
    pub(crate) where_: ContractId,
}

pub(crate) enum State {
    Trait(TraitDef),
    TraitImpl(TraitImpl),
    Fn(FnDef),
}

pub(crate) type Foralls = HashMap<String, Generic>;

pub struct Repl {
    pub(crate) tokenizer: Tokenizer,
    pub(crate) unknowns: HashMap<String, Unknown>,
    pub(crate) global_foralls: Foralls,
    pub(crate) foralls: HashMap<ContractId, Foralls>,
    pub(crate) traits: HashMap<String, TraitId>,
    pub(crate) types: Types,
    pub(crate) state: Option<State>,
    pub(crate) expects_indent: bool,
}

impl Repl {
    pub fn new() -> Repl {
        Repl {
            tokenizer: Tokenizer::new(SourceId::default()),
            unknowns: HashMap::new(),
            global_foralls: Foralls::new(),
            foralls: HashMap::new(),
            traits: HashMap::new(),
            types: Types::new(),
            state: None,
            expects_indent: false,
        }
    }

    fn print_command(&mut self, stream: &mut TokenStream) -> Result<(), Error> {
        stream.consume();
        let where_ = self.types.contracts.push(Contract::new());
        let variable = self.parse_variable(stream, where_)?;
        println!("{}", self.types.substitute(&variable));

        Ok(())
    }

    fn complete_command(&mut self, stream: &mut TokenStream) -> Result<(), Error> {
        stream.consume();
        let where_ = self.types.contracts.push(Contract::new());
        let variable = self.parse_variable(stream, where_)?;
        let complete = self.types.know(&variable, &Default::default())?;
        println!("{}", complete);

        Ok(())
    }

    fn print_trait(&self, name: &str, trait_: TraitId) -> Result<(), Error> {
        println!("trait {}", name);

        let trait_ = &self.types[trait_];
        let contract = &self.types[trait_.contract];

        if !contract.bounds.is_empty() {
            println!("where");

            for bound in contract.bounds.iter() {
                print!("| {}", bound);
            }
        }

        Ok(())
    }

    fn query_command(&mut self, stream: &mut TokenStream) -> Result<(), Error> {
        stream.consume();
        let ident = stream.expect_ident()?;

        match ident.as_str() {
            "traits" => {
                for (name, &trait_) in &self.traits {
                    self.print_trait(name, trait_)?;
                }

                Ok(())
            }
            _ => {
                let diagnostic = Diagnostic::new("expected query").with_span(stream.peek().1);
                Err(Error::from(diagnostic))
            }
        }
    }

    fn file_command(&mut self, stream: &mut TokenStream) -> Result<(), Error> {
        stream.consume();

        let path = stream.expect_string()?;
        self.run_file(Path::new(&path))?;

        Ok(())
    }

    fn solve_command(&mut self, stream: &mut TokenStream) -> Result<(), Error> {
        stream.consume();
        self.types.solve()?;

        Ok(())
    }

    fn impls_command(&mut self, stream: &mut TokenStream) -> Result<(), Error> {
        stream.consume();

        let where_id = self.types.contracts.push(Contract::new());

        let mut foralls = Foralls::new();

        if stream.take(Token::Lt) {
            loop {
                stream.expect(Token::Quote)?;
                let ident = stream.expect_ident()?;

                foralls.insert(ident, Generic::new());

                if !stream.take(Token::Comma) {
                    break;
                }
            }

            stream.expect(Token::Gt)?;
        }

        self.foralls.insert(where_id, foralls);

        let trait_ = stream.expect_ident()?;

        let trait_ = match self.traits.get(&trait_) {
            Some(trait_) => *trait_,
            None => {
                let message = format!("unknown trait {}", trait_);
                let diagnostic = Diagnostic::new(message).with_span(stream.peek().1);
                return Err(Error::from(diagnostic));
            }
        };

        let mut generics = Vec::new();

        if stream.take(Token::Lt) {
            loop {
                let variable = self.parse_variable(stream, where_id)?;
                generics.push(variable);

                if !stream.take(Token::Comma) {
                    break;
                }
            }

            stream.expect(Token::Gt)?;
        }

        if self.types[trait_].generics.len() != generics.len() {
            let message = "generic count mismatch";
            let diagnostic = Diagnostic::new(message).with_span(stream.peek().1);
            return Err(Error::from(diagnostic));
        }

        stream.expect(Token::For)?;

        let for_ = self.parse_variable(stream, where_id)?;

        //let impls = self.solver.world.implements(&for_, trait_, &generics, &[]);

        //println!("{}", impls);

        Ok(())
    }

    fn trait_statement(&mut self, stream: &mut TokenStream) -> Result<(), Error> {
        let Some(State::Trait(ref mut state)) = self.state else {
            unreachable!();
        };

        let (token, span) = stream.peek();

        match token {
            Token::Ident(ident) if ident == "type" => {
                stream.consume();
                let name = stream.expect_ident()?;

                self.types[state.trait_]
                    .assocs
                    .push(Assoc { name: name.clone() });
                state.types.push(name);

                Ok(())
            }
            _ => {
                let diagnostic = Diagnostic::new("expected type").with_span(span);
                Err(Error::from(diagnostic))
            }
        }
    }

    fn trait_impl_statement(&mut self, stream: &mut TokenStream) -> Result<(), Error> {
        let where_ = match self.state {
            Some(State::TraitImpl(ref state)) => state.contract,
            _ => unreachable!(),
        };

        let (token, span) = stream.peek();

        match token {
            Token::Ident(ident) if ident == "type" => {
                stream.consume();
                let _name = stream.expect_ident()?;

                stream.expect(Token::Eq)?;

                let variable = self.parse_variable(stream, where_)?;

                if let Some(State::TraitImpl(ref mut state)) = self.state {
                    state.types.push(variable);
                }

                Ok(())
            }
            _ => {
                let diagnostic = Diagnostic::new("expected type").with_span(span);
                Err(Error::from(diagnostic))
            }
        }
    }

    fn unify_statement(
        &mut self,
        stream: &mut TokenStream,
        contract: ContractId,
    ) -> Result<(), Error> {
        let lhs = self.parse_variable(stream, contract)?;
        stream.expect(Token::Eq)?;
        let rhs = self.parse_variable(stream, contract)?;

        self.types.unify(lhs, rhs);

        Ok(())
    }

    fn fn_statement(&mut self, stream: &mut TokenStream) -> Result<(), Error> {
        let Some(State::Fn(ref state)) = self.state else {
            unreachable!();
        };

        self.unify_statement(stream, state.where_)
    }

    fn statement(&mut self, stream: &mut TokenStream) -> Result<(), Error> {
        match self.state {
            Some(State::Trait(_)) => self.trait_statement(stream),
            Some(State::TraitImpl(_)) => self.trait_impl_statement(stream),
            Some(State::Fn(_)) => self.fn_statement(stream),
            None => {
                let diagnostic = Diagnostic::new("unexpected indent");
                Err(Error::from(diagnostic))
            }
        }
    }

    fn trait_command(&mut self, stream: &mut TokenStream) -> Result<(), Error> {
        stream.consume();

        let name = stream.expect_ident()?;

        let where_id = self.types.contracts.push(Contract::new());

        let mut foralls = Foralls::new();
        let mut generics = Vec::new();

        if stream.take(Token::Lt) {
            loop {
                stream.expect(Token::Quote)?;
                let ident = stream.expect_ident()?;

                if foralls.contains_key(&ident) {
                    let message = format!("duplicate forall {}", ident);
                    let diagnostic = Diagnostic::new(message).with_span(stream.peek().1);
                    return Err(Error::from(diagnostic));
                }

                let forall = Generic::new();
                generics.push(forall);

                foralls.insert(ident, forall);

                if !stream.take(Token::Comma) {
                    break;
                }
            }

            stream.expect(Token::Gt)?;
        }

        let trait_ = self.types.traits.push(Trait {
            self_generic: Generic::new(),
            name: Some(name.clone()),
            generics,
            contract: where_id,
            assocs: Vec::new(),
            methods: Vec::new(),
        });

        self.foralls.insert(where_id, foralls);

        self.state = Some(State::Trait(TraitDef {
            name,
            trait_,
            types: Vec::new(),
        }));

        self.expects_indent = true;

        Ok(())
    }

    fn trait_impl_command(&mut self, stream: &mut TokenStream) -> Result<(), Error> {
        stream.consume();

        let where_id = self.types.contracts.push(Contract::new());
        let mut foralls = Foralls::new();

        if stream.take(Token::Lt) {
            loop {
                stream.expect(Token::Quote)?;
                let ident = stream.expect_ident()?;

                foralls.insert(ident, Generic::new());

                if !stream.take(Token::Comma) {
                    break;
                }
            }

            stream.expect(Token::Gt)?;
        }

        self.foralls.insert(where_id, foralls);

        let trait_ = stream.expect_ident()?;
        let trait_ = match self.traits.get(&trait_) {
            Some(trait_) => *trait_,
            None => {
                let message = format!("unknown trait {}", trait_);
                let diagnostic = Diagnostic::new(message).with_span(stream.peek().1);
                return Err(Error::from(diagnostic));
            }
        };

        let mut generics = Vec::new();

        if stream.take(Token::Lt) {
            loop {
                let variable = self.parse_variable(stream, where_id)?;
                generics.push(variable);

                if !stream.take(Token::Comma) {
                    break;
                }
            }

            stream.expect(Token::Gt)?;
        }

        if self.types[trait_].generics.len() != generics.len() {
            let message = "generic count mismatch";
            let diagnostic = Diagnostic::new(message).with_span(stream.peek().1);
            return Err(Error::from(diagnostic));
        }

        stream.expect(Token::For)?;

        let for_ = self.parse_variable(stream, where_id)?;

        self.state = Some(State::TraitImpl(TraitImpl {
            trait_id: trait_,
            generics,
            contract: where_id,
            implementor: for_,
            types: Vec::new(),
            methods: Vec::new(),
        }));

        self.expects_indent = true;

        Ok(())
    }

    fn fn_command(&mut self, stream: &mut TokenStream) -> Result<(), Error> {
        stream.consume();

        let where_ = self.types.contracts.push(Contract::new());

        self.state = Some(State::Fn(FnDef { where_ }));

        self.expects_indent = true;

        Ok(())
    }

    fn end_state(&mut self) {
        match self.state.take() {
            Some(State::Trait(trait_)) => {
                self.traits.insert(trait_.name, trait_.trait_);
            }
            Some(State::TraitImpl(trait_impl)) => {
                self.types.trait_impls.push(trait_impl);
            }
            Some(State::Fn(_)) => {}
            None => {}
        }
    }

    fn ident_command(
        &mut self,
        stream: &mut TokenStream,
        ident: &str,
        span: Span,
    ) -> Result<(), Error> {
        match ident {
            "print" => self.print_command(stream),
            "complete" => self.complete_command(stream),
            "query" => self.query_command(stream),
            "file" => self.file_command(stream),
            "solve" => self.solve_command(stream),
            "impls" => self.impls_command(stream),
            _ => {
                let diagnostic = Diagnostic::new("expected command").with_span(span);
                Err(Error::from(diagnostic))
            }
        }
    }

    fn parse_where_bound(&mut self, stream: &mut TokenStream) -> Result<(), Error> {
        let where_ = match self.state {
            Some(State::TraitImpl(ref state)) => state.contract,
            Some(State::Fn(ref state)) => state.where_,
            _ => {
                let diagnostic = Diagnostic::new("unexpected where").with_span(stream.peek().1);
                return Err(Error::from(diagnostic));
            }
        };

        let base = self.parse_variable(stream, where_)?;

        stream.expect(Token::Colon)?;

        loop {
            let trait_ = stream.expect_ident()?;
            let trait_ = match self.traits.get(&trait_) {
                Some(trait_) => *trait_,
                None => {
                    let message = format!("unknown trait {}", trait_);
                    let diagnostic = Diagnostic::new(message).with_span(stream.peek().1);
                    return Err(Error::from(diagnostic));
                }
            };

            let mut generics = Vec::new();

            if stream.take(Token::Lt) {
                loop {
                    let generic = self.parse_variable(stream, where_)?;
                    generics.push(generic);

                    if !stream.take(Token::Comma) {
                        break;
                    }
                }

                stream.expect(Token::Gt)?;
            }

            let bound = TraitBound {
                base: base.clone(),
                trait_id: trait_,
                generics,
                types: Vec::new(),
            };

            self.types[where_].bounds.push(bound);

            if !stream.take(Token::Plus) {
                break;
            }
        }

        Ok(())
    }

    fn parse_command(&mut self, stream: &mut TokenStream) -> Result<(), Error> {
        if self.expects_indent {
            if stream.take(Token::Indent) {
                self.expects_indent = false;
            } else if stream.take(Token::Or) {
                return self.parse_where_bound(stream);
            } else {
                self.end_state();
            }
        }

        if self.state.is_some() {
            if stream.take(Token::Dedent) {
                self.end_state();
            } else {
                return self.statement(stream);
            };
        }

        let (token, span) = stream.peek();

        match token {
            Token::Ident(ident) => self.ident_command(stream, &ident, span),
            Token::Fn => self.fn_command(stream),
            Token::Trait => self.trait_command(stream),
            Token::Impl => self.trait_impl_command(stream),
            Token::Eof => Ok(()),
            _ => {
                let message = format!("expected statement, found {}", token);
                let diagnostic = Diagnostic::new(message).with_span(span);
                Err(Error::from(diagnostic))
            }
        }
    }

    pub fn run_line(&mut self, line: &str) -> Result<(), Error> {
        if line.trim().is_empty() {
            return Ok(());
        }

        self.tokenizer.tokenize_line(line)?;

        let mut stream = self.tokenizer.take_stream();

        self.parse_command(&mut stream)
    }

    pub fn run_file(&mut self, path: &Path) -> Result<(), Error> {
        let file = fs::read_to_string(path)?;

        let mut stream = self.tokenizer.tokenize(&file)?;

        while !stream.is_empty() {
            self.parse_command(&mut stream)?;

            if stream.is_empty() {
                break;
            }

            stream.expect(Token::Newline)?;
        }

        Ok(())
    }
}
