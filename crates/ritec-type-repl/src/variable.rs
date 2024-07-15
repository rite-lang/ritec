use ritec_diagnostic::{Diagnostic, Span};
use ritec_hir::{ContractId, Generic, Item, Partial, Projected, Projection, Type, Uid, Unknown};
use ritec_parse::{Delim, Token, TokenStream};

use crate::repl::{Error, Repl};

impl Repl {
    fn unknown(&mut self, name: String, span: Span, where_: ContractId) -> Unknown {
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
        where_: ContractId,
    ) -> Result<Type, Error> {
        match ident.as_str() {
            "tuple" => {
                stream.consume();
                self.parse_partial_variable(stream, Item::Tuple, where_)
            }
            "slice" => {
                stream.consume();
                self.parse_partial_variable(stream, Item::Slice, where_)
            }
            _ => {
                stream.consume();
                let unknown = self.unknown(ident.to_string(), span, where_);
                Ok(Type::Unknown(unknown))
            }
        }
    }

    fn parse_partial_variable(
        &mut self,
        stream: &mut TokenStream,
        item: Item,
        where_: ContractId,
    ) -> Result<Type, Error> {
        let mut params = Vec::new();

        if stream.take(Token::Bracket(Delim::Open)) {
            loop {
                if stream.take(Token::Bracket(Delim::Close)) {
                    break;
                }

                params.push(self.parse_variable(stream, where_)?);
                stream.take(Token::Comma);
            }
        }

        let partial = Partial { item, params };
        Ok(Type::Partial(partial))
    }

    fn parse_pointer_variable(
        &mut self,
        stream: &mut TokenStream,
        where_: ContractId,
    ) -> Result<Type, Error> {
        stream.expect(Token::Star)?;

        let mutable = stream.take(Token::Mut);

        let item = Item::Pointer { mutable };

        self.parse_partial_variable(stream, item, where_)
    }

    fn parse_associated_variable(
        &mut self,
        stream: &mut TokenStream,
        where_: ContractId,
    ) -> Result<Type, Error> {
        stream.expect(Token::Lt)?;

        let base = self.parse_variable(stream, where_)?;

        stream.expect(Token::As)?;

        let trait_ = stream.expect_ident()?;
        let trait_ = match self.traits.get(&trait_) {
            Some(trait_) => *trait_,
            None => {
                let message = format!("unknown trait {}", trait_);
                let diagnostic = Diagnostic::new(message);
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

        stream.expect(Token::Gt)?;
        stream.expect(Token::ColonColon)?;

        let _associated = stream.expect_ident()?;

        let projected = Projected {
            contract: where_,
            base: Box::new(base),
            projection: Projection::Associated {
                trait_id: trait_,
                generics,
                index: 0,
            },
        };

        Ok(Type::Projected(projected))
    }

    fn parse_forall_variable(
        &mut self,
        stream: &mut TokenStream,
        where_id: ContractId,
    ) -> Result<Type, Error> {
        stream.expect(Token::Quote)?;

        let name = stream.expect_ident()?;

        let (foralls, create) = match self.foralls.get_mut(&where_id) {
            Some(foralls) => (foralls, false),
            None => (&mut self.global_foralls, true),
        };

        match foralls.get(&name) {
            Some(forall) => Ok(Type::Generic(*forall)),
            None => {
                if create {
                    let forall = Generic::new();
                    foralls.insert(name.clone(), forall);
                    Ok(Type::Generic(forall))
                } else {
                    let message = format!("unknown forall {}", name);
                    let diagnostic = Diagnostic::new(message);
                    Err(Error::from(diagnostic))
                }
            }
        }
    }

    pub(crate) fn parse_variable(
        &mut self,
        stream: &mut TokenStream,
        where_id: ContractId,
    ) -> Result<Type, Error> {
        let (token, span) = stream.peek();

        let variable = match token {
            Token::Ident(ident) => self.parse_ident_variable(stream, ident, span, where_id)?,
            Token::Void => {
                stream.consume();
                self.parse_partial_variable(stream, Item::Void, where_id)?
            }
            Token::Bool => {
                stream.consume();
                self.parse_partial_variable(stream, Item::Bool, where_id)?
            }
            Token::Fn => {
                stream.consume();
                self.parse_partial_variable(stream, Item::Function, where_id)?
            }
            Token::Star => self.parse_pointer_variable(stream, where_id)?,
            Token::Lt => self.parse_associated_variable(stream, where_id)?,
            Token::Quote => self.parse_forall_variable(stream, where_id)?,
            _ => {
                let diagnostic = Diagnostic::new("expected variable").with_span(span);
                return Err(Error::from(diagnostic));
            }
        };

        Ok(variable)
    }
}
