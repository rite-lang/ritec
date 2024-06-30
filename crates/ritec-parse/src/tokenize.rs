use crate::{token::Token, TokenStream};
use ritec_span::Span;
use std::cmp::Ordering;

struct Tokenizer {
    lo: usize,
    tokens: Vec<(Token, Span)>,
    indents: Vec<usize>,
}

impl Tokenizer {
    fn new() -> Tokenizer {
        Tokenizer {
            lo: 0,
            tokens: Vec::new(),
            indents: vec![0],
        }
    }

    fn skip_whitespace(&mut self, line: &mut &str) -> () {
        let mut len = 0;

        for c in line.chars() {
            if !c.is_whitespace() {
                break;
            }

            len += c.len_utf8();
        }

        *line = &line[len..];

        self.lo += len;
    }

    fn is_ident_start(c: char) -> bool {
        c.is_alphabetic() || c == '_'
    }

    fn is_ident_continue(c: char) -> bool {
        c.is_alphanumeric() || c == '_'
    }

    fn parse_ident(&self, line: &mut &str) -> (String, usize) {
        let mut len = 0;

        for c in line.chars() {
            if !Self::is_ident_continue(c) {
                break;
            }

            len += c.len_utf8();
        }

        let ident = line[..len].to_string();

        *line = &line[len..];

        (ident, len)
    }

    fn parse_indent(&self, line: &mut &str) -> Result<(usize, usize), ()> {
        let mut len = 0;
        let mut indent = 0;

        for c in line.chars() {
            match c {
                ' ' => indent += 1,
                '\t' => indent += 4,
                _ => break,
            }

            len += c.len_utf8();
        }

        *line = &line[len..];

        Ok((indent, len))
    }

    fn parse_token(&self, line: &mut &str) -> Result<(Token, usize), ()> {
        let c = line.chars().next().unwrap();

        //if c.is_ascii_digit() {
        //    return Ok(self.parse_number(line));
        //}

        // parse a two character symbol
        if line.len() >= 2 {
            if let Some(token) = Token::from_symbol(&line[..2]) {
                *line = &line[2..];

                return Ok((token, 2));
            }
        }

        // parse a one character symbol
        if let Some(token) = Token::from_symbol(&line[..1]) {
            *line = &line[1..];

            return Ok((token, 1));
        }

        // parse an identifier
        if Self::is_ident_start(c) {
            let (ident, len) = self.parse_ident(line);

            return match Token::from_keyword(&ident) {
                Some(keyword) => Ok((keyword, len)),
                None => Ok((Token::Ident(ident), len)),
            };
        }

        let span = Span::new(self.lo, self.lo + c.len_utf8());

        // Err(ParseError::UnexpectedCharacter { c, span })
        Err(())
    }

    fn tokenize_line(&mut self, line: &mut &str) -> Result<(), ()> {
        if line.trim().is_empty() {
            // Empty lines account for one newline character in the original source.
            // But it does not produce a token.
            self.lo += line.len() + 1;
            return Ok(());
        }

        let (line_indent, len) = self.parse_indent(line)?;

        // Update indent stack.
        match line_indent.cmp(self.indents.last().unwrap()) {
            Ordering::Greater => {
                self.indents.push(line_indent);

                let span = Span::new(self.lo, self.lo + len);
                self.tokens.push((Token::Indent, span));
            }
            Ordering::Less => {
                while let Some(&last) = self.indents.last() {
                    if last <= line_indent {
                        break;
                    }

                    self.indents.pop();

                    let span = Span::new(self.lo, self.lo + len);

                    self.tokens.push((Token::Dedent, span));
                }
            }
            Ordering::Equal => {}
        }

        self.lo += len;

        // parse the tokens on the current line
        loop {
            self.skip_whitespace(line);

            if line.is_empty() {
                break;
            }

            let (token, len) = self.parse_token(line)?;

            let span = Span::new(self.lo, self.lo + len);
            self.tokens.push((token, span));

            self.lo += len;
        }

        // add a newline token at the end of the line
        let span = Span::new(self.lo, self.lo + 1);
        self.tokens.push((Token::Newline, span));

        // Add the newline character to the line offset.
        self.lo += 1;

        Ok(())
    }

    fn tokenize(&mut self, source: &str) -> Result<(), ()> {
        for mut line in source.lines() {
            self.tokenize_line(&mut line)?;
        }

        let span = Span::new(self.lo, self.lo);
        self.tokens.push((Token::Newline, span));

        // at the end of the file,
        // dedent all the way back to the first indent
        for _ in self.indents.iter().skip(1) {
            self.tokens.push((Token::Dedent, span));
        }

        Ok(())
    }

    fn finish(self) -> Vec<(Token, Span)> {
        self.tokens
    }
}

impl TokenStream {
    pub fn from_source(source: &str) -> Result<TokenStream, ()> {
        let mut tokenizer = Tokenizer::new();
        tokenizer.tokenize(source)?;

        let tokens = tokenizer.finish();

        let span = Span::new(0, source.len());
        Ok(TokenStream::new(tokens, span))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let source = r#"
            let x = 1;
            let y = 2;
        "#;

        let mut tokenizer = Tokenizer::new();
        tokenizer.tokenize(source).unwrap();

        let tokens = tokenizer.tokens;

        println!("{:?}", tokens);
    }
}
