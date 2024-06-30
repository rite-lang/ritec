use std::{cmp::Ordering, mem};

use ritec_diagnostic::{Diagnostic, Span};

use crate::{token::Token, ParseError, TokenStream};

#[derive(Clone, Debug, Default)]
pub struct Tokenizer {
    lo: usize,
    tokens: Vec<(Token, Span)>,
    indents: Vec<usize>,
}

impl Tokenizer {
    pub fn new() -> Tokenizer {
        Tokenizer {
            lo: 0,
            tokens: Vec::new(),
            indents: vec![0],
        }
    }

    fn skip_whitespace(&mut self, line: &mut &str) {
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

    fn parse_indent(&self, line: &mut &str) -> Result<(usize, usize), ParseError> {
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

    fn parse_string(&self, line: &mut &str) -> Result<(Token, usize), ParseError> {
        let mut len = 0;
        let mut buf = String::new();
        let mut escape = false;
        let mut valid = false;

        // skip the opening quote
        *line = &line[1..];

        for c in line.chars() {
            len += c.len_utf8();

            if escape {
                match c {
                    'n' => buf.push('\n'),
                    'r' => buf.push('\r'),
                    't' => buf.push('\t'),
                    '\\' => buf.push('\\'),
                    '"' => buf.push('"'),
                    _ => buf.push(c),
                }

                escape = false;
            } else {
                match c {
                    '\\' => escape = true,
                    '"' => {
                        valid = true;
                        break;
                    }
                    _ => buf.push(c),
                }
            }
        }

        if !valid {
            let span = Span::new(self.lo, self.lo + len);
            let diagnostic = Diagnostic::new("unterminated string").with_span(span);
            return Err(ParseError::from(diagnostic));
        }

        *line = &line[len..];

        Ok((Token::String(buf), len))
    }

    fn parse_integer(&self, line: &mut &str) -> (Token, usize) {
        let mut len = 0;
        let mut radix = 10;

        match line.get(..2) {
            Some("0x") => {
                radix = 16;
                len += 2;
            }
            Some("0b") => {
                radix = 2;
                len += 2;
            }
            Some("0o") => {
                radix = 8;
                len += 2;
            }
            _ => {}
        }

        let mut value = 0;

        for c in line.chars() {
            match c.to_digit(radix) {
                Some(digit) => {
                    value = value * radix + digit;
                    len += c.len_utf8();
                }
                None => break,
            }
        }

        *line = &line[len..];

        (Token::Integer(value as u64), len)
    }

    fn parse_float(&self, line: &mut &str) -> (Token, usize) {
        let mut len = 0;
        let mut buf = String::new();

        for c in line.chars() {
            match c {
                '0'..='9' | '.' | 'e' | 'E' => {
                    buf.push(c);
                    len += c.len_utf8();
                }
                _ => break,
            }
        }

        let float = buf.parse::<f64>().unwrap();

        *line = &line[len..];

        (Token::Float(float), len)
    }

    fn parse_number(&self, line: &mut &str) -> (Token, usize) {
        // Check if we see a dot or an 'e' character that is followed by another number before something else.F
        let mut has_dot = false;
        let mut has_e = false;
        let mut has_valid_e = false;

        for c in line.chars() {
            match c {
                '0'..='9' => {
                    if has_e {
                        has_valid_e = true;
                    }
                }
                '.' => {
                    if has_dot {
                        break;
                    }

                    has_dot = true;
                }
                'e' | 'E' => {
                    if has_e {
                        break;
                    }

                    has_e = true;
                }
                _ => break,
            }
        }

        if has_dot || (has_e && has_valid_e) {
            self.parse_float(line)
        } else {
            self.parse_integer(line)
        }
    }

    fn parse_token(&self, line: &mut &str) -> Result<(Token, usize), ParseError> {
        let c = line.chars().next().unwrap();

        if c.is_ascii_digit() {
            return Ok(self.parse_number(line));
        }

        // Parse a string
        if c == '"' {
            return self.parse_string(line);
        }

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

        let diagnostic = Diagnostic::new("unexpected character").with_span(span);
        Err(ParseError::from(diagnostic))
    }

    pub fn tokenize_line(&mut self, mut line: &str) -> Result<(), ParseError> {
        if line.trim().is_empty() {
            // Empty lines account for one newline character in the original source.
            // But it does not produce a token.
            self.lo += line.len() + 1;
            return Ok(());
        }

        let (line_indent, len) = self.parse_indent(&mut line)?;

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
            self.skip_whitespace(&mut line);

            if line.is_empty() {
                break;
            }

            let (token, len) = self.parse_token(&mut line)?;

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

    pub fn take_stream(&mut self) -> TokenStream {
        let span = Span::new(0, self.lo);
        TokenStream::new(mem::take(&mut self.tokens), span)
    }

    pub fn tokenize(&mut self, source: &str) -> Result<TokenStream, ParseError> {
        for line in source.lines() {
            self.tokenize_line(line)?;
        }

        let span = Span::new(self.lo, self.lo);

        // at the end of the file,
        // dedent all the way back to the first indent
        for _ in self.indents.iter().skip(1) {
            self.tokens.push((Token::Dedent, span));
        }

        Ok(self.take_stream())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let source = r#"10.10 "WOW SO COOL" "#;
        let mut tokenizer = Tokenizer::new();

        match tokenizer.tokenize(source) {
            Ok(tokens) => println!("{:?}", tokens),
            Err(e) => eprintln!("{:?}", e),
        }
    }
}
