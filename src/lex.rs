use std::cmp::Ordering;

use miette::{LabeledSpan, NamedSource, Severity};

use crate::{
    span::Span,
    token::{Token, TokenStream},
};

struct Lexer {
    file: &'static str,
    source: &'static str,
    lo: usize,
}

impl Lexer {
    fn rest(&self) -> &'static str {
        &self.source[self.lo..]
    }

    fn peek(&self) -> Option<char> {
        self.rest().chars().next()
    }

    fn next(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.lo += c.len_utf8();
        Some(c)
    }

    fn source_code(&self) -> NamedSource<&'static str> {
        NamedSource::new(self.file, self.source)
    }
}

enum Indent {
    Tabs(usize),
    Spaces(usize),
}

pub fn lex(file: &'static str, source: &'static str) -> miette::Result<TokenStream> {
    let mut tokens = Vec::new();
    let mut indents = Vec::new();
    let mut lo = 0;

    for line in source.lines() {
        if line.trim().is_empty() {
            let span = Span {
                lo,
                hi: lo + line.len() + '\n'.len_utf8(),
                file,
                source,
            };

            tokens.push((Token::Newline, span));

            lo += line.len() + '\n'.len_utf8();
            continue;
        }

        let mut lexer = Lexer { file, source, lo };

        let line_indents = lex_indents(&mut lexer, indents.iter())?;

        match line_indents.len().cmp(&indents.len()) {
            Ordering::Less => {
                for _ in 0..indents.len() - line_indents.len() {
                    let span = Span {
                        lo,
                        hi: lexer.lo,
                        file,
                        source,
                    };
                    tokens.push((Token::Dedent, span));
                }
            }
            Ordering::Greater => {
                let span = Span {
                    lo,
                    hi: lexer.lo,
                    file,
                    source,
                };
                tokens.push((Token::Indent, span));
            }
            Ordering::Equal => {}
        }

        indents = line_indents;

        loop {
            let remaining = source[lexer.lo..lo + line.len()].trim_start();

            if remaining.is_empty() {
                break;
            }

            skip_whitespace(&mut lexer);
            tokens.push(lex_token(&mut lexer)?);
        }

        let span = Span {
            lo,
            hi: lexer.lo,
            file,
            source,
        };

        tokens.push((Token::Newline, span));

        lo += line.len() + '\n'.len_utf8();
    }

    for _ in 0..indents.len() {
        let span = Span {
            lo,
            hi: lo,
            file,
            source,
        };
        tokens.push((Token::Dedent, span));
    }

    Ok(TokenStream::from_raw_parts(tokens.into(), file, source))
}

fn lex_indents<'a>(
    lexer: &mut Lexer,
    current: impl Iterator<Item = &'a Indent>,
) -> miette::Result<Vec<Indent>> {
    let mut indents = Vec::new();

    for indent in current {
        if !matches!(lexer.peek(), Some('\t') | Some(' ')) {
            break;
        }

        match *indent {
            Indent::Tabs(n) => {
                expect_indent(lexer, '\t', n)?;
                indents.push(Indent::Tabs(n));
            }
            Indent::Spaces(n) => {
                expect_indent(lexer, ' ', n)?;
                indents.push(Indent::Spaces(n));
            }
        }
    }

    if let Some(indent) = lex_indent(lexer)? {
        indents.push(indent);
    }

    Ok(indents)
}

fn expect_indent(lexer: &mut Lexer, c: char, n: usize) -> miette::Result<()> {
    for _ in 0..n {
        match lexer.peek() == Some(c) {
            true => _ = lexer.next(),
            false => {
                return Err(miette::miette!(
                    severity = Severity::Error,
                    code = "expected::tab",
                    help = "indentation must be consistent, and cannot mix tabs and spaces",
                    labels = vec![LabeledSpan::at_offset(lexer.lo, "here")],
                    "expected tab"
                )
                .with_source_code(lexer.source_code()))
            }
        }
    }

    Ok(())
}

fn lex_indent(lexer: &mut Lexer) -> miette::Result<Option<Indent>> {
    let Some(first) = lexer.peek() else {
        return Ok(None);
    };

    if !matches!(first, '\t' | ' ') {
        return Ok(None);
    }

    let mut n = 0;

    while let Some(c) = lexer.peek() {
        if matches!(c, '\t' | ' ') && c != first {
            return Err(miette::miette!(
                severity = Severity::Error,
                code = "expected::tab",
                help = "indentation must be consistent, and cannot mix tabs and spaces",
                labels = vec![LabeledSpan::at_offset(lexer.lo, "here")],
                "expected tab"
            )
            .with_source_code(lexer.source_code()));
        }

        if c != first {
            break;
        }

        _ = lexer.next();
        n += 1;
    }

    Ok(Some(match first {
        '\t' => Indent::Tabs(n),
        ' ' => Indent::Spaces(n),
        _ => unreachable!(),
    }))
}

fn skip_whitespace(lexer: &mut Lexer) {
    while let Some(c) = lexer.peek() {
        if !c.is_whitespace() {
            break;
        }

        _ = lexer.next();
    }
}

fn lex_token(lexer: &mut Lexer) -> miette::Result<(Token, Span)> {
    let c = lexer.peek().expect("lex_token called on empty input");

    if lexer.rest().len() >= 2 {
        if let Some(token) = Token::symbol_from_str(&lexer.rest()[..2]) {
            let span = Span {
                lo: lexer.lo,
                hi: lexer.lo + 2,
                file: lexer.file,
                source: lexer.source,
            };

            _ = lexer.next();
            _ = lexer.next();

            return Ok((token, span));
        }
    }

    if let Some(token) = Token::symbol_from_str(&lexer.rest()[..1]) {
        let span = Span {
            lo: lexer.lo,
            hi: lexer.lo + 1,
            file: lexer.file,
            source: lexer.source,
        };

        _ = lexer.next();

        return Ok((token, span));
    }

    if c.is_alphabetic() || c == '_' {
        let (token, span) = lex_identifier(lexer)?;

        return match Token::keyword_from_str(span.as_str()) {
            Some(keyword) => Ok((keyword, span)),
            None => Ok((token, span)),
        };
    }

    if c.is_ascii_digit() {
        return lex_integer(lexer);
    }

    Err(miette::miette!(
        severity = Severity::Error,
        code = "unexpected::token",
        help = "honestly buddy, you're on your own",
        labels = vec![LabeledSpan::at_offset(lexer.lo, "here")],
        "unexpected token"
    )
    .with_source_code(lexer.source_code()))
}

fn lex_identifier(lexer: &mut Lexer) -> miette::Result<(Token, Span)> {
    let lo = lexer.lo;

    while let Some(c) = lexer.peek() {
        if c.is_alphanumeric() || c == '_' {
            _ = lexer.next();
        } else {
            break;
        }
    }

    let hi = lexer.lo;

    let span = Span {
        lo,
        hi,
        file: lexer.file,
        source: lexer.source,
    };

    let first = span.as_str().chars().next().unwrap();
    if first.is_lowercase() || first == '_' {
        if span.as_str().contains(char::is_uppercase) {
            return Err(miette::miette! {
                severity = Severity::Error,
                code = "invalid::snake_case",
                help = "snake_case identifiers must be all lowercase",
                labels = vec![LabeledSpan::at(lo..hi, "here")],
                "invalid identifier"
            }
            .with_source_code(span));
        };

        Ok((Token::Snake, span))
    } else {
        if span.as_str().contains('_') {
            return Err(miette::miette! {
                severity = Severity::Error,
                code = "invalid::pascal_case",
                help = "PascalCase identifiers may not contain underscores",
                labels = vec![LabeledSpan::at(lo..hi, "here")],
                "invalid identifier"
            }
            .with_source_code(span));
        };

        Ok((Token::Pascal, span))
    }
}

fn lex_integer(lexer: &mut Lexer) -> miette::Result<(Token, Span)> {
    let lo = lexer.lo;

    while let Some(c) = lexer.peek() {
        if c.is_ascii_digit() {
            _ = lexer.next();
        } else {
            break;
        }
    }

    let hi = lexer.lo;

    let span = Span {
        lo,
        hi,
        file: lexer.file,
        source: lexer.source,
    };

    Ok((Token::Integer, span))
}
