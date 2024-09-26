use std::cmp::Ordering;

use miette::{LabeledSpan, NamedSource, Severity};

use crate::{
    span::Span,
    token::{Token, TokenStream},
};

#[derive(Clone, Debug, PartialEq)]
enum LexerFormatStringState {
    Idle,
    ParseString,
    StartParseExpr,
    ParseExpr,
    Stop,
}

#[derive(Clone, Debug)]
struct LexerState {
    f_string: LexerFormatStringState,
}

impl Default for LexerState {
    fn default() -> Self {
        Self {
            f_string: LexerFormatStringState::Idle,
        }
    }
}

struct Lexer {
    file: &'static str,
    source: &'static str,
    lo: usize,
    state: LexerState,
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

        let mut lexer = Lexer {
            file,
            source,
            lo,
            state: Default::default(),
        };

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

            // Only skip whitespace if we are not in a special context.
            if lexer.state.f_string == LexerFormatStringState::Idle {
                skip_whitespace(&mut lexer);
            }

            tokens.push(lex_token(&mut lexer)?);
        }

        // Cleanup LexerFormatStringState::stop if there are no more characters on the line
        if lexer.state.f_string != LexerFormatStringState::Idle {
            return Err(miette::miette!(
                severity = Severity::Error,
                code = "unexpected::brace",
                help = format!(
                    "Invalid format string current state: {:?}",
                    lexer.state.f_string
                ),
                labels = vec![LabeledSpan::at_offset(lexer.lo, "here")],
                "unexpected brace"
            )
            .with_source_code(lexer.source_code()));
        }

        // Only add a newline if the line doesn't already end with one
        if tokens.last().map(|(t, _)| *t) != Some(Token::Newline) {
            let span = Span {
                lo: lexer.lo,
                hi: lexer.lo,
                file,
                source,
            };
            tokens.push((Token::Newline, span));
        }

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

    match lexer.state.f_string {
        LexerFormatStringState::ParseString => return lex_format_string(lexer),
        LexerFormatStringState::StartParseExpr => return lex_format_string_expr_start(lexer),
        LexerFormatStringState::ParseExpr => return lex_format_string_expr(lexer),
        LexerFormatStringState::Stop => return lex_format_string_stop(lexer),
        _ => {}
    }

    if lexer.rest().starts_with("//") {
        return lex_comment(lexer);
    }

    if lexer.rest().starts_with("f\"") {
        return lex_format_string_start(lexer);
    }

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

    if c == '"' {
        return lex_string(lexer);
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

fn lex_escaped_character(lexer: &Lexer, c: char) -> miette::Result<char> {
    match c {
        'n' => Ok('\n'),
        'r' => Ok('\r'),
        't' => Ok('\t'),
        '\\' => Ok('\\'),
        '"' => Ok('"'),
        _ => Err(miette::miette!(
            severity = Severity::Error,
            code = "invalid::escape",
            help = "only \\n, \\r, \\t, \\\\ and \\\" are valid escape sequences",
            labels = vec![LabeledSpan::at_offset(lexer.lo, "here")],
            "invalid escape sequence"
        )),
    }
}

/// Lex simple string literal
fn lex_string(lexer: &mut Lexer) -> miette::Result<(Token, Span)> {
    let lo = lexer.lo;
    let mut str = String::new();

    _ = lexer.next().unwrap();

    while let Some(c) = lexer.next() {
        match c {
            '"' => break,
            '\\' => {
                let c = lexer.next().expect("unexpected end of input");
                str.push(lex_escaped_character(lexer, c)?);
            }
            c => str.push(c),
        }
    }

    Ok((
        Token::StringLiteral,
        Span {
            lo,
            hi: lexer.lo,
            file: lexer.file,
            source: lexer.source,
        },
    ))
}

/// Start lexing a format string
/// Sets next state to ParseString and emits a FormatStringStart token
fn lex_format_string_start(lexer: &mut Lexer) -> miette::Result<(Token, Span)> {
    let lo = lexer.lo;

    lexer.next();
    lexer.next();

    lexer.state.f_string = LexerFormatStringState::ParseString;

    Ok((
        Token::FormatStringStart,
        Span {
            lo,
            hi: lexer.lo,
            file: lexer.file,
            source: lexer.source,
        },
    ))
}

fn lex_format_string_stop(lexer: &mut Lexer) -> miette::Result<(Token, Span)> {
    // Consume last '"'
    lexer.next();

    lexer.state.f_string = LexerFormatStringState::Idle;

    Ok((
        Token::FormatStringEnd,
        Span {
            lo: lexer.lo,
            hi: lexer.lo,
            file: lexer.file,
            source: lexer.source,
        },
    ))
}

/// Lex a format string
fn lex_format_string(lexer: &mut Lexer) -> miette::Result<(Token, Span)> {
    let lo = lexer.lo;

    let mut str = String::new();

    while let Some(c) = lexer.peek() {
        if c == '"' {
            lexer.state.f_string = LexerFormatStringState::Stop;

            // If there is no content in the format string we do
            // not need to emit a StringLiteral token
            if str.len() == 0 {
                return lex_format_string_stop(lexer);
            }

            break;
        }

        lexer.next();

        match c {
            '{' => {
                if lexer.peek() == Some('{') {
                    _ = lexer.next();
                    str.push('{');
                } else {
                    lexer.state.f_string = LexerFormatStringState::StartParseExpr;

                    // If there is no content in the format string we do
                    // not need to emit a ParseExpr token
                    if str.len() == 0 {
                        return lex_format_string_expr_start(lexer);
                    }

                    break;
                }
            }
            '}' => {
                if lexer.peek() == Some('}') {
                    _ = lexer.next();
                    str.push('}');
                } else {
                    return Err(miette::miette!(
                        severity = Severity::Error,
                        code = "unexpected::brace",
                        help = "expected '}}' to escape '}'",
                        labels = vec![LabeledSpan::at_offset(lexer.lo, "here")],
                        "unexpected brace"
                    )
                    .with_source_code(lexer.source_code()));
                }
            }
            '\\' => {
                let c = lexer.next().expect("unexpected end of input");
                str.push(lex_escaped_character(lexer, c)?);
            }
            c => str.push(c),
        }
    }

    let span = Span {
        lo,
        hi: lexer.lo - 1,
        file: lexer.file,
        source: lexer.source,
    };

    Ok((Token::StringLiteral, span))
}

fn lex_format_string_expr_start(lexer: &mut Lexer) -> miette::Result<(Token, Span)> {
    lexer.state.f_string = LexerFormatStringState::ParseExpr;

    Ok((
        Token::FormatStringExprStart,
        Span {
            lo: lexer.lo,
            hi: lexer.lo,
            file: lexer.file,
            source: lexer.source,
        },
    ))
}

fn lex_format_string_expr(lexer: &mut Lexer) -> miette::Result<(Token, Span)> {
    lexer.state.f_string = LexerFormatStringState::Idle;

    let (token, span) = lex_token(lexer)?;

    lexer.state.f_string = LexerFormatStringState::ParseExpr;

    if token == Token::RBrace {
        lexer.state.f_string = LexerFormatStringState::ParseString;
        return Ok((Token::FormatStringExprEnd, span));
    }

    Ok((token, span))
}

fn lex_comment(lexer: &mut Lexer) -> miette::Result<(Token, Span)> {
    let lo = lexer.lo;

    _ = lexer.next();
    _ = lexer.next();

    let token = match lexer.peek() {
        Some('/') => {
            _ = lexer.next();
            Token::DocComment
        }
        Some('!') => {
            _ = lexer.next();
            Token::ModDocComment
        }
        _ => Token::Newline,
    };

    while lexer.peek() != Some('\n') {
        _ = lexer.next();
    }

    Ok((
        token,
        Span {
            lo: if token == Token::Newline {
                lexer.lo
            } else {
                lo
            },
            hi: lexer.lo,
            file: lexer.file,
            source: lexer.source,
        },
    ))
}
