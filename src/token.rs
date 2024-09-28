use std::sync::Arc;

use crate::span::Span;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Token {
    Snake,
    Pascal,
    Integer,
    Float,
    Path,

    /* strings */
    String,
    FormatStart,
    FormatEnd,
    FormatExprStart,
    FormatExprEnd,

    /* comments */
    DocComment,
    ModDocComment,

    /* control */
    Newline,
    Indent,
    Dedent,
    Eof,

    /* delimiters */
    LParen,
    LBracket,
    LBrace,
    RParen,
    RBracket,
    RBrace,

    /* keywords */
    As,
    Assert,
    Bool,
    F32,
    F64,
    False,
    Fn,
    For,
    I16,
    I32,
    I64,
    I8,
    If,
    Import,
    Int,
    Let,
    Loop,
    Match,
    Mut,
    Panic,
    Pub,
    Pure,
    Return,
    Str,
    Todo,
    True,
    Type,
    U16,
    U32,
    U64,
    U8,
    Unreachable,
    Void,

    /* two-character symbols */
    DotDot,
    EqEq,
    NotEq,
    LtEq,
    GtEq,
    AmpAmp,
    PipePipe,
    PipeGt,
    Arrow,

    /* one-character symbols */
    Colon,
    Semi,
    Dot,
    Comma,
    Eq,
    Not,
    Amp,
    Pipe,
    Lt,
    Gt,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Quote,
    Under,
    Pound,
    Question,
}

impl Token {
    pub fn keyword_from_str(s: &str) -> Option<Token> {
        Some(match s {
            "as" => Token::As,
            "assert" => Token::Assert,
            "bool" => Token::Bool,
            "f32" => Token::F32,
            "f64" => Token::F64,
            "false" => Token::False,
            "fn" => Token::Fn,
            "for" => Token::For,
            "i16" => Token::I16,
            "i32" => Token::I32,
            "i64" => Token::I64,
            "i8" => Token::I8,
            "if" => Token::If,
            "import" => Token::Import,
            "int" => Token::Int,
            "let" => Token::Let,
            "loop" => Token::Loop,
            "match" => Token::Match,
            "mut" => Token::Mut,
            "panic" => Token::Panic,
            "pub" => Token::Pub,
            "pure" => Token::Pure,
            "return" => Token::Return,
            "str" => Token::Str,
            "todo" => Token::Todo,
            "true" => Token::True,
            "type" => Token::Type,
            "u16" => Token::U16,
            "u32" => Token::U32,
            "u64" => Token::U64,
            "u8" => Token::U8,
            "unreachable" => Token::Unreachable,
            "void" => Token::Void,
            _ => return None,
        })
    }

    pub fn symbol_from_str(s: &str) -> Option<Token> {
        Some(match s {
            /* delimiters */
            "(" => Token::LParen,
            "[" => Token::LBracket,
            "{" => Token::LBrace,
            ")" => Token::RParen,
            "]" => Token::RBracket,
            "}" => Token::RBrace,

            /* two-character symbols */
            ".." => Token::DotDot,
            "==" => Token::EqEq,
            "!=" => Token::NotEq,
            "<=" => Token::LtEq,
            ">=" => Token::GtEq,
            "&&" => Token::AmpAmp,
            "||" => Token::PipePipe,
            "|>" => Token::PipeGt,
            "->" => Token::Arrow,

            /* one-character symbols */
            ":" => Token::Colon,
            ";" => Token::Semi,
            "." => Token::Dot,
            "," => Token::Comma,
            "=" => Token::Eq,
            "!" => Token::Not,
            "&" => Token::Amp,
            "|" => Token::Pipe,
            "<" => Token::Lt,
            ">" => Token::Gt,
            "+" => Token::Plus,
            "-" => Token::Minus,
            "*" => Token::Star,
            "/" => Token::Slash,
            "%" => Token::Percent,
            "'" => Token::Quote,
            "_" => Token::Under,
            "#" => Token::Pound,
            "?" => Token::Question,
            _ => return None,
        })
    }
}

#[derive(Clone)]
pub struct TokenStream {
    tokens: Arc<[(Token, Span)]>,
    index: usize,
    file: &'static str,
    source: &'static str,
}

impl TokenStream {
    pub fn from_raw_parts(
        tokens: Arc<[(Token, Span)]>,
        file: &'static str,
        source: &'static str,
    ) -> Self {
        Self {
            tokens,
            index: 0,
            file,
            source,
        }
    }

    pub fn is_eof(&self) -> bool {
        self.index >= self.tokens.len()
    }

    pub fn peek(&self) -> (Token, Span) {
        match self.tokens.get(self.index) {
            Some(token) => *token,
            None => self.eof(),
        }
    }

    pub fn consume(&mut self) -> (Token, Span) {
        let token = self.peek();

        if !self.is_eof() {
            self.index += 1;
        }

        token
    }

    pub fn is(&self, token: Token) -> bool {
        self.peek().0 == token
    }

    pub fn nth_is(&self, n: usize, token: Token) -> bool {
        match self.tokens.get(self.index + n) {
            Some((t, _)) => *t == token,
            None => false,
        }
    }

    pub fn take(&mut self, token: Token) -> Option<Span> {
        if self.is(token) {
            Some(self.consume().1)
        } else {
            None
        }
    }

    pub fn expect(&mut self, token: Token) -> miette::Result<Span> {
        let (found, span) = self.peek();

        if found == token {
            Ok(self.consume().1)
        } else {
            Err(miette::miette!(
                severity = miette::Severity::Error,
                code = "expected",
                labels = vec![span.label("here")],
                "expected {:?}, found {:?}",
                token,
                found,
            )
            .with_source_code(span))
        }
    }

    fn eof(&self) -> (Token, Span) {
        (Token::Eof, self.eof_span())
    }

    fn eof_span(&self) -> Span {
        Span {
            lo: self.source.len(),
            hi: self.source.len(),
            file: self.file,
            source: self.source,
        }
    }
}

impl std::fmt::Debug for TokenStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries(self.tokens.iter().map(|(token, _)| token))
            .finish()
    }
}
