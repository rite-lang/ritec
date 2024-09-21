use std::sync::Arc;

use crate::span::Span;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Token {
    Snake,
    Pascal,
    Integer,
    Float,

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
    Assert,
    F32,
    F64,
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
    Pub,
    Pure,
    Return,
    Type,
    U16,
    U32,
    U64,
    U8,
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
}

impl Token {
    pub fn keyword_from_str(s: &str) -> Option<Token> {
        match s {
            "assert" => Some(Token::Assert),
            "f32" => Some(Token::F32),
            "fn" => Some(Token::Fn),
            "for" => Some(Token::For),
            "i16" => Some(Token::I16),
            "i32" => Some(Token::I32),
            "i64" => Some(Token::I64),
            "i8" => Some(Token::I8),
            "if" => Some(Token::If),
            "import" => Some(Token::Import),
            "int" => Some(Token::Int),
            "let" => Some(Token::Let),
            "loop" => Some(Token::Loop),
            "match" => Some(Token::Match),
            "mut" => Some(Token::Mut),
            "pub" => Some(Token::Pub),
            "pure" => Some(Token::Pure),
            "return" => Some(Token::Return),
            "type" => Some(Token::Type),
            "u16" => Some(Token::U16),
            "u32" => Some(Token::U32),
            "u64" => Some(Token::U64),
            "u8" => Some(Token::U8),
            "void" => Some(Token::Void),
            _ => None,
        }
    }

    pub fn symbol_from_str(s: &str) -> Option<Token> {
        match s {
            /* delimiters */
            "(" => Some(Token::LParen),
            "[" => Some(Token::LBracket),
            "{" => Some(Token::LBrace),
            ")" => Some(Token::RParen),
            "]" => Some(Token::RBracket),
            "}" => Some(Token::RBrace),

            /* two-character symbols */
            ".." => Some(Token::DotDot),
            "==" => Some(Token::EqEq),
            "!=" => Some(Token::NotEq),
            "<=" => Some(Token::LtEq),
            ">=" => Some(Token::GtEq),
            "&&" => Some(Token::AmpAmp),
            "||" => Some(Token::PipePipe),
            "|>" => Some(Token::PipeGt),
            "->" => Some(Token::Arrow),

            /* one-character symbols */
            ":" => Some(Token::Colon),
            ";" => Some(Token::Semi),
            "." => Some(Token::Dot),
            "," => Some(Token::Comma),
            "=" => Some(Token::Eq),
            "!" => Some(Token::Not),
            "&" => Some(Token::Amp),
            "|" => Some(Token::Pipe),
            "<" => Some(Token::Lt),
            ">" => Some(Token::Gt),
            "+" => Some(Token::Plus),
            "-" => Some(Token::Minus),
            "*" => Some(Token::Star),
            "/" => Some(Token::Slash),
            "%" => Some(Token::Percent),
            "'" => Some(Token::Quote),
            "_" => Some(Token::Under),
            _ => None,
        }
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
