#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Delim {
    Open,
    Close,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    Ident(String),

    /* control tokens */
    Newline,
    Indent,
    Dedent,
    Eof,

    /* keywords */
    Fn,
    Let,
    If,
    Else,
    Match,
    Where,
    Struct,
    Trait,

    /* triple-character symbols */
    DotDotDot,

    /* double-character symbols */
    SlashSlash,
    FatArrot,
    Arrot,
    ColonColon,
    DotDot,
    EqEq,
    LtEq,
    GtEq,
    AndAnd,
    OrOr,

    /* single-character symbols */
    Colon,
    Semi,
    Dot,
    Comma,
    Eq,
    Lt,
    Gt,
    And,
    Or,
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    Quote,
    Under,
    Paren(Delim),
    Brace(Delim),
    Bracket(Delim),
}

impl Token {
    pub fn from_keyword(keyword: &str) -> Option<Token> {
        match keyword {
            "fn" => Some(Token::Fn),
            "let" => Some(Token::Let),
            "if" => Some(Token::If),
            "else" => Some(Token::Else),
            "match" => Some(Token::Match),
            "where" => Some(Token::Where),
            "struct" => Some(Token::Struct),
            "trait" => Some(Token::Trait),
            _ => None,
        }
    }

    pub fn from_symbol(symbol: &str) -> Option<Token> {
        match symbol {
            /* triple-character symbols */
            "..." => Some(Token::DotDotDot),

            /* double-character symbols */
            "//" => Some(Token::SlashSlash),
            "=>" => Some(Token::FatArrot),
            "->" => Some(Token::Arrot),
            "::" => Some(Token::ColonColon),
            ".." => Some(Token::DotDot),
            "==" => Some(Token::EqEq),
            "<=" => Some(Token::LtEq),
            ">=" => Some(Token::GtEq),
            "&&" => Some(Token::AndAnd),
            "||" => Some(Token::OrOr),

            /* single-character symbols */
            ":" => Some(Token::Colon),
            ";" => Some(Token::Semi),
            "." => Some(Token::Dot),
            "," => Some(Token::Comma),
            "=" => Some(Token::Eq),
            "<" => Some(Token::Lt),
            ">" => Some(Token::Gt),
            "&" => Some(Token::And),
            "|" => Some(Token::Or),
            "+" => Some(Token::Plus),
            "-" => Some(Token::Minus),
            "*" => Some(Token::Star),
            "/" => Some(Token::Slash),
            "^" => Some(Token::Caret),
            "'" => Some(Token::Quote),
            "_" => Some(Token::Under),
            "(" => Some(Token::Paren(Delim::Open)),
            ")" => Some(Token::Paren(Delim::Close)),
            "{" => Some(Token::Brace(Delim::Open)),
            "}" => Some(Token::Brace(Delim::Close)),
            "[" => Some(Token::Bracket(Delim::Open)),
            "]" => Some(Token::Bracket(Delim::Close)),
            _ => None,
        }
    }
}
