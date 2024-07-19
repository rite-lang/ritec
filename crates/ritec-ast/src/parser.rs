use std::collections::HashMap;
use std::fmt::Debug;
use std::path::PathBuf;
use ritec_diagnostic::Diagnostic;
use ritec_parse::{Tokenizer, TokenStream};
use ritec_source::{SourceId, Sources};
use crate::{Module, parse_module};


#[derive(Clone, Default)]
pub struct Parser {
    pub sources: Sources,
    modules: HashMap<String, Module>,
}

impl Parser {
    pub fn new() -> Self {
        Self {
            sources: Sources::new(),
            modules: HashMap::new(),
        }
    }

    /// Start parsing the source code from module in source_id 0.
    pub fn parse(&mut self) -> Result<Module, Diagnostic> {
        parse_module(self, &mut self.source_stream(SourceId::new(0)))
    }

    pub fn source_stream(&self, source_id: SourceId) -> TokenStream {
        let source = self.sources.get(source_id);
        let mut tokenizer = Tokenizer::new(source.index);
        tokenizer.tokenize(&source.source).unwrap()
    }

    pub fn get_module(&self, path: String) -> Option<&Module> {
        self.modules.get(&path)
    }

    pub fn add_module(&mut self, path: String, module: Module) {
        self.modules.insert(path, module);
    }
}

impl From<PathBuf> for Parser {
    /// Create a Parser from a PathBuf and use that as the entrypoint
    /// where source_id is 0.
    fn from(value: PathBuf) -> Self {
        let mut parser = Self::new();
        parser.sources.add_path(value.clone(), "".to_string());
        parser
    }
}


impl Debug for Parser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Parser")
            .field("modules", &self.modules)
            .finish()
    }
}
