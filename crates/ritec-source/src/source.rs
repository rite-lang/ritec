use std::fs;
use std::path::PathBuf;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SourceId {
    pub index: usize,
}

impl SourceId {
    pub fn new(index: usize) -> Self {
        Self { index }
    }
}

impl Default for SourceId {
    fn default() -> Self {
        Self { index: 0 }
    }
}

#[derive(Clone, Debug)]
pub struct Source {
    pub name: String,
    pub path: PathBuf,
    pub index: SourceId,
    pub source: String,
}

#[derive(Clone, Debug, Default)]
pub struct Sources {
    sources: Vec<Source>,
}

impl Sources {
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
        }
    }

    pub fn add_path(&mut self, path: PathBuf, name: String) -> &Source {
        let index = self.sources.len();

        let path = fs::canonicalize(path).unwrap();

        let source = Source {
            name,
            path: path.clone(),
            index: SourceId::new(index),
            source: fs::read_to_string(&path).unwrap(),
        };

        self.sources.push(source);

        &self.sources[index]
    }

    pub fn get(&self, id: SourceId) -> &Source {
        &self.sources[id.index]
    }
}
