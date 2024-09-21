use std::collections::HashSet;

#[derive(Debug, Default)]
pub struct Interner {
    strings: HashSet<&'static str>,
}

impl Interner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern(&mut self, string: impl ToString) -> &'static str {
        let string = string.to_string();

        if let Some(string) = self.strings.get(string.as_str()) {
            return string;
        }

        let string = Box::leak(Box::new(string));
        self.strings.insert(string);
        string
    }
}
