#[macro_export]
macro_rules! arena {
    ($name:ident [ $id:ident ]: $item:ident) => {
        #[derive(
            ::std::clone::Clone,
            ::std::marker::Copy,
            ::std::fmt::Debug,
            ::std::cmp::PartialEq,
            ::std::cmp::Eq,
            ::std::hash::Hash,
        )]
        pub struct $id {
            index: usize,
        }

        #[derive(::std::default::Default)]
        pub struct $name {
            items: ::std::vec::Vec<::std::option::Option<$item>>,
        }

        impl ::std::clone::Clone for $name
        where
            $item: ::std::clone::Clone,
        {
            fn clone(&self) -> Self {
                Self {
                    items: self.items.clone(),
                }
            }
        }

        impl ::std::fmt::Debug for $name
        where
            $item: ::std::fmt::Debug,
        {
            fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
                f.debug_map().entries(self.iter()).finish()
            }
        }

        impl $name {
            pub fn new() -> Self {
                Self::default()
            }

            pub fn alloc(&mut self) -> $id {
                let id = $id {
                    index: self.items.len(),
                };

                self.items.push(None);

                id
            }

            pub fn push(&mut self, item: $item) -> $id {
                let id = $id {
                    index: self.items.len(),
                };

                self.items.push(Some(item));

                id
            }

            pub fn insert(&mut self, id: $id, item: $item) {
                self.items[id.index] = Some(item);
            }

            pub fn get(&self, id: $id) -> Option<&$item> {
                self.items.get(id.index)?.as_ref()
            }

            pub fn get_mut(&mut self, id: $id) -> Option<&mut $item> {
                self.items.get_mut(id.index)?.as_mut()
            }

            pub fn iter(&self) -> impl Iterator<Item = ($id, &$item)> {
                self.items.iter().enumerate().filter_map(|(index, item)| {
                    let id = $id { index };

                    Some((id, item.as_ref()?))
                })
            }

            pub fn iter_mut(&mut self) -> impl Iterator<Item = ($id, &mut $item)> {
                (self.items.iter_mut().enumerate()).filter_map(|(index, item)| {
                    let id = $id { index };

                    Some((id, item.as_mut()?))
                })
            }
        }

        impl ::std::ops::Index<$id> for $name {
            type Output = $item;

            fn index(&self, id: $id) -> &Self::Output {
                self.items[id.index].as_ref().unwrap()
            }
        }

        impl ::std::ops::IndexMut<$id> for $name {
            fn index_mut(&mut self, id: $id) -> &mut Self::Output {
                self.items[id.index].as_mut().unwrap()
            }
        }
    };
}

arena!(Arena[Id]: i32);
