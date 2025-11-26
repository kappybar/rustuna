use std::collections::HashMap;

pub type Attrs = HashMap<AttrKey, String>;

#[derive(Eq, Hash, Clone, Debug, PartialEq)]
pub enum AttrKey {
    User(String),
    System(String),
}

// Compatible with CategoricalChoiceType.
// https://github.com/optuna/optuna/blob/v3.5.0/optuna/distributions.py#L18
#[derive(PartialEq, Debug, Clone)]
pub enum CategoryLabel {
    Float(f64),
    Int(i64),
    String(String),
    Bool(bool),
    None,
}
impl CategoryLabel {
    pub fn serialize(&self) -> String {
        match self {
            CategoryLabel::Float(f) => format!("f:{f:.18}"),
            CategoryLabel::Int(i) => format!("i:{i}"),
            CategoryLabel::String(s) => format!("s:{s}"),
            CategoryLabel::Bool(b) => {
                if *b {
                    String::from("true")
                } else {
                    String::from("false")
                }
            }
            CategoryLabel::None => "None".to_string(),
        }
    }
    pub fn deserialize(s: &str) -> Option<CategoryLabel> {
        if s == "None" {
            return Some(CategoryLabel::None);
        }
        if s == "true" {
            return Some(CategoryLabel::Bool(true));
        }
        if s == "false" {
            return Some(CategoryLabel::Bool(false));
        }
        if let Some(f) = s.strip_prefix("f:") {
            let f = f.parse::<f64>().ok()?;
            return Some(CategoryLabel::Float(f));
        }
        if let Some(i) = s.strip_prefix("i:") {
            let i = i.parse::<i64>().ok()?;
            return Some(CategoryLabel::Int(i));
        }
        if let Some(s) = s.strip_prefix("s:") {
            return Some(CategoryLabel::String(s.to_string()));
        }
        None // Must be unreachable.
    }
}

fn system_key_category_label(param_name: &str, choice_idx: usize) -> AttrKey {
    AttrKey::System(format!("category_labels:{param_name}:{choice_idx}"))
}

pub fn category_labels_to_attrs(param_name: &str, labels: &[CategoryLabel]) -> Attrs {
    let mut attrs = Attrs::new();
    for (i, label) in labels.iter().enumerate() {
        let key = system_key_category_label(param_name, i);
        attrs.insert(key, label.serialize().clone());
    }
    attrs
}

pub fn get_category_labels(
    attrs: &Attrs,
    param_name: &str,
    len: usize,
) -> Option<Vec<CategoryLabel>> {
    let mut labels: Vec<CategoryLabel> = Vec::with_capacity(len);
    for i in 0..len {
        let key = system_key_category_label(param_name, i);
        match attrs.get(&key) {
            Some(label) => {
                let label = CategoryLabel::deserialize(label)?;
                labels.push(label);
            }
            None => return None,
        }
    }
    Some(labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_category_label() {
        let categories = vec![
            CategoryLabel::Float(1.0),
            CategoryLabel::Int(2),
            CategoryLabel::String("3".to_string()),
            CategoryLabel::Bool(true),
            CategoryLabel::Bool(false),
            CategoryLabel::None,
        ];

        for c in categories {
            let s = c.serialize();
            let c2 = CategoryLabel::deserialize(&s).unwrap();
            assert_eq!(c, c2);
        }
    }
}
