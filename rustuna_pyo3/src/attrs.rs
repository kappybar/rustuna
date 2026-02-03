use std::sync::{Arc, RwLock};

use pyo3::class::basic::CompareOp;
use pyo3::exceptions::{PyKeyError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyIterator, PyList};
use rustuna_core::attr::{AttrKey, Attrs};
use rustuna_core::storage::Storage;

use crate::exception::err_to_exceptions;

// TODO(c-bata): Consider removing the PyDict branch if there is no significant performance
// difference. The Mapping protocol branch (else) can handle PyDict as well.
pub fn pyobj_to_attrs_with_kind(obj: &Bound<'_, PyAny>, kind: AttrKind) -> PyResult<Attrs> {
    if obj.is_instance_of::<PyDict>() {
        // Fast path for PyDict: iterate directly without calling .items() method.
        let dict = obj.cast::<PyDict>()?;
        let mut attrs = Attrs::with_capacity(dict.len());
        for (key, value) in dict {
            let key = key.extract::<String>()?;
            let value = value.extract::<String>()?;
            attrs.insert(kind.to_key(&key), value);
        }
        Ok(attrs)
    } else {
        // TODO(c-bata): Add error handling if obj does not implement Mapping protocol.
        let items = obj.call_method0("items")?;
        let items = items.extract::<Vec<(String, String)>>()?;
        let mut attrs = Attrs::with_capacity(items.len());
        for (key, value) in items {
            attrs.insert(kind.to_key(&key), value);
        }
        Ok(attrs)
    }
}

pub fn pyobj_to_attrs(
    user_attrs: &Bound<'_, PyAny>,
    system_attrs: &Bound<'_, PyAny>,
) -> PyResult<Attrs> {
    let user_attrs = pyobj_to_attrs_with_kind(user_attrs, AttrKind::User)?;
    let system_attrs = pyobj_to_attrs_with_kind(system_attrs, AttrKind::System)?;
    let cap = user_attrs.len() + system_attrs.len();
    let mut attrs = Attrs::with_capacity(cap);
    for (key, value) in user_attrs {
        attrs.insert(key, value);
    }
    for (key, value) in system_attrs {
        attrs.insert(key, value);
    }
    Ok(attrs)
}

#[derive(Clone, Copy)]
pub enum AttrKind {
    User,
    System,
}

impl AttrKind {
    fn to_key(self, key: &str) -> AttrKey {
        match self {
            AttrKind::User => AttrKey::User(key.into()),
            AttrKind::System => AttrKey::System(key.into()),
        }
    }

    fn matches(self, key: &AttrKey) -> bool {
        matches!(
            (self, key),
            (AttrKind::User, AttrKey::User(_)) | (AttrKind::System, AttrKey::System(_))
        )
    }
}

enum AttrsDictViewSource {
    Owned(Attrs),
    StorageBacked {
        storage: Arc<RwLock<dyn Storage>>,
        trial_id: u32,
    },
}

#[pyclass(name = "AttrsDictView", unsendable)]
pub struct AttrsDictView {
    source: AttrsDictViewSource,
    kind: AttrKind,
}

impl AttrsDictView {
    pub fn from_trial(trial: &rustuna_core::trial::PersistedTrial, kind: AttrKind) -> Self {
        let mut attrs = Attrs::new();
        for (key, value) in &trial.attrs {
            if kind.matches(key) {
                attrs.insert(key.clone(), value.clone());
            }
        }
        AttrsDictView {
            source: AttrsDictViewSource::Owned(attrs),
            kind,
        }
    }

    pub fn from_storage(storage: Arc<RwLock<dyn Storage>>, trial_id: u32, kind: AttrKind) -> Self {
        AttrsDictView {
            source: AttrsDictViewSource::StorageBacked { storage, trial_id },
            kind,
        }
    }

    fn with_attrs<R>(&self, f: impl FnOnce(&Attrs) -> PyResult<R>) -> PyResult<R> {
        match &self.source {
            AttrsDictViewSource::Owned(attrs) => f(attrs),
            AttrsDictViewSource::StorageBacked { storage, trial_id } => {
                let guard = storage.read().map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "Failed to acquire the storage guard: {:?}",
                        e.to_string()
                    ))
                })?;
                let trial = guard
                    .get_cached_trial(*trial_id)
                    .map_err(err_to_exceptions)?;
                f(&trial.attrs)
            }
        }
    }

    pub(crate) fn get_value(&self, key: &str) -> PyResult<Option<String>> {
        let lookup_key = self.kind.to_key(key);
        self.with_attrs(|attrs| Ok(attrs.get(&lookup_key).cloned()))
    }

    fn collect_entries(&self) -> PyResult<Vec<(String, String)>> {
        self.with_attrs(|attrs| {
            let mut entries = Vec::new();
            for (key, value) in attrs {
                if !self.kind.matches(key) {
                    continue;
                }
                let key = match key {
                    AttrKey::User(k) => k.as_str(),
                    AttrKey::System(k) => k.as_str(),
                };
                entries.push((key.to_string(), value.clone()));
            }
            Ok(entries)
        })
    }

    pub(crate) fn to_pydict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for (key, value) in self.collect_entries()? {
            dict.set_item(key, value)?;
        }
        Ok(dict.unbind())
    }

    pub(crate) fn format_as_dict(&self) -> PyResult<String> {
        let entries: Vec<String> = self
            .collect_entries()?
            .iter()
            .map(|(k, v)| format!("'{k}': '{v}'"))
            .collect();
        Ok(format!("{{{}}}", entries.join(", ")))
    }

    fn len(&self) -> PyResult<usize> {
        self.with_attrs(|attrs| Ok(attrs.iter().filter(|(k, _)| self.kind.matches(k)).count()))
    }
}

#[pymethods]
impl AttrsDictView {
    fn __len__(&self) -> PyResult<usize> {
        self.len()
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let keys = self.keys()?;
        let list = PyList::new(py, keys)?;
        let iter = PyIterator::from_object(list.as_any())?;
        Ok(iter.unbind().into())
    }

    fn __getitem__(&self, key: &str) -> PyResult<String> {
        match self.get_value(key)? {
            Some(value) => Ok(value),
            None => Err(PyKeyError::new_err(key.to_string())),
        }
    }

    fn __contains__(&self, key: &str) -> PyResult<bool> {
        Ok(self.get_value(key)?.is_some())
    }

    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        match self.get_value(key)? {
            Some(value) => Ok(value.into_pyobject(py)?.unbind().into()),
            None => Ok(default.unwrap_or_else(|| py.None())),
        }
    }

    fn keys(&self) -> PyResult<Vec<String>> {
        self.with_attrs(|attrs| {
            let mut keys = Vec::new();
            for key in attrs.keys() {
                if !self.kind.matches(key) {
                    continue;
                }
                let key = match key {
                    AttrKey::User(k) => k.as_str(),
                    AttrKey::System(k) => k.as_str(),
                };
                keys.push(key.to_string());
            }
            Ok(keys)
        })
    }

    fn values(&self) -> PyResult<Vec<String>> {
        self.with_attrs(|attrs| {
            let mut values = Vec::new();
            for (key, value) in attrs {
                if self.kind.matches(key) {
                    values.push(value.clone());
                }
            }
            Ok(values)
        })
    }

    fn items(&self) -> PyResult<Vec<(String, String)>> {
        self.collect_entries()
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        self.to_pydict(py)
    }

    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        match op {
            CompareOp::Eq | CompareOp::Ne => {
                let self_dict = self.to_pydict(py)?;
                if let Ok(other_view) = other.extract::<PyRef<AttrsDictView>>() {
                    let other_dict = other_view.to_pydict(py)?;
                    let result = self_dict.bind(py).rich_compare(other_dict.bind(py), op)?;
                    return Ok(result.unbind());
                }
                let result = self_dict.bind(py).rich_compare(other, op)?;
                Ok(result.unbind())
            }
            _ => Ok(py.NotImplemented()),
        }
    }

    fn __repr__(&self) -> PyResult<String> {
        self.format_as_dict()
    }

    fn __str__(&self) -> PyResult<String> {
        self.format_as_dict()
    }
}
