use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::PyAnyMethods;
use pyo3::types::PyDict;
use pyo3::{Bound, PyAny, PyResult};
use rustuna_core::attr::{AttrKey, Attrs};

pub fn pyobj_to_system_attrs(obj: &Bound<'_, PyAny>) -> PyResult<Attrs> {
    if !obj.is_instance_of::<PyDict>() {
        return Err(PyRuntimeError::new_err("attrs must be a dict"));
    }
    let user_attrs = obj.downcast::<PyDict>()?;
    let mut attrs = Attrs::with_capacity(user_attrs.len()?);
    for (key, value) in user_attrs {
        let key = key.extract::<String>()?;
        let value = value.extract::<String>()?;
        attrs.insert(AttrKey::System(key), value);
    }
    Ok(attrs)
}

pub fn pyobj_to_user_attrs(obj: &Bound<'_, PyAny>) -> PyResult<Attrs> {
    if !obj.is_instance_of::<PyDict>() {
        return Err(PyRuntimeError::new_err("attrs must be a dict"));
    }
    let system_attrs = obj.downcast::<PyDict>()?;
    let mut attrs = Attrs::with_capacity(system_attrs.len()?);
    for (key, value) in system_attrs {
        let key = key.extract::<String>()?;
        let value = value.extract::<String>()?;
        attrs.insert(AttrKey::User(key), value);
    }
    Ok(attrs)
}

pub fn pyobj_to_attrs(
    user_attrs: &Bound<'_, PyAny>,
    system_attrs: &Bound<'_, PyAny>,
) -> PyResult<Attrs> {
    if !user_attrs.is_instance_of::<PyDict>() || !system_attrs.is_instance_of::<PyDict>() {
        return Err(PyRuntimeError::new_err(
            "user_attrs and system_attrs must be a dict",
        ));
    }
    let user_attrs = user_attrs.downcast::<PyDict>()?;
    let system_attrs = system_attrs.downcast::<PyDict>()?;
    let cap = user_attrs.len()? + system_attrs.len()?;
    let mut attrs = Attrs::with_capacity(cap);
    for (key, value) in user_attrs {
        let key = key.extract::<String>()?;
        let value = value.extract::<String>()?;
        attrs.insert(AttrKey::User(key), value);
    }
    for (key, value) in system_attrs {
        let key = key.extract::<String>()?;
        let value = value.extract::<String>()?;
        attrs.insert(AttrKey::System(key), value);
    }
    Ok(attrs)
}
