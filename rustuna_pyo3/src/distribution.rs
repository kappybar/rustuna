use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyList;
use pyo3::types::{PyBool, PyFloat, PyInt, PyString, PyType};

use rustuna_core::attr::{get_category_labels, Attrs, CategoryLabel};
use rustuna_core::distribution::Distribution;

#[derive(Clone, Debug)]
#[pyclass(name = "Distribution")]
#[pyo3(module = "rustuna")]
pub struct PyDistribution {
    pub distribution: Distribution,
    pub category_labels: Option<Vec<CategoryLabel>>,
}
impl PyDistribution {
    pub fn new(distribution: Distribution, name: &str, study_attrs: &Attrs) -> PyDistribution {
        match distribution {
            Distribution::Categorical { cardinality } => {
                let category_labels = get_category_labels(study_attrs, name, cardinality);
                PyDistribution {
                    distribution,
                    category_labels,
                }
            }
            _ => PyDistribution {
                distribution,
                category_labels: None,
            },
        }
    }
}
impl From<PyDistribution> for Distribution {
    fn from(val: PyDistribution) -> Self {
        val.distribution
    }
}
#[pymethods]
impl PyDistribution {
    #[classmethod]
    #[pyo3(signature = (low, high, log=false, step=None))]
    pub fn float(
        _cls: &Bound<'_, PyType>,
        low: f64,
        high: f64,
        log: bool,
        step: Option<f64>,
    ) -> Self {
        PyDistribution {
            distribution: Distribution::Float {
                low,
                high,
                step,
                log,
            },
            category_labels: None,
        }
    }

    #[classmethod]
    #[pyo3(signature = (low, high, log=false, step=1))]
    pub fn int(_cls: &Bound<'_, PyType>, low: i64, high: i64, log: bool, step: i64) -> Self {
        PyDistribution {
            distribution: Distribution::Int {
                low,
                high,
                step,
                log,
            },
            category_labels: None,
        }
    }

    #[classmethod]
    pub fn categorical(_cls: &Bound<'_, PyType>, choices: Vec<PyObject>) -> PyResult<Self> {
        let cardinality = choices.len();
        let category_labels = Python::with_gil(|py| {
            let mut labels: Vec<CategoryLabel> = Vec::with_capacity(choices.len());
            for choice in choices {
                match pyobject_to_category_label(py, choice) {
                    Ok(label) => labels.push(label),
                    Err(e) => return Err(e),
                }
            }
            Ok(labels)
        })?;
        let py_dist = PyDistribution {
            distribution: Distribution::Categorical { cardinality },
            category_labels: Some(category_labels),
        };
        Ok(py_dist)
    }

    pub fn to_dict(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| match &self.distribution {
            Distribution::Float {
                low,
                high,
                step,
                log,
            } => {
                let dist = PyDict::new_bound(py);
                dist.set_item("type", "FloatDistribution")?;
                dist.set_item("low", low)?;
                dist.set_item("high", high)?;
                dist.set_item("log", log)?;
                if let Some(step) = step {
                    dist.set_item("step", step)?;
                } else {
                    dist.set_item("step", py.None())?;
                }
                Ok(dist.into())
            }
            Distribution::Int {
                low,
                high,
                step,
                log,
            } => {
                let dist = PyDict::new_bound(py);
                dist.set_item("type", "IntDistribution")?;
                dist.set_item("low", low)?;
                dist.set_item("high", high)?;
                dist.set_item("log", log)?;
                dist.set_item("step", step)?;
                Ok(dist.into())
            }
            Distribution::Categorical { cardinality } => {
                let dist = PyDict::new_bound(py);
                dist.set_item("type", "CategoricalDistribution")?;

                let mut elements: Vec<PyObject> = Vec::with_capacity(*cardinality);
                let labels = match self.category_labels {
                    Some(ref labels) => labels.clone(),
                    None => {
                        let mut labels: Vec<CategoryLabel> = Vec::with_capacity(*cardinality);
                        for i in 0..*cardinality {
                            labels.push(CategoryLabel::Int(i as i64));
                        }
                        labels
                    }
                };
                for i in 0..*cardinality {
                    let c = labels.get(i).ok_or(PyValueError::new_err(
                        "Internal representation of categorical value is out of range",
                    ))?;
                    elements.push(category_label_to_pyobject(py, c));
                }
                let choices = PyList::new_bound(py, &elements);
                dist.set_item("choices", choices)?;
                Ok(dist.into())
            }
        })
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let class_obj = slf.get_type();
        let class_name = class_obj.name()?;
        Ok(format!("{}({:?})", class_name, slf.borrow().__str__()?))
    }

    fn __str__(&self) -> PyResult<String> {
        Python::with_gil(|_py| Ok(self.to_dict()?.to_string()))
    }
}

pub fn category_label_to_pyobject(py: Python, label: &CategoryLabel) -> PyObject {
    match label {
        CategoryLabel::String(s) => s.into_py(py),
        CategoryLabel::Int(i) => i.into_py(py),
        CategoryLabel::Float(f) => f.into_py(py),
        CategoryLabel::Bool(b) => b.into_py(py),
        CategoryLabel::None => py.None(),
    }
}

pub fn pyobject_to_category_label(py: Python, obj: PyObject) -> PyResult<CategoryLabel> {
    let py_any = obj.bind(py);
    if py_any.is_instance_of::<PyBool>() {
        let x = py_any
            .extract::<bool>()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to extract bool: {e:?}")))?;
        Ok(CategoryLabel::Bool(x))
    } else if py_any.is_instance_of::<PyInt>() {
        let x = py_any
            .extract::<i64>()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to extract i64: {e:?}")))?;
        Ok(CategoryLabel::Int(x))
    } else if py_any.is_instance_of::<PyFloat>() {
        let x = py_any
            .extract::<f64>()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to extract f64: {e:?}")))?;
        Ok(CategoryLabel::Float(x))
    } else if py_any.is_instance_of::<PyString>() {
        let x = py_any
            .extract::<String>()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to extract String: {e:?}")))?;
        Ok(CategoryLabel::String(x))
    } else if py_any.is_none() {
        Ok(CategoryLabel::None)
    } else {
        Err(PyRuntimeError::new_err(
            "Unsupported type for categorical choice",
        ))
    }
}

pub fn py_to_external_repr(
    dist: &Distribution,
    internal_repr: f64,
    param_name: &str,
    study_attrs: &Attrs,
) -> PyResult<PyObject> {
    match dist {
        Distribution::Float { .. } => {
            let external_value = Python::with_gil(|py| internal_repr.into_py(py));
            Ok(external_value)
        }
        Distribution::Int { .. } => {
            let external_value = Python::with_gil(|py| (internal_repr as i64).into_py(py));
            Ok(external_value)
        }
        Distribution::Categorical { cardinality } => {
            match get_category_labels(study_attrs, param_name, *cardinality) {
                Some(labels) => {
                    let c = labels
                        .get(internal_repr as usize)
                        .ok_or(PyValueError::new_err(
                            "Internal representation of categorical value is out of range",
                        ))?;
                    let external_value = Python::with_gil(|py| category_label_to_pyobject(py, c));
                    Ok(external_value)
                }
                None => {
                    let external_value = Python::with_gil(|py| (internal_repr as i64).into_py(py));
                    Ok(external_value)
                }
            }
        }
    }
}
