use pyo3::prelude::*;

/// Placeholder to exercise the Rust build and release pipeline until the
/// alphadia-search-rs code is moved into this crate.
#[pyfunction]
fn hello_world() -> &'static str {
    "Hello from alphadia Rust!"
}

#[pymodule]
#[pyo3(name = "_rust")]
fn alphadia_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_world, m)?)?;
    Ok(())
}
