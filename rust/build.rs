fn main() {
    // conda's libpython has an `@rpath` install_name, without an rpath `cargo test` aborts on macOS
    pyo3_build_config::add_libpython_rpath_link_args();
}
