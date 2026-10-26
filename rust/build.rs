fn main() {
    // Conda builds libpython with install_name `@rpath/libpython3.x.dylib` and Cargo
    // records no LC_RPATH, so `cargo test` and the CLI binaries abort under dyld on a
    // conda macOS install. Framework and system Pythons use an absolute install_name,
    // which is why CI never hits it. This helper emits the rpath link args with the
    // per-platform guards maintained upstream.
    pyo3_build_config::add_libpython_rpath_link_args();
}
