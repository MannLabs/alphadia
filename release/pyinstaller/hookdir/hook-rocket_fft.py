from PyInstaller.compat import is_win

print("==================================")
print("hook-rocket_fft.py")
print("==================================")

hiddenimports = ["rocket_fft"]
module_collection_mode = "py"

try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata


dist_files = importlib_metadata.files("rocket-fft")

collected_runtime_files = []

if dist_files is not None:
    runtime_dll_files = [
        f
        for f in dist_files
        if f.as_posix().endswith(".so")
        or f.as_posix().endswith(".dll")
        or f.as_posix().endswith(".pyd")
    ]
    print("runtime_dll_files:", runtime_dll_files)
    collected_runtime_files = [
        (runtime_dll_file.locate(), runtime_dll_file.parent.as_posix())
        for runtime_dll_file in runtime_dll_files
    ]
    print("collected_runtime_files:", collected_runtime_files)


# On Windows, collect runtime DLL file(s) as binaries; on other OSes, collect them as data files, to prevent fatal
# errors in binary dependency analysis.
if is_win:
    binaries = collected_runtime_files
else:
    datas = collected_runtime_files
