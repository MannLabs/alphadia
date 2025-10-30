# -*- mode: python ; coding: utf-8 -*-

import pkgutil
import os
import sys
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT, BUNDLE, TOC
import PyInstaller.utils.hooks
from PyInstaller.utils.hooks import copy_metadata
import pkg_resources
import importlib.metadata


##################### User definitions
exe_name = 'alphadia'
script_name = 'cli_hook.py'
if sys.platform[:6] == "darwin":
	icon = '../logos/alphadia.icns'
else:
	icon = '../logos/alphadia.ico'
block_cipher = None
location = os.getcwd()
project = "alphadia"
remove_tests = True
bundle_name = "alphadia"
#####################


datas, binaries, hidden_imports = PyInstaller.utils.hooks.collect_all(
	project,
	include_py_files=True
)

alpha_x = ['alphabase', 'alpharaw','alphatims','peptdeep','alphadia-search-rs']
for alpha_package in alpha_x:
	_datas, _binaries, _hidden_imports = PyInstaller.utils.hooks.collect_all(
		alpha_package,
		include_py_files=True
	)
	datas+=_datas
	binaries+=_binaries
	hidden_imports+=_hidden_imports

hidden_imports = list(set(hidden_imports) & {'clr','rocket_fft','tokenizers'})
hidden_imports = [h for h in hidden_imports if "__pycache__" not in h]
datas = [d for d in datas if ("__pycache__" not in d[0]) and (d[1] not in [".", "Resources", "scripts"])]

a = Analysis(
	[script_name],
	pathex=[location],
	binaries=binaries,
	datas=datas,
	hiddenimports=hidden_imports,
	hookspath=['./release/pyinstaller/hookdir'],
	runtime_hooks=[],
	excludes=[],
    win_no_prefer_redirects=False,
	win_private_assemblies=False,
	cipher=block_cipher,
	noarchive=False
)
pyz = PYZ(
	a.pure,
	a.zipped_data,
	cipher=block_cipher
)

if sys.platform[:5] == "linux":
	exe = EXE(
		pyz,
		a.scripts,
		a.binaries,
		a.zipfiles,
		a.datas,
		name=bundle_name,
		debug=False,
		bootloader_ignore_signals=False,
		strip=False,
		upx=True,
		console=True,
		upx_exclude=[],
		icon=icon
	)
else: # non-linux
	exe = EXE(
		pyz,
		a.scripts,
		# a.binaries,
		a.zipfiles,
		# a.datas,
		exclude_binaries=True,
		name=exe_name,
		debug=False,
		bootloader_ignore_signals=False,
		strip=False,
		upx=True,
		console=True,
		icon=icon
	)
	coll = COLLECT(
		exe,
		a.binaries,
		# a.zipfiles,
		a.datas,
		strip=False,
		upx=True,
		upx_exclude=[],
		name=exe_name
	)
