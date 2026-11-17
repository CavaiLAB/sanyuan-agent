# -*- mode: python ; coding: utf-8 -*-
"""Native one-file build for the no-tool desktop-run entry point."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

spec_dir = Path(SPEC).resolve().parent
harness_root = spec_dir.parent
entry = harness_root / "sanyuan_agent" / "local_desktop" / "bundled_entry.py"

datas = []
binaries = []
hiddenimports = []
for package in ("litellm", "langchain_core", "langgraph"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas.extend(package_datas)
    binaries.extend(package_binaries)
    hiddenimports.extend(package_hiddenimports)
hiddenimports.extend(collect_submodules("tiktoken_ext"))

analysis = Analysis(
    [str(entry)],
    pathex=[str(harness_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "jupyter",
        "matplotlib",
        "notebook",
        "pytest",
        "tkinter",
    ],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="iclaw-desktop-harness",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=True,
)
