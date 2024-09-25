# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['LeagueMachine.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\LowQi\\IdeaProjects\\leagueMachine1\\venv\\Lib\\site-packages\\ultralytics\\cfg\\default.yaml', './ultralytics/cfg'), ('matchesjson/*', '.')],
    hiddenimports=['msgpack'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='LeagueMachine',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
