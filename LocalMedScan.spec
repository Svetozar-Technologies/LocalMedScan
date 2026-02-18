# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for LocalMedScan desktop application."""

import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('assets', 'assets'),
        ('i18n', 'i18n'),
        ('models/.gitkeep', 'models'),
    ],
    hiddenimports=[
        'torch',
        'torchvision',
        'torchxrayvision',
        'pytorch_grad_cam',
        'cv2',
        'PIL',
        'pydicom',
        'numpy',
        'reportlab',
        'skimage',
        'skimage.transform',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'scipy.spatial.cKDTree',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LocalMedScan',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.icns' if sys.platform == 'darwin' else 'assets/icon.png',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='LocalMedScan',
)

if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='LocalMedScan.app',
        icon='assets/icon.icns',
        bundle_identifier='com.svetozar.localmedscan',
        info_plist={
            'CFBundleName': 'LocalMedScan',
            'CFBundleDisplayName': 'LocalMedScan',
            'CFBundleVersion': '1.0.0',
            'CFBundleShortVersionString': '1.0.0',
            'NSHighResolutionCapable': True,
            'LSMinimumSystemVersion': '10.15',
        },
    )
