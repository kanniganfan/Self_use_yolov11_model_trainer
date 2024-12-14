import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('weights', 'weights'),
        ('src/ui', 'src/ui'),
    ],
    hiddenimports=[
        'ultralytics',
        'torch',
        'torchvision',
        'PyQt5',
        'cv2',
        'numpy',
        'PIL',
        'PIL._imaging',
        'PIL.Image',
        'PIL.ImageQt',
        'PIL.ImageColor',
        'PIL.ImageFilter',
        'PIL.ImageEnhance',
        'PIL.ImageDraw',
        'PIL.ImageFile',
        'PIL.JpegImagePlugin',
        'PIL.PngImagePlugin',
        'PIL.GifImagePlugin',
        'PIL.BmpImagePlugin',
        'PIL.TiffImagePlugin',
        'PIL.WebPImagePlugin',
        'PIL.IcoImagePlugin',
        'PIL.DngImagePlugin',
        'PIL.MpoImagePlugin',
        'PIL.PfmImagePlugin',
        'PIL.HeicImagePlugin',
        'PIL.ImImagePlugin',
        'PIL.IcnsImagePlugin',
        'PIL.TgaImagePlugin',
        'PIL.XbmImagePlugin',
        'PIL.XpmImagePlugin',
        'PIL.XVThumbImagePlugin',
        'PIL.WmfImagePlugin',
        'PIL.SunImagePlugin',
        'PIL.PsdImagePlugin',
        'PIL.PpmImagePlugin',
        'PIL.PcxImagePlugin',
        'PIL.MspImagePlugin',
        'PIL.McIdasImagePlugin',
        'PIL.IptcImagePlugin',
        'PIL.ImtImagePlugin',
        'PIL.GribStubImagePlugin',
        'PIL.FpxImagePlugin',
        'PIL.FliImagePlugin',
        'PIL.FitsStubImagePlugin',
        'PIL.EpsImagePlugin',
        'PIL.DcxImagePlugin',
        'PIL.CurImagePlugin',
        'PIL.BufrStubImagePlugin',
        'PIL.BlpImagePlugin',
        'PIL.SgiImagePlugin',
        'PIL.SpiderImagePlugin',
    ] + collect_submodules('ultralytics'),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='YOLO训练器',
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
    entitlements_file=None
) 