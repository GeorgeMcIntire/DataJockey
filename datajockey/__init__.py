from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("datajockey")
except PackageNotFoundError:  # during editable dev installs without a built dist
    __version__ = "0.0.0"