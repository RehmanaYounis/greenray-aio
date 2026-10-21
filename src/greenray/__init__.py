"""GreenRay AIO: hands-on cluster simulator and mock exam for NCP-AIO (AI Operations) preparation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("greenray-aio")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0+local"
