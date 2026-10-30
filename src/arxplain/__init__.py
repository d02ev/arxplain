"""ArXplain — a CLI tool that explains a research paper at your level.

This package is the CLI's implementation only; it is not a library and
exposes no public Python API. The single entry point is the `arxplain`
command (see `arxplain --help`), which writes an explanation file for
the reader. Internal modules (ingestion, agents, pipeline, renderers)
exist to serve that command and may change without notice.
"""
__version__ = "0.1.0"

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    from .cli import main as _cli_main
    return _cli_main(argv)
