import sys
from warnings import warn

from rich.console import Console as RichConsole
from rich.markdown import Markdown
from tqdm import tqdm as tqdm_base, trange as trange_base


class Console(RichConsole):
    def __init__(self, *args, **kwargs):
        self.quiet = False
        super().__init__(*args, **kwargs)

    def set_quiet(self, quiet=True):
        self.quiet = quiet

    def print(self, *args, **kwargs):
        if not self.quiet:
            super().print(*args, **kwargs)

    def warn(self, *args, **kwargs):
        warn(*args, **kwargs)

    def _h(self, text, level):
        """Print header with given level (e.g. `h1` for `level = 1`)."""
        if not isinstance(level, int) or not 1 <= level <= 6:
            raise ValueError(f"Invalid header level: {level}")
        md = f"{'#'*level} {text}"
        self.print(Markdown(md))

    def h1(self, text):
        self._h(text, 1)

    def h2(self, text):
        self._h(text.upper(), 2)

    def h3(self, text):
        self._h(text, 3)

    def h4(self, text):
        self._h(text, 4)

    tqdm_default_kwargs = {
        "file": sys.stdout,
        "disable": None,
    }

    class _Tqdm(tqdm_base):
        def __init__(self, *args, outer: "Console", **kwargs):
            if outer is not None and outer.quiet:
                kwargs.update(disable=True)

            super().__init__(*args, **{**outer.tqdm_default_kwargs, **kwargs})

    def tqdm(self, *args, **kwargs):
        return self._Tqdm(*args, outer=self, **kwargs)

    def trange(self, *args, **kwargs):
        if self.quiet:
            kwargs.update(disable=True)
        return trange_base(*args, **{**self.tqdm_default_kwargs, **kwargs})


console = Console()
