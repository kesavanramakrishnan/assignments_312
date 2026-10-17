"""Shrink figure canvases at save time so text stays legible when a figure is scaled to the page width."""
import matplotlib as mpl
import matplotlib.figure as mf

WIDE_CANVAS_IN = 11.   # strips of two or more panels
GRID_CANVAS_IN = 8.    # multi-row grids
SINGLE_CANVAS_IN = 6.2  # single panels
mpl.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "figure.titlesize": 10})
_original_savefig = mf.Figure.savefig


def _savefig(self, *args, **kwargs):
    width, height = self.get_size_inches()
    target = WIDE_CANVAS_IN if width / height >= 2 else GRID_CANVAS_IN if width >= 9 else SINGLE_CANVAS_IN
    if width > target and not getattr(self, "_shrunk", False):
        self.set_size_inches(target, height * target / width)
        self._shrunk = True
        rect = [0, .05 if self.legends else 0, 1, .95 if self._suptitle is not None else 1]
        try:
            self.tight_layout(rect=rect)
        except Exception:
            pass
    return _original_savefig(self, *args, **kwargs)


mf.Figure.savefig = _savefig
