"""KnowMoreDiRT public package interface.

Only the two intended public functions are exported:

- :func:`initialize`
- :func:`question`
"""

from .public import initialize, question

__all__ = ["initialize", "question"]
