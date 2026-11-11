"""Per-bundle explanations of submission checks and actions."""

from typing import Literal

from pydantic import BaseModel


class SubmissionDecision(BaseModel):
    """Describe what submitting a bundle would do, or has done.

    Parameters
    ----------
    **data : object
        Values for the attributes below.

    Attributes
    ----------
    bundle_id : str
        Full scientific bundle identity, or the input path if loading failed.
    action : str
        ``new``, ``retry``, or ``resume`` describes a preview. ``submitted``
        and ``resumed`` mean a launch was confirmed. ``monitor``, ``succeeded``,
        and ``cancelled`` need no new launch. ``blocked`` means evidence or
        inputs are insufficient; ``error`` means an operation failed.
    reason : str
        Explanation of the decision and, when needed, what to check next.
    attempt : str or None
        Attempt directory relative to the store, absent before a first launch
        or when the input cannot be read. A retry preview identifies the
        previous attempt; after submission this identifies the new attempt.
    resource : str or None
        Execution profile name, absent when no profile could be selected.
    """

    bundle_id: str
    action: Literal[
        "new", "retry", "resume", "submitted", "resumed", "monitor",
        "succeeded", "cancelled", "blocked", "error",
    ]
    reason: str
    attempt: str | None = None
    resource: str | None = None
