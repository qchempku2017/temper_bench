"""Exceptions distinguishing rejected launches from uncertain
outcomes."""


class SubmissionRejected(RuntimeError):
    """Report a confirmed scheduler or preflight rejection.

    Diagnostics are retained in the attempt's dispatcher state and local
    submission files. A transport exception is never converted to this
    error.

    Parameters
    ----------
    *args : object
        Exception message components, normally a single diagnostic
        string.

    Attributes
    ----------
    args : tuple[object, ...]
        Message components supplied when the rejection was raised.
    """
