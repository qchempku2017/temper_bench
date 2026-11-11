"""Attempts presented for explicit abandonment by the user."""

from pydantic import BaseModel, ConfigDict


class AbandonmentCandidate(BaseModel):
    """Describe one unresolved attempt selected for user review.

    Parameters
    ----------
    **data : object
        Values for the attributes below.

    Attributes
    ----------
    bundle_id : str
        Full scientific identity of the bundle containing the attempt.
    attempt : str
        Attempt directory relative to the local job store.
    resource : str
        Saved execution profile name identifying where the job may exist.
    job_id : str or None
        Known remote job identifiers, separated by commas if more than one
        is recorded. None means no identifier was saved, not that no job ran.
    reason : str
        Explanation of why the attempt remains unresolved.
    review_token : str
        Fingerprint of the saved attempt at review time. Used to reject a
        changed selection before abandonment; callers must pass it unchanged.
        It is not a remote job identifier or an authentication credential.
    """

    model_config = ConfigDict(frozen=True)

    bundle_id: str
    attempt: str
    resource: str
    job_id: str | None
    reason: str
    review_token: str
