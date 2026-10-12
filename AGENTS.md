# Docstring preferences

Write docstrings for the person using the object. Explain what a value contains,
where its paths are resolved, when it is absent, and what that absence means for
the caller. Describe return values and side effects concretely. Avoid compressed
implementation jargon. Preserve user-written explanations when updating APIs.

For example, `MLFFTrainBundle.files` is absent on a newly constructed local bundle;
packaging fills it with copied input paths and output locations, and the runtime
cannot train or evaluate an unpackaged bundle.
