"""Submit bundles and retain recoverable histories of remote execution.

Use ``manager.JobManager`` to submit, inspect, synchronize, and retry attempts.
``store.JobStore`` reads their local JSON records, represented by
``records.JobRecord``. Configure remote machines and MLFF software through
``resources.RemoteResource`` and ``resources.RuntimeEnvironment``.

The ``temper.entrypoints.job`` module supplies the ``temper job`` commands.
Remote dependencies are loaded only when the manager needs
``dispatcher.DispatcherBackend``;
reading local history does not require dpdispatcher or a remote connection.
"""
