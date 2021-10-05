"""Defines extension mechanism for third party to hook into Orion"""


class EventDelegate:
    """Allow extensions to listen to incoming events from Orion.
    Orion broadcasts events which trigger extensions callbacks.

    Parameters
    ----------
    name: str
        name of the event we are creating, this is useful for error reporting

    deferred: bool
        if false events are triggered as soon as broadcast is called
        if true the events will need to be triggered manually
    """

    def __init__(self, name, deferred=False) -> None:
        self.handlers = []
        self.deferred_calls = []
        self.name = name
        self.deferred = deferred
        self.bad_handlers = []
        self.manager = None

    def remove(self, function) -> bool:
        """Remove an event handler from the handler list"""
        try:
            self.handlers.remove(function)
            return True
        except ValueError:
            return False

    def add(self, function):
        """Add an event handler to our handler list"""
        self.handlers.append(function)

    def broadcast(self, *args, **kwargs):
        """Broadcast and event to all our handlers"""
        if not self.deferred:
            self._execute(args, kwargs)
            return

        self.deferred_calls.append((args, kwargs))

    def _execute(self, args, kwargs):
        for fun in self.handlers:
            try:
                fun(*args, **kwargs)
            except Exception as err:
                if self.manager:
                    self.manager.on_extension_error.broadcast(
                        self.name, fun, err, args=(args, kwargs)
                    )

    def execute(self):
        """Execute all our deferred handlers if any"""
        for args, kwargs in self.deferred_calls:
            self._execute(args, kwargs)


class _DelegateStartEnd:
    def __init__(self, start, error, end, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.start = start
        self.end = end
        self.error = error

    def __enter__(self):
        self.start.broadcast(*self.args, **self.kwargs)
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.end.broadcast(*self.args, **self.kwargs)

        if exception_value is not None:
            self.error.broadcast(
                *self.args,
                exception_type,
                exception_value,
                exception_traceback,
                **self.kwargs
            )


class OrionExtensionManager:
    """Manages third party extensions for Orion"""

    def __init__(self):
        self._events = {}
        self._get_event("on_extension_error")

        # -- Trials
        self._get_event("new_trial")
        self._get_event("on_trial_error")
        self._get_event("end_trial")

        # -- Experiments
        self._get_event("start_experiment")
        self._get_event("on_experiment_error")
        self._get_event("end_experiment")

    def experiment(self, *args, **kwargs):
        """Initialize a context manager that will call start/error/end events automatically"""
        return _DelegateStartEnd(
            self.start_experiment,
            self.on_experiment_error,
            self.end_experiment,
            *args,
            **kwargs
        )

    def trial(self, *args, **kwargs):
        """Initialize a context manager that will call start/error/end events automatically"""
        return _DelegateStartEnd(
            self.new_trial, self.on_trial_error, self.end_trial, *args, **kwargs
        )

    def __getattr__(self, name):
        if name in self._events:
            return self._get_event(name)

    def _get_event(self, key):
        """Retrieve or generate a new event delegate"""
        delegate = self._events.get(key)

        if delegate is None:
            delegate = EventDelegate(key)
            delegate.manager = self
            self._events[key] = delegate

        return delegate

    def register(self, ext):
        """Register a new extensions

        Parameters
        ----------
        ext
            object implementing :class`OrionExtension` methods

        Returns
        -------
        the number of calls that was registered
        """
        registered_callbacks = 0
        for name, delegate in self._events.items():
            if hasattr(ext, name):
                delegate.add(getattr(ext, name))
                registered_callbacks += 1

        return registered_callbacks

    def unregister(self, ext):
        """Remove an extensions if it was already registered"""
        unregistered_callbacks = 0
        for name, delegate in self._events.items():
            if hasattr(ext, name):
                delegate.remove(getattr(ext, name))
                unregistered_callbacks += 1

        return unregistered_callbacks


class OrionExtension:
    """Base orion extension interface you need to implement"""

    def on_extension_error(self, name, fun, exception, args):
        """Called when an extension callbakc raise an exception

        Parameters
        ----------
        fun: callable
            handler that raised the error

        exception:
            raised exception

        args: tuple
            tuple of the arguments that were used
        """
        return

    def on_trial_error(
        self, trial, exception_type, exception_value, exception_traceback
    ):
        """Called when a error occur during the optimization process"""
        return

    def new_trial(self, trial):
        """Called when the trial starts with a new configuration"""
        return

    def end_trial(self, trial):
        """Called when the trial finished"""
        return

    def on_experiment_error(
        self, experiment, exception_type, exception_value, exception_traceback
    ):
        """Called when a error occur during the optimization process"""
        return

    def start_experiment(self, experiment):
        """Called at the begin of the optimization process before the worker starts"""
        return

    def end_experiment(self, experiment):
        """Called at the end of the optimization process after the worker exits"""
        return
