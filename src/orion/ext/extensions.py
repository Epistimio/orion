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

    parent: optional
        Used to specify a hierachy of callbacks for debugging
    """
    def __init__(self, name, deferred=False, parent=None) -> None:
        self.handlers = []
        self.deferred_calls = []
        self.name = name
        self.parent = parent
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
                fun(*args, _parent=self.parent, **kwargs)
            except Exception as err:
                if self.manager:
                    self.manager.broadcast(self.name, fun, err, args=(args, kwargs))

    def execute(self):
        """Execute all our deferred handlers if any"""
        for args, kwargs in self.deferred_calls:
            self._execute(args, kwargs)


class OrionExtensionManager:
    """Manages third party extensions for Orion"""

    def __init__(self):
        self._events = {}

        self._get_event('error')
        self._get_event('start_experiment')
        self._get_event('new_trial')
        self._get_event('end_trial')
        self._get_event('end_experiment')

    def __getattribute__(self, name):
        if name not in ('register', 'unregister'):
            return self._get_event(name)
       
        return super().__getattribute__(name)

    def _get_event(self, key):
        """Retrieve or generate a new event delegate"""
        delegate = self._events.get(key)

        if delegate is None:
            delegate = EventDelegate(key)
            delegate.manager = self
            self._events[key] = delegate

        return delegate

    def register(self, ext):
        """Register a new extensions"""
        for name, delegate in self._events.items():
            if hasattr(ext, name):
                delegate.add(getattr(ext, name))

    def unregister(self, ext):
        """Remove an extensions if it was already registered"""
        for name, delegate in self._events.items():
            if hasattr(ext, name):
                delegate.remove(getattr(ext, name))


class OrionExtension:
    """Base orion extension interface you need to implement"""

    def error(self, *args, **kwargs):
        """Called when a error occur during the optimization process"""
        return

    def start_experiment(self, *args, **kwargs):
        """Called at the begin of the optimization process before the worker starts"""
        return

    def new_trial(self, *args, **kwargs):
        """Called when the trial starts with a new configuration"""
        return

    def end_trial(self, *args, **kwargs):
        """Called when the trial finished"""
        return

    def end_experiment(self, *args, **kwargs):
        """Called at the end of the optimization process after the worker exits"""
        return

