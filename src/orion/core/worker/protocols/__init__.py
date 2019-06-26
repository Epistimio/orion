from orion.core.worker.protocols.debug import DebugProtocol
from orion.core.worker.protocols.track import TrackProtocol

_protocols = {
    '__default__': DebugProtocol,
    'debug': DebugProtocol,
    'track': TrackProtocol
}


def make_protocol(uri, **kwargs):
    values = uri.split(':', maxsplit=1)

    proto = _protocols.get(values[0])
    if proto is None:
        proto = _protocols.get('__default__')

    return proto(uri=values[1:], **kwargs)
