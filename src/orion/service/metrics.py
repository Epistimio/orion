
from orion.core import __version__

from prometheus_client import start_http_server, Summary, Info


info = Info('orion', 'Orion service info')
info.info({
    'version': __version__,
})


def initialize_metrics(port):
    start_http_server(port)
