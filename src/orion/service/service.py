import logging
import os
import traceback

import falcon

from orion.service.auth import AuthenticationService
from orion.service.broker.broker import RequestContext, ServiceContext
from orion.service.broker.local import LocalExperimentBroker

log = logging.getLogger(__file__)


class QueryRoute:
    """Base route handling, makes sure status is always set"""

    def __init__(self, broker, auth) -> None:
        self.service = ServiceContext()
        self.broker = broker
        self.auth = auth

    def on_post(self, req: falcon.Request, resp: falcon.Response) -> None:
        """Force status to be set, and send back an error on exception"""
        resp.media = dict()

        try:
            msg = req.get_media()
            token = msg.pop("token", None)

            if token is None:
                resp.status = falcon.HTTP_401
                return

            credentials = self.auth.authenticate(token)
            if credentials is None:
                resp.status = falcon.HTTP_401
                return

            ctx = RequestContext(
                self.service,
                credentials[0],
                credentials[1],
                data=msg,
                token=token,
                request=req,
                response=resp,
            )

            self.on_post_request(ctx)
            resp.status = falcon.HTTP_200
            return

        except Exception as err:
            log.error(traceback.format_exc())

            resp.status = falcon.HTTP_200
            resp.media["status"] = 1
            resp.media["error"] = str(err)

    def on_post_request(self, ctx: RequestContext) -> None:
        """Request specific handling to implement here"""
        raise NotImplementedError()


class OrionService:
    """Orion service API implementation"""

    def __init__(self) -> None:
        self.app = falcon.App()
        self.broker = LocalExperimentBroker() #
        self.auth = AuthenticationService()
        OrionService.add_routes(self.app, self.broker, self.auth)

    @staticmethod
    def add_routes(app: falcon.App, broker, auth) -> None:
        """Add the routes to a given falcon App"""
        app.add_route("/experiment", OrionService.NewExperiment(broker, auth))
        app.add_route("/suggest", OrionService.Suggest(broker, auth))
        app.add_route("/observe", OrionService.Observe(broker, auth))
        app.add_route("/is_done", OrionService.IsDone(broker, auth))
        app.add_route("/heartbeat", OrionService.Heartbeat(broker, auth))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        log.info("Shutting down")
        self.broker.stop()

    class NewExperiment(QueryRoute):
        """Create or set the experiment"""

        def on_post_request(self, ctx: RequestContext) -> None:
            result = self.broker.new_experiment(ctx)

            ctx.response.status = falcon.HTTP_200
            ctx.response.media = result

    class Suggest(QueryRoute):
        """Suggest new trials for a given experiment"""

        def on_post_request(self, ctx: RequestContext) -> None:
            result = self.broker.suggest(ctx)
            ctx.response.media = result

    class Observe(QueryRoute):
        """Observe results for a given trial"""

        def on_post_request(self, ctx: RequestContext) -> None:
            pass

    class IsDone(QueryRoute):
        """Query the status of a given experiment"""

        def on_post_request(self, ctx: RequestContext) -> None:
            pass

    class Heartbeat(QueryRoute):
        """Notify the server than the trial is still running"""

        def on_post_request(self, ctx: RequestContext) -> None:
            pass

    def run(self, hostname: str, port: int) -> None:
        """Run the server forever"""
        try:
            from wsgiref.simple_server import make_server

            with make_server(hostname, port, self.app) as httpd:
                print(f"{os.getpid()}: Serving on port {port}...")
                httpd.serve_forever()

        except KeyboardInterrupt:
            pass


def main(address: str = "localhost", port: int = 8080) -> None:
    logging.basicConfig(level=logging.DEBUG)

    with OrionService() as service:
        service.run(address, port)


if __name__ == "__main__":
    main()
