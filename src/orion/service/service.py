"""Boilerplate for Falcon server

All the logic is implemented through the broker interface
"""
import logging
import os
import traceback

import falcon

from orion.service.auth import NO_CREDENTIAL, AuthenticationService
from orion.service.broker.broker import RequestContext, ServiceContext
from orion.service.broker.local import LocalExperimentBroker

log = logging.getLogger(__file__)


class AuthenticationError(Exception):
    pass


class QueryRoute:
    """Base route handling, makes sure status is always set"""

    def __init__(self, ctx: ServiceContext) -> None:
        self.service = ctx

    @property
    def broker(self):
        """Return orion experiment broker"""
        return self.service.broker

    @property
    def auth(self):
        """returns the authentication service"""
        return self.service.auth

    def authenticate(self, token):
        """Authenticate a user given its API token"""
        if token is None:
            raise AuthenticationError("Missing authentication token")

        credentials = self.auth.authenticate(token)
        if credentials == NO_CREDENTIAL:
            raise AuthenticationError("Incorrect authentication token")

        return credentials

    def on_post(self, req: falcon.Request, resp: falcon.Response) -> None:
        """Force status to be set, and send back an error on exception"""
        resp.media = dict()

        try:
            msg = req.get_media()

            token = msg.pop("token", None)
            credentials = self.authenticate(token)

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

    def __init__(self, ctx) -> None:
        self.ctx = ctx
        ctx.broker = LocalExperimentBroker(ctx)
        ctx.auth = AuthenticationService()

        self.app = falcon.App()
        OrionService.add_routes(self.app, ctx)

    @staticmethod
    def add_routes(app: falcon.App, ctx) -> None:
        """Add the routes to a given falcon App"""
        app.add_route("/experiment", OrionService.NewExperiment(ctx))
        app.add_route("/suggest", OrionService.Suggest(ctx))
        app.add_route("/observe", OrionService.Observe(ctx))
        app.add_route("/is_done", OrionService.IsDone(ctx))
        app.add_route("/heartbeat", OrionService.Heartbeat(ctx))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        log.info("Shutting down")
        self.ctx.broker.stop()

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
            result = self.broker.observe(ctx)
            ctx.response.media = result

    class IsDone(QueryRoute):
        """Query the status of a given experiment"""

        def on_post_request(self, ctx: RequestContext) -> None:
            result = self.broker.is_done(ctx)
            ctx.response.media = result

    class Heartbeat(QueryRoute):
        """Notify the server than the trial is still running"""

        def on_post_request(self, ctx: RequestContext) -> None:
            result = self.broker.heartbeat(ctx)
            ctx.response.media = result

    def run(self, hostname: str, port: int) -> None:
        """Run the server forever"""
        try:
            from wsgiref.simple_server import make_server

            with make_server(hostname, port, self.app) as httpd:
                print(f"{os.getpid()}: Serving on port {port}...")
                httpd.serve_forever()

        except KeyboardInterrupt:
            pass


def main(
    address: str = "localhost", port: int = 8080, servicectx=ServiceContext()
) -> None:

    logging.basicConfig(
        format="%(asctime)-15s::%(levelname)s::%(name)s::%(message)s",
        level=logging.DEBUG,
        force=True,
    )

    with OrionService(servicectx) as service:
        service.run(address, port)


if __name__ == "__main__":
    main()
