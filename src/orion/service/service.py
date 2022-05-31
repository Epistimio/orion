import falcon

import os
import traceback


from orion.storage.legacy import Legacy
from orion.core.io.database.mongodb import MongoDB


class QueryRoute:
    """Base route handling, makes sure status is always set


    """

    def on_post(self, req: falcon.Request, resp: falcon.Response) -> None:
        """Force status to be set, and send back an error on exception"""
        try:
            resp.media = dict()
            self.on_post_request(req, resp)

            resp.status = falcon.HTTP_200
            resp.media["status"] = 0
            return

        except Exception as err:
            traceback.print_exc()

            resp.status = falcon.HTTP_200
            resp.media["status"] = 1
            resp.media["error"] = str(err)

    def on_post_request(self, req: falcon.Request, resp: falcon.Response) -> None:
        """Request specific handling to implement here"""
        raise NotImplementedError()

# this could be stored in the mongodb itself
# the users only know they token, the password is never shared outside our infra
tok_to_user = {
    "Tok1": ('User1', 'Pass1'),
    "Tok2": ('User2', 'Pass2'),
    "Tok3": ('User3', 'Pass3'),
}


class OrionService:
    """Orion service API implementation"""

    def __init__(self) -> None:
        self.app = falcon.App()
        OrionService.add_routes(self.app)

    @staticmethod
    def add_routes(app: falcon.App) -> None:
        """Add the routes to a given falcon App"""
        app.add_route("/experiment", OrionService.NewExperiment())
        app.add_route("/suggest", OrionService.Suggest())
        app.add_route("/observe", OrionService.Observe())
        app.add_route("/is_done", OrionService.IsDone())
        app.add_route("/heartbeat", OrionService.Heartbeat())

    class NewExperiment(QueryRoute):
        """Create or set the experiment"""

        def on_post_request(self, req: falcon.Request, resp: falcon.Response) -> None:
            msg = req.get_media()
            token = msg.pop('token', None)

            if token is None:
                resp.status = falcon.HTTP_401
                return

            credientials = tok_to_user.get(token)
            if credientials is None:
                resp.status = falcon.HTTP_401
                return

            # force the singleton to use this database with our credentials
            # bypassing the config
            print(f"{os.getpid()} Connecting to db")

            # this bypass the Database singleton logic
            db = MongoDB(
                name='orion',
                host='192.168.0.116',
                port=8124,
                username=credientials[0],
                password=credientials[1]
            )

            # this bypass the Storage singleton logic
            storage = Legacy(
                database_instance=db,
                # Skip setup, this is a shared database
                # we might not have the permissions to do the setup
                # and the setup should already be done anyway
                setup=False
            )

            print(msg)
            from orion.client import build_experiment
            client = build_experiment(**msg, storage_instance=storage)

            resp.status =  falcon.HTTP_200
            resp.media = dict(experiment_id=str(client.id))

    class Suggest(QueryRoute):
        """Suggest new trials for a given experiment"""

        def on_post_request(self, req: falcon.Request, resp: falcon.Response) -> None:
            resp.media["trials"] = []

    class Observe(QueryRoute):
        """Observe results for a given trial"""

        def on_post_request(self, req: falcon.Request, resp: falcon.Response) -> None:
            pass

    class IsDone(QueryRoute):
        """Query the status of a given experiment"""

        def on_post_request(self, req: falcon.Request, resp: falcon.Response) -> None:
            pass

    class Heartbeat(QueryRoute):
        """Notify the server than the trial is still running"""

        def on_post_request(self, req: falcon.Request, resp: falcon.Response) -> None:
            pass

    def run(self, hostname: str, port: int) -> None:
        """Run the server forever"""
        try:
            from wsgiref.simple_server import make_server

            with make_server(hostname, port, self.app) as httpd:
                print(f"{os.getpid()}: Serving on port {port}...")
                httpd.serve_forever()

        except KeyboardInterrupt:
            print("Stopping server")


def main(hostname: str = "", port: int = 8080) -> None:
    service = OrionService()
    service.run(hostname, port)


if __name__ == "__main__":
    main()
