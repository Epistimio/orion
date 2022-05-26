import falcon


class QueryRoute:
    """Base route handling, makes sure status is always set"""

    def on_post(self, req: falcon.Request, resp: falcon.Response) -> None:
        """Force status to be set, and send back an error on exception"""
        try:
            resp.media = dict()
            self.on_post_request(req, resp)

            resp.status = falcon.HTTP_200
            resp.media["status"] = 0
            return

        except Exception as err:
            resp.status = falcon.HTTP_500
            resp.media["status"] = 1
            resp.media["error"] = type(err)

    def on_post_request(self, req: falcon.Request, resp: falcon.Response) -> None:
        """Request specific handling to implement here"""
        raise NotImplementedError()


class OrionService:
    """Orion service API implementation"""

    def __init__(self) -> None:
        self.app = falcon.App()
        self.app.add_route("/experiment", OrionService.NewExperiment())
        self.app.add_route("/suggest", OrionService.Suggest())
        self.app.add_route("/observe", OrionService.Observe())
        self.app.add_route("/is_done", OrionService.IsDone())

    class NewExperiment(QueryRoute):
        """Create or set the experiment"""

        def on_post_request(self, req: falcon.Request, resp: falcon.Response) -> None:
            resp.status = falcon.HTTP_200
            resp.media = dict(experiment_id=1123)

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

    class Heatbeat(QueryRoute):
        """Notify the server than the trial is still running"""

        def on_post_request(self, req: falcon.Request, resp: falcon.Response) -> None:
            pass

    def run(self, hostname: str, port: int) -> None:
        """Run the server forever"""
        try:
            from wsgiref.simple_server import make_server

            with make_server(hostname, port, self.app) as httpd:
                print(f"Serving on port {port}...")
                httpd.serve_forever()

        except KeyboardInterrupt:
            print("Stopping server")


def main(hostname: str = "", port: int = 8080) -> None:
    service = OrionService()
    service.run(hostname, port)


if __name__ == "__main__":
    main()
