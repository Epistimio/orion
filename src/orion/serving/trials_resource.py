# -*- coding: utf-8 -*-
"""
:mod:`orion.serving.trials_resource` -- Module responsible for the trials/ REST endpoint
========================================================================================

.. module:: trials_resource
   :platform: Unix
   :synopsis: Serves all the requests made to trials/ REST endpoint
"""
import json

from falcon import Request, Response
import falcon.status_codes

from orion.core.io import experiment_builder
from orion.core.utils.exceptions import NoConfigurationError
from orion.core.worker.trial import Trial
from orion.serving.parameters import verify_query_parameters
from orion.storage.base import get_storage

SUPPORTED_PARAMETERS = {
    'ancestors': 'bool',
    'status': 'str',
    'version': 'int'
}


class TrialsResource(object):
    """Serves all the requests made to trials/ REST endpoint"""

    def __init__(self):
        self.storage = get_storage()

    def on_get_trials_for_experiment(self, req: Request, resp: Response, experiment_name: str):
        """
        Handle GET requests for trials/:experiment where ``experiment`` is
        the user-defined name of the experiment.
        """
        error_message = verify_query_parameters(req.params, SUPPORTED_PARAMETERS)
        if error_message:
            response = {
                "message": error_message
            }
            resp.status = falcon.HTTP_400
            resp.body = json.dumps(response)
            return

        try:
            status = req.params['status'] if 'status' in req.params else None
            if status and status not in Trial.allowed_stati:
                message = f"Invalid status value. Expected one of {list(Trial.allowed_stati)}"
                resp.status = falcon.HTTP_400
                resp.body = json.dumps({'message': message})
                return


            version = int(req.params['version']) if 'version' in req.params else None
            with_ancestors = \
                req.params['ancestors'].lower() == 'true' if 'ancestors' in req.params else False

            experiment = experiment_builder.build_view(experiment_name, version)
            if version and experiment.version != version:
                resp.status = falcon.HTTP_404
                resp.body = json.dumps(
                    {
                        'message': f"Experiment '{experiment_name}' has no version '{version}'"
                    }
                )
                return

            if status:
                trials = experiment.fetch_trials_by_status(status, with_ancestors)
            else:
                trials = experiment.fetch_trials(with_ancestors)

            response = []
            for trial in trials:
                response.append({
                    'id': trial.id
                })

            resp.body = json.dumps(response)

        except NoConfigurationError:
            response = {
                'message': f"Experiment '{experiment_name}' does not exist"
            }
            resp.status = falcon.HTTP_404
            resp.body = json.dumps(response)

        except ValueError as exception:
            resp.body = json.dumps(str(exception))
            resp.status = falcon.HTTP_400
