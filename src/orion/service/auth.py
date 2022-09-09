import logging
import secrets

import pymongo

logger = logging.getLogger(__name__)


NO_CREDENTIAL = None, None


class AuthenticationServiceInterface:
    def __init__(self, config) -> None:
        pass

    def authenticate(self, token):
        raise NotImplementedError()


class AuthenticationMongoDB(AuthenticationServiceInterface):
    """Authentication service using mongodb"""

    def __init__(self, config) -> None:
        self.authconfig = config.authentication
        self.mongo = pymongo.MongoClient(
            host=self.authconfig.host,
            port=self.authconfig.port,
            username=self.authconfig.username,
            password=self.authconfig.password,
        )

    def add_user(self, username, password):
        token = secrets.token_hex(32)

        self.mongo[self.authconfig.database].insert_one(
            {
                "username": username,
                "password": password,
                "token": token,
            }
        )

        return token

    def authenticate(self, token):
        user = self.mongo[self.authconfig.database].find_one(
            {
                "token": token,
            },
        )

        # Get access config

        if user is None:
            return NO_CREDENTIAL

        username, password = user.get("username"), user.get("password")
        return username, password
