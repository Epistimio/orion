import logging

logger = logging.getLogger(__name__)


NO_CREDENTIAL = None, None


class AuthenticationService:
    def __init__(self) -> None:
        self.tok_to_user = {
            "Tok1": ("User1", "Pass1"),
            "Tok2": ("User2", "Pass2"),
            "Tok3": ("User3", "Pass3"),
        }

    def authenticate(self, token):
        username, password = self.tok_to_user.get(token, NO_CREDENTIAL)

        logger.debug("Authenticated %s => %s", token, username)
        return username, password
