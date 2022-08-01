class AuthenticationService:
    def __init__(self) -> None:
        self.tok_to_user = {
            "Tok1": ("User1", "Pass1"),
            "Tok2": ("User2", "Pass2"),
            "Tok3": ("User3", "Pass3"),
        }

    def authenticate(self, token):
        return self.tok_to_user.get(token)
