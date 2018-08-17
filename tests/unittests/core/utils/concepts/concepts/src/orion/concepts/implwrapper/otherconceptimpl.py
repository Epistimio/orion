# -*- coding: utf-8 -*-
# flake8: noqa
from orion.concepts.base import BaseConcept


class OtherConceptImpl(BaseConcept):
    """Define a concept implementation inside a submodule"""

    def __init__(self, arg=0):
        """Initialize the concept"""
        super(OtherConceptImpl, self).__init__(arg=arg)
