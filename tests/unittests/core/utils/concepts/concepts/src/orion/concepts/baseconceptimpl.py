# -*- coding: utf-8 -*-
# flake8: noqa
from orion.concepts.base import BaseConcept


class BaseConceptImpl(BaseConcept):
    """Define a dummy implementation of the base concept"""

    def __init__(self, arg=0):
        """Initialize the dummy implementation"""
        super(BaseConceptImpl, self).__init__(arg=arg)
