# -*- coding: utf-8 -*-
# flake8: noqa
from orion.concepts.base import BaseConcept


class EntryPointConcept(BaseConcept):
    """Define a concept that will be found using an entry point"""

    def __init__(self):
        """Initialize the concept"""
        super(BaseConcept, self).__init__()
