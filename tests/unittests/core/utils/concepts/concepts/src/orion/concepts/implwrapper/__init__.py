# -*- coding: utf-8 -*-
# flake8: noqa
from orion.concepts.base import BaseConcept


class ImplWrapper(BaseConcept):
    """Define a wrapper for the local concept"""

    implementation_module = "orion.concepts.implwrapper"

    def __init__(self, **concept):
        """Initialize the local concept"""
        self.concept = None

        super(ImplWrapper, self).__init__(concept=concept)
