# -*- coding: utf-8 -*-
# flake8: noqa
from orion.concepts.base import BaseConcept


class SubConcept(BaseConcept):
    """Define an implementation for cascading concepts creation"""

    def __init__(self, **nested_concept):
        """Initialize the concept"""
        super(SubConcept, self).__init__(**nested_concept)
