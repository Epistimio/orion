# -*- coding: utf-8 -*-
from orion.core.utils import Concept
from abc import ABCMeta


class BaseConcept(Concept, metaclass=ABCMeta):
    """Define a dummy concept with local implementation for test purposes"""

    name = "LocalConcept"
    implementation_module = "orion.concepts"

    def __init__(self, **kwargs):
        """Initialize the concept"""
        super(BaseConcept, self).__init__(**kwargs)


class BaseConceptWrapper(BaseConcept):
    """Define a wrapper for the local concept"""

    def __init__(self, concept):
        """Initialize the local concept"""
        self.concept = None

        super(BaseConceptWrapper, self).__init__(concept=concept)
