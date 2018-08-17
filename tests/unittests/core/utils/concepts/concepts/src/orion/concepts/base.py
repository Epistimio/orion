# -*- coding: utf-8 -*-
# flake8: noqa
from abc import ABCMeta

from orion.core.utils import Concept


class BaseConcept(Concept, metaclass=ABCMeta):
    """Define a dummy base concept with for test purposes"""

    name = "BaseConcept"
    implementation_module = "orion.concepts"

    def __init__(self, **kwargs):
        """Initialize the concept"""
        super(BaseConcept, self).__init__(**kwargs)


class BaseConceptWrapper(BaseConcept):
    """Define a wrapper for the concept"""

    def __init__(self, concept):
        """Initialize the concept"""
        self.concept = None

        super(BaseConceptWrapper, self).__init__(concept=concept)
