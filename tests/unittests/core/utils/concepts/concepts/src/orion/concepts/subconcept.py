# -*- coding: utf-8 -*-

from orion.concepts.base import BaseConcept


class SubConcept(BaseConcept):

    def __init__(self, **nested_concept):
        super(SubConcept, self).__init__(**nested_concept)
