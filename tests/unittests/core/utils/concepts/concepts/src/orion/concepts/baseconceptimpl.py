# -*- coding: utf-8 -*-

from orion.concepts.base import BaseConcept


class BaseConceptImpl(BaseConcept):

    def __init__(self, arg=0):
        super(BaseConceptImpl, self).__init__(arg=arg)
