# -*- coding: utf-8 -*-
"""
:mod:`orion.algo.mutate_functions` --
Different mutate functions
===========================================================================================

.. module:: mutate_functions
    :platform: Unix
    :synopsis: Implement evolution to exploit configurations with fixed resource efficiently

"""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def default_mutate(self, winner_id, loser_id, multiply_factor, add_factor):
    """Get a default mutate function"""
    select_genes_key_list = np.random.choice(self.search_space_remove_fidelity,
                                             self.eves.nums_mutate_gene,
                                             replace=False)
    self.copy_winner(winner_id, loser_id)
    for i, _ in enumerate(select_genes_key_list):
        lower_bound = -np.inf
        upper_bound = np.inf
        if self.space.values()[select_genes_key_list[i]].type == "real":
            if self.space.values()[select_genes_key_list[i]].prior.name == "uniform" or \
               self.space.values()[select_genes_key_list[i]].prior.name == "loguniform":
                lower_bound = self.space.values()[select_genes_key_list[i]].prior.a
                upper_bound = self.space.values()[select_genes_key_list[i]].prior.b

            factors = (1.0 / multiply_factor + (multiply_factor - 1.0 / multiply_factor) *
                       np.random.random())
            if lower_bound <= \
                self.eves.population[select_genes_key_list[i]][loser_id] * factors \
                           <= upper_bound:
                self.eves.population[select_genes_key_list[i]][loser_id] *= factors
            elif lower_bound > \
                    self.eves.population[select_genes_key_list[i]][loser_id] * factors:
                self.eves.population[select_genes_key_list[i]][loser_id] = \
                    lower_bound + self.eves.volatility * np.random.random()
            else:
                self.eves.population[select_genes_key_list[i]][
                    loser_id] = upper_bound - self.eves.volatility * np.random.random()
        elif self.space.values()[select_genes_key_list[i]].type == "integer":
            if self.space.values()[select_genes_key_list[i]].prior.name == "uniform" or \
               self.space.values()[select_genes_key_list[i]].prior.name == "loguniform":
                lower_bound = self.space.values()[select_genes_key_list[i]].prior.a
                upper_bound = self.space.values()[select_genes_key_list[i]].prior.b

            factors = int(add_factor * (2 * np.random.randint(2) - 1))
            if lower_bound <= \
                self.eves.population[select_genes_key_list[i]][loser_id] + factors \
                           <= upper_bound:
                self.eves.population[select_genes_key_list[i]][loser_id] += factors
            elif lower_bound > \
                    self.eves.population[select_genes_key_list[i]][loser_id] + factors:
                self.eves.population[select_genes_key_list[i]][loser_id] = int(lower_bound)
            else:
                self.eves.population[select_genes_key_list[i]][loser_id] = int(upper_bound)
        elif self.space.values()[select_genes_key_list[i]].type == "categorical":
            sample_index = \
                np.where(np.random.multinomial(1,
                                               list(self.space.values()
                                                    [select_genes_key_list[
                                                        i]].get_prior)) == 1)[0][0]
            self.eves.population[select_genes_key_list[i]][loser_id] = \
                self.space.values()[select_genes_key_list[i]].categories[sample_index]

    self.eves.performance[loser_id] = -1
