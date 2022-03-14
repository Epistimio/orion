from __future__ import annotations
from typing import *



# - Registry
# - RegistryMapping
# - Algo:
#   - Refactor _trials_info -> Registry
#   - state_dict/set_state
# - AlgoWrapper
#   - Init
#   - suggest/observe
#   - state_dict/set_state
#   - is_done
# - _instantiate_algo has to be changed to look like the `create_algo` from below
# - Check-out Hyperband + PBT
# 	- Keep get_id, gets ID of the transformed trial (without reversing the transform, original space)
# - Commencer avec HyperBand ou PBT, vu que ce sont les plus chiants.



class Registry:
  	def __init__(self):
      	self._trials_info: dict[str, Trial] = {}
          
    def __contains__(self, trial: str | Trial) -> bool:
        trial_id = trial.id if isinstance(trial, Trial) else trial
        return trial_id in self._trials_info

    def __getitem__(self, item: str) -> Trial:
        if not isinstance(item, str):
            raise KeyError(item)
        return self._trials_info[item]

    def state_dict(self):
      return

    def set_state(self, statedict):
      pass
        
	def has_observed():
    	pass
      
    def has_suggested(trial: Trial) -> bool:
    	return trial in self

    def get_trial(self, id: str) -> Trial:
    	return
      
    def register(self, trial: Trial) -> str:
      	return id

      
class RegistryMapping:
	def __init__(self, orignal_registry, transformed_registry):
		self.orignal_registry
        self.original_registery = original_registery
        self.transformed_registry = transformed_registry
        self._mapping: dict[str, set[str]] = defaultdict(set)

    def state_dict(self):
      return
    
    def set_state(self, statedict):
      pass
    
    def __contains__(self, trial: Trial):
        return trial in self.original_registry
	
    def __getitem__(self, item: Trial) -> list[Trial]:
      	if item.id not in self._mapping:
          return []
        return [self.transformed_registry.get_trial(transformed_id) for transformed_id in self._mapping[item.id]]

    def get_trials(self, original_trial: Trial) -> list[Trial]:
        return self[original_trial]
        return list(self._mapping.get(original_trial, set()))
      
    def register(self, original_trial: Trial, transformed_trial: Trial) -> str:
        # NOTE: Peut-être pas .id, faut voir comment on compute le ID.
        self._mapping[original_trial.id].add(transformed_trial.id)
      	return self.original_registry.register(original_trial)
       
        
def create_algo(algo_constructor: type[Algorithm], space: Space, **algorithm_config) -> Algorithm:
   requirements = backward.get_algo_requirements(algo_constructor) 
   transformed_space = build_required_space(space, **requirements)
   algorithm = algo_constructor(transformed_space, **algorithm_config)
   return SpaceTransformAlgoWrapper(algorithm, space)


class BaseAlgorithm(ABC):
	def __init__(self, space: Space, **configuration):
  		self._space = space
      	self._registry = Registry()
        self._kwargs = configuration
Algorithm = BaseAlgorithm


class SpaceTransformAlgoWrapper(Algorithm):
  # ...
  
	def __init__(self, algorithm: Algorithm, space: Space):
    	super().__init__(space=space)
        self.algorithm = algorithm
        self.registry_mapping = RegistryMapping(original_registry=self.registry, transformed_registry=self.algorithm.registry)
  
  	@property
    def transformed_space(self) -> Space:
        return self.algo.space
  
  	@property
  	def original_space(self) -> Space:
        return self.space
  
	def observe(self, trials: list[Trial]) -> None:
        """Observe evaluated trials.

        .. seealso:: `orion.algo.base.BaseAlgorithm.observe`
        """
        transformed_trials = []
        for trial in trials:
            self._verify_trial(trial, space=self.space)
            # AVANT: transformed_trials.append(self.transformed_space.transform(trial))
            transformed_trials.extend(self.registry_mapping.get_trials(trial))
        self.algorithm.observe(transformed_trials)
      
	def suggest(self, num: int) -> list[Trial] | None:
        """ Suggest some new trials if possible. """
        transformed_trials = self.algorithm.suggest(num=num)
        trials = []
        for transformed_trial in transformed_trials:
            original: Trial = self.transformed_space.reverse(transformed_trial)
            if original not in self.registry:
                trials.append(original)
            self.registry_mapping.register(original, transformed_trial)
            # self.registry_mapping[original].append(transformed_trial)
		return trials
      
    @property
    def is_done(self):
        """Whether the algorithm is done and will not make further suggestions.

        Return True, if an algorithm holds that there can be no further improvement.
        By default, the cardinality of the specified search space will be used to check
        if all possible sets of parameters has been tried.
        """
        if self.n_suggested >= self.original_space.cardinality:
            return True

        if self.n_observed >= getattr(self, "max_trials", float("inf")):
            return True

        return self.algorithm.is_done


class MappingRegistry(Registry):
	def __init__(self):
    	super().__init__()
        self._converted_trials = dict()
        self._original_trials = dict()

    def statedict(self):
      return
    
    def setstate(self, statedict):
      pass
        
	def has_observed(trial: Trial) -> bool:
        transformed_trial = self.transformed_space.transform(trial)
        return self.original_registry.has_observed(transformed_trial)
    	pass
      
    def has_suggested():
    	pass
    
    def get_trial():
    	pass
      
    def register():
      	return converted_trial