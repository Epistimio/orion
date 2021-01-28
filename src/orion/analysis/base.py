import itertools

from orion.core.worker.transformer import build_required_space


def flatten_params(space, params=None):
    keys = set(space.keys())
    flattened_keys = set(
        build_required_space(
            space,
            dist_requirement="linear",
            type_requirement="numerical",
            shape_requirement="flattened",
        ).keys()
    )

    if params is None:
        return sorted(flattened_keys)

    flattened_params = []
    for param in params:
        if param not in flattened_keys and param not in keys:
            raise ValueError(
                f"Parameter {param} not contained in space: {flattened_keys}"
            )
        elif param not in flattened_keys and param in keys:
            dim = space[param]
            flattened_params += [
                f'{dim.name}[{",".join(map(str, index))}]'
                for index in itertools.product(*map(range, dim.shape))
            ]
        else:
            flattened_params.append(param)

    return flattened_params
