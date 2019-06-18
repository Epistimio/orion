# -*- coding: utf-8 -*-
import yaml

from orion.core.io.database import Database


def config_checks():
    return _CONFIG_CHECKS


_CONFIG_CHECKS = []


def _register_check(*checklists):
    def wrap(func):
        for checklist in checklists:
            checklist.append(func)

        def wrapped_func(*args):
            func(*args)

        return wrapped_func
    return wrap


@_register_check(_CONFIG_CHECKS)
def check_if_config_present(shared_dict):
    """Check if --config is present... """
    if 'config' in shared_dict and shared_dict['config'] is not None:
        return False, ""
    else:
        return True, "The `--config` argument is missing."


@_register_check(_CONFIG_CHECKS)
def check_if_config_has_database_section(shared_dict):
    """Check if the database section is present... """
    shared_dict.update(**yaml.safe_load(shared_dict['config']))

    if 'database' in shared_dict and shared_dict['database'] is not None:
        return False, ""
    else:
        return True, "No database section inside config file."


@_register_check(_CONFIG_CHECKS)
def check_if_database_section_is_valid(shared_dict):
    """Check if the required information for database creation is present... """
    database = shared_dict['database']

    values = ['type', 'host', 'name']

    for v in values:
        if v not in database:
            True, "Missing section '{}' for database configuration.".format(v)
        elif database[v] is None:
            True, "Missing value for section '{}' in database configuration.".format(v)

    return False, ""


@_register_check(_CONFIG_CHECKS)
def check_database_creation(shared_dict):
    """Check if database of specified type can be create... """
    database = shared_dict['database']
    db_type = database.pop('type')

    try:
        db = Database(of_type=db_type, **database)
    except ValueError as ex:
        return True, str(ex)

    if not db.is_connected:
        True, "Database failed to connect after creation."

    shared_dict['instance'] = db

    return False, ""


@_register_check(_CONFIG_CHECKS)
def check_write(shared_dict):
    """Check if database supports write operation... """
    database = shared_dict['instance']

    try:
        database.write('test', {'index': 'value'})
    except Exception as ex:
        return True, str(ex)

    return False, ""


@_register_check(_CONFIG_CHECKS)
def check_read(shared_dict):
    """Check if database supports read operation... """
    database = shared_dict['instance']

    try:
        result = database.read('test', {'index': 'value'})
    except Exception as ex:
        return True, str(ex)

    value = result[0]['index']
    if value != 'value':
        return True, "Expected 'value', received {}.".format(value)

    return False, ""


@_register_check(_CONFIG_CHECKS)
def check_count(shared_dict):
    """Check if database supports count operation... """
    database = shared_dict['instance']

    count = database.count('test', {'index': 'value'})

    if count != 1:
        return True, "Expected 1 hit, received {}.".format(count)

    return False, ""


@_register_check(_CONFIG_CHECKS)
def check_remove(shared_dict):
    """Check if database supports delete operation... """
    database = shared_dict['instance']

    database.remove('test', {'index': 'value'})
    remaining = database.count('test', {'index': 'value'})

    if remaining:
        return True, "{} items remaining.".format(remaining)

    return False, ""
