#!/bin/bash

# Setup a mongodb server for testing
# this setup is done my IT when the DB is deployed
# set -evx
SCRIPT_DIR="$(dirname -- "${BASH_SOURCE:-0}";)";
source $SCRIPT_DIR/mongodb.sh

function launch {
    # Setup a mongodb for testing
    start_init_mongod
    add_admin_user
    ensure_indexes
    stop_mongod

    # Start mongodb with access control
    start_mongod

    add_user User1 Pass1 Tok1
    add_user User2 Pass2 Tok2
    add_user User3 Pass3 Tok3
}

export MONGO_RUNNING="${DB_PATH}"
launch

