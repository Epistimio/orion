#!/bin/bash

# Setup a mongodb server for testing
# this setup is done my IT when the DB is deployed
# set -evx

PORT=${MONGO_PORT:-"8123"}
ADDRESS=${MONGO_ADDRESS:-"localhost"}
ADMIN=${MONGO_ADMIN:-"god"}
PASSWORD=${MONGO_PASS:-"god123"}
DB_PATH=${MONGO_PATH:-"/tmp/db"}

function start_init_mongod {
    # Start mongodb without Access Control, this is used to insert the admin user
    # the db is then stopped and started with Access Control

    if test -f "$DB_PATH/pid"; then
        pid=$(cat $DB_PATH/pid)
        kill -3 $pid || true
    fi

    rm -rf $DB_PATH
    mkdir -p $DB_PATH

    mongod --dbpath $DB_PATH/ --wiredTigerCacheSizeGB 1 --port $PORT --bind_ip localhost --pidfilepath $DB_PATH/pid &
    sleep 1
}

function start_mongod {
    # start mongodb with Access Control
    mongod --auth --dbpath $DB_PATH/ --wiredTigerCacheSizeGB 1 --port $PORT --bind_ip $ADDRESS --pidfilepath $DB_PATH/pid --fork --logpath $DB_PATH/log.txt
    sleep 1
}

function stop_mongod  {
    # mongo --dbpath $DB_PATH --shutdown

    echo "$(pwd)"
    pid=$(cat $DB_PATH/pid)
    kill -s TERM $pid
    rm -rf $DB_PATH/pid
    sleep 1
}

function add_admin_user {
    # create an admin user
    # userAdminAnyDatabase: create users, grant & revoke roles, create and modify roles

    CMD=$(cat <<EOM
    use admin
    db.createUser({
        user: "$ADMIN",
        pwd: "$PASSWORD",
        roles: [
            { role: "userAdminAnyDatabase", db: "admin" },
            { role: "readWriteAnyDatabase", db: "admin" },
        ]
    })
EOM
    )

    echo "$CMD" | mongo --port $PORT

    add_user $ADMIN $PASSWORD
}

function add_user {
    # Create a user for the orion database

    username=$1
    password=$2

    CMD=$(cat << EOM
    use orion
    db.createUser({
        user: "$username",
        pwd: "$password",
        roles: [
            { role: "readWrite", db: "orion" }
        ]
    })
EOM
    )

    echo "$CMD" | mongo "mongodb://$ADDRESS:$PORT" --authenticationDatabase "admin" -u $ADMIN -p $PASSWORD
}


function ensure_indexes {
    # User will have limited access to the collection
    # so orion's client cannot do this

    CMD=$(cat <<EOM
    use orion

EOM
    )

    echo "$CMD" | mongo --port $PORT
}

function launch {
    # Setup a mongodb for testing
    start_init_mongod
    add_admin_user
    ensure_indexes
    stop_mongod

    # Start mongodb with access control
    start_mongod

    add_user User1 Pass1
    add_user User2 Pass2
    add_user User3 Pass3
}

export MONGO_RUNNING="${DB_PATH}"
launch

