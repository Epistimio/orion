#!/bin/sh

# Setup a mongodb server for testing
# this setup is done my IT when the DB is deployed


PORT=${PORT:-8124}
ADDRESS=${ADDRESS:-192.168.0.116}
ADMIN=${ADMIN:-god}
PASSWORD=${PASS:-god123}

set -evx

function start_init_mongod() {
    # Start mongodb without Access Control, this is used to insert the admin user
    # the db is then stopped and started with Access Control

    if test -f "db/pid"; then
        pid=$(cat db/pid)
        kill -3 $pid || true
    fi

    rm -rf db
    mkdir -p db

    mongod --dbpath db/ --wiredTigerCacheSizeGB 1 --port $PORT --bind_ip localhost --pidfilepath db/pid &
    sleep 1
}

function start_mongod() {
    # start mongodb with Access Control
    mongod --auth --dbpath db/ --wiredTigerCacheSizeGB 1 --port $PORT --bind_ip $ADDRESS --pidfilepath db/pid &
    sleep 1
}

function stop_mongod() {
    echo "$(pwd)"
    pid=$(cat $(pwd)/db/pid)
    kill -s TERM $pid
    rm -rf db/pid
    sleep 1
}

function add_admin_user() {
    # create an admin user
    # userAdminAnyDatabase: create users, grant & revoke roles, create and modify roles

    CMD=$(cat <<EOM
    use admin
    db.createUser({
        user: "$ADMIN",
        pwd: "$PASSWORD",
        roles: [
            { role: "userAdminAnyDatabase", db: "admin" },
            { role: "readWriteAnyDatabase", db: "admin" }
        ]
    })
EOM
    )

    echo "$CMD" | mongo --port $PORT
}

function add_user() {
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

    echo "$CMD" | mongo "mongodb://$ADDRESS:$PORT" -u $ADMIN -p $PASSWORD
}


function ensure_indexes() {
    # User will have limited access to the collection
    # so orion's client cannot do this

    CMD=$(cat <<EOM
    use orion

EOM
    )

    echo "$CMD" | mongo --port $PORT
}

function launch() {
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

launch