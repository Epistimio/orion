#!/bin/bash

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
    #
    #   How to make a group setup
    #       - make a Token => Group
    #       - Token => User => Group

    username=$1
    password=$2
    token=$3

    CMD=$(cat << EOM
    use orion
    db.createUser({
        user: "$username",
        pwd: "$password",
        roles: [
            { role: "readWrite", db: "orion" }
        ]
    })

    db.users.insertOne({
        username: "$username",
        password: "$password",
        token: "$token",
    })
EOM
    )

    echo "$CMD" | mongo "mongodb://$ADDRESS:$PORT" --authenticationDatabase "admin" -u $ADMIN -p $PASSWORD
}


ASCENDING=1
DESCENDING=-1

function ensure_indexes {
    # User will have limited access to the collection
    # so orion's client cannot do this

    CMD=$(cat <<EOM
    use orion

    db.experiments.createIndex(
        {
            "name": 1,
            "version": 1,
            "owner_id": 1
        },
        {
            unique: true
        }
    )
    db.experiments.createIndex(
        {
            "metadata.datetime": 1
        }
    )

    db.trials.createIndex(
        {
            "experiment": 1,
            "id": 1
        },
        {
            unique: true
        }
    )
    db.trials.createIndex(
        {
            "experiment": 1
        }
    )
    db.trials.createIndex(
        {
            "status": 1
        }
    )
    db.trials.createIndex(
        {
            "results": 1
        }
    )
    db.trials.createIndex(
        {
            "start_time": 1
        }
    )
    db.trials.createIndex(
        {
            "end_time": -1
        }
    )
    db.algo.createIndex(
        {
            "experiment": 1
        }
    )
EOM
    )

    echo "$CMD" | mongo --port $PORT
}
