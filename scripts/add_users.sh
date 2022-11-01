#!/bin/bash

SCRIPT_DIR="$(dirname -- "${BASH_SOURCE:-0}";)";
source $SCRIPT_DIR/mongodb.sh

add_user User1 Pass1 Tok1
add_user User2 Pass2 Tok2
add_user User3 Pass3 Tok3

