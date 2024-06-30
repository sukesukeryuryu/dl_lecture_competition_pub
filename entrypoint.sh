#!/bin/bash

# USERID=1000
# GROUPID=1000
# echo "Create User = $USERID. Group = $GROUPID"
# groupadd -g $GROUPID dockeruser
# useradd -m -s /bin/bash -u $USERID -g $GROUPID dockeruser
gosu dockeruser "$@"