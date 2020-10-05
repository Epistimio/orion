#!/bin/bash -e
# IBM_SOURCE_PROLOG_BEGIN_TAG
# *****************************************************************
#
# IBM Confidential
# OCO Source Materials
#
# (C) Copyright IBM Corp. 2018
#
# The source code for this program is not published or otherwise
# divested of its trade secrets, irrespective of what has been
# deposited with the U.S. Copyright Office.
#
# *****************************************************************
# IBM_SOURCE_PROLOG_END_TAG

ROOT_DIR=`git rev-parse --show-toplevel`;

VISION_TOOLS_DIR=vision-tools
if [ ! -d ${VISION_TOOLS_DIR} ]; then
    git clone --depth 1 --single-branch -b master git@github.ibm.com:aivision/vision-tools.git ${VISION_TOOLS_DIR} || exit -1
else
    cd $VISION_TOOLS_DIR
    #git pull origin master
    cd $ROOT_DIR
fi

# Set path env for license files used by copyright tool
export LICENSE_FILE_DIR=$ROOT_DIR/vision-tools/prolog-tool/
COPYRIGHT_TOOL=$ROOT_DIR/vision-tools/prolog-tool/addCopyright.pl

# Default to NO SHIP for all files
# Note bash injected problems when I created the param string via an array/variable etc.
# Somewhere we'd have to add a new directory, so adding a new line to this is not any more work
echo "Adding default prolog to all relevant files in repo..."
find . -type f  ! -path "./vision-tools/*" \
                ! -path "./build/*" \
                ! -path "./.idea/*" \
                ! -path "./.git/*" \
                ! -path "./doc/*" \
                ! -path "./docs/*" \
                ! -path "./node_modules/*" \
                ! -path "./dist/*" \
                ! -path "./output/*" \
                ! -path "./src/libraries/*" \
    | xargs ${COPYRIGHT_TOOL} update || exit -1