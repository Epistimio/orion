#!/bin/bash -e
# IBM_SOURCE_PROLOG_BEGIN_TAG
# *****************************************************************
#
# IBM Confidential
# OCO Source Materials
#
# (C) Copyright IBM Corp. 2018,2020
#
# The source code for this program is not published or otherwise
# divested of its trade secrets, irrespective of what has been
# deposited with the U.S. Copyright Office.
#
# *****************************************************************
# IBM_SOURCE_PROLOG_END_TAG


usage(){
    echo "Usage: build.sh [-h|--help] | [ [-v version |--version version] [-l label | --label label ] ]"
}

while [ $# -gt 0 ]
do
  key="$1"

  case $key in
    -v|--version)
    VERSION="$2"
    shift # past argument
    shift # past value
    ;;
    -l|--label)
    DOCK_LABEL="$2"
    shift # past argument
    shift # past value
    ;;
    -h|--help)
    usage
    exit 0
    ;;
    *)    # unknown option
    echo "Unknown option: $key"
    usage
    exit 1
    ;;
  esac
done

if [ -z ${VERSION} ]
then
    echo "VERSION is not defined"
    exit 1
fi

if [ -z ${DOCK_LABEL} ]
then
    # if a docker label was not specified, then construct
    # the label based on the repo used to create the image
    # Record build information to be included in docker image
    UI_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    UI_HASH=$(git rev-parse --short HEAD)
    DOCK_LABEL="${UI_BRANCH}:${UI_HASH}"
fi

# Let's see if we are building on an x86 system to tag the docker image appropriately.
ARCH_TAG=""
ARCH=$(uname -i)
if [[ "${ARCH}" == "x86_64" ]]
then
    ARCH_TAG="-x86"
fi

VISION_COMPONENT=vision-ui

if [ ! -z $JENKINS_URL ]
then
   current_folder=$(cd `dirname $0`; pwd)
   export GIT_JSON=$(python vision-ci/changelog/git_json.py --base-dir $current_folder  --json-file $current_folder/vision-ci/changelog/components.json --component vision-ui)
   echo "***GIT_JSON:$GIT_JSON****"
fi

# Multi stage build for static UI content and nginx container
docker build --label "build=${DOCK_LABEL}" \
        --label "com.ibm.aivision.git-info=${GIT_JSON}" \
        -t ${VISION_COMPONENT}${ARCH_TAG}:${VERSION} \
        -f Dockerfile .

exit 0
