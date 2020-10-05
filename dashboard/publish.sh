#!/bin/bash -xe
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
    echo "Usage: publish.sh [-h|--help] | [-v version |--version version]"
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

if [ -z ${ARTIFACTORY_CREDS_USR} ] || [ -z ${ARTIFACTORY_CREDS_PSW} ]
then
    echo "ARTIFACTORY_CREDS_USR and ARTIFACTORY_CREDS_PSW need to be defined with Artifactory creds"
    echo "If you are running this through CI set the creds in Jenkins env {} para."
    exit 1
fi

# See if we are going to publish the ppc64le (no tag) or x85 image (with '-x86' tag)
ARCH_TAG=""
ARCH=$(uname -i)
if [[ "${ARCH}" == "x86_64" ]]
then
    ARCH_TAG="-x86"
fi

VISION_COMPONENT=vision-ui
DOCKER_VISION_IMAGE=${VISION_COMPONENT}${ARCH_TAG}:${VERSION}

# Log into Artifactory docker repo

# the ARTIFACTORY_DOCKER variable is set in the environment by jenkins, 
# if its not set then raise an error.
if [ -z "$ARTIFACTORY_DOCKER" ]
then
    echo "ARTIFACTORY_DOCKER env variable has to be set to specify the repo to which to publish"
    exit 1
fi

if ! docker login --username="${ARTIFACTORY_CREDS_USR}" --password="${ARTIFACTORY_CREDS_PSW}" ${ARTIFACTORY_DOCKER}
then
    echo "Failed to log into Artifactory docker repo: ${ARTIFACTORY_DOCKER}"
    exit 1
fi

if ! docker images ${DOCKER_VISION_IMAGE} | grep ${VISION_COMPONENT}
then
    echo "The expected docker image, ${DOCKER_VERSION_IMAGE}, can not be found"
    exit 1
fi

docker tag ${DOCKER_VISION_IMAGE} ${ARTIFACTORY_DOCKER}/${DOCKER_VISION_IMAGE}
docker push ${ARTIFACTORY_DOCKER}/${DOCKER_VISION_IMAGE}

exit 0
