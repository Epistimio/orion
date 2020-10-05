# wmla-gui
IBM Watson Machine Learning Accelerator Front-End Web UI

[![Build Status](https://powerai-jenkins.swg-devops.com/buildStatus/icon?job=Power-AI/Production/vision/vision-ui/master)](https://powerai-jenkins.swg-devops.com/job/Power-AI/job/Production/job/vision/job/vision-ui/job/master/)

IBM Watson Machine Learning Accelerator front-end UI is built using IBM Carbon, React and Redux.

## Daily Development
Developers should be able to run the UI on a local system (or any system with npm) and proxy a remote API endpoint (even an endpoint on their local system).

To get started for daily development:

#### Deployed Backend Target

```
# install dependencies
npm install

# optionally, set backend vision-service api url
# NOTE! This MUST be https!
# and should not have a trailing "/api" e.g. https://power9.ibm.com/powerai-vision
export VISION_SERVICE_API=https://servername/contextroot

# run in debug mode with hot reload
npm start

# build for production with minification
npm run build
```

#### Local Backend Target

If you start vision-service via `gretty`, follow the same steps as above but change env var.

```
export VISION_SERVICE_API=http://localhost:9080/vision-service
```

## Production Containers
Watson Machine Learning Accelerator generates and is packaged as a Docker container in final production deployments. See the [Dockerfile] for that build. Daily developers shouldn't need to modify the Dockerfile. This is used to generate the final, clean build. This Dockerfile takes advantage of [multi-stage builds](https://docs.docker.com/develop/develop-images/multistage-build/) which are a new feature in Docker to avoid having a separate "Dockerfile.build" and a "Dockerfile" for production.
