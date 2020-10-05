vision_ci_common = 'vision-ci/jenkins/common/ci_common.groovy'

def isProdBranch(){
    vision_ci_common_functions = load vision_ci_common
    return vision_ci_common_functions.isProdBranch()
}

def cloneVisionCI(){
    load_vision_ci_functions = load 'ci/jenkins/load_vision_ci.groovy'
        load_vision_ci_functions.cloneVisionCI()
}

def getDockerTag(){
	    vision_ci_common_functions = load vision_ci_common
	        return vision_ci_common_functions.getDockerTag()
}

def stashVisionCI() {
    stash includes: "vision-ci/**/*", name: "ci_cache_vision_ci"
}

def unstashVisionCI() {
    sh "rm -rf vision-ci"
    echo "Unstashing vision-ci"
    unstash "ci_cache_vision_ci"
}

def sendNotification() {
    if ( ! isProdBranch() )
    {
        vision_ci_common_functions = load vision_ci_common
        return vision_ci_common_functions.sendNotification()
    }
}

def setArtRepo(){
    vision_ci_common_functions = load vision_ci_common
    return vision_ci_common_functions.setArtRepo()
}

def setBuildStatus(CONTEXT, STATUS) {
    vision_ci_common_functions = load vision_ci_common
    vision_ci_common_functions.setBuildStatus(CONTEXT, STATUS)
}


return this