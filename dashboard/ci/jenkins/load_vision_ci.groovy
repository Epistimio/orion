OWNER = 'aivision'
// The branch of vision_ci depends on which branch the PR is opened on for PR builds
// for other builds it will be same as the source branch
BRANCH = env.BRANCH_NAME.startsWith("PR-")?env.CHANGE_TARGET:env.BRANCH_NAME
DIR_LOC  = 'vision-ci'

def cloneVisionCI() {
    sh "rm -rf $DIR_LOC"
    dir(DIR_LOC)
    {
        checkout changelog: false,
            poll: false,
            scm: [$class: 'GitSCM', branches: [[name: "*/${BRANCH}"]],
                doGenerateSubmoduleConfigurations: false,
                extensions: [[$class: 'SubmoduleOption',
                    disableSubmodules: false,
                    parentCredentials: true,
                    recursiveSubmodules: true,
                    reference: '', trackingSubmodules: false]],
                submoduleCfg: [],
                userRemoteConfigs: 
                    [[credentialsId: 'ae7b0f81-e5e5-4792-9af9-a5f5901e74ff',
                    url: " https://github.ibm.com/${OWNER}/vision-ci"]]]
    }
}

return this
