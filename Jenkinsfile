pipeline {
    agent none
    stages {
        stage('Parallel Stages') {
            parallel {
                stage('linux') {
                    agent {
                        docker { 
                            label 'docker'
                            image 'pritt/base'
                        }
                    }
                    stages {
                        stage('Build') {
                            steps {
                                timeout(60) {
                                    echo 'Building..'
                                    cmakeBuild buildDir: 'build', buildType: 'Release', cmakeArgs: '-DBUILD_TARGET=CPU -DBUILD_SUPERBUILD=ON -DBUILD_TESTS=ON -DBUILD_HPC=OFF -DBUILD_DIST=OFF -DBUILD_RUNTIME=OFF', installation: 'InSearchPath', sourceDir: '.', cleanBuild: true, steps: [
                                        [args: '--parallel 4', withCmake: true]
                                    ]
                                }
                            }
                        }
                        stage('Test') {
                            steps {
                                timeout(15) {
                                    echo 'Testing..'
                                    ctest arguments: '-C Release -VV', installation: 'InSearchPath', workingDir: 'build'
                                }
                            }
                        }
                        stage('linux_end') {
                            steps {
                                echo 'Success!'
                            }
                        }
                    }
                }
                stage('windows') {
                    agent {
                        label 'windows'
                    }
                    stages {
                        stage('Build') {
                            steps {
                                timeout(60) {
                                    echo 'Building..'
                                    cmakeBuild buildDir: 'build', buildType: 'Release', cmakeArgs: '-DBUILD_TARGET=CPU -DBUILD_SHARED_LIBS=OFF -DBUILD_SUPERBUILD=ON -DBUILD_TESTS=ON -DBUILD_HPC=OFF -DBUILD_DIST=OFF -DBUILD_RUNTIME=OFF',  installation: 'InSearchPath', sourceDir: '.', cleanBuild: true, steps: [
                                        [args: '--config Release --parallel 4', withCmake: true]
                                    ]
                                }
                            }
                        }
                        stage('Test') {
                            steps {
                                timeout(15) {
                                    echo 'Testing..'
                                    bat 'cd build && ctest -C Release -VV'
                                }
                            }
                        }
                        stage('windows_end') {
                            steps {
                                echo 'Success!'
                            }
                        }
                    }
                }
                stage('linux_gpu') {
                    agent {
                        docker { 
                            label 'docker && gpu'
                            image 'pritt/base-cuda'
                            args '--gpus 1'
                        }
                    }
                    stages {
                        stage('Build') {
                            steps {
                                timeout(60) {
                                    echo 'Building..'
                                    cmakeBuild buildDir: 'build', buildType: 'Release', cmakeArgs: '-DBUILD_TARGET=GPU -DBUILD_TESTS=ON -DBUILD_SUPERBUILD=ON -DBUILD_HPC=OFF -DBUILD_DIST=OFF -DBUILD_RUNTIME=OFF', installation: 'InSearchPath', sourceDir: '.', cleanBuild: true, steps: [
                                        [args: '--parallel 4', withCmake: true]
                                    ]
                                }
                            }
                        }
                        stage('Test') {
                            steps {
                                timeout(15) {
                                    echo 'Testing..'
                                    ctest arguments: '-C Release -VV', installation: 'InSearchPath', workingDir: 'build'
                                }
                            }
                        }
                        stage('linux_gpu_end') {
                            steps {
                                echo 'Success!'
                            }
                        }
                    }
                }
                stage('windows_gpu') {
                    agent {
                        label 'windows && gpu'
                    }
                    stages {
                        stage('Build') {
                            steps {
                                timeout(60) {
                                    echo 'Building..'
                                    cmakeBuild buildDir: 'build', buildType: 'Release', cmakeArgs: '-DBUILD_TARGET=GPU -DBUILD_TESTS=ON -DBUILD_SUPERBUILD=ON -DBUILD_HPC=OFF -DBUILD_DIST=OFF -DBUILD_RUNTIME=OFF', installation: 'InSearchPath', sourceDir: '.', cleanBuild: true, steps: [
                                        [args: '--config Release --parallel 4', withCmake: true]
                                    ]
                                }
                            }
                        }
                        stage('Test') {
                            steps {
                                timeout(15) {
                                    echo 'Testing..'
                                    bat 'cd build && ctest -C Release -VV'
                                }
                            }
                        }
                        stage('windows_gpu_end') {
                            steps {
                                echo 'Success!'
                            }
                        }
                    }
                }
            }
        }
    }
}
