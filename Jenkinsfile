pipeline {
    agent none
    stages {
        stage('Parallel Stages') {
            parallel {
                stage('linux') {
                    agent {
                        docker { 
                            label 'docker'
                            image 'stal12/ubuntu18-gcc5'
                        }
                    }
                    stages {
                        stage('Build') {
                            steps {
                                echo 'Building..'
                                cmakeBuild buildDir: 'build', installation: 'InSearchPath', sourceDir: '.', cleanBuild: true, steps: [[withCmake: true]]
                             }
                        }
                        stage('Test') {
                            steps {
                                echo 'Testing..'
                                ctest arguments: '-C Debug -VV', installation: 'InSearchPath', workingDir: 'build'
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
                                echo 'Building..'
                                cmakeBuild buildDir: 'build', installation: 'InSearchPath', sourceDir: '.', cleanBuild: true, steps: [[withCmake: true]]
                            }
                        }
                        stage('Test') {
                            steps {
                                echo 'Testing..'
                                bat 'cd build && ctest -C Debug -VV'
                            }
                        }
                        stage('windows_end') {
                            steps {
                                echo 'Success!'
                            }
                        }
                    }
                }
                stage('documentation') {
                    when { 
                        branch 'master' 
                        beforeAgent true
                    }
                    agent {
                        label 'windows && eddl_doxygen'
                    }
                    stages {
                        stage('Update documentation') {
                            steps {
                                bat 'cd doc\\doxygen && doxygen'
                                bat 'powershell -Command "(gc %EDDL_DOXYGEN_INPUT_COMMANDS%) -replace \'@local_dir\', \'doc\\html\' | Out-File commands_out.txt"'
                                bat 'winscp /ini:nul /script:commands_out.txt'
                            }
                        }
                    }
                }
            }
        }
    }
}
