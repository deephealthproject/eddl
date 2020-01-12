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
								timeout(15) {
									echo 'Building..'
									cmakeBuild buildDir: 'build', cmakeArgs: '-D BUILD_TESTS=ON', installation: 'InSearchPath', sourceDir: '.', cleanBuild: true, steps: [[withCmake: true]]
								}
                            }
                        }
                        stage('Test') {
                            steps {
								timeout(15) {
									echo 'Testing..'
									ctest arguments: '-C Debug -VV', installation: 'InSearchPath', workingDir: 'build'
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
								timeout(15) {
									echo 'Building..'
									cmakeBuild buildDir: 'build', cmakeArgs: '-D BUILD_TESTS=ON',  installation: 'InSearchPath', sourceDir: '.', cleanBuild: true, steps: [[withCmake: true]]
								}
                            }
                        }
                        stage('Test') {
                            steps {
								timeout(15) {
									echo 'Testing..'
									bat 'cd build && ctest -C Debug -VV'
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
                            image 'stal12/cuda10-gcc5'
                            args '--gpus 1'
                        }
                    }
                    stages {
                        stage('Build') {
                            steps {
								timeout(15) {
									echo 'Building..'
									cmakeBuild buildDir: 'build', cmakeArgs: '-D BUILD_TARGET=GPU -D BUILD_TESTS=ON', installation: 'InSearchPath', sourceDir: '.', cleanBuild: true, steps: [[withCmake: true]]
								}
							}
                        }
                        stage('Test') {
                            steps {
								timeout(15) {
									echo 'Testing..'
									ctest arguments: '-C Debug -VV', installation: 'InSearchPath', workingDir: 'build'
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
								timeout(15) {
									echo 'Building..'
									cmakeBuild buildDir: 'build', cmakeArgs: '-D BUILD_TARGET=GPU -D BUILD_TESTS=ON',  installation: 'InSearchPath', sourceDir: '.', cleanBuild: true, steps: [[withCmake: true]]
								}
                            }
                        }
                        stage('Test') {
                            steps {
								timeout(15) {
									echo 'Testing..'
									bat 'cd build && ctest -C Debug -VV'
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
								timeout(15) {
									bat 'cd docs\\doxygen && doxygen'
									bat 'powershell -Command "(gc %EDDL_DOXYGEN_INPUT_COMMANDS%) -replace \'@local_dir\', \'docs\\build\\html\' | Out-File commands_out.txt"'
									bat 'winscp /ini:nul /script:commands_out.txt'
								}
                            }
                        }
                    }
                }
            }
        }
    }
}
