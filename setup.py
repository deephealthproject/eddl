#! /usr/bin/env python3

import os
import re
import sys
import sysconfig
import platform
import subprocess

from setuptools import setup, Extension, distutils, find_packages
from distutils import core, dir_util
from distutils.core import Distribution
from distutils.errors import DistutilsArgError
import setuptools.command.build_ext
import setuptools.command.install
import distutils.command.clean
import distutils.sysconfig
import filecmp
import subprocess
import shutil
import sys
import os
import json
import glob
import importlib

from distutils.version import LooseVersion
from setuptools.command.build_ext import build_ext
from shutil import copyfile, copymode, move

################################################################################
# Parameters parsed from environment
################################################################################

RUN_BUILD_DEPS = True
RERUN_CMAKE = True
filtered_args = []
for i, arg in enumerate(sys.argv):
    if arg == '--cmake':
        RERUN_CMAKE = True
        continue
    if arg == "--":
        filtered_args += sys.argv[i:]
        break
    if arg == 'clean':
        RUN_BUILD_DEPS = False
    filtered_args.append(arg)
sys.argv = filtered_args

################################################################################
# Constants
################################################################################
cwd = os.path.dirname(os.path.abspath(__file__))
third_party_path = os.path.join(cwd, "third_party")


################################################################################
# Classes and functions
################################################################################


class CMakeExtension(Extension):
    def __init__(self, name, dest_dir='', source_dir='', **kwargs):
        Extension.__init__(self, name, sources=[])
        self.dest_dir = os.path.abspath(dest_dir)
        self.source_dir = os.path.abspath(source_dir)
        self.extra = kwargs


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                                   out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + ext.dest_dir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable
                      ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                cfg.upper(),
                ext.dest_dir)]
            if sys.maxsize > 2 ** 32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.source_dir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)

        # Copy *_test file to tests directory
        test_bin = os.path.join(self.build_temp, 'eddl_test')
        self.copy_test_file(test_bin)
        print()  # Add empty line for nicer output

    def copy_test_file(self, src_file):
        # Create directory if needed
        dest_dir = os.path.abspath("./bin")
        if dest_dir != "" and not os.path.exists(dest_dir):
            print("creating directory {}".format(dest_dir))
            os.makedirs(dest_dir)

        # Copy/Move file
        dest_file = os.path.join(dest_dir, os.path.basename(src_file))
        print("moving {} -> {}".format(src_file, dest_file))
        move(src_file, dest_file)
        # copyfile(src_file, dest_file)
        # copymode(src_file, dest_file)


class install(setuptools.command.install.install):
    def run(self):
        setuptools.command.install.install.run(self)


class clean(distutils.command.clean.clean):
    def run(self):
        import glob
        import re
        with open('.gitignore', 'r') as f:
            ignores = f.read()
            pat = re.compile(r'^#( BEGIN NOT-CLEAN-FILES )?')
            for wildcard in filter(None, ignores.split('\n')):
                match = pat.match(wildcard)
                if match:
                    if match.group(1):
                        # Marker is found and stop reading .gitignore.
                        break
                    # Ignore lines which begin with '#'.
                else:
                    for filename in glob.glob(wildcard):
                        try:
                            os.remove(filename)
                        except OSError:
                            shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


# Checks before runnning setup
def build_deps():
    def check_file(f):
        if not os.path.exists(f):
            print("Could not find {}".format(f))
            print("Did you run 'git submodule update --init --recursive'?")
            sys.exit(1)

    check_file(os.path.join(third_party_path, "pybind11", "CMakeLists.txt"))
    check_file(os.path.join(third_party_path, "catch2", "CMakeLists.txt"))


################################################################################
# Declare extensions and package
################################################################################
# Packages
package_dir = {'': 'src/python/'}
packages = find_packages('src/python/')

# Extensions
extensions = []
if RERUN_CMAKE:
    C = CMakeExtension(name='_C', dest_dir='src/python/pyeddl', source_dir='.')
    extensions.append(C)

# If doesn't exists a compatible shared library, compile C++ EDDL
ext_modules = []

cmdclass = {
    'build_ext': CMakeBuild,
    'clean': clean,
    'install': install,
}

entry_points = {
    'console_scripts': []
}

################################################################################
# More package stuff
################################################################################

# Get Python requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

long_description = (
        open('README.txt').read()
        + '\n' +
        open('LICENSE').read()
        + '\n')

if __name__ == '__main__':
    # Parse the command line and check the arguments
    # before we proceed with building deps and setup
    dist = Distribution()
    dist.script_name = sys.argv[0]
    dist.script_args = sys.argv[1:]
    try:
        ok = dist.parse_command_line()
    except DistutilsArgError as msg:
        raise SystemExit(core.gen_usage(dist.script_name) + "\nerror: %s" % msg)
    if not ok:
        sys.exit()

    if RUN_BUILD_DEPS:
        build_deps()
    setup(
        name='pyeddl',
        version='0.1',
        author='Salva Carrion',
        author_email='salcarpo@prhlt.upv.es',
        description='Python wrapper for the European Distributed Deep Learning Library',
        long_description=long_description,
        keywords='deep learning, neural networks',
        download_url="https://github.com/deephealthproject/EDDLL",
        license="MIT",
        platforms="Unix",
        packages=packages,  # Load python packages from the python folder
        package_dir=package_dir,
        ext_modules=extensions,  # Name of the shared library
        cmdclass=cmdclass,
        test_suite='tests',
        install_requires=requirements,
        zip_safe=False,
        include_package_data=True,
    )
