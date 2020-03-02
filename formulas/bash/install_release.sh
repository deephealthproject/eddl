VERSION=0.4.3
sudo apt-get install wget
wget "https://github.com/deephealthproject/eddl/archive/$VERSION.tar.gz"
tar -xf "$VERSION.tar.gz"
cd "eddl-$VERSION"
mkdir build
cd build
cmake ..
make install