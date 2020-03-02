# Homebrew Formula for EDDL
# Usage: brew install eddl

class Eddl < Formula
  desc "European Distributed Deep Learning Library (EDDL)"
  homepage "https://github.com/deephealthproject/eddl"
  url "https://github.com/deephealthproject/eddl/archive/0.4.3.tar.gz"
  sha256 "07492ae01d13a8b9c686ea5578dfc8740be751dc9b82d61ef669699054b9dd7f"

  depends_on "cmake" => :build
  depends_on "eigen" => :build
  depends_on "graphviz" => :build
  depends_on "openblas" => :build
  depends_on "wget" => :build
  depends_on "zlib" => :build
  depends_on "protobuf" => :build

  def install
    mkdir "build" do
      system "cmake", "..", "-DBUILD_PROTOBUF=ON", "-DBUILD_EXAMPLES=OFF", *std_cmake_args
      system "make", "install", "PREFIX=#{prefix}"
    end
  end

  test do
    (testpath/"CMakeLists.txt").write <<~EOS
      cmake_minimum_required(VERSION 3.9.2)
      project(test)

      set(CMAKE_CXX_STANDARD 11)

      find_package(eddl)

      add_executable(test test.cpp)
      target_link_libraries(test eddl)
    EOS

    (testpath/"test.cpp").write <<~EOS
      #include <iostream>
      #include <eddl/tensor/tensor.h>
      int main(){
        Tensor *t1 = Tensor::ones({5, 5});
        std::cout << t1->sum() << std::endl;
      }
    EOS

    system "cmake", "."
    system "make"
    assert_equal "25", shell_output("./test").strip
  end
end
