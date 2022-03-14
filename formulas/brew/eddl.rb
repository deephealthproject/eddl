# Homebrew Formula for EDDL
# Usage: brew install eddl

class Eddl < Formula
  desc "European Distributed Deep Learning Library (EDDL)"
  homepage "https://github.com/deephealthproject/eddl"
  url "https://github.com/deephealthproject/eddl/archive/v1.1b.tar.gz"
  sha256 "b9fe2bdc63808ae8e1a8eec96f66106c49b7a5ce9ee32ffe17fd6cf9d1b2c4ec"

  depends_on "cmake" => :build
  depends_on "eigen" => :build
  depends_on "protobuf" => :build
  depends_on "zlib" => :build
  depends_on "openssl@1.1" => :build
  depends_on "graphviz" => :build
  depends_on "wget" => :build

  def install
    mkdir "build" do
      system "cmake", "..", "-DBUILD_SUPERBUILD=OFF", "-DBUILD_EXAMPLES=OFF", "-DBUILD_TESTS=OFF", "-DBUILD_OPENMP=OFF" *std_cmake_args
      system "make", "-j", "install", "PREFIX=#{prefix}"
    end
  end

  test do
    (testpath/"CMakeLists.txt").write <<~EOS
      cmake_minimum_required(VERSION 3.9.2)
      project(test)

      set(CMAKE_CXX_STANDARD 11)

      add_executable(test test.cpp)

      find_package(EDDL REQUIRED)
      target_link_libraries(test PUBLIC EDDL::eddl)
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
