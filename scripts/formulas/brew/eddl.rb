# Homebrew Formula for EDDL
# Usage: brew install eddl

class Eddl < Formula
  desc "European Distributed Deep Learning Library (EDDL)"
  homepage "https://github.com/deephealthproject/eddl"
  url ""
  sha256 ""

  depends_on "cmake" => :build
  depends_on "eigen" => :build
  depends_on "graphviz" => :build
  depends_on "openblas" => :build
  depends_on "wget" => :build
  depends_on "zlib" => :build

  def install
    mkdir "build" do
      system "cmake", "..", *std_cmake_args
      system "make", "install", "PREFIX=#{prefix}"
    end
  end

  test do
    (testpath/"test.cpp").write <<~EOS
      #include <iostream>
      #include <eddl/tensor/tensor.h>
      int main(){
        Tensor *t1 = Tensor::ones({5, 1});
        std::cout << t1->sum() << std::endl;
      }
    EOS
    system ENV.cxx, "test.cpp", "-I#{include}/eddl", "-o", "test"
    assert_equal "5", shell_output("./test").split
  end
end
