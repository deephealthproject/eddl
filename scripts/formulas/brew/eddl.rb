# Homebrew Formula for Google Test
# Usage: brew install eddl

class Eddl < Formula
  desc "European Distributed Deep Learning (EDDL) Library"
  homepage "https://github.com/deephealthproject/eddl"
  url "https://github.com/deephealthproject/eddl/archive/v0.3.1.tar.gz"
  sha256 "f439f9cf01e95e0eb7af8ab1bf2dee5d390023aa8bb1514e34e063552865b7ee"

  depends_on "cmake" => :build
  depends_on "openblas" => :build
  depends_on "eigen" => :build
  depends_on "zlib" => :build
  depends_on "graphviz" => :build
  depends_on "wget" => :build

  def install
    mkdir "build" do
      system "cmake", "..", *std_cmake_args
      system "make", "install", "PREFIX=#{prefix}"
    end
  end

  test do
  end
end
