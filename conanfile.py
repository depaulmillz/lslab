from conans import ConanFile, CMake

class LSlabConan(ConanFile):
    name = "lslab"
    version = "3.0.0"
    author = "dePaul Miller"
    url = "https://github.com/depaulmillz/lslab"
    license = "MIT"
    settings={"os" : ["Linux"], "compiler" : None, "build_type" : None, "arch": ["x86_64"] }
    requires="unifiedmemorygroupallocation/1.1"
    generators="cmake_find_package"
    description = """The LSlab GPU hashmap was designed in the paper
    "KVCG: A Heterogeneous Key-Value Store for Skewed Workloads" by dePaul Miller, 
    Jacob Nelson, Ahmed Hassan, and Roberto Palmieri."
    """
    topic = ("data structure", "gpu programming", "hashmap", "hashtable")
    
    exports_sources = "CMakeLists.txt", "cmake/*", "benchmark/*", "include/*", "test/*", "LICENSE"
    options = {"cuda_arch" : "ANY", "cuda_compiler" : "ANY"}

    def configure(self):
        if self.options.cuda_arch == None:
            self.options.cuda_arch = '70;75;80;86'
        if self.options.cuda_compiler == None:
            self.options.cuda_compiler = "nvcc"

    def _configure_cmake(self):
        cmake = CMake(self)
        cmake.definitions["CMAKE_CUDA_ARCHITECTURES"] = str(self.options.cuda_arch)
        cmake.definitions["CMAKE_CUDA_COMPILER"] = str(self.options.cuda_compiler)
        cmake.definitions["USING_CONAN"] = "ON"
        cmake.definitions["CMAKE_EXPORT_COMPILE_COMMANDS"] = "ON"
        cmake.configure()
        return cmake

    def build(self):
        cmake = self._configure_cmake()
        cmake.build()
        cmake.test()

    def package(self):
        cmake = self._configure_cmake()
        cmake.install()

    def package_id(self):
        self.info.header_only()

    def package_info(self):
        self.cpp_info.names["cmake_find_package"] = "LSlab"
        self.cpp_info.names["cmake_find_package_multi"] = "LSlab"
