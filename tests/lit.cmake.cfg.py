# -*- Python -*-

import os

import lit.formats
import lit.util

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "MLIR_TUTORIAL"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir", ".py"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.project_binary_dir, "tests")

config.substitutions.append(('%PYTHON', sys.executable))
config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(("%project_source_dir", config.project_source_dir))

# Add Torch-MLIR and MLIR Python bindings
torch_mlir_build_dir = os.path.join(config.project_source_dir, "externals", "torch-mlir", "build")
torch_mlir_python_dir = os.path.join(torch_mlir_build_dir, "tools", "torch-mlir", "python_packages", "torch_mlir")
mlir_python_dir = os.path.join(torch_mlir_build_dir, "tools", "mlir", "python_packages", "mlir_core")

if llvm_config:
    llvm_config.with_environment("PYTHONPATH", torch_mlir_python_dir, append_path=True)
    llvm_config.with_environment("PYTHONPATH", mlir_python_dir, append_path=True)

#torch_mlir_python_path = os.path.join(config.project_binary_dir, "externals", "torch-mlir", 'build', 'tools', 'torch-mlir', 'python_packages')
#sys.path.insert(0, torch_mlir_python_path)
#config.environment['PYTHONPATH'] = torch_mlir_python_path + os.pathsep + \
#    config.environment.get('PYTHONPATH', '')

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ["Inputs", "Examples", "CMakeLists.txt", "README.txt", "LICENSE.txt", "lit.cmake.cfg.py", "get_fx_graph.py"]

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.project_binary_dir, "test")
config.project_tools_dir = os.path.join(config.project_binary_dir, "tools")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.project_tools_dir, config.llvm_tools_dir]
tools = [
    "mlir-opt",
    "mlir-runner",
    "torch-mlir-opt",
]
print("The path:", config.llvm_tools_dir)
llvm_config.add_tool_substitutions(tools, tool_dirs)

#llvm_config.with_environment(
#    "PYTHONPATH",
#    [
#        os.path.join(config.torch_mlir_python_packages_dir, "torch_mlir"),
#    ],
#    append_path=True,
#)

#torch_mlir_python_path = os.path.join(config.project_binary_dir, 'externals', 'torch-mlir', 'build', 'tools', 'mlir', 'python_packages', 'mlir_core')
#config.environment['PYTHONPATH'] = torch_mlir_python_path + ':' + config.environment.get('PYTHONPATH', '')

#torch_mlir_python_path = os.path.join(config.project_binary_dir, 'externals', 'torch-mlir', 'build', 'tools', 'torch-mlir', 'python_packages', 'torch_mlir')
#config.environment['PYTHONPATH'] = torch_mlir_python_path + ':' + config.environment.get('PYTHONPATH', '')
