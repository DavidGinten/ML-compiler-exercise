llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
  TORCH_MLIR_ENABLE_REFBACKEND
  TORCH_MLIR_ENABLE_STABLEHLO
)

configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set (MLIR_TUTORIAL_TEST_DEPENDS
    FileCheck count not
    #TorchMLIRPythonModules
    mlir-opt
    torch-mlir-opt
    #mlir-runner
)

add_lit_testsuite(check-mlir-tutorial "Running the MLIR tutorial regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${MLIR_TUTORIAL_TEST_DEPENDS}
)
