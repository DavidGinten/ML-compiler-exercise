add_mlir_library(ConvertMatMulToBlas
    ConvertMatMulToBlas.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/lib/ConvertMatMulToBlas

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRPass
)
