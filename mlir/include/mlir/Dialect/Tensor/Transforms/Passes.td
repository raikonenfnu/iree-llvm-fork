//===-- Passes.td - pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TENSOR_TRANSFORMS_PASSES
#define MLIR_DIALECT_TENSOR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def TensorBufferize : Pass<"tensor-bufferize", "FuncOp"> {
  let summary = "Bufferize the `tensor` dialect";
  let constructor = "mlir::createTensorBufferizePass()";
}

def TensorSplitPadding : Pass<"tensor-split-padding", "FuncOp"> {
  let summary = "Split `tensor.pad` op into padding-unnecessary and "
                "padding-needed cases";
  let description = [{
     This pass creates scf.if ops to wrap tensor.pad ops to allow handle
     padding-elided and padding-needed cases separately.
  }];
  let constructor = "mlir::createTensorSplitPaddingPass()";
}

#endif // MLIR_DIALECT_TENSOR_TRANSFORMS_PASSES
