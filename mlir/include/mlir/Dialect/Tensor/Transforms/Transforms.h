//===- Transforms.h - Tensor Transformation Patterns ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TENSOR_TRANSFORMS_TRANSFORMS_H
#define MLIR_DIALECT_TENSOR_TRANSFORMS_TRANSFORMS_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace tensor {

/// Populates `patterns` with patterns that split linalg.pad_tensor ops by
/// creating scf.if ops to wrap linalg.pad_tensor ops and handle
/// padding-unncessary and padding-needed cases separately.
void populateSplitPaddingPatterns(RewritePatternSet &patterns,
                                  PatternBenefit baseBenefit = 1);

} // namespace tensor
} // namespace mlir

#endif // MLIR_DIALECT_TENSOR_TRANSFORMS_TRANSFORMS_H
