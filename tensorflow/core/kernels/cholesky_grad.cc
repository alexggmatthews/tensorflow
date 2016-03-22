/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "third_party/eigen3/Eigen/Core"

#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {

template <typename T>
class CholeskyGrad : public OpKernel {
 public:
  explicit CholeskyGrad(OpKernelConstruction* context) : OpKernel(context) {}
  using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ConstMatrixMap = Eigen::Map<const Matrix>;
  using MatrixMap = Eigen::Map<Matrix>;
  using ConstRef = Eigen::Ref<const Matrix>;
  using Ref = Eigen::Ref<Matrix>;

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor_l = context->input(0);
    const Tensor& input_tensor_grad = context->input(1);
    // Check that input tensors represent a matrix.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_tensor_l.shape()), errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_tensor_grad.shape()), errors::InvalidArgument("In[1] is not a matrix"));
    // Check that input tensors are square.
    OP_REQUIRES(context, input_tensor_l.dim_size(0) == input_tensor_l.dim_size(1), errors::InvalidArgument("Input matrix must be square."));
    OP_REQUIRES(context, input_tensor_grad.dim_size(0) == input_tensor_grad.dim_size(1), errors::InvalidArgument("Input matrix must be square."));

    // Check that input tensors are of same size.
    OP_REQUIRES(context, input_tensor_l.dim_size(0) == input_tensor_grad.dim_size(0), errors::InvalidArgument("Input matrices must be of same size."));

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor_grad.shape(), &output_tensor));

    if (output_tensor->NumElements() == 0) {
      // the output shape is a 0-element matrix, so there is nothing to do.
      return;
    }
    // The next three lines are necessary to get Eigen matrix behaviour.
    const ConstMatrixMap input_matrix_l(input_tensor_l.flat<T>().data(), input_tensor_l.dim_size(0), input_tensor_l.dim_size(1));
    const ConstMatrixMap input_matrix_grad(input_tensor_grad.flat<T>().data(), input_tensor_grad.dim_size(0), input_tensor_grad.dim_size(1));
    MatrixMap output_matrix(output_tensor->template flat<T>().data(), input_tensor_l.dim_size(0), input_tensor_l.dim_size(1) );

    const int64 kMatrixSize = input_matrix_l.rows();
    const int64 kMaxBlockSize = 32;

    output_matrix = input_matrix_grad.template triangularView<Eigen::Lower>();
    for ( int64 block_end = kMatrixSize ; block_end > 0ll; block_end-= kMaxBlockSize ) {
      const int64 block_begin = std::max(0ll, block_end - kMaxBlockSize);
      const int64 block_size = block_end - block_begin;
      const int64 trailing_size = kMatrixSize - block_size;
      output_matrix.block(block_end, block_begin, trailing_size , block_size) = input_matrix_l.block(block_begin, block_begin, block_size, block_size)
                                                                                                 .adjoint()
                                                                                                 .template triangularView<Eigen::Upper>()
                                                                                                 .solve(output_matrix.block(block_end, block_begin, trailing_size, block_size)
                                                                                                 .adjoint() )
                                                                                                 .adjoint();

      output_matrix.block(block_begin, block_begin, block_size, block_size) -= (output_matrix.block(block_end, block_begin, trailing_size, block_size)
                                                                                                 .adjoint()
                                                                                                 * input_matrix_l.block(block_end, block_begin, trailing_size, block_size))
                                                                                                 .template triangularView<Eigen::Lower>();
      output_matrix.block(block_end, 0, trailing_size, block_begin)  -=  output_matrix.block(block_end, block_begin, trailing_size, block_size)
                                                                                          * input_matrix_l
                                                                                          .block(block_begin, 0, block_size, block_begin);

      output_matrix.block(block_begin, 0, block_size, block_begin) -= output_matrix.block(block_end, block_begin, trailing_size, block_size)
                                                                                      .adjoint()
                                                                                      * input_matrix_l.block(block_end, 0, trailing_size, block_begin);
      CholeskyGradUnblocked(input_matrix_l.block(block_begin, block_begin, block_size, block_size), output_matrix
                                          .block(block_begin, block_begin, block_size, block_size));
      output_matrix.block(block_begin, 0, block_size, block_begin) -= (output_matrix.block(block_begin, block_begin, block_size, block_size)
                                                                                        +output_matrix.block(block_begin, block_begin, block_size, block_size)
                                                                                        .adjoint() )
                                                                                        * input_matrix_l
                                                                                        .block(block_begin, 0, block_size, block_begin);
    }
    output_matrix = (0.5 * (output_matrix +  output_matrix.transpose())).eval();
  }
  void CholeskyGradUnblocked(const ConstRef l_block, Ref grad_block) {
    const int64 kMatrixSize = l_block.rows();
    for (int64 k = kMatrixSize-1; k >= 0; k--) {
      grad_block(k, k) -= (l_block.block(k+1, k, kMatrixSize-(k+1), 1).adjoint() * grad_block.block(k+1, k, kMatrixSize-(k+1), 1))(0, 0) / l_block(k, k);
      grad_block.block(k, k, kMatrixSize-k, 1) /= l_block(k, k);
      grad_block.block(k, 0, 1, k) -= grad_block.block(k, k, kMatrixSize-k, 1).adjoint() * l_block.block(k, 0, kMatrixSize-k, k);
      grad_block.block(k+1, 0, kMatrixSize-(k+1), k) -= grad_block.block(k+1, k , kMatrixSize-(k+1), 1) * l_block.block(k, 0, 1, k);
      grad_block(k, k) *= 0.5;
    }
  }
};

REGISTER_LINALG_OP("CholeskyGrad", (CholeskyGrad<float>), float);
REGISTER_LINALG_OP("CholeskyGrad", (CholeskyGrad<double>), double);
}  // namespace tensorflow
