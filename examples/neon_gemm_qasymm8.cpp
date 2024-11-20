/*
 * Copyright (c) 2020-2021 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"

#include <cstdlib>
#include <vector>
  
using namespace arm_compute;
using namespace std;

int main(){
    vector<float> input(1*32*3*100, 1);
    vector<float> out(32*100*64, 0);
    vector<float> weights(9*3*64, 1);
    vector<float> biases(64, 1);

    Tensor input_tensor;
    Tensor weights_tensor;
    Tensor biases_tensor;
    Tensor out_tensor;

    input_tensor.allocator()->init(TensorInfo(TensorShape(1, 3, 32, 100), 3, DataType::F32, DataLayout::NCHW));
    TensorShape weights_sh(64, 3, 3, 3);
    TensorShape biases_sh(64);
    TensorShape out_sh(1, 64, 32, 100);

    TensorInfo input_info = TensorInfo(TensorShape(1, 3, 32, 100), 3, DataType::F32, DataLayout::NCHW);
    TensorInfo weights_info = TensorInfo(TensorShape(64, 3, 3, 3), 1, DataType::F32);
    TensorInfo biases_info = TensorInfo(TensorShape(64), 1, DataType::F32);
    TensorInfo out_info = TensorInfo(TensorShape(1, 64, 32, 100), 1, DataType::F32);


    Status status = NEConvolutionLayer::validate(&input_info, &weights_info, &biases_info, &out_info, PadStrideInfo(1, 1, -1, -1));
    cout << status.error_description() << endl;

    weights_tensor.allocator()->init(TensorInfo(weights_sh, 1, DataType::F32));
    biases_tensor.allocator()->init(TensorInfo(biases_sh, 1, DataType::F32));
    out_tensor.allocator()->init(TensorInfo(out_sh, 1, DataType::F32));

    NEConvolutionLayer neConv;
    neConv.configure(&input_tensor, &weights_tensor, &biases_tensor, &out_tensor, PadStrideInfo(1, 1, -1, -1));
    
    input_tensor.allocator()->import_memory(input.data());
    weights_tensor.allocator()->import_memory(weights.data());
    biases_tensor.allocator()->import_memory(biases.data());
    out_tensor.allocator()->import_memory(out.data());
    
    cout << out[650] << endl;//0
    neConv.run();
    cout << out[650] << endl;//
    
    return 0;
}