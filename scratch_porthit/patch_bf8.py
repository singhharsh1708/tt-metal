import sys
p = sys.argv[1].rstrip("/") + "/ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp"; s = open(p).read()
inc = '#include "ttnn/operations/copy/typecast/typecast.hpp"\n'
call = "    return prim::unary(\n        input_tensor,\n        op_chain,\n        output_dtype,\n"
assert s.count(inc) == 1, ("include", s.count(inc))
assert s.count(call) == 1, ("call", s.count(call))
guard = '''    // A block-float output shares one exponent per 16 values. Tile padding is zero, and an op with a pole
    // at zero turns it into inf, which flushes the real values of the same group to zero when packed.
    Tensor input = input_tensor;
    const auto first_op = op_chain.front().type();
    const bool pole_at_zero = first_op == unary::UnaryOpType::RECIP || first_op == unary::UnaryOpType::RSQRT ||
                              first_op == unary::UnaryOpType::LOG || first_op == unary::UnaryOpType::LOG2 ||
                              first_op == unary::UnaryOpType::LOG10;
    if (pole_at_zero && (output_dtype == DataType::BFLOAT8_B || output_dtype == DataType::BFLOAT4_B) &&
        input_tensor.layout() == ttnn::TILE_LAYOUT &&
        input_tensor.logical_shape()[-1] != input_tensor.padded_shape()[-1]) {
        input = ttnn::fill_implicit_tile_padding(input_tensor, 1.0f);
    }

'''
s = s.replace(inc, inc + '#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"\n')
s = s.replace(call, guard + "    return prim::unary(\n        input,\n        op_chain,\n        output_dtype,\n")
open(p, "w").write(s)
print("bf8 patch applied")
