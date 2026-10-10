import sys
root = sys.argv[1].rstrip("/")


def edit(rel, pairs):
    p = f"{root}/{rel}"; s = open(p).read()
    for old, new, n in pairs:
        assert s.count(old) == n, (rel, old[:60], s.count(old))
        s = s.replace(old, new)
    open(p, "w").write(s)


# index_fill: the kernels read the index as a flat array, so hand them a ROW_MAJOR index
edit("ttnn/cpp/ttnn/operations/index_fill/index_fill.cpp", [
    ('#include "ttnn/operations/index_fill/device/index_fill_device_operation.hpp"\n',
     '#include "ttnn/operations/core/core.hpp"\n#include "ttnn/operations/index_fill/device/index_fill_device_operation.hpp"\n', 1),
    ("    return ttnn::prim::index_fill(input, dim, index, value, memory_config);\n",
     "    // The kernels read the index as a flat array; a TILE index is padded to a whole tile, so untilize it first.\n"
     "    if (index.layout() == ttnn::TILE_LAYOUT) {\n"
     "        return ttnn::prim::index_fill(\n"
     "            input, dim, ttnn::to_layout(index, ttnn::ROW_MAJOR_LAYOUT), value, memory_config);\n"
     "    }\n"
     "    return ttnn::prim::index_fill(input, dim, index, value, memory_config);\n", 1),
])
# gather: rank-1 output and int32 indices
edit("ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp", [
    ('#include "ttnn/operations/core/core.hpp"\n',
     '#include "ttnn/operations/copy/typecast/typecast.hpp"\n#include "ttnn/operations/core/core.hpp"\n', 1),
    ("        output_tensor = ttnn::squeeze_from_4D(output_tensor, orig_rank);\n",
     "        // A rank-1 tile tensor is padded on dim 2, which squeeze_from_4D cannot drop; reshape instead.\n"
     "        output_tensor = orig_rank == 1 ? ttnn::reshape(output_tensor, original_lshape)\n"
     "                                       : ttnn::squeeze_from_4D(output_tensor, orig_rank);\n", 1),
    ("    namespace gather_ns = operations::data_movement::gather;\n\n    const bool use_codegen =\n",
     "    namespace gather_ns = operations::data_movement::gather;\n\n"
     "    // Indices are never negative, so a signed index tensor is read as unsigned.\n"
     "    if (input_index_tensor.dtype() == ttnn::DataType::INT32) {\n"
     "        return gather(\n"
     "            input_tensor,\n"
     "            dim,\n"
     "            ttnn::typecast(input_index_tensor, ttnn::DataType::UINT32),\n"
     "            sparse_grad,\n"
     "            memory_config,\n"
     "            std::move(optional_output_tensor),\n"
     "            sub_core_grids);\n"
     "    }\n\n"
     "    const bool use_codegen =\n", 1),
])
# embedding_bw: return the gradient in the weight's own rank
edit("ttnn/cpp/ttnn/operations/embedding_backward/embedding_backward.cpp", [
    ("    return ttnn::prim::embedding_backward(\n"
     "        input_tensor, output_gradient_tensor_arg, output_mem_config, out_dtype, num_embeddings, optional_output_tensor);\n",
     "    auto grad = ttnn::prim::embedding_backward(\n"
     "        input_tensor, output_gradient_tensor_arg, output_mem_config, out_dtype, num_embeddings, optional_output_tensor);\n"
     "    // The device op works on [1, 1, V, H]; a rank-2 weight gets a rank-2 gradient back.\n"
     "    if (weight_tensor_arg.logical_shape().rank() == 2 && !optional_output_tensor.has_value()) {\n"
     "        return ttnn::reshape(grad, weight_tensor_arg.logical_shape());\n"
     "    }\n"
     "    return grad;\n", 1),
])
print("small patch applied")
