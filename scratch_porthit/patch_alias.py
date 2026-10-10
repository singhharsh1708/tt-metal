import sys
p = sys.argv[1]; s = open(p).read()
a = "    static ttsl::hash::hash_t compute_mesh_workload_hash(\n"
helper = '''    // Which tensor arguments are the same tensor object: for each Tensor leaf, the index of the first
    // leaf sharing its MeshTensor. A cached entry binds tensors by position, so a call that passes one
    // tensor for two parameters and a call that passes two tensors must not share an entry.
    static std::vector<uint32_t> tensor_alias_signature(const tensor_args_t& tensor_args) {
        std::vector<const void*> seen;
        std::vector<uint32_t> signature;
        const auto visit = [&](const ttnn::Tensor& t) {
            const void* id = t.storage_type() == StorageType::DEVICE
                                 ? static_cast<const void*>(&t.mesh_tensor())
                                 : nullptr;
            uint32_t first = static_cast<uint32_t>(seen.size());
            for (uint32_t i = 0; id != nullptr && i < seen.size(); ++i) {
                if (seen[i] == id) {
                    first = i;
                    break;
                }
            }
            seen.push_back(id);
            signature.push_back(first);
        };
        ttsl::reflection::visit_object_of_type<ttnn::Tensor>(visit, tensor_args);
        return signature;
    }

'''
b = "        ttsl::hash::hash_t hash = compute_program_hash(attrs, tensor_args);\n"
b2 = b + "        for (const uint32_t first : tensor_alias_signature(tensor_args)) {\n            hash = ttsl::hash::hash_objects(hash, first);\n        }\n"
c = "            key += ttsl::hash::canonical_key(attrs, tensor_args);\n"
c2 = c + "            for (const uint32_t first : tensor_alias_signature(tensor_args)) {\n                key += ttsl::hash::canonical_key(first);\n            }\n"
for name, x in (("helper anchor", a), ("hash anchor", b), ("key anchor", c)):
    assert s.count(x) == 1, (name, s.count(x))
s = s.replace(a, helper + a).replace(b, b2).replace(c, c2)
open(p, "w").write(s)
