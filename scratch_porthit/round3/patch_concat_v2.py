path = "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/reader_concat_stick_layout_interleaved_start_id.cpp"
s = open(path).read()
old = """                    tt::data_movement::common::tt_memmove<false, false, false, 0>(
                        noc, l1_write_addr, scratch_addr, page_size);
"""
new = """                    auto* dst = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(l1_write_addr);
                    auto* src = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(scratch_addr);
                    for (uint32_t b = 0; b < page_size; ++b) {
                        dst[b] = src[b];
                    }
                    (void)dst[page_size - 1];
"""
assert s.count(old) == 1, (path, "bounced piece copy")
open(path, "w").write(s.replace(old, new))
print("patched", path)
