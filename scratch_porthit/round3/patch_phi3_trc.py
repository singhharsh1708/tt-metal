import sys

path = "models/tt_transformers/tt/model_config.py"
s = open(path).read()
old = '''        if self.base_model_name in ["Phi-3-mini-128k-instruct"]:
            self.trust_remote_code_hf = True
'''
assert s.count(old) == 1, (path, "trust_remote_code for Phi-3-mini-128k")
open(path, "w").write(s.replace(old, ""))
print("patched", path)
