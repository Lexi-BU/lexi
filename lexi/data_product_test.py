import importlib

import lxi_read_binary_data as lrbd

importlib.reload(lrbd)

file_val = "/home/vetinari/Desktop/git/Lexi-Bu/lexi/data/from_PIT/20230816/"
multiple_files = True
t_start = "2024-05-23 10:26:20"
t_end = "2024-05-23 10:38:20"

file_name, df = lrbd.read_binary_file(
    file_val=file_val, t_start=t_start, t_end=t_end, multiple_files=multiple_files
)
