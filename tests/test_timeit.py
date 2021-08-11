

import timeit


setup_code = """
name = "Pylenin"
result_list = []
"""


main_code = """
for char in name:
    result_list.append(char)
"""

print(timeit.timeit(stmt=main_code,
                    setup=setup_code,
                    number=10000))
