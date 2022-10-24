from classes import *
test_statement = Statement("a&b' + b&c' + a&c'")
test_statement.print_truth_table()
test_statement.simplify()

print(test_statement.s)

