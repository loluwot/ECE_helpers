from collections import defaultdict
import numpy as np
from SetCoverPy import setcover
MAX_VARS = 10
count_cache = [bin(x).count('1') for x in range(1 << MAX_VARS)]
class Statement:
    alias = {'|' : '+', '&' : '*', '+': '+', '*': '*'}
    operator_order = {'+' : 0, '*' : 1, ')' : -1, '(' : -2}
    operator_func = {'+' : lambda x, y: x | y, '*' : lambda x, y: x & y}

    @classmethod
    def from_table(cls, table, variables):
        
        pass

    def __init__(self, s):
        self.s = s
        self.variables = sorted(list(set([x for x in s if x.isalpha()])))
        self.initialize()

    def initialize(self):
        self.table, self.table_ints = self.truth_table()

    def parse(self, values):
        alias, operator_order, operator_func = Statement.alias, Statement.operator_order, Statement.operator_func
        operator_stack = []
        variable_stack = []
        def eval_op():
            operator = operator_stack.pop()
            variable_stack.append(operator_func[operator](variable_stack.pop(), variable_stack.pop()))
        for c in self.s:
            if c == '(':
                operator_stack.append(c)
            elif c == "'":
                variable_stack.append(1 - variable_stack.pop())
            elif c in operator_order or c in alias:
                while operator_stack and operator_order[operator_stack[-1]] >= operator_order[alias[c]]:
                    eval_op()
                operator_stack.append(alias[c])
            elif c in self.variables:
                variable_stack.append(values[c])
        while operator_stack:
            eval_op()
        return variable_stack[-1]
    
    def truth_table(self):
        table = []
        table_ints = []
        for ii in range(1 << len(self.variables)):
            values = defaultdict(int)
            x = len(self.variables) - 1
            while ii >= 1:
                values[self.variables[x]] = ii % 2
                ii //= 2
                x -= 1
            table_ints.append(self.parse(values))
            table.append((values, self.parse(values)))
        return table, table_ints

    def print_truth_table(self):
        print(' '.join(self.variables), '| f')
        print('-'*(len(self.variables)*2+3))
        for values, result in self.table:
            print(' '.join([str(values[x]) for x in self.variables]), '|', result)

    def minterms(self):
        return [i for i, v in enumerate(self.table_ints) if v == 1]

    def simplify(self):
        indices = self.minterms()
        counts = [[[] for _ in range(max(indices).bit_length()+1)] for zz in range(1 << max(indices).bit_length())]
        for i in indices:
            counts[0][count_cache[i] if i < len(count_cache) else bin(i).count('1')].append(MergedMinterm(i))
        queue = [0]
        visited = set([0])
        merged = set()
        while queue:
            cur_merge = queue.pop()
            for count in range(len(counts[cur_merge]) - 1):
                for idx_1 in counts[cur_merge][count]:
                    for idx_2 in counts[cur_merge][count+1]:
                        if (z := idx_1 | idx_2):
                            new_merge = (cur_merge | z[1])
                            z[0].dont_care = new_merge
                            counts[new_merge][count].append(z[0])
                            #merged
                            if new_merge not in visited:
                                queue.append(new_merge)
                                visited.add(new_merge)
                            merged.add(idx_1)
                            merged.add(idx_2)
        # print(counts)
        unmerged = set()
        for i, z in enumerate(counts):
            for l in z:
                for idx in l:
                    if idx not in merged:
                        unmerged.add(idx)
        rows = np.array([[1 if z in x.minterms else 0 for z in indices] for x in unmerged])
        T = setcover.SetCover(rows.T, np.ones(len(unmerged)))
        _, _ = T.SolveSCP()
        self.s = '+'.join([j.expression(self.variables) for i, j in zip(T.s, unmerged) if i])
        self.initialize()
    

class MergedMinterm:
    def __init__(self, index, minterms=None, dont_care=0):
        self.index = index
        self.minterms = minterms or set([index])
        self.dont_care = dont_care
    
    def merge(self, other):
        if not (z := self.index ^ other.index) & (z - 1):
            return MergedMinterm(self.index, self.minterms | other.minterms), z
        return None
    
    def expression(self, variables):
        exp = []
        index = self.index
        care = self.dont_care
        counter = len(variables) - 1
        while index >= 1:
            if care % 2 == 0:
                exp = [variables[counter] + ('\'' if not index % 2 else '')] + exp
            index >>= 1
            care >>= 1
            counter -= 1
        return '&'.join(exp)

    def __or__(self, other):
        return self.merge(other)
    
    def __str__(self):
        return f'{self.index} : {self.minterms}'

    def __hash__(self):
        return hash(frozenset(self.minterms))
    
    def __eq__(self, o):
        return self.minterms == o.minterms


