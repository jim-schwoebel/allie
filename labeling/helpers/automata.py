''' Cellular automata

Could be useful for audio applications.

REFERENCES

https://faingezicht.com/articles/2017/01/23/wolfram/
http://mathworld.wolfram.com/Rule30.html
'''
def window(iterable, stride=3):
    for index in range(len(iterable) - stride + 1):
        yield iterable[index:index + stride]

def generate_pattern(state, rule, MAX_TIME):
    for time in range(MAX_TIME):
        print(state)
        patterns = window(state)
        state = ''.join(rule[pat] for pat in patterns)
        state = '0{}0'.format(state)
    print(state)

'''
window function creates list of all possible states 

list(window('footbar'))
['foo', 'oot', 'otb', 'tba', 'bar']'''


#rule 30, 90, 110, 184
RULES = {30: {"111": '0', "110": '0', "101": '0', "000": '0',
              "100": '1', "011": '1', "010": '1', "001": '1'},

         90: {"111": "0", "110": "1", "101": "0", "100": "1",
              "011": "1", "010": "0", "001": "1", "000": "0"},

         110: {"111": '0', "110": '1', "101": '1', "100": '0',
               "011": '1', "010": '1', "001": '1', "000": '0'},

         184: {"111": "1", "110": "0", "101": "1", "100": "1",
               "011": "1", "010": "0", "001": "0", "000": "0"}
         }


initial_state = '00000000000000000000100000000000000000000'

    
list(window(initial_state))
generate_pattern(initial_state, RULES[30], 30)


# figure out the rules of the system in speech propagation....
