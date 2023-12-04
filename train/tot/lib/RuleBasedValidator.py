import re

class RuleBasedValidator:

    def __init__(self, t):
        self.tokenizer = t

    # returns 1 (valid) if the current step is a solution that has the 
    # correct length, 0 (neutral) if the current step is an intermediate
    # step, and -1 (invalid) if the current step is a solution that has
    # and incorrect length
    def __call__(self, ex, clue):
        ans = re.findall(r'^[A-Z\- ]+$', ex)
        val = 0
        if ans[0] == ex:
            val = 1
            length = re.findall(r'\([1-9,\- ]+\)', clue)
            length = re.findall(r'[^\(\)]+', length[0])[0]
            if "," in length:
                length = length.split(",")
                ans = ans.split(" ")
                for i in range(len(length)):
                    if len(ans[i]) != int(length[i]):
                        val = -1
            elif "-" in length:
                length = length.split("-")
                ans = ans.split("-")
                for i in range(len(length)):
                    if len(ans[i]) != int(length[i]):
                        val = -1
            else:
                if len(ans) != int(length):
                        val = -1
        return val
            
