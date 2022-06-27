from hanspell import spell_checker

class Solution:
    def __init__(self) -> None:
        pass
    def grammar(self):
        sent = "맞춤법 틀리면 외 않되? 쓰고싶은대로쓰면돼지 "
        spelled_sent = spell_checker.check(sent)

        hanspell_sent = spelled_sent.checked
        print(hanspell_sent)
    
if __name__=='__main__':
    Solution().grammar()