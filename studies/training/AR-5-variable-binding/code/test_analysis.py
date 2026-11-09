import unittest
from analyze import grade_case,decode_ids,GRID
from grader import qualify
def case():return dict(name='f',arguments=['u','v'],prompt='def f(u, v):\n    return',examples=[([x,y],x-y) for x,y in GRID])
class AnalysisTests(unittest.TestCase):
    def test_positional_binding(self):
        c=case();self.assertTrue(grade_case(c,' u-v\n',True)['passed']);self.assertFalse(grade_case(c,' v-u\n',True)['passed'])
    def test_unknown_names_and_eos(self):
        c=case();r=grade_case(c,' a-b\n',True);self.assertTrue(r['syntax']);self.assertEqual(r['undeclared_names'],['a','b']);self.assertFalse(r['passed'])
        self.assertFalse(grade_case(c,' u-v\n',False)['passed'])
    def test_bad_bytes_are_not_valid_code(self):
        self.assertEqual(decode_ids([2,0],{'2':'75'}),('u',True,None))
        self.assertEqual(decode_ids([3,0],{'3':'ff'})[2],'invalid_bytes')
        self.assertEqual(decode_ids([1,0],{})[2],'special_token')
    def test_safe_interpreter_fixtures(self):
        self.assertEqual(qualify(),7)
        self.assertFalse(grade_case(case(),' __import__("os").system("never")\n',True)['supported'])
if __name__=='__main__':unittest.main()
