import copy,unittest
from analyze import check_trace
from tasks import grade
class AuditTests(unittest.TestCase):
    def test_trace_rejects_policy_or_token_switch(self):
        t=dict(chosen=0,base_top=1,base_max=.6,neighbor_distances=[0.]*8,neighbor_values=[0]*8,neighbor_support=[True]*8,
               support_weight=1.,memory_top=0,memory_max=1.,candidate_base_probs={'0':.4,'1':.6},weight=.25)
        check_trace(t,'fixed',0,1e-10)
        with self.assertRaises(AssertionError):check_trace(t,'model_confidence',0,1e-10)
        gated=copy.deepcopy(t);gated['weight']=0.
        with self.assertRaises(AssertionError):check_trace(gated,'model_confidence',0,1e-10)
        gated['chosen']=1;check_trace(gated,'model_confidence',1,1e-10)
    def test_grading_requires_complete_exact_answer(self):
        c=dict(prompt='TASK RECALL\nFACTS a=2;b=7\nASK a\nANSWER:')
        self.assertTrue(grade(c,[50,259])['passed'])
        self.assertFalse(grade(c,[50])['passed'])
        self.assertFalse(grade(c,[128,259])['passed'])
        self.assertFalse(grade(c,[55,259])['passed'])
    def test_last_write_oracle(self):
        from tasks import oracle
        self.assertEqual(oracle('TASK UPDATE\nINIT a=1;b=0\nWRITE a=3;a=7;b=2\nASK a\nANSWER:'),'7')
    def test_pair_rejects_extra_change(self):
        from tasks import check_pair
        a='TASK RECALL\nFACTS a=2;b=5\nASK a\nANSWER:'
        b=a.replace('a=2','a=3').replace('b=5','b=9')
        with self.assertRaises(AssertionError):check_pair(dict(family='memory_recall',past=dict(prompt=a,answer='2'),current=dict(prompt=b,answer='3')))
if __name__=='__main__':unittest.main()
