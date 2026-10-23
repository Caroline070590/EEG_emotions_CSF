import unittest
import numpy as np
from recurrence import runs,quantify,build
from train import Model
import torch
class Checks(unittest.TestCase):
    def test_lines(self):
        np.testing.assert_array_equal(runs([1,1,0,1,0,1,1,1]),[2,1,3])
        self.assertEqual(quantify(np.zeros((5,5),bool)),[0,0,0,0,0,0])
    def test_graphs(self):
        x=np.random.default_rng(2).normal(size=(3500,3));g,r=build(x)
        self.assertEqual(g.shape,(3,3,3));self.assertEqual(r.shape,(3,6))
        np.testing.assert_allclose(g,g.transpose(0,2,1),atol=1e-6)
        self.assertTrue(np.isfinite(r).all());self.assertTrue(np.all(np.diagonal(g,axis1=1,axis2=2)==0))
    def test_gradient(self):
        m=Model((0,1,2),True);out=m(torch.eye(30).repeat(2,3,1,1),torch.randn(2,30,36))
        out.sum().backward();self.assertEqual(out.shape,(2,9));self.assertTrue(all(p.grad is not None for p in m.parameters()))
if __name__=='__main__':unittest.main()
