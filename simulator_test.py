import unittest
import numpy as np
from simulator import RetailerEnv

class TestSimulator(unittest.TestCase):

    def test_init(self):
        env = RetailerEnv()
        action = np.random.randint(10)
        time_step = env.reset()
        next_time_step = env.step(action)

    def test_reset(self):
        env = RetailerEnv()
        action = np.random.randint(10)
        next_time_step = env.step(action)
        reset_time_step = env.reset()


        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_step(self):
        #generate step and calculate transition manually
        pass

if __name__ == '__main__':
    unittest.main()