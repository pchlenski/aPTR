from src.tests.torch_solver_tests import *

if __name__ == "__main__":
    assert test_init()
    assert test_set_vals()
    assert test_forward_vector()
    assert test_forward_tabular()
    assert test_train_vector()
    assert test_train_tabular()
    assert test_early_stopping()
    assert test_gradients()
