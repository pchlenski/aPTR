from src.db import RnaDB
from src.test.tests import *
import sys
sys.path.append(".")

db = RnaDB(f"./data/db.pkl", f"./data/collisions.pkl")

# test_1()
# test_2()
# test_3()
# test_4()
# test_5()
# test_6()
# test_7(db)
test_8(db)