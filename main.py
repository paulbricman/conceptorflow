from conceptorflow import Conceptor
import conceptorflow as cf
import numpy as np

state_cloud = np.random.rand(3, 3)
C1 = Conceptor().from_states(state_cloud, 0.1)
print(C1.conceptor_matrix)

state_cloud = np.random.rand(3, 3)
C2 = Conceptor().from_states(state_cloud, 0.1)
print(C2.conceptor_matrix)

C3 = cf.conjunction([C1, C2])
C4 = cf.disjunction([C1, C2])

print(cf.compare(C1, C3), cf.compare(C1, C4))
