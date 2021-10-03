from conceptorflow import Conceptor, alignment, similarity
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

C5 = cf.aperture_adaptation(cf.aperture_adaptation(C1, 0.2), 0.1)
print(C1.conceptor_matrix)
print(C5.conceptor_matrix)

print('SVD')
print(C1.conceptor_matrix)

print('similarity')
print(similarity(C1, C1))
print(similarity(C1, C4))

print('alignment')
print(alignment(C2, state_cloud[0]))
print(alignment(C3, state_cloud[0]))
