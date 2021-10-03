

states1 = np.random.rand(3, 3)
c1 = Conceptor().from_states(states1, 0.1)
print(c1.conceptor_matrix)

states2 = np.random.rand(3, 3)
c2 = Conceptor().from_states(states2, 0.1)
print(c2.conceptor_matrix)

