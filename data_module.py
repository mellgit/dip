# #    from numpy import array
# import numpy
   
#    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
#    training_set_outputs = array([[0, 1, 1, 0]]).T

#    print(training_set_inputs)

observation_vector = [
[1, 1, 0, 0, 1, 1, 0], 
[0, 1, 1, 0, 1, 1, 0], 
[1, 1, 1, 0, 0, 1, 0], 
[1, 0, 1, 1, 0, 0, 1], 
[1, 1, 1, 1, 0, 1, 1], 
[1, 0, 1, 1, 1, 1, 0], 
[0, 0, 0, 0, 1, 0, 1], 
[0, 1, 1, 1, 1, 0, 0]
]

required_response = [[1, 0, 1, 1, 1, 1, 0, 0]]

new_situation = [1, 1, 1, 1, 1, 0, 0]