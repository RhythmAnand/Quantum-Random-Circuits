import numpy as np
import random 
import matplotlib.pyplot as plt

###############################################
# SECTION - 1 Defining variables and operators 
###############################################
# generations
generations=[0,1,2,3,4]

# defining the computational basis
zero=np.array([[1],[0]])
one=np.array([[0],[1]])

# defining the hadamard transformed x basis
plus=(1/np.sqrt(2))*(zero + one)
minus = (1/np.sqrt(2))*(one - zero)

# function for kronecker product for n input
def kronproduct(state):
    product = 1
    for counter in range(len(state)):
        product=np.kron(product, state[counter])
    return product

# defining the dot product for n inputs
def dotproduct(state):
    product=state[0]
    for counter in range(1,len(state)):
        product=np.dot(product,state[counter])
    return(product)


#common matrix operations
identity=np.array ([[1,0],[0,1]])
paulix=np.array ([[0,1],[1,0]])
pauliz=np.array ([[1,0],[0,-1]])

# controlled hadamard gate
C_H=([1,0,0,0],[0,1,0,0],[0,0,1/np.sqrt(2),1/np.sqrt(2)],[0,0,1/np.sqrt(2),-1/np.sqrt(2)])


###################################################
# SECTION - 2 Unitary evolution
###################################################

# SECTION - 2.1 States under the unitary operations 

# choosing an arbitrary intial state
initial_state=kronproduct([plus,minus,plus,minus])

#operators applied in the brick wall network
operator_gen1=kronproduct([C_H,C_H])
operator_gen2=kronproduct([identity,C_H,identity])
operator_gen3=operator_gen1
operator_gen4=operator_gen2

# evolution of the system under unitary gates
state_1 = (np.dot(operator_gen1,initial_state))
state_2 =(np.dot(operator_gen2,state_1))
state_3=(np.dot(operator_gen3,state_2))
state_4=(np.dot(operator_gen4,state_3))

# list of states after every generation on applying unitary gates
unitary_state_list=[initial_state,state_1,state_2,state_3,state_4]

######################################################

# SECTION - 2.2  Fidelity of unitarily evolved quantum states
''' Projection of the unitarily evolved state on the initial state, yileds the probabilty amplitude
(ranging from 0 to 1) of getting that state under the unitary evolution'''

# takes the inner product of the operationally obtained quantum states witht the initial state
def inner_product(state_list):
    projection_list=[]
    for counter in range (len(state_list)):
        projection_element=np.dot(np.dot(np.transpose(initial_state), state_list[counter]),np.dot(np.transpose(state_list[counter]),initial_state))
        projection_list.append(float(projection_element[0]))
    return (projection_list)

# probability of getting a particular state under the given unitary evolution 
fidelity_unitary_evolution=inner_product(unitary_state_list)
#print("Fidelity of state evolved under unitary gates:",fidelity_unitary_evolution)

######################################################
# SECTION - 3 Projections
######################################################

# SECTION - 3.1 Randomising the selection of a projection operator (defining functions)
'''
(i) projection operator for a definite qubit and a definite projection basis
(ii) randomise the selection of the projection basis based on the probability amplitude 
of the state on the respective basis and setting up an arbitrary parameter which 
indeterministically chooses the basis on which it collapses
(iii) successive application of projection operators, generate projected states which
depend further on the history of application of the operators
'''

# projection matrices corresponding to the computational basis, 0 and 1
projection_zero=([1,0],[0,0])
projection_one=([0,0],[0,1])

# deterministically defines a projection operator by specifying the projection basis and the
# qubit on which it acts 
def projection_operator(i,basis):
    operator=[]
    for counter in range(4):
        if counter+1==i:
            operator.append(basis)
        else:
            operator.append(identity)
    final_operator=kronproduct(operator)
    return final_operator

# calculates the probability of the given state collapsing on 0 or 1
# where the projection operator is defined considering both the basis and the particular qubit
def probability(i,state):
    probability_zero=(np.dot(np.transpose(state), np.dot(projection_operator(i,projection_zero),state)))
    probability_one=1-probability_zero
    probability_state=np.dot(np.transpose(state),state)
    return [i,probability_zero,probability_one,probability_state]


# (measure in the sequence 4,3,1,2)

# indeterminstic selection of the projection basis
def random_projection_generator(probability_matrix):
    i=random.randint(0, 100)/100
    final_projection_operator=0
    if i<probability_matrix[1]:
        final_projection_operator=projection_operator(probability_matrix[0],projection_zero)
    else:
        final_projection_operator=projection_operator(probability_matrix[0],projection_one)
    return final_projection_operator

######################################################################

# SECTION - 3.2 Purely projective states

# SECTION - 3.2.1 States under the projections

#generation1:
pure_P_1=random_projection_generator(probability(4,initial_state))
pure_projected_state_1=np.dot(pure_P_1, initial_state)

#generation2:
pure_P_2=random_projection_generator(probability(3,pure_projected_state_1))
pure_projected_state_2=np.dot(pure_P_2,pure_projected_state_1)

#generation3:
pure_P_3=random_projection_generator(probability(1,pure_projected_state_2))
pure_projected_state_3=np.dot(pure_P_3,pure_projected_state_2)

#generation4:
pure_P_4=random_projection_generator(probability(2,pure_projected_state_3))
pure_projected_state_4=np.dot(pure_P_4,pure_projected_state_3)

# list of all the generations of projected states
pure_projection_list=[initial_state,pure_projected_state_1,pure_projected_state_2,pure_projected_state_3,pure_projected_state_4]

#############################################################

# SECTION - 3.2.2 Fidelity under purely projective operations

# probability of getting a particular state under the given indeterministic projections
fidelity_pure_projection=inner_product(pure_projection_list)

#print("Fidelity of states evolved under pure projections: ",fidelity_pure_projection)

#############################################################

# SECTION - 4 Random brickwork circuit
'''
(i) apply the unitary gate on the pure state
(ii) apply projection and repeat the sequence
'''

# SECTION - 4.1 States under the repeated application of the operators and projections

#generation1:
P_1=random_projection_generator(probability(4,state_1))
projected_state_1=np.dot(P_1, state_1)

#generation2:
projected_evolved_state_1=(np.dot(operator_gen2,projected_state_1))
P_2=random_projection_generator(probability(3,projected_evolved_state_1))
projected_state_2=np.dot(P_2,projected_evolved_state_1)

#generation3:
projected_evolved_state_2=(np.dot(operator_gen1,projected_state_2))
P_3=random_projection_generator(probability(1,projected_evolved_state_2))
projected_state_3=np.dot(P_3,projected_evolved_state_2)

#generation4:
projected_evolved_state_3=(np.dot(operator_gen2,projected_state_3))
P_4=random_projection_generator(probability(2,projected_evolved_state_3))
projected_evolved_state_4=np.dot(P_4,projected_evolved_state_3)

# list of generation of states on applying the corresponding unitary gate and the projection
projected_evolved_list=[initial_state,projected_evolved_state_1,projected_evolved_state_2,projected_evolved_state_3,projected_evolved_state_4]

# SECTION - 4.2 Fidelity of the brickwork network

fidelity_projection_evolved=inner_product(projected_evolved_list)

#print("Fidelity of states evolved under the brickwork network: ",fidelity_projection_evolved)


'''
plt.scatter(generations,fidelity_unitary_evolution,s=10, label="Unitary")
plt.scatter(generations,fidelity_pure_projection,s=10,label="Pure projections")
plt.scatter(generations,fidelity_projection_evolved,s=10,label="Unitary-projections" )

plt.xlabel('Generations')
plt.ylabel('Fidelity')

plt.legend()

plt.title('')
plt.legend()
plt.show()
'''

#################################################### 
# SECTION - 5 Calculating Probability of obtaining definite states
####################################################

'''
(i) Generate all possible permutations of the sequential application of the 
projection operator on the definite qubits
(ii) All the indeterministic sequence of the projection operators is a subset 
of the deterministically constructed permutations
(iii) Use the permutation to construct the projection operators acting on a
definite sequence of qubits (here, 4->3->1)
(iv) Taking the dot product of all the projection operators gives the composite
projection operator that takes the initial state to the final state
(v) Calculate the probability of obtaining a given state ket under the specified
permutation by taking an inner product of the final state with itself
(vi) Evaluated the expectation value of projecting the final state kets randomly 
on pauli z basis
'''


# program to generate all possible permutations possible of the successively applied projection operators
projection_list=[1,0]
combination_list=[]
projection_operator_list=[]

for i in range(2):
    for counter_i in range(3):
        combination=[]
        projection_combination=[]
        for counter_total in range(3):
           
            if counter_i==counter_total:
                if i==0:
                    combination.insert(counter_i,0)
                else:
                    combination.insert(counter_i,1)
                combination_list.append(combination)
                     
            else:
               if i==0:
                   combination.insert(counter_total,1)
               else:
                   combination.insert(counter_total,0)
               
# appending the two trivial homogeneous states
combination_list.append([1,1,1])
combination_list.append([0,0,0])

#print(combination_list)

# generating the operator corresponding to the given permutation on specified qubit
def successive_projection_operators(combination_list,qubit_states):
    operator_list=[]
    for counter in range(len(combination_list)):
        for qubit_index in qubit_states:
            operator=[]
            if combination_list[counter]==0:
                operator=projection_operator(qubit_index,projection_zero)
            
            else:
               operator=projection_operator(qubit_index,projection_one)
            operator_list.append(operator)
    return(operator_list)

# generates a compiled sequential list corresponding to each projected state specified in the permutation list
operator_list=(successive_projection_operators(combination_list, [4,3,1]))

# program to group the projection operators in groups of 3, i.e. the number of generations over which they are applied
final_operator_list=[]

for i in range(8):
    final_operator_list_element=[]
    operator_element=0
    for counter in range(3):
        operator_element=operator_list[3*i+counter]
        final_operator_list_element.append(operator_element)
    final_operator_list.append(final_operator_list_element)

# multiplying the projection operators corresponding to a permutation and applying 
# final compostition to the initial state
projection_operator_list=[]
final_projected_state_list=[]

for counter in range (8):
    projection_operator_element=1
    final_projected_state_element=0
    
    projection_operator_element=dotproduct(final_operator_list[counter])
    projection_operator_list.append(projection_operator_element)
    
    final_projected_state_element=np.dot(projection_operator_list[counter],initial_state)
    final_projected_state_list.append(final_projected_state_element)

#print(projection_operator_list)
#print(final_projected_state_list)

# probabilty of obtaining a final state under the composite projections
# expectation value of projecting the final state ket randomly on pauli z basis
probability_list=[]
expectation_sigmaz=[]

for counter in range(8):
    expectation_element=0
    probability_element=0
    expectation_element=dotproduct([np.transpose(final_projected_state_list[counter]),random_projection_generator(probability(2,pure_projected_state_3)),final_projected_state_list[counter]])
    probability_element=np.dot(np.transpose(final_projected_state_list[counter]),final_projected_state_list[counter])
    expectation_sigmaz.append(expectation_element)
    probability_list.append((probability_element))
    
#print(probability_list)
#print(expectation_sigmaz)


################################################################
#SECTION 6: Fidelity trend
################################################################
 
operator_1=operator_gen1
operator_2=operator_gen2


 
def unitary_evolved_state_list(initial_state,n): 
    state_list=[initial_state]
    for counter in range(n):
        element=0
        if counter % 2==0:
            element=np.dot(operator_1,state_list[counter])
        else:
            element=np.dot(operator_2,state_list[counter])
        state_list.append(element)
    return state_list

state_list_ngen=unitary_evolved_state_list(initial_state, 50)     
state_list_ngen_fidelity=inner_product(state_list_ngen)  

generations_list=[]
for counter in range(51):
    generations_list.append(counter)

#plt.plot(generations_list,state_list_ngen_fidelity)

plt.xlabel('Generations')
plt.ylabel('Fidelity')

plt.legend()

plt.title('')
plt.legend()
plt.show()


###########################################################
#SECTION 7: Local Operator spread: expectation value
###########################################################

operator_expectation=pauliz

initial_state_expectation=kronproduct([plus,zero,plus,minus])
state_list_expectation=unitary_evolved_state_list(initial_state_expectation,50)

operator_q1_expectation=kronproduct([pauliz,identity,identity,identity])
operator_q2_expectation=kronproduct([identity,pauliz,identity,identity])
operator_q3_expectation=kronproduct([identity,identity,pauliz,identity])
operator_q4_expectation=kronproduct([identity,identity,identity,pauliz])

def expectation_value(state_list,operator):
    expectation_list=[]
    for counter in range(len(state_list)):
        value=abs(dotproduct([np.transpose(state_list[counter]),operator,state_list[counter]]))
        expectation_list.append(float(value[0]))
    return expectation_list

expectation_list_q1=expectation_value(state_list_expectation, operator_q1_expectation)
expectation_list_q2=expectation_value(state_list_expectation, operator_q2_expectation)
expectation_list_q3=expectation_value(state_list_expectation, operator_q3_expectation)
expectation_list_q4=expectation_value(state_list_expectation, operator_q4_expectation)


#plt.plot(generations_list,expectation_list_q1,label='qubit 1')
#plt.plot(generations_list,expectation_list_q2,label='qubit 2')
#plt.plot(generations_list,expectation_list_q3,label='qubit 3')
#plt.plot(generations_list,expectation_list_q4,label='qubit 4')

plt.ylabel('Expectation value over generations')
plt.xlabel('Generations')

plt.legend()

plt.title('')
plt.legend()
plt.show()

############################################################

