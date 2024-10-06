
import numpy as np
import random 
import matplotlib.pyplot as plt
import seaborn as sn


# BASICS

# defining the computational basis
zero=np.array([[1],[0]])
one=np.array([[0],[1]])

# defining the hadamard transformed x basis
plus=(1/np.sqrt(2))*(zero + one)
minus = (1/np.sqrt(2))*(one - zero)

# function for kronecker product for n input
def kronproduct(state):
    product = state[0]
    for counter in range(1,len(state)):
        product=np.kron(product, state[counter])
    return product

# defining the dot product for n inputs
def dotproduct(state):
    product=state[0]
    for counter in range(1,len(state)):
        product=np.dot(product,state[counter])
    return(product[0])


def outerproduct(x,y):
    element=np.dot(x,np.matrix.getH(y))
    return element


def expectation_value(state,operator):
    val=dotproduct(([np.matrix.getH(state),operator,state]))
    return val

#common matrix operations
identity=np.array ([[1,0],[0,1]])
paulix=np.array ([[0,1],[1,0]])
pauliz=np.array ([[1,0],[0,-1]])

# controlled gates
C_H=([1,0,0,0],[0,1,0,0],[0,0,1/np.sqrt(2),1/np.sqrt(2)],[0,0,1/np.sqrt(2),-1/np.sqrt(2)])
C_Y=([1,0,0,0],[0,1,0,0],[0,0,0,complex(0,-1)],[0,0,complex(0,1),0])
C_Z=([1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1])
C_S=([1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,complex(0,1)])

#SECTION 1: Operator spread and random gates

# 1.1 without measurements

initial_state=kronproduct([plus,minus,plus,minus,zero,minus,plus,minus])

# 1.1.1 operator is alternate hadamard gate

operator_odd_C_H=kronproduct([C_H,C_H,C_H,C_H])
operator_even_C_H=kronproduct([identity,C_H,C_H,C_H,identity])

def projection_operator_random_qubit(qubit_i,operator):
    projection_operator_Qi=[]
    for counter in range(8):     #no of qubits, can be made more general
        if counter+1==qubit_i:
            projection_operator_Qi.append(operator)
        else:
            projection_operator_Qi.append(identity)
    return kronproduct(projection_operator_Qi)

 
def unitary_evolved_state_list(operator_odd,operator_even,initial_state,n): 
    state_list=[initial_state]
    for counter in range(n):
        element=0
        if counter % 2==0:
            element=np.dot(operator_odd,state_list[counter])
        else:
            element=np.dot(operator_even,state_list[counter])
        state_list.append(element)
    return state_list


hadamard_evolved_state_list=unitary_evolved_state_list(operator_odd_C_H, operator_even_C_H, initial_state,50)

def expectation_value_pauliz(qubit_i,state_list):
    expectation_value_Qi_list=[]
    for state in (state_list):
        expectation_value_Qi_list.append((abs(expectation_value(state,projection_operator_random_qubit(qubit_i,pauliz))))[0])
    return expectation_value_Qi_list


generations_list=[]
for counter in range(51):
    generations_list.append(counter)

#for counter in range (1,9):
 #   plt.plot(generations_list,expectation_value_pauliz(counter))

data=[]
for counter in range (1,9):
    data.append(expectation_value_pauliz(counter,hadamard_evolved_state_list))

hm_expectation_hadamard=sn.heatmap(data=data)

hm_expectation_hadamard.invert_yaxis()

hm_expectation_hadamard.set_yticklabels(np.arange(1, len(data) + 1))

hm_expectation_hadamard.set_xlabel('Generations')
hm_expectation_hadamard.set_ylabel('Qubits')

hm_expectation_hadamard.collections[0].colorbar.set_label('Expectation value')
#hm_expectation_hadamard.set_title('Operator spread using controlled hadamard gate in brickwork without measurements')


plt.show()


# 1.1.2 Random operators

random_operators=[C_H,C_Z,C_Y,C_S] # randomisation based on that of list index

def random_operator_sequence(n):   # to select a list of consecutive random operators, input being the length of sequence
    operator_list=[]
    for counter in range(n):
        random_number=random.randint(0,3)
        random_element=random_operators[random_number]
        operator_list.append(random_element)
    return operator_list
    

def random_operator_list(generations):  # to add a loop after every step such that the choice of operator after every generation is randomised
    operator_list=[]
    for generation_counter in range(generations):
        if generation_counter%2==0:
            odd_operator=kronproduct(random_operator_sequence(4))
            operator_list.append(odd_operator)
        else:
            even_operator_list=random_operator_sequence(3)
            even_operator=kronproduct([identity,even_operator_list[0],even_operator_list[1],even_operator_list[2],identity])
            operator_list.append(even_operator)
    return (operator_list)


random_evolved_state_list=[initial_state]   # for loop to apply the operator to state 
for counter in range(len(random_operator_list(50))):
    random_evolved_state_list_element=np.dot(random_operator_list(50)[counter],random_evolved_state_list[counter])
    random_evolved_state_list.append(random_evolved_state_list_element)

data_random_operators=[]

for counter in range (1,9):
    data_random_operators.append(expectation_value_pauliz(counter,random_evolved_state_list))

hm_expectation_random=sn.heatmap(data=data_random_operators)
hm_expectation_random.invert_yaxis()

hm_expectation_random.set_yticklabels(np.arange(1, len(data) + 1))

hm_expectation_random.set_xlabel('Generations')
hm_expectation_random.set_ylabel('Qubits')

hm_expectation_random.collections[0].colorbar.set_label('Expectation value')
#hm_expectation_random.set_title('Operator spread using random gates in brickwork without measurements')

plt.show()


# 1.2 Spread of operator with measurements

# 1.2.1 Hadamard evolved

measurement_operator_qubit=(identity+paulix)/2
      
def measurement_operator_list(generations):
    operator_list=[]
    for counter in range (generations):
        qubit=random.randint(1,8)
        list_element=projection_operator_random_qubit(qubit, measurement_operator_qubit)
        operator_list.append(list_element)
    return operator_list

def hadamard_operator_list(generations):
    operator_list=[]
    for counter in range(generations):
        if counter%2==0:
            operator_list.append(operator_odd_C_H)
        else:
            operator_list.append(operator_even_C_H)
    return (operator_list)

def unitary_measurement_evolved_state_list(initial_state, unitary_operator_list, measurement_operator_list):
    state_list = [initial_state]
    for counter in range(len(unitary_operator_list)):
        unitary_state_element = np.dot(unitary_operator_list[counter], state_list[counter])
        expectation_val = abs(expectation_value(unitary_state_element, measurement_operator_list[counter]))[0]
        
        if expectation_val == 0:
            measured_state_element = np.zeros_like(unitary_state_element)
        else:
            measured_state_element = np.dot(measurement_operator_list[counter], unitary_state_element) / np.sqrt(expectation_val)
 
        state_list.append(measured_state_element)
        print(expectation_val)
    return state_list

hadamard_measurement_evolved_state_list=unitary_measurement_evolved_state_list(initial_state, hadamard_operator_list(50), measurement_operator_list(50))

data_expectation_value_hadamard_measurement=[]

for counter in range(1,9):
    data_expectation_value_hadamard_measurement.append(expectation_value_pauliz(counter, hadamard_measurement_evolved_state_list))

hm_expectation_hadamard_measured=sn.heatmap(data=data_expectation_value_hadamard_measurement)
hm_expectation_hadamard_measured.invert_yaxis()

hm_expectation_hadamard_measured.set_yticklabels(np.arange(1, len(data) + 1))

hm_expectation_hadamard_measured.set_xlabel('Generations')
hm_expectation_hadamard_measured.set_ylabel('Qubits')

hm_expectation_hadamard_measured.collections[0].colorbar.set_label('Expectation value')
#hm_expectation_hadamard_measured.set_title('Operator spread using controlled hadamard gate in brickwork with measurements')

plt.show()


# 1.2.2 Random operators

random_measurement_evolved_state_list=unitary_measurement_evolved_state_list(initial_state, random_operator_list(50), measurement_operator_list(50))

data_expectation_value_randomoperators_measurement=[]


for counter in range(1,9):
    data_expectation_value_randomoperators_measurement.append(expectation_value_pauliz(counter, random_measurement_evolved_state_list))

hm_expectation_random_measured=sn.heatmap(data=data_expectation_value_randomoperators_measurement)
hm_expectation_random_measured.invert_yaxis()


hm_expectation_random_measured.set_yticklabels(np.arange(1, len(data) + 1))

hm_expectation_random_measured.set_xlabel('Generations')
hm_expectation_random_measured.set_ylabel('Qubits')

hm_expectation_random_measured.collections[0].colorbar.set_label('Expectation value')
#hm_expectation_random_measured.set_title('Operator spread using random gates in brickwork with measurements')

plt.show()


# SECTION 2: Fidelity

# 2.1 without measurements

def fidelity(state_list):
    projection_list=[]
    for counter in range (len(state_list)):
        projection_element=np.dot(np.dot(np.transpose(initial_state), state_list[counter]),np.dot(np.transpose(state_list[counter]),initial_state))
        projection_list.append(float(abs(projection_element[0])))
    return (projection_list)

# 2.1.1 Fidelity of purely Hadamard evolved states

fidelity_hadamard_evolved_states=fidelity(hadamard_evolved_state_list)
plt.plot(generations_list,fidelity_hadamard_evolved_states)
plt.show()
# 2.1.2 Fidelity of randomly evolved states

fidelity_randomly_evolved_states=fidelity(random_evolved_state_list)
plt.plot(generations_list,fidelity_randomly_evolved_states)
plt.show()
# 2.2 with measurements

# 2.2.1 Fidelity of pure Hadamard evolved states

fidelity_hadamard_evolved_measured_states=fidelity(hadamard_measurement_evolved_state_list)
plt.plot(generations_list,fidelity_hadamard_evolved_measured_states)
plt.show()
# 2.2.2 Fidelity of randomly evolved states

fidelity_random_evolved_measured_states=fidelity(random_measurement_evolved_state_list)
plt.plot(generations_list,fidelity_random_evolved_measured_states)
plt.show()

unitary_measurement_evolved_state_list(initial_state, random_operator_list(50), measurement_operator_list(50))


