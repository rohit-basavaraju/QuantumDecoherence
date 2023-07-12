#!/usr/bin/env python
# coding: utf-8

# The purpose of this notebook is to model the decoherence of a qubit. We use a rotationally invariant linear product of spins in the Pauli matrix representation as the interaction Hamiltonian. The expectation is that time evolution under such a Hamiltonian will result in entanglement between the qubit and its environment, yielding a reduced mixed-state density matrix recognized by non-zero eigenvalues.
# 
# N > 1: The qubit is entangled with a multi-qubit environment. 
# 
# v4: wrap simulation in function and call large number of times (100-1000) to calculate aggregate stats e.g. histograms of mean magnitude of off-diagonal reduced density matrix, histogram of eigenvalues of reduced density matrix

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import itertools


# In[2]:


# helper functions
def create_env_basis_list(N):
    """
    Create 2^N different basis states from 0, 1

    Arguments:
    - N: The number of elements in the string to create.

    Returns:
    - perms: List of 2^N strings '0100...'
    """

    perms = list(itertools.product('01', repeat=N))
    
    return perms

def create_ham_order_list(N):
    """
    Create different permutations from 0, 1 for Hamiltonian

    Arguments:
    - N: The number of elements in the string to create.

    Returns:
    - perms: List of strings '0100...'
    """

    perms = list(itertools.permutations('0'*(N-1)+'1'))
    
    return perms

def run_quantum_sim(dt, num_steps, N, ham, q_global):
    """
    Runs the quantum sim and produces reduced density matrix

    Arguments:
    - dt: The timestep for solver.
    - num_steps: The number of timesteps over which to iterate.
    - N: The number of environmental qubits.
    - ham: The global Hamiltonian matrix.
    - q_global: The global quantum state (at the start of the sim and will be updated).

    Returns:
    - eigs_list: list of eigenvalues of the reduced density matrix across sim time 
    - rho_red_od_ll: list of 10 reduced density matrix elements across sim time
    - rho_red_od_ur: list of 01 reduced density matrix elements across sim time
    - rho_red_od_lr: list of 11 reduced density matrix elements across sim time
    - rho_red_od_ul: list of 00 reduced density matrix elements across sim time
    """
        
    # time evolve the global state d|psi> = -i*H|psi>dt

    # record the eigenvalues of the reduced density matrix
    eigs_list = []
    eigvects_list = []
    rho_red_od_ll = []
    rho_red_od_ur = []
    rho_red_od_lr = []
    rho_red_od_ul = []

    for step in range(num_steps):
        q_update = -i*dt*np.dot(ham, q_global)
        heun_term = np.linalg.inv(np.identity(ham.shape[0]) + .5*i*dt*ham) # trying 2nd order solver
        q_global = np.matmul(heun_term, (q_global + q_update*.5))
        q_global_norm = np.sqrt(np.dot(q_global.conj().T, q_global))
        q_global = q_global/(q_global_norm) # need to explicitly normalize since linearization is non-unitary at 2nd order
    
        # form the density matrix of the time-evolved global state
        rho_global = np.outer(q_global, q_global.conj().T)
    
        # trace out the environmental state
        # environmental basis states in global hilbert space
        list_of_env_basis_states = create_env_basis_list(N)
        rho_red = 0
        for perm in list_of_env_basis_states:
            env_basis_state = id_mat
            for j in perm: # note that order of elements in kronecker product will be reverse of perm, but will not matter in trace
                if j == '1':
                    env_basis_state = np.kron(up, env_basis_state)
                elif j == '0':
                    env_basis_state = np.kron(down, env_basis_state)
            rho_red = np.matmul(env_basis_state.conj().T, np.matmul(rho_global, env_basis_state)) + rho_red
    
        # calculate the eigenvalues of the reduced density matrix
        eigs = np.linalg.eigvals(rho_red)
        eigs_list.append(eigs)
    
        # calculate the eigenvectors of the reduced density matrix
        w, eigvects = np.linalg.eig(rho_red)
        eigvects_list.append(eigvects)
    
        # store the off-diagonal terms of the reduced density matrix
        rho_red_od_ll.append(np.linalg.norm(rho_red[1,0]))
        rho_red_od_ur.append(np.linalg.norm(rho_red[0,1]))
    
        # store the off-diagonal terms of the reduced density matrix
        rho_red_od_lr.append(np.linalg.norm(rho_red[1,1]))
        rho_red_od_ul.append(np.linalg.norm(rho_red[0,0]))
        
    return eigs_list, rho_red_od_ll, rho_red_od_ur, rho_red_od_lr, rho_red_od_ul


# In[3]:


# define the qubit (our system)
up = np.array([1,0]).reshape(2,1)
down = np.array([0,1]).reshape(2,1)
    
c1 = np.sqrt(.25)
c2 = np.sqrt(.75)
qubit = np.array([c1, c2]) # c1|0> + c2|1>
qubit = qubit.reshape(2,1) # reshape as a 2x1 column vector


# In[4]:


# define the environment (let's try N>1 now)
N = 5 # number of environmental qubits

q_env = down # get the ball rolling 
for n in range(N-1):
    #if np.round(np.random.rand()) == 0:
    if n%2 == 1:
        q_env = np.kron(down, q_env)
    else:
        q_env = np.kron(up, q_env)


# In[5]:


# define the interaction Hamiltonian between qubit and evironment (linear product of spins)
# pauli matrices
i = complex(0,1)
id_mat = np.array([[1,0],[0,1]])
sigma_x = np.array([[0, 1], [1,0]])
sigma_y = np.array([[0, -i], [i,0]])
sigma_z = np.array([[1, 0], [0,-1]])

ham = 0
coupling = 100 # coupling strength between environment and qubit
'''
list_of_ham_orders = create_ham_order_list(N)
for perm in list_of_ham_orders:
    ham_x = sigma_x
    ham_y = sigma_y
    ham_z = sigma_z
    for j in perm: # note that order of elements in kronecker product will be reverse of perm, but will not matter in sum
        if j == '1':
            ham_x = np.kron(sigma_x, ham_x)
            ham_y = np.kron(sigma_y, ham_y)
            ham_z = np.kron(sigma_z, ham_z)
        elif j == '0':
            ham_x = np.kron(id_mat, ham_x)
            ham_y = np.kron(id_mat, ham_y)
            ham_z = np.kron(id_mat, ham_z)
    ham = ham + coupling*(ham_x + ham_y + ham_z)
'''


# In[6]:


# define the global combined state of qubit and environment
q_global = np.kron(q_env, qubit)


# In[ ]:


# time evolve the global state d|psi> = -i*H|psi>dt
dt = .1
num_steps = 5000 #50000

# run the sim and return the reduced density matrix elements + eigenvalues
num_sims = 100
all_eigs = []
all_rho_red_od_ll = []
all_rho_red_od_ur = []
all_rho_red_od_lr = []
all_rho_red_od_ul = []

for sim_count in range(num_sims):
    # trying arbitrary hermitian matrices as the hamiltonian
    ham = coupling*np.random.rand(2**(N+1), 2**(N+1)) + i*(np.random.rand(2**(N+1), 2**(N+1)))
    ham = (ham + ham.conj().T)/2
    eigs_list, rho_red_od_ll, rho_red_od_ur, rho_red_od_lr, rho_red_od_ul\
    = run_quantum_sim(dt, num_steps, N, ham, q_global)
    all_eigs.append(np.mean(eigs_list,0))
    all_rho_red_od_ll.append(np.mean(rho_red_od_ll))
    all_rho_red_od_ur.append(np.mean(rho_red_od_ur))
    all_rho_red_od_lr.append(np.mean(rho_red_od_lr))
    all_rho_red_od_ul.append(np.mean(rho_red_od_ul))


# In[ ]:


# prep arrays for plotting
eigs_arr = np.array(eigs_list)
eig1 = np.real(np.min(eigs_arr, 1))
eig2 = np.real(np.max(eigs_arr, 1))
t_arr = np.arange(0,dt*num_steps,dt)

smooth_window = 100 # units of dt
eig1_smooth = pd.Series(eig1).rolling(smooth_window, min_periods=1).mean()
eig2_smooth = pd.Series(eig2).rolling(smooth_window, min_periods=1).mean()
rho_red_od_ll_smooth = pd.Series(rho_red_od_ll).rolling(smooth_window, min_periods=1).mean()
rho_red_od_ur_smooth = pd.Series(rho_red_od_ur).rolling(smooth_window, min_periods=1).mean()
rho_red_od_lr_smooth = pd.Series(rho_red_od_lr).rolling(smooth_window, min_periods=1).mean()
rho_red_od_ul_smooth = pd.Series(rho_red_od_ul).rolling(smooth_window, min_periods=1).mean()


# In[ ]:


# plot the eigenvalues as a function of time
plt.figure(1)
plt.subplot(2,2,1)
plt.plot(t_arr, eig2,  label='eig2 ($\mu$='+str(np.round(np.mean(eig2),2))+\
         ', '+'$\sigma$='+str(np.round(np.std(eig2),2))+\
         ','+' $T_w$='+str(int(smooth_window*dt))+')')
plt.plot(t_arr, eig1, label='eig1 ($\mu$='+str(np.round(np.mean(eig1),2))+\
         ', '+'$\sigma$='+str(np.round(np.std(eig1),2))+\
         ','+' $T_w$='+str(int(smooth_window*dt))+')')
plt.plot(t_arr, eig1_smooth, color='black')
plt.plot(t_arr, eig2_smooth, color='black')
plt.grid()
plt.xlabel('time')
plt.ylabel('classical probability')
plt.title('Reduced Density Matrix Eigenvalues over Time ($n_e$='+str(N)+\
          r', $\alpha$='+str(coupling)+', dt='+str(dt)+')')
plt.ylim([0,1])
plt.legend()

plt.subplot(2,2,2)
plt.plot(t_arr, rho_red_od_ur, label=r'$|\rho_{01}|$ ($\mu$='+str(np.round(np.mean(rho_red_od_ll),2))+\
         ', '+'$\sigma$='+str(np.round(np.std(rho_red_od_ll),2))+\
         ','+' $T_w$='+str(int(smooth_window*dt))+')')
plt.plot(t_arr, rho_red_od_ll, label=r'$|\rho_{10}|$ ($\mu$='+str(np.round(np.mean(rho_red_od_ll),2))+\
         ', '+'$\sigma$='+str(np.round(np.std(rho_red_od_ll),2))+\
         ','+' $T_w$='+str(int(smooth_window*dt))+')')
plt.plot(t_arr, rho_red_od_ll_smooth, color='black')
plt.plot(t_arr, rho_red_od_ur_smooth, color='black')
plt.grid()
plt.xlabel('time')
plt.ylabel('norm of off-diagonal terms (10 and 01 elements)')
plt.title('Reduced Density Matrix Off-Diagonal Terms over Time ($n_e$='+str(N)+\
          r', $\alpha$='+str(coupling)+', dt='+str(dt)+')')
plt.ylim([0,1])
plt.legend()

plt.subplot(2,2,3)
plt.plot(t_arr, rho_red_od_ul, label=r'$|\rho_{00}|$ ($\mu$='+str(np.round(np.mean(rho_red_od_ul),2))+\
         ', '+'$\sigma$='+str(np.round(np.std(rho_red_od_ul),2))+\
         ','+' $T_w$='+str(int(smooth_window*dt))+')')
plt.plot(t_arr, rho_red_od_lr, label=r'$|\rho_{11}|$ ($\mu$='+str(np.round(np.mean(rho_red_od_lr),2))+\
         ', '+'$\sigma$='+str(np.round(np.std(rho_red_od_lr),2))+\
         ','+' $T_w$='+str(int(smooth_window*dt))+')')
plt.plot(t_arr, rho_red_od_ul_smooth, color='black')
plt.plot(t_arr, rho_red_od_lr_smooth, color='black')
plt.grid()
plt.xlabel('time')
plt.ylabel('norm of diagonal terms (00 and 11 elements)')
plt.title('Reduced Density Matrix Diagonal Terms over Time ($n_e$='+str(N)+\
          r', $\alpha$='+str(coupling)+', dt='+str(dt)+')')
plt.ylim([0,1])
plt.legend()


# In[ ]:


# plot the histograms
binwidth = .01

# histogram of eigenvalues
plt.subplot(2,2,4)
plt.hist(all_eigs, label=r'$\mu$='+str(np.mean(np.real(all_eigs)))+r', $\sigma$='+str(np.std(all_eigs)))
plt.xlim([0,1])
plt.ylim([0, num_sims])
plt.grid()
plt.title('Reduced Density Matrix Eigenvalues ($N_{sims}$='+str(num_sims)+', $n_e$='+str(N)+\
          r', $\alpha$='+str(coupling)+', dt='+str(dt)+')')
plt.legend()

# histogram of rho_00
plt.figure(2)
plt.subplot(2,2,1)
plt.hist(all_rho_red_od_ul, label=r'$\mu$='+str(np.round(np.mean(all_rho_red_od_ul),2))+\
         r', $\sigma$='+str(np.round(np.std(all_rho_red_od_ul),2)))
plt.xlim([0,1])
plt.ylim([0, num_sims])
plt.grid()
plt.title(r'Reduced Density Matrix $\rho_{00}$ Diagonal Term ($N_{sims}$='+str(num_sims)+', $n_e$='+str(N)+\
          r', $\alpha$='+str(coupling)+', dt='+str(dt)+')')
plt.legend()

# histogram of rho_11
plt.subplot(2,2,4)
plt.hist(all_rho_red_od_lr, label=r'$\mu$='+str(np.round(np.mean(all_rho_red_od_lr),2))+\
         r', $\sigma$='+str(np.round(np.std(all_rho_red_od_lr),2)))
plt.xlim([0,1])
plt.ylim([0, num_sims])
plt.grid()
plt.title(r'Reduced Density Matrix $\rho_{11}$ Diagonal Term ($N_{sims}$='+str(num_sims)+', $n_e$='+str(N)+\
          r', $\alpha$='+str(coupling)+', dt='+str(dt)+')')
plt.legend()

# histogram of rho_01
plt.subplot(2,2,2)
plt.hist(all_rho_red_od_ur, label=r'$\mu$='+str(np.round(np.mean(all_rho_red_od_ur),2))+\
         r', $\sigma$='+str(np.round(np.std(all_rho_red_od_ur),2)))
plt.xlim([0,1])
plt.ylim([0, num_sims])
plt.grid()
plt.title(r'Reduced Density Matrix $\rho_{01}$ Off-Diagonal Term ($N_{sims}$='+str(num_sims)+', $n_e$='+str(N)+\
          r', $\alpha$='+str(coupling)+', dt='+str(dt)+')')
plt.legend()

# histogram of rho_10
plt.subplot(2,2,3)
plt.hist(all_rho_red_od_ll, label=r'$\mu$='+str(np.round(np.mean(all_rho_red_od_ll),2))+\
         r', $\sigma$='+str(np.round(np.std(all_rho_red_od_ll),2)))#,\
#         bins=np.arange(0, 1 + binwidth, binwidth))
plt.xlim([0,1])
plt.ylim([0, num_sims])
plt.grid()
plt.title(r'Reduced Density Matrix $\rho_{10}$ Off-Diagonal Term ($N_{sims}$='+str(num_sims)+', $n_e$='+str(N)+\
          r', $\alpha$='+str(coupling)+', dt='+str(dt)+')')
plt.legend()

plt.show()