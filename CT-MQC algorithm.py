## File that user has to provide for simulations
## Contains parameters for simulation, potential and the initial conditions

import time as time_module
import numpy as np
import matplotlib.pyplot as plt




nquant=2 ## Quantum states
nclass=1 ## number of classical d.o.f.
ntraj=10 ## number of trajectories

#*************************************** in loops => i -> ntraj; j -> nquant; k -> nclass   *****************


## State of the system variables
mass=np.zeros(nclass)
x=np.zeros(nclass)
v=np.zeros(nclass)
ci=np.zeros(nquant,dtype = complex)

## Parameters of potential (in atomic units)
mass[0]=2000
hbar=1.0
AA=6.e-4
BB=0.1
CC=0.9

## Parameters of simulation
dt= 0.1                  ## Time step
total_time= 300.0           ## Total simulation time
ntim=int(total_time/dt)    ## Number of steps
time = np.zeros(ntim)
time[0] = 0
for t in range(ntim):
    if t == 0 : continue
    time[t] = time[t-1] + dt

###############################################
## Input: x(nclass)
## Returns: BO Potential Ei(nquant), its derivative grad_Ei(nquant,nclass) and derivative coupling dij(nquant,nquant,nclass)
def pot(x):
    V=np.zeros((nquant,nquant),dtype=np.complex_)
    dv_dx=np.zeros((nquant,nquant,nclass),dtype=np.complex_)
    V[0,0]=AA#*np.tanh(BB*x[0])
    V[1,1]=-V[0,0]
    # if(x[0]<0.0):
    if(x < 0.0 ):
      V[0,1]=BB*np.exp(CC*x)
    else:
      V[0,1]=BB*(2-np.exp(-CC*x))
    V[1,0]=V[0,1]

    dv_dx[0,0,0]=0.0
    dv_dx[1,1,0]=0.0
    # if(x[0]<0.0):
    if (x < 0.0):
      dv_dx[0,1,0]=BB*CC*np.exp(CC*x)
    else:
      dv_dx[0,1,0]=BB*CC*np.exp(-CC*x)

    dv_dx[1,0,0]=dv_dx[0,1,0]

    Ei,phi=np.linalg.eigh(V)

    grad_Ei=np.zeros((nquant,nclass),dtype=np.complex_)
    for j in range(nquant):
      for i in range(nclass):
        grad_Ei[j,i]=sum(phi[:,j]*np.matmul(dv_dx[:,:,i],phi[:,j]))

    # note i shifted the nclass from third position to first position
    dij=np.zeros((nclass, nquant, nquant),dtype=np.complex_)

    # hence loop for nclass was last loop before if conditon and after j loop
    # also changed the dij => previous it was dij[i,j,k]
    for k in range(nclass):
        for i in range(nquant):
          for j in range(nquant):
            if(i!=j):
                dij[k, i,j]=sum(phi[:,i]*np.matmul(dv_dx[:,:,k],phi[:,j]))/(Ei[j]-Ei[i])
    return (Ei,grad_Ei,dij)
###################################################

## Initial condition
## Returns position, velocity, initial state, and ci's
def init_cond(ntraj, nclass , nquant):
    k= 30          #np.sqrt(2*mass[0]*0.03)   # k = 10
    sig_x= 20/k    # 1.0/np.sqrt(2.0)
    sig_v=hbar/(2* mass[0]*sig_x )

    R_I_v = np.zeros((ntraj,nclass),dtype=np.complex_)
    for i in range(ntraj):
        for k in range(nclass):
            R_I_v[i][k] =   0 + sig_x*np.random.normal()
            print("position is: ",sig_x*np.random.normal())

    v = np.zeros((ntraj,nclass))
      # velocity(ntraj , nclass)
    for i in range(ntraj):
        for k in range(nclass):
            v[i][k] = hbar*k/mass[k] + sig_v*np.random.normal()

    C_I_l = np.zeros((ntraj, nquant), dtype=np.complex_)
     # C = 1 for ground state and C= 0 for ex state for each traj
    C_I_l[:,0] = 0.9
    C_I_l[:,1] = 0.43

    return(R_I_v, v ,C_I_l )
################################################################################

def user_input():
    nquant = 0 # no. of quantum states   (l)
    nclass = 0 # no. of classical degrees of freedom  (v)
    ntraj = 0  # no. of trajectories    (I)

    print("enter the number of classical degrees of freedom: ")
    nclass = int(input())
    print("enter the number of quantum states: ")
    nquant = int(input())
    print("enter the number of trajectories: ")
    ntraj = int(input())
    return (nquant, nclass, ntraj)

########################################################################
def cal_force(force_I_l_v, R_I_v ,dt , ntraj, nquant, nclass):

    # force = initial force + f_dot * dt
    # f_dot = - dell(v) * epsilon(I,l,t)
    for i in range(ntraj):
        x = R_I_v[i, 0]
        (Ei, grad_Ei, dij) = pot(x)

        for k in range(nclass):
            for j in range(nquant):
                force_I_l_v[i][j][k] = force_I_l_v[i][j][k] - grad_Ei[j][k] * dt   #grad_Ei=np.zeros((nquant,nclass))

    return force_I_l_v

###############################################################################
#func to calculate the classical momentum
def calc_clas_momentum(P_I_v_dot, clas_P_I_v ,dt ,ntraj,nclass):
    clas_P_I_v = np.zeros((ntraj,nclass),dtype=np.complex_)

    for k in range(nclass):
        for i in range(ntraj):
            clas_P_I_v[i][k] = clas_P_I_v[i][k] + P_I_v_dot[i][k] * dt


    return clas_P_I_v



##############################################################################
#function to calculate the classical force
def deriv_clas_momentum(ntraj, nquant, nclass,R_I_v, rho_I_l_l, force_I_l_v , P_I_v):
    P_I_v_dot = np.zeros((ntraj,nclass),dtype=np.complex_)
    for v in range(nclass):  # correspond the research paper subscripts
        for i in range(ntraj):
            a = 0
            b = 0
            c = 0

            x = R_I_v[i, 0]
            (Ei, grad_Ei, dij) = pot(x)

            for k in range(nquant):
                a += rho_I_l_l[i][k][k] * grad_Ei[k][v]     # grad_Ei (nquant,nclass)

            for k in range(nquant):
                for l in range(nquant):
                    b += rho_I_l_l[i][l][k] * (Ei[k] - Ei[l]) * dij[v][l][k]  # condition of l != k is removed

            for l in range(nquant):

                e = 0
                d = 0
                for v_prime in range(nclass):
                    e += (2 / (hbar * mass[v_prime])) * P_I_v[i][v_prime] * force_I_l_v[i][l][v_prime]

                for k in range(nquant):
                    d += (rho_I_l_l[i][k][k] * force_I_l_v[i][k][v])   # could be error

                c += rho_I_l_l[i][l][l] * e * (d - force_I_l_v[i][l][v])

            P_I_v_dot[i][v] = (-1)*a - b - c

    return P_I_v_dot


################################################################################
# function to calculate the quantum momentum
def quan_momentum(C_I_l, var_sigma_sqr,R_I_v, rho_I_l_l ,force_I_l_v , nclass,ntraj):

    P_I_v = np.zeros((ntraj, nclass),dtype=np.complex_)
    for k in range(nclass):
        for i in range(ntraj):
            a = 0
            b = 0

            # calculation of part 'a'
            for j in range(2): # range of quantum states

                a += ((complex.conjugate(C_I_l[i][j]))* C_I_l[i][j]) / ((var_sigma_sqr[j][k] )**2)    # could be error here

            c = 0 #denominator
            for j in range(ntraj):
                c += rho_I_l_l[j][0][0] * rho_I_l_l[j][1][1] * ( force_I_l_v[j][0][k] - force_I_l_v[j][1][k])

            for j in range(ntraj):
                b += ( R_I_v[j][k] * rho_I_l_l[j][0][0] * rho_I_l_l[j][1][1] * ( force_I_l_v[j][0][k] - force_I_l_v[j][1][k])) / c

            P_I_v[i][k] = hbar * a * (R_I_v[i][k] - b )

    return P_I_v

#################################################################################
# function to calculate mean pos and its variance through equation 23
def mean_pos_And_var(rho_I_l_l, R_I_v ,ntraj , nquant ,nclass):

    # initialising The mean position  R_l_v =>
    Rv_mean= np.zeros((nquant,nclass),dtype=np.complex_)
    var_sigma_sqr = np.zeros((nquant,nclass),dtype=np.complex_) # var_sigma_sqr[j][k]



    # i -> ntraj ; j -> nquant ; nclass -> k
    for j in range(nquant):
        total_rho = complex(0, 0)
        for i in range(ntraj):
            total_rho += rho_I_l_l[i][j][j]  # fixed l and loop over trajectory

        for k in range(nclass):  # for nclass -> v
            for i in range(ntraj):  # for loop over -> I

                Rv_mean[j][k] +=  (R_I_v[i][k]  *  rho_I_l_l[i][j][j] / total_rho)

                # calculate variance => sigma^2
            for i in range(ntraj):
                var_sigma_sqr[j][k] += ( (R_I_v[i][k] - Rv_mean[j][k] )**2 ) * rho_I_l_l[i][j][j] / total_rho
    return (Rv_mean, var_sigma_sqr)


#################################################################################
#function to calculate the derivative of coefficients
def derivative_coeff(R_I_v, C_I_l, P_I_v ,clas_P_I_v ,force_I_l_v ,ntraj, nquant, nclass):
    C_dot_I_l = np.zeros((ntraj, nquant),dtype=np.complex_)

    for i in range(ntraj):
        x = R_I_v[i, 0]
        (Ei, grad_Ei, dij) = pot(x)
        for j in range(nquant):
            a = 0

            c = 0

            e = 0
            iota_hbar = complex(0,1/hbar)



            for j1 in range(nquant):
                b = 0
                for k in range(nclass):
                    b += (clas_P_I_v[i][k] / mass[k]) * dij[k][j][j1]
                a += C_I_l[i][j1] * b


            for k in range(nclass):
                d = 0
                for j1 in range(nquant):
                    d += ((abs(C_I_l[i][j1]))**2) * force_I_l_v[i][j1][k]

                c += ( P_I_v[i][k] / (hbar * mass[k]) * (d - force_I_l_v[i][j][k]) * C_I_l[i][j] )

            C_dot_I_l[i][j] = -iota_hbar * Ei[j] * C_I_l[i][j] - a - c


    return C_dot_I_l


####################################################################################################
def evolve_parameters(R_I_v, C_I_l, clas_P_I_v, P_I_v_dot, C_dot_I_l ,dt , ntraj, nquant, nclass):


    for i in range(ntraj):
        for k in range(nclass):
            R_I_v[i][k] = R_I_v[i][k] + (clas_P_I_v[i][k] /mass[k] ) * dt

            clas_P_I_v[i][k] = clas_P_I_v[i][k] + P_I_v_dot[i][k]* dt

        for j in range(nquant):

            C_I_l[i][j] = C_I_l[i][j] + C_dot_I_l[i][j] * dt



    return (R_I_v,clas_P_I_v ,C_I_l)

############################################################################3###
def testing_pot():
    ## Testing potential - not a subroutine of main function
    pos = np.zeros(100)
    en = np.zeros((100, nquant))
    deriv = np.zeros((100, nquant, nquant, nclass))

    Ei,grad_Ei,dij = pot()

    # for i in range(100):
    #     x[0] = -15.0 + 30.0 * i / 100.0
    #     Ei, grad_Ei, dij = pot(x)
    #     pos[i] = x[0]
    #     en[i, :] = Ei
    #     deriv[i, :, :, :] = dij
    #
    # plt.plot(pos, en[:, 0])
    # plt.plot(pos, en[:, 1])
    # plt.plot(pos, deriv[:, 0, 1, 0])

    # plt.show()

    return

################################################################################
def test():
    i = -15
    arr = np.zeros(31)
    grad_Ei_matrix = np.zeros((31,2),dtype=np.complex_)
    y = 0
    while(i <= 15):
        (Ei, grad_Ei, dij) = pot(i)
        grad_Ei_matrix[y][0] = grad_Ei[0][0]
        grad_Ei_matrix[y][1] = grad_Ei[1][0]
        print("ground grad_Ei: ", grad_Ei[0][0])
        print("ex grad_Ei: ", grad_Ei[1][0])
        arr[y] = i
        y += 1
        i += 1

    plt.plot(arr, grad_Ei_matrix[:,0])
    plt.plot(arr, grad_Ei_matrix[:,1])

    # testing_pot()
    plt.xlabel("R (au)")
    plt.ylabel("GRAD")
    plt.show()
    return

###########################################################################################
#main function - code start from main()
def main():

    start = time_module.time() # for testing speed of implemention of code
    # (nquant, nclass, ntraj) = user_input()
    (R_I_v, v, C_I_l) = init_cond(ntraj, nclass , nquant)


    # calculate the classical momentum
    clas_P_I_v = np.zeros((ntraj,nclass),dtype=np.complex_)

    for i in range(ntraj):
        for k in range(nclass):
            clas_P_I_v[i][k] = v[i][k] * mass[k]

    # rho = C_star X C
    rho_I_l_l = np.zeros((ntraj,nquant, nquant),dtype=np.complex_)


    force_I_l_v = np.zeros((ntraj, nquant, nclass),dtype=np.complex_)

    population_ll = np.zeros((nquant,ntim), dtype=np.complex_)

    #storing populations with respect to time for plotting on pyplot
    x = np.zeros(ntim)
    y = np.zeros(ntim)
    # loop over time
    for t in range(ntim): # time evolution

        force_I_l_v = cal_force(force_I_l_v, R_I_v ,dt , ntraj, nquant, nclass)

        #for calculation of rho
        for i in range(ntraj):
            for j1 in range(nquant):
                for j2 in range(nquant):
                    rho_I_l_l[i][j1][j2] = np.conj(C_I_l[i][j1]) * C_I_l[i][j2]

        # eq 23 , 24
        # first calculation of mean pos by eqn 23
        (mean_pos,var_sigma_sqr) = mean_pos_And_var(rho_I_l_l,R_I_v, ntraj , nquant ,nclass)


        # calculation of equation 27 => quantum momentum
        P_I_v = quan_momentum(C_I_l, var_sigma_sqr,R_I_v, rho_I_l_l,force_I_l_v, nclass,ntraj)


        # Classical force
        P_I_v_dot = deriv_clas_momentum(ntraj, nquant, nclass,R_I_v, rho_I_l_l, force_I_l_v , P_I_v)



        # derivative of coefficients => C_dot_I_l
        C_dot_I_l = derivative_coeff(R_I_v, C_I_l, P_I_v ,clas_P_I_v ,force_I_l_v ,ntraj, nquant, nclass)


        # calculating average population in lth state

        for j in range(nquant):
            a = 0
            for i in range(ntraj):
                a += rho_I_l_l[i][j][j]
            population_ll[j][t] = a/ntraj
            y[t] = population_ll[0][t] + population_ll[1][t]


        (R_I_v, clas_P_I_v ,C_I_l) = evolve_parameters(R_I_v, C_I_l, clas_P_I_v, P_I_v_dot, C_dot_I_l ,dt , ntraj, nquant, nclass)

        print(".")


    for j in range(nquant):
        x = population_ll[j,:]
        plt.plot(time,x)

    plt.xlabel("Time (au)")
    plt.ylabel("Populations")
    plt.legend(["Ground State", "Excited State"], loc="right")

    end = time_module.time()
    print(end - start)

    plt.show()

    return
##################################################################################

main()

# test()


# for j in range(nclass):
#     for i in range(ntraj):
#         print(R_I_v[j][i], end=" ")
#
#     print(end="\n")
