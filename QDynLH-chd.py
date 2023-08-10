#/usr/bin/env python3.6
import numpy as np
import sympy as sym
from numpy.polynomial import chebyshev as cb
import scipy as scp
import scipy.linalg as la
from scipy import special as spe
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import scipy.integrate as intr
import scipy.signal as signal
from scipy import interpolate as itp
import timeit
from scipy.interpolate import splrep, splev, CubicSpline, bisplrep, bisplev
import sys

import cupy as cp
import cupy.linalg as cla

###### Author: Edith Leal-Sanchez
#
#For any comments or contributions communicate to:
#--------edithalicialeal@gmail.com-----------#
#
#
## Important References:
# J. Chem. Phys. 91, 3571 (1989); https://doi.org/10.1063/1.456888
# Kosloff, R., J. Phys. Chem. 1988, 92, 2087-2100
#J. Chem. Phys. 81, 3967 (1984); https://doi.org/10.1063/1.448136
# Tannor, D.J., "Introduction to Quantum Mechanics. A Time Perspective", University  Science Books, USA, 2007. Specially Chapter 11
###Agregar la cita de kineo...

##-----Kosloff FGH's routines for the evaluation of the first (DF) and second (TFG) derivatives.
def TFG(xx):                                    #xx -> 1-D array of the grid in the position representation.
    N = float(len(xx))
    #print(int(N)//2)
    dk = (2.*np.pi)/(N*(xx[1]-xx[0]))
    TF = np.zeros(int(N),dtype=np.complex_)
    TF2 = [(0.5*(float(l)*dk)**2.) for l in range(-(int(N)//2)+1,int(N)//2+1) ]
    TF[:(int(N)//2)+1] = TF2[int(N)//2-1:]
    TF[(int(N)//2)+1:] = (TF2[:int(N)//2-1])
    TF = TF*np.eye(int(N))
    ww = []
    for a in range(len(xx)):
        wi = []
        wij = np.exp(1j*2.*np.pi/N)
        for b in range(len(xx)):
            wi.append(wij**(float(a)*float(b)))
        ww.append(wi)

    ww = 1./np.sqrt(N)*np.array(ww,dtype=np.complex_)
    Tm= np.matmul(ww,np.matmul(TF,np.conj(ww).T))

    return(Tm)

def DF(xx):
    N = float(len(xx))
    #print(int(N)//2)
    dk = (2.*np.pi)/(N*(xx[1]-xx[0]))
    DF = np.zeros(int(N),dtype=np.complex_)
    DF2 = [(1j*(float(l)*dk)) for l in range(-(int(N)//2)+1,int(N)//2+1) ]
    DF[:(int(N)//2)+1] = DF2[int(N)//2-1:]
    DF[(int(N)//2)+1:] = (DF2[:int(N)//2-1])
    DF = DF*np.eye(int(N))
    ww = []
    for a in range(len(xx)):
        wi = []
        wij = np.exp(1j*2.*np.pi/N)
        for b in range(len(xx)):
            wi.append(wij**(float(a)*float(b)))
        ww.append(wi)

    ww = 1./np.sqrt(N)*np.array(ww,dtype=np.complex_)
    Dm= np.matmul(ww,np.matmul(DF,np.conj(ww).T))

    return(Dm)

####para fft/gpus
def DF_2(xx):
    N = float(len(xx))
    #print(int(N)//2)
    dk = (2.*np.pi)/(N*(xx[1]-xx[0]))
    DF = np.zeros(int(N),dtype=np.complex_)
    DF2 = [(1j*(float(l)*dk)) for l in range(-(int(N)//2)+1,int(N)//2+1) ]
    DF[:(int(N)//2)+1] = DF2[int(N)//2-1:]
    DF[(int(N)//2)+1:] = (DF2[:int(N)//2-1])
    
    return(DF)

def TFG_2(xx):                                    #xx -> 1-D array of the grid in the position representation.
    N = float(len(xx))                      
    #print(int(N)//2)
    dk = (2.*np.pi)/(N*(xx[1]-xx[0]))           
    TF = np.zeros(int(N),dtype=np.complex_)
    TF2 = [(0.5*(float(l)*dk)**2.) for l in range(-(int(N)//2)+1,int(N)//2+1) ]
    TF[:(int(N)//2)+1] = TF2[int(N)//2-1:]
    TF[(int(N)//2)+1:] = (TF2[:int(N)//2-1])
    #print(TF)
    
    return(TF)




####----General Chebyshev propagator, usable with a square Hamiltonian Matrix for  one dimension problems (whenever psi is a vector) 
def cheb_prop(PSI, Ene1, Ene2, Ham, time1, time2):  ##time1 < time2

    Emaxr = np.max([Ene1.real,Ene2.real])
    Eminr = np.min([Ene1.real,Ene2.real])
    Emaxi = np.max([Ene1.imag,Ene2.imag])
    Emini = np.min([Ene1.imag,Ene2.imag])
    Emax = Emaxr+1j*Emaxi
    Emin = Eminr+1j*Emini

    DE = Emax - Emin
    sem_DE = 0.5*DE
    exp_arg = (sem_DE) + Emin

    #--------------------------Beginning of the approximate propagator---------------------------------#
    ham = np.array(Ham,dtype=np.complex_) 
    dt = time2-time1

    alpha = 0.5*DE*dt
    
    exp_prop = np.exp(-1j*exp_arg*dt) 
    #print(DE, exp_arg, Emaxi, Emini)
       
    psi_Mdt_m1 = np.array(PSI,dtype=np.complex_)

    phi0=psi_Mdt_m1
    psi_Mdt = np.zeros(len(psi_Mdt_m1), dtype=np.complex_)
    psi_Mdt+=(spe.jv(0., alpha)*phi0)

    phi1 = ((-1j/sem_DE)*(np.matmul(ham, psi_Mdt_m1) - exp_arg*psi_Mdt_m1))
    psi_Mdt+=(2.*spe.jv(1., alpha)*phi1)
    
    #print(2.*spe.jv(34., alpha)
    for k in range(2, 35):
        psi_k = (-2j/sem_DE)*( np.matmul(ham, phi1) - (exp_arg*phi1) )
        phi_k = psi_k + phi0
        psi_Mdt+=(2.*spe.jv(float(k),alpha)*phi_k)
        phi0 =phi1
        phi1=phi_k
    
   
    psi_mdt = exp_prop*psi_Mdt
   
    norm = np.dot(np.conj(psi_mdt),psi_mdt)
    print(norm)
    psi_mdt_norm = (1./np.sqrt(norm))*psi_mdt
    
    return(psi_mdt_norm)

#Chebyshev propagator for rectangular 2D PES of coupled electronic states in the adiabatic representation.
##poner una rutina con, segun el DE determinar cuantos polinomios se usaran para la expansion de Chebyshev--> recuerda tomar en cuenta la parte imaginaria...
def cheb_prop_internal_nl_nacme_cpu(r, R, PSI, Enes, K_r, K_R, D_r, D_R, Gs, Vs, vio, Tsup, Tinf, Lambs_r, Lambs_R, n, ef, time1, time2):
    ''' for non gpu calculations (THREADING is used, as in MKL, OMP libraries)'''
    '''time1 < time2 for forward time propagation'''
    ''' The original (Psi(r,R)) [[Psi(r1,R1), Psi(r1,R2),...,Psi(r1,Rn)],'''
    '''                          [...],'''
    '''                          [Psi(rm,R1), Psi(rm,R2),...,Psi(rm,Rn)]]'''
    ''' then K1 = K_r and K2 = K_R refer to the second derivative terms of the kinetic energy matrices, please give them in said order. D_r and D_R are the first derivative matrices.'''
    '''Enes is a list that contains the estimations on minimum (potential) and maximum(potential and kinetic) energies that are contained in the grid i.e. estimation on the spectrum of eigenvalues of the Hamiltonian. '''
    ''' n  is the number of levels'''
    ''' Vs, Tsup, Tinf, Lambs_r, Lambs_R are lists of the arrays of the Hamiltonian matrix elements for each level. E.g. V1, potential for S0, 12 the array on that position of the n-level system Hamiltonian '''
    ''' Vs = [ V1, V2, V3, ..., Vn] Diagonal Potential energy matrix elements '''
    ''' vio = CAP for the Mandelshtam recurrence, if direct substitution is used, the CAP must be included in the Vs elements.'''
    '''All the contributions to the off-diagonal, multiplicative contributions to the Hamiltonian must be containend in the lists of arrays Tsup, for the superior triangular and in Tinf for the inferior triangular of the matrix. The lists can contain NACME, NACME derivatives, etc. but if the contribution to the matrix is antihermitean its signs must be adressed beforehand. The order of the elements in the list must be: '''
    '''Tsup order = [ 12,13,...,1n, 23,..,2n,34,...,...,n-1n] '''
    '''Tinf order = [n1,n2,..,nn-1, n-11,n-12,...,n-1n-2,...,n-21,n-22,...,n-2n-3,...,...,21] ----> e.g. n=3: [ L31, L32, L21] Triangular inferior '''
    ''' Separately, the non-adiabatic coupling matrix elements are given, in order to operate the derivative, off-diagonal part. Here only the superior triangular elements are needed. The sign is considered in this algorithm, do not change it beforehand. The list needs to be given in the following order: '''
    '''Lambs_r o Lambs_R = [ 12,13,...,1n, 23,...,2n,34,...,...,n-1n ] Non-adiabatic coupling elements '''
    ''' The separation of terms is only between multiplicative and derivative, diagonal or off-diagonal, so multiplicative terms of the kinetic energy matrix must be adressed in Vs, Tsup or Tinf accordingly. '''
    '''Gs = { 'Grr' : Grr, 'GRR': GRR, 'GrR' : GrR, 'dGrr_dr' : dGrr_dr, 'dGRR_dR' : dGRR_dR, 'dGrR_dr' : dGrR_dr, 'dGrR_dR' : dGrR_dR } G matrix elements'''

    Enes=np.array(Enes,dtype=np.complex_)
    Emax_re = np.max(Enes.real)  #The maximum routine selects the
    Emin_re = np.min(Enes.real)  #element with the maximum real part,
    Emax_im = np.max(Enes.imag)  #not the one with the greater magnitude
    Emin_im = np.min(Enes.imag)  #
    Emin = Emin_re + 1j*Emin_im  
    Emax = Emax_re + 1j*Emax_im

    DE = Emax-Emin

    sem_DE = 0.5*DE
    exp_arg = (sem_DE) + Emin
    print(DE, Emin,Emax, exp_arg)

    #--------------------------Beginning of the approximate propagator---------------------------------#

    dt = time2-time1

    alpha = 0.5*DE*dt

    exp_prop = np.exp(-1j*exp_arg*dt)
    #---
    Qn_y = 1. #np.exp(0.050*vio)        #Multiplying first by the cap in order to eliminate the aliased part.
    #---
    psi_Mdt_m1 = Qn_y*np.array(PSI)

    #phi = []                                                    #All the polynomials are stored here.
    #phi.append(psi_Mdt_m1)                                      #the zeroth Chebychev polynomial is the unity.
    #print(n)
    psi_Mdt = np.zeros(psi_Mdt_m1.shape, dtype=np.complex_)
    psi_Mdt+=(psi_Mdt_m1*spe.jv(0.,complex(alpha)))
    phi0 = psi_Mdt_m1

    psi_split = np.split(psi_Mdt_m1,n)                          #The Psi is split in the number of levels in order to operate the Ham contributions individually.
    VPdiag = np.zeros(np.shape(psi_split),dtype=np.complex_)    #Array of arrays of zeros in the shape of [ np.zeros(psi_split[0].shape), ...,np.zeros(psi_split[n-1].shape)] is used to later contain the corresponding contributions to the Psi.
    Kpsi = np.zeros(np.shape(psi_split),dtype=np.complex_)
    for g5 in range(n):                                         #Evaluation of the diagonal derivative terms.
        K1psi = Gs['Grr']*np.matmul(K_r, psi_split[g5])     #The K's already have the -0.5 factor for the kinetic contribution (TFG routine)
        K2psi = Gs['GRR']*np.matmul(K_R, psi_split[g5].T).T 
        KrRpsi =-Gs['GrR']*np.matmul(D_r, np.matmul(D_R, psi_split[g5].T).T) 
        Drpsi = -0.5*(Gs['dGrr_dr']+Gs['dGrR_dR'])*np.matmul(D_r, psi_split[g5]) + (-0.5*(Gs['dGRR_dR']+Gs['dGrR_dr'])*np.matmul(D_R, psi_split[g5].T).T )
        Kpsi[g5] = K1psi+K2psi+KrRpsi+Drpsi

        VPdiag[g5] = Vs[g5]*psi_split[g5]                        #VPdiag refers to the result of operating the diagonal (multiplicative) part of H 
    Kpsi = np.vstack(tuple(Kpsi))


    VPfd = np.zeros(np.shape(psi_split),dtype=np.complex_)      #VPfd refers to the multiplicative, off-diagonal contribution.
    ix = 1                                                      #
    iy = 0
    iz = 0
    Dpsifd = np.zeros(np.shape(psi_split),dtype=np.complex_)            
    for g2 in range(n):                                         #Number of levels                            
        for g3 in range(ix,n):                                  #Number of operations in each level, for the superior triangular part. 
            Vpfis = Tsup[iy]*psi_split[g3]                      #iy is advancing on the Tsup list elements, it runs separately, but the times as the total number of g3s.
            VPfd[g2]+=Vpfis                                     #                               
            Vpfii = Tinf[iy]*psi_split[g3-ix]                   #The same is done for the Tinf part, just starting by the last row.
            VPfd[-ix]+=Vpfii                                    #
            Dpfds = -1.*(Gs['GrR']*Lambs_R[iy] + Gs['Grr']*Lambs_r[iy])*np.matmul(D_r,psi_split[g3]) - ( (Gs['GrR']*Lambs_r[iy] + Gs['GRR']*Lambs_R[iy])*np.matmul(D_R, psi_split[g3].T ).T)   #Lo que multiplica a D_R fue corregido.
            #when the np object is a matrix, the operand * is read as matmul, not as elementwise multiplication, as in np array. Neither off-diagonal derivative needs to be multiplicated by 0.5.
            Dpsifd[g2] += Dpfds
            #print(iy,g3,g2,g3-ix,-ix,'g3')
            iy+=1
        if g2 != n-1:
            for gl in range(ix):                                #Evaluating the derivative triangular inferior part, considering the appropiate signs.
                Dpfdi = 1.*(Gs['GrR']*Lambs_R[iz] + Gs['Grr']*Lambs_r[iz])*np.matmul(D_r,psi_split[gl]) + ( (Gs['GrR']*Lambs_r[iz] + Gs['GRR']*Lambs_R[iz])*np.matmul(D_R,psi_split[gl].T).T) 
                Dpsifd[ix]+= Dpfdi
                #print(gl,iz,ix,'gl')
                iz+=1
        ix+=1                                                   #ix changes with g2.

    VPsi = np.zeros(np.shape(psi_split),dtype=np.complex_)      #Sum of the potential and off-diagonal contributions. Only the diagonal derivative
    for g4 in range(len(VPfd)):                                 #terms are missing for the evaluation of the complete Hamiltonian.
        VPsi[g4]+= (VPfd[g4] + VPdiag[g4] + Dpsifd[g4])         
    Upsi = np.vstack(tuple(VPsi))                               #The list of arrays is fused so that it has the same shape as the original PSI

    phi1 = (-1j/sem_DE)*Qn_y*( Kpsi - (exp_arg*psi_Mdt_m1) + Upsi) #The first Chebychev polynomial, T_1, with x being the normalized Hamiltonian. The linear transform of the normalization is done here.
    psi_Mdt+= (phi1*spe.jv(1.,complex(alpha))*2.)

    #The K1psi, K2psi,Upsi,KrRpsi and Drpsi togheter correspond to the application of the Hamiltonian to the wavefunction. As it is needed that it is applied in the recurrences, the steps to obtain it are repeated, multiplicated by the corresponding factor.
    a_m = 2.*spe.jv(26.,complex(alpha))
    print(a_m, alpha)
    for k in range(2, 27):
        psi_split = np.split(phi1,n)
        VPdiag = np.zeros(np.shape(psi_split),dtype=np.complex_)
        Kpsi = np.zeros(np.shape(psi_split),dtype=np.complex_)
        for g11 in range(n):
            K1psi = Gs['Grr']*np.matmul(K_r, psi_split[g11]) 
            K2psi = Gs['GRR']*np.matmul(K_R,psi_split[g11].T).T 
            KrRpsi = - Gs['GrR']*np.matmul(D_r, np.matmul(D_R, psi_split[g11].T).T) 
            Drpsi = - 0.5*(Gs['dGrr_dr']+Gs['dGrR_dR'])*np.matmul(D_r, psi_split[g11]) + ( -0.5*(Gs['dGRR_dR']+Gs['dGrR_dr'])*np.matmul(D_R, psi_split[g11].T).T )
            Kpsi[g11] = K1psi+K2psi+KrRpsi+Drpsi
            VPdiag[g11] = Vs[g11]*psi_split[g11]

        Kpsi = np.vstack(tuple(Kpsi))

        VPfd = np.zeros(np.shape(psi_split),dtype=np.complex_)
        ix = 1
        iy = 0
        iz = 0
        Dpsifd = np.zeros(np.shape(psi_split),dtype=np.complex_)
        for g8 in range(n):
            for g9 in range(ix,n):
                Vpfis = Tsup[iy]*psi_split[g9]
                VPfd[g8]+=Vpfis
                Vpfii = Tinf[iy]*psi_split[g9-ix]
                VPfd[-ix]+=Vpfii
                Dpfds = -1.*(Gs['GrR']*Lambs_R[iy] + Gs['Grr']*Lambs_r[iy])*np.matmul(D_r,psi_split[g9]) - ( (Gs['GrR']*Lambs_r[iy] + Gs['GRR']*Lambs_R[iy])*np.matmul(D_R, psi_split[g9].T).T  )
                Dpsifd[g8] += Dpfds
                #print(iy,g3,g2,g3-ix,-ix,'g3')
                iy+=1
            if g8 != n-1:
                for gm in range(ix):
                    Dpfdi = 1.*(Gs['GrR']*Lambs_R[iz] + Gs['Grr']*Lambs_r[iz])*np.matmul(D_r,psi_split[gm]) + ( (Gs['GrR']*Lambs_r[iz] + Gs['GRR']*Lambs_R[iz])*np.matmul(D_R,psi_split[gm].T).T )
                    Dpsifd[ix]+= Dpfdi
                    #print(gl,iz,ix,'gl')
                    iz+=1
            ix+=1

        VPsi = np.zeros(np.shape(psi_split),dtype=np.complex_)
        for g10 in range(len(VPfd)):
            VPsi[g10]+= (VPfd[g10] + VPdiag[g10] + Dpsifd[g10])
        Upsi = np.vstack(tuple(VPsi))

        psi_k = (-2j/sem_DE)*( Kpsi - ((exp_arg)*phi1) + Upsi) #nth Chebychev polynomial -> -2j*H_{norm}T_n(x)
        phi_k = Qn_y*(psi_k + Qn_y*phi0)        #T_{n+1} = -2j*T_n + T_{n-1}   
        psi_Mdt+= (2.*phi_k*spe.jv(float(k),complex(alpha)))
        phi0=phi1
        phi1=phi_k
   
    psi_mdt = exp_prop*psi_Mdt                      #Offset for the shifting made in the normalization of the Hamiltonian.
    norm= np.tensordot(np.conj(psi_mdt),psi_mdt)    #As it is a polynomial expansion it is not norm conserving, however, due to their
    print(norm,exp_prop)                            #orthogonal interval, the deviation from the obtained norm and 1 is a measurement of the error
    psi_mdt_norm = (1./np.sqrt(norm))*psi_mdt

    return(psi_mdt_norm)

def cheb_prop_internal_nl_nacme_gpu(r, R, PSI, Enes, K_r, K_R, D_r, D_R, Gs, Vs, vio, Tsup, Tinf, Lambs_r, Lambs_R, n, ef, time1, time2):
    '''To be used with cuda'''
    '''time1 < time2 '''
    ''' The original (Psi(r,R)) [[Psi(r1,R1), Psi(r1,R2),...,Psi(r1,Rn)],'''
    '''                          [...],'''
    '''                          [Psi(rm,R1), Psi(rm,R2),...,Psi(rm,Rn)]]'''
    ''' then K1 = K_r and K2 = K_R refer to the diagonal, second derivative term of the kinetic energy matrices, please give them in said order. D_r and D_R are the first derivative matrices.'''
    '''Enes is a list that contains the estimations on minimum (potential) and maximum(potential and kinetic) energies that are contained in the grid i.e. estimation on the spectrum of eigenvalues of the Hamiltonian. '''
    ''' n  is the number of levels'''
    ''' Vs, Tsup, Tinf, Lambs_r, Lambs_R are lists of the arrays of the Hamiltonian matrix elements for each level. E.g. V1, potential for S0, 12 the array on that position of the n-level system Hamiltonian '''
    ''' Vs = [ V1, V2, V3, ..., Vn] Diagonal Potential energy matrix elements '''
    ''' vio = CAP for the Mandelshtam recurrence, if direct substitution is used, the CAP must be included in the Vs elements.'''
    '''All the contributions to the off-diagonal, multiplicative contributions to the Hamiltonian must be containend in the lists of arrays Tsup, for the superior triangular and in Tinf for the inferior triangular of the matrix. The lists can contain NACME, NACME derivatives, etc. but if the contribution to the matrix is antihermitean its signs must be adressed beforehand. The order of the elements in the list must be: '''
    '''Tsup order = [ 12,13,...,1n, 23,..,2n,34,...,...,n-1n] '''
    '''Tinf order = [n1,n2,..,nn-1, n-11,n-12,...,n-1n-2,...,n-21,n-22,...,n-2n-3,...,...,21] ----> e.g. n=3: [ L31, L32, L21] Triangular inferior '''
    ''' Separately, the non-adiabatic coupling matrix elements are given, in order to operate the derivative, off-diagonal part. Here only the superior triangular elements are needed. The sign is considered in this algorithm, do not change it beforehand. The list needs to be given in the following order: '''
    '''Lambs_r o Lambs_R = [ 12,13,...,1n, 23,...,2n,34,...,...,n-1n ] Non-adiabatic coupling elements '''
    ''' The separation of terms is only between multiplicative and derivative, diagonal or off-diagonal, so multiplicative terms of the kinetic energy matrix must be adressed in Vs, Tsup or Tinf accordingly. '''
    '''Gs = { 'Grr' : Grr, 'GRR': GRR, 'GrR' : GrR, 'dGrr_dr' : dGrr_dr, 'dGRR_dR' : dGRR_dR, 'dGrR_dr' : dGrR_dr, 'dGrR_dR' : dGrR_dR } G matrix elements'''

    Enes=np.array(Enes,dtype=np.complex_)
    Emax_re = np.max(Enes.real)  #The maximum routine selects the
    Emin_re = np.min(Enes.real)  #element with the maximum real part,
    Emax_im = np.max(Enes.imag)  #not the one with the greater magnitude
    Emin_im = np.min(Enes.imag)  #
    Emin = Emin_re + 1j*Emin_im  
    Emax = Emax_re + 1j*Emax_im

    DE = Emax-Emin

    sem_DE = 0.5*DE
    exp_arg = (sem_DE) + Emin
    print(DE, Emin,Emax, exp_arg)

    #--------------------------Beginning of the approximate propagator---------------------------------#

    dt = time2-time1

    alpha = 0.5*DE*dt

    exp_prop = np.exp(-1j*exp_arg*dt)
    #---
    Qn_y = 1. #cp.exp(0.050*vio)        #Multiplying first by the cap in order to eliminate the aliased part.
    #---
    psi_Mdt_m1 = Qn_y*cp.array(PSI)

    #phi = []                                                    #All the polynomials are stored here.
    #phi.append(psi_Mdt_m1)                                      #the zeroth Chebychev polynomial is the unity.
    psi_Mdt = cp.zeros( psi_Mdt_m1.shape, dtype=cp.complex_)
    psi_Mdt+= psi_Mdt_m1*spe.jv(0.,complex(alpha))
    phi0 = psi_Mdt_m1
    
    #print(n)
    psi_split = cp.split(psi_Mdt_m1,n)                          #The Psi is split in the number of levels in order to operate the Ham contributions individually.
    psi_split = cp.array(psi_split)
    VPdiag = cp.zeros(cp.shape(psi_split),dtype=cp.complex_)    #Array of arrays of zeros in the shape of [ np.zeros(psi_split[0].shape), ...,np.zeros(psi_split[n-1].shape)] is used to later contain the corresponding contributions to the Psi.
    Kpsi = cp.zeros(cp.shape(psi_split),dtype=cp.complex_)
    for g5 in range(n):                                         #Evaluation of the diagonal derivative terms.
        K1psi = Gs['Grr']*cp.matmul(K_r, psi_split[g5])     #The K's already have the -0.5 factor for the kinetic contribution (TFG routine)
        K2psi = Gs['GRR']*cp.matmul(K_R, psi_split[g5].T).T 
        KrRpsi =-Gs['GrR']*cp.matmul(D_r, cp.matmul(D_R, psi_split[g5].T).T) 
        Drpsi = -0.5*(Gs['dGrr_dr']+Gs['dGrR_dR'])*cp.matmul(D_r, psi_split[g5]) + (-0.5*(Gs['dGRR_dR']+Gs['dGrR_dr'])*cp.matmul(D_R, psi_split[g5].T).T )
        Kpsi[g5] = K1psi+K2psi+KrRpsi+Drpsi

        VPdiag[g5] = Vs[g5]*psi_split[g5]                        #VPdiag refers to the result of operating the diagonal (multiplicative) part of H 
    Kpsi = cp.vstack(tuple(Kpsi))


    VPfd = cp.zeros(cp.shape(psi_split),dtype=cp.complex_)      #VPfd refers to the multiplicative, off-diagonal contribution.
    ix = 1                                                      #
    iy = 0
    iz = 0
    Dpsifd = cp.zeros(cp.shape(psi_split),dtype=cp.complex_)            
    for g2 in range(n):                                         #Number of levels                            
        for g3 in range(ix,n):                                  #Number of operations in each level, for the superior triangular part. 
            Vpfis = Tsup[iy]*psi_split[g3]                      #iy is advancing on the Tsup list elements, it runs separately, but the times as the total number of g3s.
            VPfd[g2]+=Vpfis                                     #                               
            Vpfii = Tinf[iy]*psi_split[g3-ix]                   #The same is done for the Tinf part, just starting by the last row.
            VPfd[-ix]+=Vpfii                                    #
            Dpfds = -1.*(Gs['GrR']*Lambs_R[iy] + Gs['Grr']*Lambs_r[iy])*np.matmul(D_r,psi_split[g3]) - ( (Gs['GrR']*Lambs_r[iy] + Gs['GRR']*Lambs_R[iy])*np.matmul(D_R, psi_split[g3].T ).T)   #Lo que multiplica a D_R fue corregido.
            #when the np object is a matrix, the operand * is read as matmul, not as elementwise multiplication, as in np array. Neither off-diagonal derivative needs to be multiplicated by 0.5.
            Dpsifd[g2] += Dpfds
            #print(iy,g3,g2,g3-ix,-ix,'g3')
            iy+=1
        if g2 != n-1:
            for gl in range(ix):                                #Evaluating the derivative triangular inferior part, considering the appropiate signs.
                Dpfdi = 1.*(Gs['GrR']*Lambs_R[iz] + Gs['Grr']*Lambs_r[iz])*np.matmul(D_r,psi_split[gl]) + ( (Gs['GrR']*Lambs_r[iz] + Gs['GRR']*Lambs_R[iz])*np.matmul(D_R,psi_split[gl].T).T) 
                Dpsifd[ix]+= Dpfdi
                #print(gl,iz,ix,'gl')
                iz+=1
        ix+=1                                                   #ix changes with g2.

    VPsi = cp.zeros(cp.shape(psi_split),dtype=cp.complex_)      #Sum of the potential and off-diagonal contributions. Only the diagonal derivative
    for g4 in range(len(VPfd)):                                 #terms are missing for the evaluation of the complete Hamiltonian.
        VPsi[g4]+= (VPfd[g4] + VPdiag[g4] + Dpsifd[g4])         
    Upsi = cp.vstack(tuple(VPsi))                               #The list of arrays is fused so that it has the same shape as the original PSI

    phi1 = (-1j/sem_DE)*Qn_y*( Kpsi - (exp_arg*psi_Mdt_m1) + Upsi) #The first Chebychev polynomial, T_1, with x being the normalized Hamiltonian. The linear transform of the normalization is done here.
    psi_Mdt+= (phi1*spe.jv(1.,complex(alpha))*2.)
    #phi.append(phi1)

    #The K1psi, K2psi,Upsi,KrRpsi and Drpsi togheter correspond to the application of the Hamiltonian to the wavefunction. As it is needed that it is applied in the recurrences, the steps to obtain it are repeated, multiplicated by the corresponding factor.
    a_m = 2.*spe.jv(26.,complex(alpha))
    print(a_m, alpha)
    for k in range(2, 27):
        psi_split = cp.split(phi1,n)
        psi_split = cp.array(psi_split) 
        VPdiag = cp.zeros(cp.shape(psi_split),dtype=cp.complex_)
        Kpsi = cp.zeros(cp.shape(psi_split),dtype=cp.complex_)
        for g11 in range(n):
            K1psi = Gs['Grr']*cp.matmul(K_r, psi_split[g11]) 
            K2psi = Gs['GRR']*cp.matmul(K_R,psi_split[g11].T).T 
            KrRpsi = - Gs['GrR']*cp.matmul(D_r, cp.matmul(D_R, psi_split[g11].T).T) 
            Drpsi = - 0.5*(Gs['dGrr_dr']+Gs['dGrR_dR'])*cp.matmul(D_r, psi_split[g11]) + ( -0.5*(Gs['dGRR_dR']+Gs['dGrR_dr'])*cp.matmul(D_R, psi_split[g11].T).T )
            Kpsi[g11] = K1psi+K2psi+KrRpsi+Drpsi
            VPdiag[g11] = Vs[g11]*psi_split[g11]

        Kpsi = cp.vstack(tuple(Kpsi))

        VPfd = cp.zeros(cp.shape(psi_split),dtype=cp.complex_)
        ix = 1
        iy = 0
        iz = 0
        Dpsifd = cp.zeros(cp.shape(psi_split),dtype=cp.complex_)
        for g8 in range(n):
            for g9 in range(ix,n):
                Vpfis = Tsup[iy]*psi_split[g9]
                VPfd[g8]+=Vpfis
                Vpfii = Tinf[iy]*psi_split[g9-ix]
                VPfd[-ix]+=Vpfii
                Dpfds = -1.*(Gs['GrR']*Lambs_R[iy] + Gs['Grr']*Lambs_r[iy])*np.matmul(D_r,psi_split[g9]) - ( (Gs['GrR']*Lambs_r[iy] + Gs['GRR']*Lambs_R[iy])*np.matmul(D_R, psi_split[g9].T).T  )
                Dpsifd[g8] += Dpfds
                #print(iy,g3,g2,g3-ix,-ix,'g3')
                iy+=1
            if g8 != n-1:
                for gm in range(ix):
                    Dpfdi = 1.*(Gs['GrR']*Lambs_R[iz] + Gs['Grr']*Lambs_r[iz])*np.matmul(D_r,psi_split[gm]) + ( (Gs['GrR']*Lambs_r[iz] + Gs['GRR']*Lambs_R[iz])*np.matmul(D_R,psi_split[gm].T).T )
                    Dpsifd[ix]+= Dpfdi
                    #print(gl,iz,ix,'gl')
                    iz+=1
            ix+=1

        VPsi = cp.zeros(cp.shape(psi_split),dtype=cp.complex_)
        for g10 in range(len(VPfd)):
            VPsi[g10]+= (VPfd[g10] + VPdiag[g10] + Dpsifd[g10])
        Upsi = cp.vstack(tuple(VPsi))

        psi_k = (-2j/sem_DE)*( Kpsi -((exp_arg)*phi1) + Upsi ) #nth Chebychev polynomial -> -2j*H_{norm}T_n(x)
        phi_k = Qn_y*(psi_k + Qn_y*phi0)        #T_{n+1} = -2j*T_n + T_{n-1}   
        psi_Mdt+= (2.*phi_k*spe.jv(float(k),complex(alpha)))
        phi0=phi1
        phi1=phi_k
        

    psi_mdt = exp_prop*psi_Mdt                      #Offset for the shifting made in the normalization of the Hamiltonian.
    norm= cp.tensordot(cp.conj(psi_mdt),psi_mdt)    #As it is a polynomial expansion it is not norm conserving, however, due to their
    print(norm,exp_prop)                            #orthogonal interval, the deviation from the obtained norm and 1 is a measurement of the error
    psi_mdt_norm = (1./cp.sqrt(norm))*psi_mdt

    return(psi_mdt_norm)

def cheb_prop_internal_nl_nacme_gpu_fft(r, R, PSI, Enes, K_r, K_R, D_r, D_R, Gs, Vs, vio, Tsup, Tinf, Lambs_r, Lambs_R, n, ef, time1, time2):
    '''Only on gpus the routine with fft is faster in wall time than with matrix multiplication'''
    '''time1 < time2 '''
    ''' The original (Psi(r,R)) [[Psi(r1,R1), Psi(r1,R2),...,Psi(r1,Rn)],'''
    '''                          [...],'''
    '''                          [Psi(rm,R1), Psi(rm,R2),...,Psi(rm,Rn)]]'''
    ''' then K1 = K_r and K2 = K_R refer to the second derivative term of the kinetic energy operator IN MOMENTUM SPACE, please give them in said order. This routine receives the vector, as it performs a multiplication instead of a matrix multiplication. D_r and D_R are the first derivative vectors in momentum space.'''
    '''Enes is a list that contains the estimations on minimum (potential) and maximum(potential and kinetic) energies that are contained in the grid i.e. estimation on the spectrum of eigenvalues of the Hamiltonian. '''
    ''' n  is the number of levels'''
    ''' Vs, Tsup, Tinf, Lambs_r, Lambs_R are lists of the arrays of the Hamiltonian matrix elements for each level. E.g. V1, potential for S0, 12 the array on that position of the n-level system Hamiltonian '''
    ''' Vs = [ V1, V2, V3, ..., Vn] Diagonal Potential energy matrix elements '''
    ''' vio = CAP for the Mandelshtam recurrence, if direct substitution is used, the CAP must be included in the Vs elements.'''
    '''All the contributions to the off-diagonal, multiplicative contributions to the Hamiltonian must be containend in the lists of arrays Tsup, for the superior triangular and in Tinf for the inferior triangular of the matrix. The lists can contain NACME, NACME derivatives, etc. but if the contribution to the matrix is antihermitean its signs must be adressed beforehand. The order of the elements in the list must be: '''
    '''Tsup order = [ 12,13,...,1n, 23,..,2n,34,...,...,n-1n] '''
    '''Tinf order = [n1,n2,..,nn-1, n-11,n-12,...,n-1n-2,...,n-21,n-22,...,n-2n-3,...,...,21] ----> e.g. n=3: [ L31, L32, L21] Triangular inferior '''
    ''' Separately, the non-adiabatic coupling matrix elements are given, in order to operate the derivative, off-diagonal part. Here only the superior triangular elements are needed. The sign is considered in this algorithm, do not change it beforehand. The list needs to be given in the following order: '''
    '''Lambs_r o Lambs_R = [ 12,13,...,1n, 23,...,2n,34,...,...,n-1n ] Non-adiabatic coupling elements '''
    ''' The separation of terms is only between multiplicative and derivative, diagonal or off-diagonal, so multiplicative terms of the kinetic energy matrix must be adressed in Vs, Tsup or Tinf accordingly. '''
    '''Gs = { 'Grr' : Grr, 'GRR': GRR, 'GrR' : GrR, 'dGrr_dr' : dGrr_dr, 'dGRR_dR' : dGRR_dR, 'dGrR_dr' : dGrR_dr, 'dGrR_dR' : dGrR_dR } G matrix elements'''

    Enes=np.array(Enes,dtype=np.complex_)
    Emax_re = np.max(Enes.real)  #The maximum routine selects the
    Emin_re = np.min(Enes.real)  #element with the maximum real part,
    Emax_im = np.max(Enes.imag)  #not the one with the greater magnitude
    Emin_im = np.min(Enes.imag)  #
    Emin = Emin_re + 1j*Emin_im  
    Emax = Emax_re + 1j*Emax_im

    DE = Emax-Emin

    sem_DE = 0.5*DE
    exp_arg = (sem_DE) + Emin
    print(DE, Emin,Emax, exp_arg)

    #--------------------------Beginning of the approximate propagator---------------------------------#

    dt = time2-time1

    alpha = 0.5*DE*dt

    exp_prop = np.exp(-1j*exp_arg*dt)
    #---
    Qn_y = 1. #cp.exp(0.050*vio)        #Multiplying first by the cap in order to eliminate the aliased part.
    #---
    psi_Mdt_m1 = Qn_y*cp.array(PSI)

    #phi = []                                                    #All the polynomials are stored here.
    #phi.append(psi_Mdt_m1)                                      #the zeroth Chebychev polynomial is the unity.
    psi_Mdt = cp.zeros( psi_Mdt_m1.shape, dtype=cp.complex_)
    psi_Mdt+= psi_Mdt_m1*spe.jv(0.,complex(alpha))
    phi0 = psi_Mdt_m1
    
    #print(n)
    psi_split = cp.split(psi_Mdt_m1,n)                          #The Psi is split in the number of levels in order to operate the Ham contributions individually.
    psi_split = cp.array(psi_split)
    VPdiag = cp.zeros(cp.shape(psi_split),dtype=cp.complex_)    #Array of arrays of zeros in the shape of [ np.zeros(psi_split[0].shape), ...,np.zeros(psi_split[n-1].shape)] is used to later contain the corresponding contributions to the Psi.
    Kpsi = cp.zeros(cp.shape(psi_split),dtype=cp.complex_)
    for g5 in range(n):                                         #Evaluation of the diagonal derivative terms.
        #dP0fft = scp.fft.ifft((DFT*scp.fft.fft(Psi0t.T))).T
        K1psi = Gs['Grr']*cp.fft.ifft((K_r*cp.fft.fft(psi_split[g5].T))).T   #The K's already have the -0.5 factor for the kinetic contribution (TFG routine)
        K2psi = Gs['GRR']*cp.fft.ifft(K_R*cp.fft.fft(psi_split[g5])) 
        #KrRpsi =-Gs['GrR']*cp.matmul(D_r, cp.matmul(D_R, psi_split[g5].T).T) 
        KrRpsi = cp.fft.ifft(D_R*cp.fft.fft(psi_split[g5])) 
        KrRpsi = -Gs['GrR']*cp.fft.ifft((D_r*cp.fft.fft(KrRpsi.T))).T
        Drpsi = -0.5*(Gs['dGrr_dr']+Gs['dGrR_dR'])*cp.fft.ifft((D_r*cp.fft.fft(psi_split[g5].T))).T + (-0.5*(Gs['dGRR_dR']+Gs['dGrR_dr'])*cp.fft.ifft(D_R*cp.fft.fft(psi_split[g5])) )
        Kpsi[g5] = K1psi+K2psi+KrRpsi+Drpsi

        VPdiag[g5] = Vs[g5]*psi_split[g5]                        #VPdiag refers to the result of operating the diagonal (multiplicative) part of H 
    Kpsi = cp.vstack(tuple(Kpsi))


    VPfd = cp.zeros(cp.shape(psi_split),dtype=cp.complex_)      #VPfd refers to the multiplicative, off-diagonal contribution.
    ix = 1                                                      #
    iy = 0
    iz = 0
    Dpsifd = cp.zeros(cp.shape(psi_split),dtype=cp.complex_)            
    for g2 in range(n):                                         #Number of levels                            
        for g3 in range(ix,n):                                  #Number of operations in each level, for the superior triangular part. 
            Vpfis = Tsup[iy]*psi_split[g3]                      #iy is advancing on the Tsup list elements, it runs separately, but the times as the total number of g3s.
            VPfd[g2]+=Vpfis                                     #                               
            Vpfii = Tinf[iy]*psi_split[g3-ix]                   #The same is done for the Tinf part, just starting by the last row.
            VPfd[-ix]+=Vpfii                                    #
            #dP0fft = scp.fft.ifft((DFT*scp.fft.fft(Psi0t.T))).T
            Dpfds = -1.*(Gs['GrR']*Lambs_R[iy] + Gs['Grr']*Lambs_r[iy])*cp.fft.ifft((D_r*cp.fft.fft(psi_split[g3].T))).T - ( (Gs['GrR']*Lambs_r[iy] + Gs['GRR']*Lambs_R[iy])*cp.fft.ifft(D_R*cp.fft.fft(psi_split[g3])) )  #Lo que multiplica a D_R fue corregido.
            #when the np object is a matrix, the operand * is read as matmul, not as elementwise multiplication, as in np array. Neither off-diagonal derivative needs to be multiplicated by 0.5.
            Dpsifd[g2] += Dpfds
            #print(iy,g3,g2,g3-ix,-ix,'g3')
            iy+=1
        if g2 != n-1:
            for gl in range(ix):                                #Evaluating the derivative triangular inferior part, considering the appropiate signs.
                Dpfdi = 1.*(Gs['GrR']*Lambs_R[iz] + Gs['Grr']*Lambs_r[iz])*cp.fft.ifft((D_r*cp.fft.fft(psi_split[gl].T))).T + ( (Gs['GrR']*Lambs_r[iz] + Gs['GRR']*Lambs_R[iz])*cp.fft.ifft(D_R*cp.fft.fft(psi_split[gl])) )
                Dpsifd[ix]+= Dpfdi
                #print(gl,iz,ix,'gl')
                iz+=1
        ix+=1                                                   #ix changes with g2.

    VPsi = cp.zeros(cp.shape(psi_split),dtype=cp.complex_)      #Sum of the potential and off-diagonal contributions. Only the diagonal derivative
    for g4 in range(len(VPfd)):                                 #terms are missing for the evaluation of the complete Hamiltonian.
        VPsi[g4]+= (VPfd[g4] + VPdiag[g4] + Dpsifd[g4])         
    Upsi = cp.vstack(tuple(VPsi))                               #The list of arrays is fused so that it has the same shape as the original PSI

    phi1 = (-1j/sem_DE)*Qn_y*( Kpsi - (exp_arg*psi_Mdt_m1) + Upsi) #The first Chebychev polynomial, T_1, with x being the normalized Hamiltonian. The linear transform of the normalization is done here.
    psi_Mdt+= (phi1*spe.jv(1.,complex(alpha))*2.)
    #phi.append(phi1)

    #The K1psi, K2psi,Upsi,KrRpsi and Drpsi togheter correspond to the application of the Hamiltonian to the wavefunction. As it is needed that it is applied in the recurrences, the steps to obtain it are repeated, multiplicated by the corresponding factor.
    a_m = 2.*spe.jv(56,complex(alpha))
    print(a_m, alpha)
    for k in range(2, 57):
        psi_split = cp.split(phi1,n)
        psi_split = cp.array(psi_split) 
        VPdiag = cp.zeros(cp.shape(psi_split),dtype=cp.complex_)
        Kpsi = cp.zeros(cp.shape(psi_split),dtype=cp.complex_)
        for g11 in range(n):
            #dP0fft = scp.fft.ifft((DFT*scp.fft.fft(Psi0t.T))).T
            K1psi = Gs['Grr']*cp.fft.ifft((K_r*cp.fft.fft(psi_split[g11].T))).T   #The K's already have the -0.5 factor for the kinetic contribution (TFG routine)
            K2psi = Gs['GRR']*cp.fft.ifft(K_R*cp.fft.fft(psi_split[g11]))  
            #KrRpsi =-Gs['GrR']*cp.matmul(D_r, cp.matmul(D_R, psi_split[g5].T).T) 
            KrRpsi = cp.fft.ifft(D_R*cp.fft.fft(psi_split[g11])) 
            KrRpsi = -Gs['GrR']*cp.fft.ifft((D_r*cp.fft.fft(KrRpsi.T))).T
            Drpsi = -0.5*(Gs['dGrr_dr']+Gs['dGrR_dR'])*cp.fft.ifft((D_r*cp.fft.fft(psi_split[g11].T))).T + (-0.5*(Gs['dGRR_dR']+Gs['dGrR_dr'])*cp.fft.ifft(D_R*cp.fft.fft(psi_split[g11])) )
            Kpsi[g11] = K1psi+K2psi+KrRpsi+Drpsi
            VPdiag[g11] = Vs[g11]*psi_split[g11]

        Kpsi = cp.vstack(tuple(Kpsi))

        VPfd = cp.zeros(cp.shape(psi_split),dtype=cp.complex_)
        ix = 1
        iy = 0
        iz = 0
        Dpsifd = cp.zeros(cp.shape(psi_split),dtype=cp.complex_)
        for g8 in range(n):
            for g9 in range(ix,n):
                Vpfis = Tsup[iy]*psi_split[g9]
                VPfd[g8]+=Vpfis
                Vpfii = Tinf[iy]*psi_split[g9-ix]
                VPfd[-ix]+=Vpfii
                Dpfds = -1.*(Gs['GrR']*Lambs_R[iy] + Gs['Grr']*Lambs_r[iy])*cp.fft.ifft((D_r*cp.fft.fft(psi_split[g9].T))).T - ( (Gs['GrR']*Lambs_r[iy] + Gs['GRR']*Lambs_R[iy])*cp.fft.ifft((D_R*cp.fft.fft(psi_split[g9])))  )
                Dpsifd[g8] += Dpfds
                #print(iy,g3,g2,g3-ix,-ix,'g3')
                iy+=1
            if g8 != n-1:
                for gm in range(ix):
                    Dpfdi = 1.*(Gs['GrR']*Lambs_R[iz] + Gs['Grr']*Lambs_r[iz])*cp.fft.ifft((D_r*cp.fft.fft(psi_split[gm].T))).T  + ( (Gs['GrR']*Lambs_r[iz] + Gs['GRR']*Lambs_R[iz])*cp.fft.ifft((D_R*cp.fft.fft(psi_split[gm]))) )
                    Dpsifd[ix]+= Dpfdi
                    #print(gl,iz,ix,'gl')
                    iz+=1
            ix+=1

        VPsi = cp.zeros(cp.shape(psi_split),dtype=cp.complex_)
        for g10 in range(len(VPfd)):
            VPsi[g10]+= (VPfd[g10] + VPdiag[g10] + Dpsifd[g10])
        Upsi = cp.vstack(tuple(VPsi))

        psi_k = (-2j/sem_DE)*( Kpsi -((exp_arg)*phi1) + Upsi ) #nth Chebychev polynomial -> -2j*H_{norm}T_n(x)
        phi_k = Qn_y*(psi_k + Qn_y*phi0)        #T_{n+1} = -2j*T_n + T_{n-1}   
        psi_Mdt+= (2.*phi_k*spe.jv(float(k),complex(alpha)))
        phi0=phi1
        phi1=phi_k
        

    psi_mdt = exp_prop*psi_Mdt                      #Offset for the shifting made in the normalization of the Hamiltonian.
    norm= cp.tensordot(cp.conj(psi_mdt),psi_mdt)    #As it is a polynomial expansion it is not norm conserving, however, due to their
    print(norm,exp_prop)                            #orthogonal interval, the deviation from the obtained norm and 1 is a measurement of the error
    psi_mdt_norm = (1./cp.sqrt(norm))*psi_mdt

    return(psi_mdt_norm)

def nacme_calc(tipo,mlist,nac_cv,alpha,beta,gamma,r_cv):
    '''tipo is the type of the internal coordinate for which the non-adiabatic coupling (NAC) matrix element is being transformed'''
    '''tipo can be 'r1','r2','r3','phi1','phi2' or 'tau', notice the type is a string.'''
    '''provided there are four atoms connected in the manner a1-a2-a3-a4 where 'ai' dennotes an atom and '-' a bond '''
    '''r1 corresponds to the distance between a1-a2 '''
    '''r2 corresponds to the distance between a2-a3 '''
    '''r3 corresponds to the distance between a3-a4  '''
    '''phi1 corresponds to the bond angle formed between a1-a2-a3 '''
    '''phi2 corresponds to the bond angle formed between a2-a3-a4 '''
    '''tau is the corresponding dihedral angle '''
    '''If there are less than four atoms involved introduce mi == 0. with i=3,4'''
    '''nac_cv is the vector containing the non-adiabatic couplings in cartesian coordinates, in the following manner:'''
    '''[a1_x,a1_y,a1_z, a2_x,a2_y,a2_z, a3_x,a3_y,a3_z, a4_x,a4_y,a4_z] '''
    '''if there are less than four atoms, introduce the ai_alpha = 0. with i=3,4 and alpha = x,y,z'''
    '''alfa, beta and gamma are the corresponding Euler angles'''
    '''r_cv is a vector containing the positions of the atoms, given in the same order as the  vector '''
    '''Please provide the coordinates of r_cv so that the resulting distances are given in Bohr radii '''
    '''The routine will return a number, corresponding to the projected NAC matrix element for the introduced geometry'''


    if len(mlist) == 2:
        0.

    elif len(mlist) == 3:
        0.
        #aqui si hay vth y vph
    
    else:
        m1 = float(mlist[0])
        m2 = float(mlist[1])
        m3 = float(mlist[2])
        m4 = float(mlist[3])
        M = m1+m2+m3+m4
        W1_inv = np.zeros((12,12))              #3N dimensions for the analytic transformation, and here 4 atoms are being accounted.
        W1_inv[:3,:3] = m1/M*np.eye(3)          
        W1_inv[:3,3:6] = m2/M*np.eye(3)         #The Jacobian with respect to the transformed intermediate coordinates is easier to compute,
        W1_inv[:3,6:9] = m3/M*np.eye(3)         #so it is calculated in this way and later inverted.
        W1_inv[:3,9:] = m4/M*np.eye(3)

        W1_inv[3:6,:3], W1_inv[6:9,6:9], W1_inv[9:,3:6] = np.eye(3), np.eye(3), np.eye(3)  
        W1_inv[3:6,3:6], W1_inv[6:9,:3], W1_inv[9:,9:] = -np.eye(3), -np.eye(3), -np.eye(3)
        
        W1 = la.inv(W1_inv)                     #Obtaining the Jacobian for the needed transformation

        S = [ [ np.cos(alpha)*np.cos(gamma)-np.sin(alpha)*np.cos(beta)*np.sin(gamma), -np.cos(alpha)*np.sin(gamma)-np.sin(alpha)*np.cos(beta)*np.cos(gamma), np.sin(alpha)*np.sin(beta)], [ np.sin(alpha)*np.cos(gamma)+np.cos(alpha)*np.cos(beta)*np.sin(gamma), -np.sin(alpha)*np.sin(gamma)+np.cos(alpha)*np.cos(beta)*np.cos(gamma), -np.cos(alpha)*np.sin(beta)], [ np.sin(beta)*np.sin(gamma), np.sin(beta)*np.cos(gamma), np.cos(beta) ]]

        dS_da = [ [ -np.sin(alpha)*np.cos(gamma)-np.cos(alpha)*np.cos(beta)*np.sin(gamma), np.sin(alpha)*np.sin(gamma)-np.cos(alpha)*np.cos(beta)*np.cos(gamma), np.cos(alpha)*np.sin(beta)], [ np.cos(alpha)*np.cos(gamma)-np.sin(alpha)*np.cos(beta)*np.sin(gamma), -np.cos(alpha)*np.sin(gamma)-np.sin(alpha)*np.cos(beta)*np.cos(gamma), np.sin(alpha)*np.sin(beta)], [ 0.,0.,0]]

        dS_db = [ [ np.sin(alpha)*np.sin(beta)*np.sin(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma), np.sin(alpha)*np.cos(beta)], [ -np.cos(alpha)*np.sin(beta)*np.sin(gamma), -np.cos(alpha)*np.sin(beta)*np.cos(gamma), -np.cos(alpha)*np.cos(beta)], [np.cos(beta)*np.sin(gamma), np.cos(beta)*np.cos(gamma), -np.sin(gamma)]]

        dS_dg = [ [-np.cos(alpha)*np.sin(gamma)-np.sin(alpha)*np.cos(beta)*np.cos(gamma), -np.cos(alpha)*np.cos(gamma)+np.sin(alpha)*np.cos(beta)*np.sin(gamma), 0.], [-np.sin(alpha)*np.sin(gamma)+np.cos(alpha)*np.cos(beta)*np.cos(gamma), -np.sin(alpha)*np.cos(gamma) -np.cos(alpha)*np.cos(beta)*np.sin(gamma), 0.], [ np.sin(beta)*np.cos(gamma), -np.sin(beta)*np.sin(gamma), 0.]]

        a1 = np.array(r_cv[:3])
        a2 = np.array(r_cv[3:6])
        a3 = np.array(r_cv[6:9])
        a4 = np.array(r_cv[9:])

        R1_v = a1-a2
        R1 = la.norm(R1_v)

        R2_v = a3-a2
        R2 = la.norm(R2_v)

        R3_v = a4-a3
        R3 = la.norm(R3_v)

        Phi1 = np.arccos(np.dot(R1_v,R2_v)/(R1*R2))
        Phi2 = np.arccos(np.dot(-R2_v,R3_v)/(R2*R3))

        n1 = np.cross( -R1_v/R1, R2_v/R2 )
        n2 = np.cross(  R2_v/R2, R3_v/R3 )

        tau = np.arccos(np.dot(n1,n2)/(la.norm(n1)*la.norm(n2)))
        
        th = np.pi-np.arccos(R2_v[0]/la.norm(R2_v)) 
        ph = -np.pi+np.arctan2(R2_v[2],R2_v[1])  

        r1 = R2
        r2 = R3
        r3 = R1

        phi2 = Phi2
        phi3 = Phi1

        tau3 = tau 
        
        dp1 = [-r1,0.,0.]                               #Rotated distance vectors. 
        dp2 = [-r2*np.cos(phi2),  r2*np.sin(phi2), 0.]   
        dp3 = [ r3*np.cos(phi3),  r3*np.sin(phi3)*np.cos(tau3), -r3*np.sin(phi3)*np.sin(tau3)] 
        
        W2 = np.zeros((12,12))
        #eta1
        W2[3:6,3] = np.matmul(dS_da,dp1)
        W2[3:6,4] = np.matmul(dS_db,dp1)
        W2[3:6,5] = np.matmul(dS_dg,dp1)
        #eta2
        W2[6:9,3] = np.matmul(dS_da,dp2)
        W2[6:9,4] = np.matmul(dS_db,dp2)
        W2[6:9,5] = np.matmul(dS_dg,dp2)
        #eta3 
        W2[9:,3] = np.matmul(dS_da,dp3)
        W2[9:,4] = np.matmul(dS_db,dp3)
        W2[9:,5] = np.matmul(dS_dg,dp3)
        #sigma2_1
        W2[3,6] = -1.
        W2[3:6,6:9] = np.matmul(S,W2[3:6,6:9])
        #sigma2_2
        W2[6:9,7] = [-np.cos(phi2),  np.sin(phi2), 0.]
        W2[6:9,8] = [r2*np.sin(phi2), r2*np.cos(phi2), 0.]
        W2[6:9,6:9] = np.matmul(S,W2[6:9,6:9])
        #sigma3_2
        W2[9:,9] =  [ np.cos(phi3), np.sin(phi3)*np.cos(tau3), -np.sin(phi3)*np.sin(tau3) ]
        W2[9:,10] = [ -r3*np.sin(phi3), r3*np.cos(phi3)*np.cos(tau3), -r3*np.cos(phi3)*np.sin(tau3)]
        W2[9:,11] = [ 0., -r3*np.sin(phi3)*np.sin(tau3), -r3*np.sin(phi3)*np.cos(tau3)]
        W2[9:,9:] = np.matmul(S,W2[9:,9:])

        W = np.matmul(W1, W2)

        R1 = [ [1., 0., 0.], [ 0., np.cos(ph), np.sin(ph)], [0., -np.sin(ph), np.cos(ph)]]
        R2 = [ [np.cos(th), np.sin(th), 0.], [ -np.sin(th), np.cos(th), 0.], [0., 0., 1.]]
        all_nacm_c = np.zeros(12)

        for mm3 in range(4):
            all_nacm_c[(mm3*3):(mm3+1)*3] = np.matmul(R1,nac_cv[(mm3*3):(mm3+1)*3]) 
        for mm3 in range(4):
            all_nacm_c[(mm3*3):(mm3+1)*3] = np.matmul(R2,all_nacm_c[(mm3*3):(mm3+1)*3])

        if tipo == 'r1':
            #nacme = np.dot(W[-6], all_nacm_c)
            nacme = np.dot(W[-3], all_nacm_c)

        elif tipo == 'r2':
            #nacme = np.dot(W[-5], all_nacm_c)
            nacme = np.dot(W[-6], all_nacm_c)

        elif tipo == 'phi1':
            #nacme = np.dot(W[-4], all_nacm_c)
            nacme = np.dot(W[-2], all_nacm_c)

        elif tipo == 'r3':
            #nacme = np.dot(W[-3], all_nacm_c)
            nacme = np.dot(W[-5], all_nacm_c)

        elif tipo == 'phi2':
            #nacme = np.dot(W[-2], all_nacm_c)
            nacme = np.dot(W[-4], all_nacm_c)

        elif tipo == 'tau':
            nacme = np.dot(W[-1], all_nacm_c)


        return(nacme)

def cot(ang):
    return((1./np.tan(ang)))


def G_mel(r,R,Ms,r_cv):
    '''Ms is a list which contains the masses'''
    '''the order and types are the same as in the function nacme_calc'''
    '''r_cv es el mismo vector que en nacme_calc'''
    m1 = float(Ms[0])
    m2 = float(Ms[1])
    if (r,R) == ('r1','r2') or (r,R) == ('r2','r1'):

        R1 = r_cv[:3]
        R2 = r_cv[3:6]
        R3 = r_cv[6:9]

        r1 = R1-R2
        r2 = R3-R2

        phi213 = np.arccos(np.dot(r1,r2)/(la.norm(r1)*la.norm(r2)))

        r1 = la.norm(r1)
        r2 = la.norm(r2)

        G_rr = G_RR = 1/m1 + 1/m2
        G_rR = (1./m1)*np.cos(phi213)
        
        Vp = (1./(m1*r1*r2))*np.cos(phi213)

        if (r,R) == ('r1','r2'):
            
            return(G_rr, G_RR, G_rR, 0., 0., 0., 0., Vp)

        elif(r,R) == ('r2','r1'):
            #G_d = { 'Grr' : G_RR, 'GRR': G_rr, 'GrR' : G_rR, 'dGrr_dr' : 0., 'dGRR_dR' : 0., 'dGrR_dr' : 0., 'dGrR_dR' : 0. }
            return(G_RR, G_rr, G_rR, 0., 0., 0., 0., Vp)

    elif (r,R) == ('r1','phi1') or (r,R) == ('phi1','r1'):
        m3 = float(Ms[2])

        R1 = r_cv[:3]
        R2 = r_cv[3:6]
        R3 = r_cv[6:9]

        r1 = R2-R1
        r2 = R3-R2

        phi123 = np.arccos(np.dot(r1,r2)/(la.norm(r1)*la.norm(r2)))
        
        r1 = la.norm(r1)
        r2 = la.norm(r2)

        G_rR = -(1./(m2*r2))*np.sin(phi123)
        dGrp_dp = -(1./(m2*r2))*np.cos(phi123)
        Vp_rp = -(1./(m2*r2*r1))*np.cos(phi123)

        G_rr = 1./m1 + 1./m2

        G_RR = (1./(m1*r1**2.)) + (1./(m3*r2**2.)) + ((1./m2)*( (1./r1**2.) + (1./r2**2.) - ((2.*np.cos(phi123))/(r1*r2))))
        dGRR_dR = ((2.*np.sin(phi123))/(m2*r1*r2))
        Vp_RR = (np.cos(phi123)/(2.*m2*r1*r2)) - ( G_RR/8. *(2+(cot(phi123)**2.)))

        Vp = Vp_rp+Vp_RR

        if (r,R) == ('r1','phi1'):
            return(G_rr,G_RR,G_rR,0., dGRR_dR, 0., dGrp_dp,Vp)
        elif (r,R) == ('phi1','r1'):
            return(G_RR,G_rr,G_rR,dGRR_dR,0.,dGrp_dp,0.,Vp)

    elif (r,R) == ('phi1','phi2') or (r,R) == ('phi2','phi1'):
        m3 = float(Ms[2])
        m4 = float(Ms[3])

        R1 = r_cv[:3]
        R2 = r_cv[3:6]
        R3 = r_cv[6:9]
        R4 = r_cv[9:12]

        r12 = R2-R1
        r23 = R3-R2
        r14 = R4-R1

        phi123 = np.arccos(np.dot(r12,r23)/(la.norm(r12)*la.norm(r23)))
        phi214 = np.arccos(np.dot(-r12,r14)/(la.norm(r12)*la.norm(r14)))

        n1 = np.cross(-r23/la.norm(r23),-r12/la.norm(r12))
        n2 = np.cross(-r12/la.norm(r12),r14/la.norm(r14))

        tau3214 = np.arccos(np.dot(n1,n2)/(la.norm(n1)*la.norm(n2)))

        r12 = la.norm(r12)
        r23 = la.norm(r23)
        r14 = la.norm(r14)

        Grr = (1./(m1*r12**2.)) + (1./(m3*r23**2.)) + ( (1./m2)*( 1./r12**2. + 1./r23**2. -( (2.*np.cos(phi123))/(r12*r23)) ))
        dGrr_dr = (2.*np.sin(phi123))/(m2*r12*r23)
        Vp_rr = (np.cos(phi123)/(2.*m2*r12*r23)) - (Grr/8.*(2.+(cot(phi123)**2.))) 

        GRR = (1./(m2*r12**2.)) + (1./(m4*r14**2.)) + ( (1./m1)*( 1./r12**2. + 1./r14**2. -( (2.*np.cos(phi214))/(r12*r14)) ))
        dGRR_dR = (2.*np.sin(phi214))/(m1*r12*r14)

        lamb214=(1./np.sin(phi214))*( (1/r12) - (np.cos(phi214)/r14))
        lamb123=(1./np.sin(phi123))*( (1/r12) - (np.cos(phi123)/r23))

        GrR = (-np.cos(tau3214)/r12)*( (lamb214*np.sin(phi214))/m1 + (lamb123*np.sin(phi123))/m2 )
        dGrR_dr = (-np.cos(tau3214)/r12)*((lamb123*np.cos(phi123))/m2 )
        dGrR_dR = (-np.cos(tau3214)/r12)*((lamb214*np.cos(phi214))/m1 )


    elif (r,R) == ('r1','tau') or (r,R) == ('tau','r1'):
        m3 = float(Ms[2])
        m4 = float(Ms[3])

        R1 = r_cv[:3]
        R2 = r_cv[3:6]
        R3 = r_cv[6:9]
        R4 = r_cv[9:12]

        r12 = R2-R1
        r23 = R3-R2
        r34 = R4-R3

        phi123 = np.arccos( np.dot(r12,r23)/(la.norm(r12)*la.norm(r34)))
        phi234 = np.arccos( np.dot(r23,r34)/(la.norm(r12)*la.norm(r34)))
        
        n1 = np.cross(r12/la.norm(r12),r23/la.norm(r23))
        n2 = np.cross(r23/la.norm(r23),r34/la.norm(r34))

        tau1234 = np.arccos(np.dot(n1,n2)/(la.norm(n1)*la.norm(n2)))

        r12 = la.norm(r12)
        r23 = la.norm(r23)
        r34 = la.norm(r34)
        
        lam123 = ((1./r12) - (np.cos(phi123)/r23))*(1./np.sin(phi123))
        lam432 = ((1./r34) - (np.cos(phi123)/r23))*(1./np.sin(phi234))

        G_rr =  1./m1 + 1./m2
        
        G_rt = -(1./(m2*r23))*np.sin(phi123)*cot(phi234)*np.sin(tau1234)
        dGrt_dt = -(1./(m2*r23))*np.sin(phi123)*cot(phi234)*np.cos(tau1234)
        Vrt = (1./(2.*m2*r12*r23))*np.sin(phi123)*cot(phi234)*np.cos(tau1234)

        G_tt = (1./(m1*((r12*np.sin(phi123))**2.))) + (1./(m4*((r34*np.sin(phi234))**2.))) + ((1./m2)*(lam123**2. + (cot(phi234)/r23)**2.)) + ((1./m3)*(lam432**2. + (cot(phi123)/r23)**2.)) - ( 2.*( (np.cos(tau1234)/r23)*( (lam123/m2)*cot(phi234) + (lam432/m3)*cot(phi123) )))
        dGtt_dt = 2.*( (np.sin(tau1234)/r23)*( (lam123/m2)*cot(phi234) + (lam432/m3)*cot(phi123) )) 

        if (r,R) == ('r1','tau'):

            #G_d = { 'Grr' : G_rr, 'GRR': G_tt, 'GrR' : G_rt, 'dGrr_dr' : 0., 'dGRR_dR' : dGtt_dt, 'dGrR_dr' : 0., 'dGrR_dR' : dGrt_dt }

            return(G_rr, G_tt, G_rt, 0., dGtt_dt, 0., dGrt_dt, Vrt)

        elif (r,R) == ('tau','r1'):

            return(G_tt, G_rr, G_rt, dGtt_dt, 0., dGrt_dt, 0., Vrt)

    else:
        print('Sorry! Not available')
        print('exit')
        break


def triang_calc(r, R, nacm_r, nacm_R, G_d):

    ns = len(nacm_r)
    
    tsup = np.zeros( (ns, len(r), len(R)) )
    tinf = np.zeros( (ns, len(r), len(R)) )
    Lsq = np.zeros( (ns,len(r),len(R)) )

    ind = [0]
    for l in range(-1,-ns,-1):
        ind.append(l)
    
    for l in range(ns):
        Lsq[ind[l-2]] = 0.5*((-1.)**(1.+l))*( G_d['Grr']*nacm_r[ind[l]]*nacm_r[ind[l-1]] + G_d['GRR']*nacm_R[ind[l]]*nacm_R[ind[l-1]] + 0.5*G_d['GrR']*nacm_r[ind[l]]*nacm_R[ind[l-1]] + 0.5*G_d['GrR']*nacm_R[ind[l]]*nacm_r[ind[l-1]] )

    l= 0
    for ij in range(ns):

        fijr = itp.RectBivariateSpline(r,R,nacm_r[ij])
        fijR = itp.RectBivariateSpline(r,R,nacm_R[ij])

        dLijr_dr = fijr(r,R,dx=1,dy=0)
        dLijr_dR = fijr(r,R,dx=0,dy=1)
        
        dLijR_dr = fijR(r,R,dx=1,dy=0) 
        dLijR_dR = fijR(r,R,dx=0,dy=1) 

        Lsq_ij_sup = np.zeros((len(r),len(R)))
        #[12--> -13 23 ,13--> 12 23 , 23 --> - 12 13, ]
        #[31, 32, 12]
        #[-2
        #[12, 13, 23]
        #[0,  1,  2]
        #[-3, -2, -1]
        
        tsup[ij] += - 0.5*G_d['GrR']*dLijr_dR - 0.5*G_d['GrR']*dLijR_dr - 0.5*G_d['GRR']*dLijR_dR - 0.5*G_d['Grr']*dLijr_dr - 0.5*G_d['dGrr_dr']*nacm_r[ij] - 0.5*G_d['dGRR_dR']*nacm_R[ij] - 0.5*G_d['dGrR_dr']*nacm_R[ij] - 0.5*G_d['dGrR_dR']*nacm_r[ij] + Lsq[ij]


        #iy = (-ns+1)-ij
        #Lsq_ij_inf = np.zeros((len(r), len(R)))
        #l=0
        #while l < ns:
        #    Lsq_ij_inf += (-1.)**(1+l)*( (nacm_r[-ns+1+l]*nacm_r[-1+l] ) + nacm_R[-ns+1+l]*nacm_R[-1+l] + nacm_r[-ns+1+l]*nacm_R[-1+l] + nacm_R[-ns+1+l]*nacm_r[-1+l ] )   
        #ij  --> iy
        #-2 1--> 0 -3
        #-1 2--> 1 -2
        #-3 0--> 2 -1
        #0 1 2 --> -1 -3 -2 201
        if ij == 0:
            tinf[ns-1] += + 0.5*G_d['GrR']*dLijr_dR + 0.5*G_d['GrR']*dLijR_dr + 0.5*G_d['GRR']*dLijR_dR + 0.5*G_d['Grr']*dLijr_dr + 0.5*G_d['dGrr_dr']*nacm_r[ij] + 0.5*G_d['dGRR_dR']*nacm_R[ij] + 0.5*G_d['dGrR_dr']*nacm_R[ij] + 0.5*G_d['dGrR_dR']*nacm_r[ij] + Lsq[ij]

        else:
            tinf[ij-1] += + 0.5*G_d['GrR']*dLijr_dR + 0.5*G_d['GrR']*dLijR_dr + 0.5*G_d['GRR']*dLijR_dR + 0.5*G_d['Grr']*dLijr_dr + 0.5*G_d['dGrr_dr']*nacm_r[ij] + 0.5*G_d['dGRR_dR']*nacm_R[ij] + 0.5*G_d['dGrR_dr']*nacm_R[ij] + 0.5*G_d['dGrR_dR']*nacm_r[ij] + Lsq[ij]
            

    return(tsup,tinf)
     

    
###If a Butterscotch-type filter is wanted to be used:
def DF_s(xx):
    N = float(len(xx))
    #print(int(N)//2)
    dk = (2.*np.pi)/(N*(xx[1]-xx[0]))
    DF = np.zeros(int(N),dtype=np.complex_)
    DF2 = [(1j*(float(l)*dk)) for l in range(-(int(N)//2)+1,int(N)//2+1) ]
    DF[:(int(N)//2)+1] = DF2[int(N)//2-1:]
    DF[(int(N)//2)+1:] = (DF2[:int(N)//2-1])
    DF = DF*np.eye(int(N))
    ww = []
    for a in range(len(xx)):
        wi = []
        wij = np.exp(1j*2.*np.pi/N)
        for b in range(len(xx)):
            wi.append(wij**(float(a)*float(b)))
        ww.append(wi)

    ww = 1./np.sqrt(N)*np.array(ww,dtype=np.complex_)
    Dm= np.matmul(ww,np.matmul(DF,np.conj(ww).T))

    return(ww,DF) #Returns Fourier's transform frequencies and the derivative in momentum representation.


def butter_phi(ww,psi,DN):
    ''' This rountine is for filtering in the R coordinate direction of the matrices'''
    '''Nor is the number of elements corresponding to the length of the non-extended grid'''
    '''N is the real length'''
    '''DN=N-Nor ---> but it should be provided as it is the desired non-zero elements in the momentum representation array'''
    N = len(ww)
    #Nor = int((60.*(np.pi/180.))/(ang2[1]-ang2[0]))
    #DN = N-Nor
    Nor = N-DN
    #print(Nor,DN)

    #print(int(Nor),N,int(N-Nor))
    lp = np.zeros(N,dtype=np.complex_)
    lp[Nor//2:-Nor//2] =np.ones(int(DN))
    lp[:Nor//2] = np.exp((Rn[:Nor//2]-Rn[Nor//2]))
    lp[-Nor//2:] = np.exp(-(Rn[-Nor//2:]-Rn[-Nor//2]))
    lp = lp*np.ones(psi.shape)
    ps0_k = np.matmul(np.conj(ww).T,psi.T)

    psp_ko=np.zeros(psi.T.shape,dtype=np.complex_)
    psp_ko[int(N)//2-1:] = ps0_k[:(int(N)//2)+1]
    psp_ko[:int(N)//2-1] = ps0_k[(int(N)//2)+1:]
    filt = lp.T*psp_ko
    ps0_k[:(int(N)//2)+1] = filt[int(N)//2-1:]
    ps0_k[(int(N)//2)+1:] = filt[:int(N)//2-1]

    dps0 = np.matmul(ww,ps0_k).T

    return(dps0)


def butter_tau(wwt,psi,DNt):
    ''' This rountine is for filtering in the r coordinate direction of the matrices'''
    '''Ntor is the number of elements corresponding to the length of the non-extended grid'''
    '''Nt is the real length'''
    '''DNt=Nt-Ntor ---> but it should be provided as it is the desired non-zero elements in the momentum representation array'''

    Nt = len(wwt)
    #Ntor = int((540.*(np.pi/180.))/(di1[1]-di1[0]))
    #DNt = Nt-Ntor
    Ntor = Nt-DNt
    #print(Ntor,DNt)

    lpt = np.zeros(Nt,dtype=np.complex_)
    lpt[Ntor//2:-Ntor//2] =np.ones(int(DNt))
    lpt[:Ntor//2] = np.exp((rn[:Ntor//2]-rn[Ntor//2]))
    lpt[-Ntor//2:] = np.exp((rn[-Ntor//2:]-rn[-Ntor//2]))
    lpt = lpt*np.ones(psi.T.shape)
    ps0_kt = np.matmul(np.conj(wwt).T,psi)

    psp_kt=np.zeros(psi.shape,dtype=np.complex_)
    psp_kt[int(Nt)//2-1:] = ps0_kt[:(int(Nt)//2)+1]
    psp_kt[:int(Nt)//2-1] = ps0_kt[(int(Nt)//2)+1:]
    filtt = lpt*psp_kt.T
    filtt= filtt.T
    ps0_kt[:(int(Nt)//2)+1] = filtt[int(Nt)//2-1:]
    ps0_kt[(int(Nt)//2)+1:] = filtt[:int(Nt)//2-1]

    #fig2 = plt.figure()
    #ax2 = fig2.add_subplot(111, projection='3d')
    #ax2.plot_surface(Rmn, rmn, filtt)
    #plt.show()
    #plt.close()

    dps02 = np.matmul(wwt,ps0_kt)

    return(dps02)

#-------------------Other useful routines
def takefirst(lista):
    return(lista[0])

def takedos(lista):
    return(lista[0],lista[1])

#-------------------Instructions to introduce the input--------------------------------------------------------#:
'''Usage of the program: python3 QDynLH.py energs.txt geoms.txt nacmes_ij.txt'''
'''energs.txt is a file containing the positions and energies in the order  r  R  E1 E2... En '''
'''r is the coordinate that changes with changing rows (if rR = np.array((len(r),len(R))) r= rR[:,m]) where m is any index in 0<=m<=len(R)  '''
'''R is the coordinate that changes with changing columns (R = rR[n] with 0<=n<= len(r))'''
'''the first line of energs.txt must be: r R E1  ... En (2D case)'''
'''                                  or: r E1 E2 ... En (1D case)'''
'''where r and R can be r1,r2,r3,phi1,phi2,tau (the coordinates are defined below)'''
'''An input with the xyz coordinates of the atoms involved in the definitions of the internal cordinates must be provided.'''
'''The first line of the geoms file must be given in the same way as the first line of the nacme files'''
'''The geoms file must contain the x y z coordinates for each atom on the same line. The geometries are to be put in an order that coincides with the corresponding line in the energy file, e.g the second line on the geoms.txt file must correspond to the geometry of the PES at the second line in the energs.txt file .'''
'''nacmes_ij.txt are the separate files containinig the nacmes for the coupled states i and j. The files must be provided in the following order: nacm_12.txt,nacm_13.txt,...,nacm_1n.txt, nacm_23.txt,...,nacm_2n.txt,nacm_34.txt,...,...,nacm_n-1n.txt '''
''' The first line of each of the files should contain the number of atoms involved followed by each of their masses and finally the types of coordinates to be calculated'''
'''For example if the calculation involves a distance r12 and a dihedral tau1234 for the atoms A1 A2 A3 A4'''
'''the first line should be 4 m_a1 m_a2 m_a3 m_a4 r1 tau.'''
'''The masses must be given in multiples of the electron mass (1 uma = 1822.88839 me).'''
'''The types of coordinates can be: r1, r2, or r3 for bond distances, phi1 or phi2 for bond angles and tau for dihedral angles.'''
'''The following lines should contain the NAC matrix elements corresponding to each atom in the following order: '''
'''A1_x A1_y A1_z A2_x A2_y A2_z A3_x A3_y A3_z A4_x A4_y A4_z.'''
'''where A1,A2,...,An means atom 1, atom 2, ..., atom n'''
'''r1 = bond distance between A1 and A2'''
'''r2 = bond distance between A2 and A3'''
'''r3 = bond distance between A3 and A4'''
'''phi1 = bond angle between the atoms A1, A2 and A3'''
'''phi2 = bond angle between the atoms A2, A3 and A4'''
'''tau  = (proper) dihedral angle formed with the atoms at A1,A2, A3 and A4 '''
'''They must be written in the same order as in the geoms file.'''
'''The n-th line must correspond to the same (r,R) than the n-th line in the energs file.'''


inputs = sys.argv
file_energs = open(inputs[1],'r')
fe = file_energs.readlines()

file_geoms = open(inputs[2],'r')
fg = file_geoms.readlines()

nacmes = []
for a1 in range(3,len(inputs)):
    fi_nac = open(inputs[a1],'r')
    fnac = fi_nac.readlines()
    nacmes.append(fnac)

print('For how much time is the wavefunction going to be propagated? Please provide a number in femtoseconds')
t_mx = float(input())*41.34
print('Which is the time step? Please provide a number in femtoseconds')
dt = float(input())*41.34

time = np.arange(0.,t_mx+dt,dt)
cord_type = ['r1','r2','r3','phi1','phi2','tau']
firstl = fe[0].split()
if firstl[1] in cord_type:
    alls = {}
    all_g = {}
    for a2 in range(1,len(fe)):
        elem = np.array(fe[a2].split(),dtype=float)
        ind = ( np.around(elem[0],decimals=3), np.around(elem[1],decimals=3) ) #The rounding is in order to obtain the correct number of non repeated elements
        alls[ind] = elem[2:]

        all_g[ind] = np.array(fg[a2].split(),dtype=float)



    r_R = list(alls.keys())
    r = list(set( [ r_R[a3][0] for a3 in range(len(r_R))]))
    R = list(set( [ r_R[a3][1] for a3 in range(len(r_R))]))

    r.sort()
    R.sort()
    

    R_m,r_m=np.meshgrid(R,r)

    T_r = TFG(r)
    T_R = TFG(R)

    D_dr = DF(r)
    D_dR = DF(R)
    
    T_r = TFG_2(r)
    T_R = TFG_2(R)

    D_dr = DF_2(r)
    D_dR = DF_2(R)
    
    Grr_d = {}
    GRR_d = {}
    GrR_d = {}
    dGrr_dr_d = {}
    dGRR_dR_d = {}
    dGrR_dr_d = {}
    dGrR_dR_d = {}
    Vpd = {}
    ind3 = list(all_g.keys())
    ms = fg[0].split()
    for a10 in range(len(ind3)):
        rvec = all_g[ind3[a10]]
        Grr_d[ind3[a10]],GRR_d[ind3[a10]],GrR_d[ind3[a10]],dGrr_dr_d[ind3[a10]],dGRR_dR_d[ind3[a10]],dGrR_dr_d[ind3[a10]],dGrR_dR_d[ind3[a10]],Vpd[ind3[a10]] = G_mel(firstl[0],firstl[1],ms[1:int(ms[0])+1],rvec)

    Grr = np.zeros((len(r),len(R)))
    GRR = np.zeros((len(r),len(R)))
    GrR = np.zeros((len(r),len(R)))
    dGrr_dr = np.zeros((len(r),len(R)))
    dGRR_dR = np.zeros((len(r),len(R)))
    dGrR_dr = np.zeros((len(r),len(R)))
    dGrR_dR = np.zeros((len(r),len(R)))
    Vp = np.zeros((len(r),len(R)))

    for a11 in range(len(r)):
        for a12 in range(len(R)):

            Grr[a11,a12] = Grr_d[(np.around(r[a11],decimals=3),np.around(R[a12],decimals=3))]
            GRR[a11,a12] = GRR_d[(np.around(r[a11],decimals=3),np.around(R[a12],decimals=3))] 
            GrR[a11,a12] = GrR_d[(np.around(r[a11],decimals=3),np.around(R[a12],decimals=3))] 
            dGrr_dr[a11,a12] = dGrr_dr_d[(np.around(r[a11],decimals=3),np.around(R[a12],decimals=3))]
            dGRR_dR[a11,a12] = dGRR_dR_d[(np.around(r[a11],decimals=3),np.around(R[a12],decimals=3))]
            dGrR_dr[a11,a12] = dGrR_dr_d[(np.around(r[a11],decimals=3),np.around(R[a12],decimals=3))]
            dGrR_dR[a11,a12] = dGrR_dR_d[(np.around(r[a11],decimals=3),np.around(R[a12],decimals=3))]
            Vp[a11,a12] = Vpd[(np.around(r[a11],decimals=3),np.around(R[a12],decimals=3))]
    
    Gs_d = { 'Grr' : Grr, 'GRR': GRR, 'GrR' : GrR, 'dGrr_dr' : dGrr_dr, 'dGRR_dR' : dGRR_dR, 'dGrR_dr' : dGrR_dr, 'dGrR_dR' : dGrR_dR }


    Energs = np.zeros((len(elem[2:]),len(r),len(R)))

    for a2 in range(len(Energs)):
        
        for a3 in range(len(r)):
            for a4 in range(len(R)):


                Energs[a2,a3,a4] = alls[( np.around(r[a3],decimals=3), np.around(R[a4],decimals=3) ) ][a2] + Vp[a3,a4]
    

    print('Do you want to use a Gaussian wave packet as the initial condition?')
    res = input()
    if res in 'yes' or res in 'si':
        print('Introduce the r around which the Gaussian is centered, the sigma and the momentum for the r degree of freedom, separated by a blank space. Signs are alowed.')
        r0,sigma_r,p_r = input().split()
        r0 = float(r0.replace(',',''))
        sigma_r = float(sigma_r.replace(',',''))
        p_r = float(p_r.replace(',',''))
        print('Introduce the R around which the Gaussian is centered, the sigma and the momentum for the R degree of freedom, separated by a blank space. Signs are alowed.')
        R0,sigma_R,p_R = input().split()
        R0 = float(R0.replace(',',''))
        sigma_R = float(sigma_R.replace(',',''))
        p_R = float(p_R.replace(',',''))

        Psi0 = np.exp( -(r_m-r0)**2./sigma_r )*np.exp( -(R_m-R0)**2./sigma_R )*np.exp(1j*p_r*(r_m-r0))*np.exp(1j*p_R*(R_m-R0))    #Initial psi as a Gaussian wavepacket. Change if wanted

        print('Introduce the electronic state in which the wave packet starts, 0 for ground state, 1 for S1, ..., n for Sn')
        st = int(input())

    else:
            print('function still not available!')
   
    if len(nacmes) > 0:
        
        nacms_r = {}
        nacms_R = {}
        for a5 in range(len(nacmes)):
            ms = nacmes[a5][0].split()
            ind2 = list(alls.keys())
            for a6 in range(1,len(nacmes[a5])):
                nac_v = np.array(nacmes[a5][a6].split(),dtype=float)  #Notice that it is very important for the data in the nacme files to have the same order as the data in the energs file.
                if a5 == 0:
                    rvec = all_g[ind2[a6-1]]
                    nacms_r[ind2[a6-1]] = [nacme_calc(ms[-2],ms[1:int(ms[0])+1],nac_v,0.,0.,0.,rvec)]
                    nacms_R[ind2[a6-1]] = [nacme_calc(ms[-1],ms[1:int(ms[0])+1],nac_v,0.,0.,0.,rvec)]

                else:
                    rvec = all_g[ind2[a6-1]]
                    nacms_r[ind2[a6-1]].append(nacme_calc(ms[-2],ms[1:int(ms[0])+1],nac_v,0.,0.,0.,rvec))
                    nacms_R[ind2[a6-1]].append(nacme_calc(ms[-1],ms[1:int(ms[0])+1],nac_v,0.,0.,0.,rvec))


                    #checar multiplicar los nacmes por la transformacion inversa a ver que me da...


        Nacms_r = np.zeros((len(nacmes), len(r),len(R)))
        Nacms_R = np.zeros((len(nacmes), len(r),len(R)))

        for a7 in range(len(nacmes)):

            for a8 in range(len(r)):
                for a9 in range(len(R)):

                    Nacms_r[a7,a8,a9] = nacms_r[( np.around(r[a8],decimals=3), np.around(R[a9],decimals=3) )][a7]
                    Nacms_R[a7,a8,a9] = nacms_R[( np.around(r[a8],decimals=3), np.around(R[a9],decimals=3) )][a7]



        for a14 in range(len(Energs)): 
            for a13 in range(len(r)):
                #Tsup_l[a14][a13] = signal.savgol_filter(Tsup_l[a14][a13], 9,2)
                #Tinf_l[a14][a13] = signal.savgol_filter(Tinf_l[a14][a13], 9,2)
                Nacms_r[a14][a13] =signal.savgol_filter(Nacms_r[a14][a13],11,2)
                Nacms_R[a14][a13] =signal.savgol_filter(Nacms_R[a14][a13],11,2)
            for a13 in range(len(R)):
                #Tsup_l[a14][:,a13] = signal.savgol_filter(Tsup_l[a14][:,a13],19,2)
                #Tinf_l[a14][:,a13] = signal.savgol_filter(Tinf_l[a14][:,a13],19,2)
                Nacms_r[a14][:,a13] =signal.savgol_filter(Nacms_r[a14][:,a13],11,2)
                Nacms_R[a14][:,a13] =signal.savgol_filter(Nacms_R[a14][:,a13],11,2)
        
         
        Tsup_l, Tinf_l = triang_calc(r, R, Nacms_r, Nacms_R, Gs_d) 

        for a14 in range(len(Energs)): 
            for a13 in range(len(r)):
                Tsup_l[a14][a13] = signal.savgol_filter(Tsup_l[a14][a13],15,3)
                Tinf_l[a14][a13] = signal.savgol_filter(Tinf_l[a14][a13],15,3)
            for a13 in range(len(R)):
                Tsup_l[a14][:,a13] = signal.savgol_filter(Tsup_l[a14][:,a13],15,3)
                Tinf_l[a14][:,a13] = signal.savgol_filter(Tinf_l[a14][:,a13],15,3)
            
        #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        rn = np.linspace(3.4,7.8,300)
        Rn = np.linspace(-0.1,np.max(R)+0.1,640)
        Rmn,rmn = np.meshgrid(Rn,rn)
        Tsup_ln = np.zeros((len(Energs), len(rn),len(Rn)))
        Tinf_ln = np.zeros((len(Energs), len(rn),len(Rn)))
        Energsn = np.zeros((len(Energs), len(rn),len(Rn)))
        Nacms_rn = np.zeros((len(Energs), len(rn),len(Rn)))
        Nacms_Rn = np.zeros((len(Energs), len(rn),len(Rn)))
        for a12 in range(len(Energs)):
            fsln = itp.RectBivariateSpline(r,R,Tsup_l[a12])
            filn = itp.RectBivariateSpline(r,R,Tinf_l[a12])
            feln = itp.RectBivariateSpline(r,R,Energs[a12])
            fnrln = itp.RectBivariateSpline(r,R,Nacms_r[a12])
            fnRln = itp.RectBivariateSpline(r,R,Nacms_R[a12])

            Tsup_ln[a12] = fsln(rn,Rn)
            Tinf_ln[a12] = filn(rn,Rn)
            Energsn[a12] = feln(rn,Rn)
            Nacms_rn[a12] = fnrln(rn,Rn)
            Nacms_Rn[a12] = fnRln(rn,Rn)

            for a13 in range(len(rn)):
                Tsup_ln[a12][a13] = signal.savgol_filter(Tsup_ln[a12][a13],19,2)
                Tinf_ln[a12][a13] = signal.savgol_filter(Tinf_ln[a12][a13],19,2)
                Energsn[a12][a13] = signal.savgol_filter(Energsn[a12][a13],13,2)
                Nacms_rn[a12][a13] =signal.savgol_filter(Nacms_rn[a12][a13],19,2)
                Nacms_Rn[a12][a13] =signal.savgol_filter(Nacms_Rn[a12][a13],19,2)
            for a13 in range(len(Rn)):
                Tsup_ln[a12][:,a13] = signal.savgol_filter(Tsup_ln[a12][:,a13],25,2)
                Tinf_ln[a12][:,a13] = signal.savgol_filter(Tinf_ln[a12][:,a13],25,2)
                Energsn[a12][:,a13] = signal.savgol_filter(Energsn[a12][:,a13],13,2)
                Nacms_rn[a12][:,a13] =signal.savgol_filter(Nacms_rn[a12][:,a13],25,2)
                Nacms_Rn[a12][:,a13] =signal.savgol_filter(Nacms_Rn[a12][:,a13],25,2)

        Grrn = np.zeros((len(rn),len(Rn))) 
        GRRn = np.zeros((len(rn),len(Rn)))
        GrRn = np.zeros((len(rn),len(Rn)))
        dGrr_drn = np.zeros((len(rn),len(Rn)))
        dGRR_dRn = np.zeros((len(rn),len(Rn)))
        dGrR_drn = np.zeros((len(rn),len(Rn)))
        dGrR_dRn = np.zeros((len(rn),len(Rn)))
        fgrln = itp.RectBivariateSpline(r,R,Grr)
        fgRln = itp.RectBivariateSpline(r,R,GRR)
        fgrRln = itp.RectBivariateSpline(r,R,GrR)
        fdgrln = itp.RectBivariateSpline(r,R,dGrr_dr)
        fdgRln = itp.RectBivariateSpline(r,R,dGRR_dR)
        fdgrRln1 = itp.RectBivariateSpline(r,R,dGrR_dr)
        fdgrRln2 = itp.RectBivariateSpline(r,R,dGrR_dR)
        Grrn = fgrln(rn,Rn) 
        GRRn = fgRln(rn,Rn) 
        GrRn = fgrRln(rn,Rn)
        dGrr_drn = fdgrln(rn,Rn)
        dGRR_dRn = fdgRln(rn,Rn)
        dGrR_drn = fdgrRln1(rn,Rn)
        dGrR_dRn = fdgrRln2(rn,Rn)

        v_iop = np.zeros(( len(Energsn), len(rn), len(Rn)),dtype=np.complex_)
        for a15 in range(len(Energsn)):
            for v1 in range(len(rn)):
                for v2 in range(len(Rn)):
                    if rn[v1] < np.min(r):
                        r00 = np.min(r)
                        v_iop[a15,v1,v2]+= -50.5j*(rn[v1]-r00)**2
                        Tsup_ln[a15,v1,v2] = 0.
                        Tinf_ln[a15,v1,v2] = 0.
                        Nacms_rn[a15,v1,v2] = 0.
                        Nacms_Rn[a15,v1,v2] = 0.
                    if Rn[v2] < np.min(R):
                        R00 = np.min(R)
                        v_iop[a15,v1,v2]+= -230.5j*(Rn[v2]-R00)**2
                        Tsup_ln[a15,v1,v2] = 0.
                        Tinf_ln[a15,v1,v2] = 0.
                        Nacms_rn[a15,v1,v2] = 0.
                        Nacms_Rn[a15,v1,v2] = 0.
                    if rn[v1] > np.max(r):
                        r00 =np.max(r)
                        v_iop[a15,v1,v2]+= -50.5j*(rn[v1]-r00)**2
                        Tsup_ln[a15,v1,v2] = 0.
                        Tinf_ln[a15,v1,v2] = 0.
                        Nacms_rn[a15,v1,v2] = 0.
                        Nacms_Rn[a15,v1,v2] = 0.
                    if Rn[v2] > np.max(R):
                        R00 =np.max(R)
                        v_iop[a15,v1,v2]+= -230.5j*(Rn[v2]-R00)**2
                        Tsup_ln[a15,v1,v2] = 0.
                        Tinf_ln[a15,v1,v2] = 0.
                        Nacms_rn[a15,v1,v2] = 0.
                        Nacms_Rn[a15,v1,v2] = 0.

        for a1 in range(len(rn)):
            for b1 in range(len(Rn)):
                if v_iop[0,a1,b1].imag < v_iop[0,50,0].imag:
                    v_iop[0,a1,b1] = v_iop[0,50, 0]
                    v_iop[1,a1,b1] = v_iop[1,50, 0]
                    v_iop[2,a1,b1] = v_iop[2,50, 0]


        wwr,DFr = DF_s(rn)
        wwR,DFR = DF_s(Rn)
 
        Energsn=np.array(Energsn,dtype=np.complex_)
        Energsn+=v_iop


        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.plot_surface(Rmn, rmn, (Energsn[0].real-np.min(Energsn[0]))*27.211 )
        ax2.set_zlim((-0.05,12.))
        ax2.set_xlabel('$\%s$' %'tau')
        ax2.set_ylabel('$r$')
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.plot_surface(Rmn, rmn, (Energsn[1].real-np.min(Energsn[0]))*27.211, color='orange')
        ax2.set_zlim((-0.05,12.))
        ax2.set_xlabel('$\%s$' %'tau')
        ax2.set_ylabel('$r$')
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.plot_surface(Rmn, rmn, (Energsn[2].real-np.min(Energsn[0]))*27.211, color='green')
        ax2.set_zlim((-0.05,12.))
        ax2.set_xlabel('$\%s$' %'tau')
        ax2.set_ylabel('$r$')
        plt.show()
        plt.close() 
        
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.plot_surface(Rmn, rmn, Tsup_ln[0].real)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.plot_surface(Rmn, rmn, Tsup_ln[1].real)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.plot_surface(Rmn, rmn, Tsup_ln[2].real)
        plt.show()
        plt.close() 

        Psi0n = np.exp( -(rmn-r0)**2./sigma_r )*np.exp( -(Rmn-R0)**2./sigma_R )*np.exp(1j*p_r*(rmn-r0))*np.exp(1j*p_R*(Rmn-R0))    #Initial psi as a Gaussian wavepacket. Change if wanted
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.plot_surface(Rmn, rmn, Psi0n)
        plt.show()
        plt.close() 
        psi_t_p = np.zeros( (len(Energs)*len(rn), len(Rn)), dtype=np.complex_)

        psi_t_p[st*len(rn):(st+1)*len(rn)] = Psi0n/np.sqrt(np.tensordot(np.conj(Psi0n),Psi0n))

        T_r = TFG_2(rn)
        T_R = TFG_2(Rn)

        D_dr = DF_2(rn)
        D_dR = DF_2(Rn)
        
        Gs_d = { 'Grr' : cp.asarray(Grrn), 'GRR': cp.asarray(GRRn), 'GrR' : cp.asarray(GrRn), 'dGrr_dr' : cp.asarray(dGrr_drn), 'dGRR_dR' : cp.asarray(dGRR_dRn), 'dGrR_dr' : cp.asarray(dGrR_drn), 'dGrR_dR' : cp.asarray(dGrR_dRn) }
        
        dr = rn[1]-rn[0]
        dR = Rn[1]-Rn[0]
        Emax = Gs_d['Grr']*(np.pi**2./dr**2) + Gs_d['GRR']*(np.pi**2./dR**2.) + cp.asarray(Energsn) + 2.5
        Eall = [ cp.asnumpy(Emax), cp.asnumpy(Energsn) -2.5 ]
 

        fu = open('poblaciones.txt','w')
        fu.write('time \t P0 \t P1 \t P2\n')
        fil_e = open('energs_tot.txt','w')
        fil_e.write('time \t Energy_r \t Energy_i \n')
        for p in range(1,len(time)):
            psi_t_p = cheb_prop_internal_nl_nacme_gpu_fft(rn, Rn, cp.asarray(psi_t_p), Eall, cp.asarray(T_r), cp.asarray(T_R), cp.asarray(D_dr), cp.asarray(D_dR), Gs_d, cp.asarray(Energsn), 0., cp.asarray(Tsup_ln), cp.asarray(Tinf_ln), cp.asarray(Nacms_rn), cp.asarray(Nacms_Rn), len(Energsn), 0., time[p-1], time[p] )


            psi_t_p = cp.asnumpy(psi_t_p)
            r = cp.asnumpy(rn)
            R = cp.asnumpy(Rn)
            psia = psi_t_p[:len(Psi0n)]
            psib = psi_t_p[len(Psi0n):2*len(Psi0n)]
            psic = psi_t_p[2*len(Psi0n):]
            fu.write('%.2f \t' %time[p].real)
            fu.write('%.8f \t' %np.tensordot(np.conj(psia),psia).real)
            fu.write('%.8f \t' %np.tensordot(np.conj(psib),psib).real)
            fu.write('%.8f \n' %np.tensordot(np.conj(psic),psic).real)
            #print('ptau,pphi', ptau_av_b, ktmax, pphi_av_b, kpmax)
            if p% 8 ==0:	
                    fig2 = plt.figure()
                    ax2 = fig2.add_subplot(111, projection='3d')
                    ax2.plot_surface(Rmn, rmn, (np.conj(psia)*psia).real, color='y')
                    ax2.view_init(elev=43,azim=-28)
                    ax2.set_title('%.6f fs' %(time[p].real/41.34))
                    #ax2.set_zlim((0.,0.5e-7))
                    plt.savefig('psi-2d-trial-s0-'+str(p)+'.png')
                    plt.close()
                    fig2 = plt.figure()
                    ax2 = fig2.add_subplot(111, projection='3d')
                    ax2.plot_surface(Rmn, rmn, (np.conj(psib)*psib).real, color='orange')
                    ax2.view_init(elev=43,azim=-28)
                    ax2.set_title('%.6f fs' %(time[p].real/41.34))
                    #ax2.set_zlim((0.,0.02))
                    plt.savefig('psi-2d-trial-s1-'+str(p)+'.png')
                    plt.close()
                    fig2 = plt.figure()
                    ax2 = fig2.add_subplot(111, projection='3d')
                    ax2.plot_surface(Rmn, rmn, (np.conj(psic)*psic).real, color='slategray')
                    ax2.view_init(elev=43,azim=-28)
                    ax2.set_title('%.6f fs' %(time[p].real/41.34))
                    #ax2.set_zlim((0.,0.02))
                    plt.savefig('psi-2d-trial-s2-'+str(p)+'.png')
                    plt.close()

                    plt.contour(Rn,rn, (np.conj(psia)*psia).real, 35, cmap='rainbow')
                    plt.title('%.6f fs' %(time[p].real/41.34))
                    plt.colorbar()
                    plt.savefig('psi-2d-cont-s0-'+str(p)+'.png')
                    plt.clf()
                    plt.contour(Rn,rn, (np.conj(psib)*psib).real, 35, cmap='jet')
                    plt.title('%.6f fs' %(time[p].real/41.34))
                    plt.colorbar()
                    plt.savefig('psi-2d-cont-s1-'+str(p)+'.png')
                    plt.clf()
                    plt.contour(Rn,rn, (np.conj(psic)*psic).real, 35, cmap='turbo')
                    plt.title('%.6f fs' %(time[p].real/41.34))
                    plt.colorbar()
                    plt.savefig('psi-2d-cont-s2-'+str(p)+'.png')
                    plt.clf()
            if p%15 == 0:
                psia = butter_tau(wwr,psia,230)
                psib = butter_tau(wwr,psib,230)
                psic = butter_tau(wwr,psic,230)

                psia = butter_phi(wwR,psia,450)
                psib = butter_phi(wwR,psib,450)
                psic = butter_phi(wwR,psic,450)

                psi_t_p = np.vstack((psia,psib,psic))
                psi_t_p = cp.asarray(psi_t_p)




    #elif len(nacmes) == 0:

        #for a10 in range(1,len(time)):
        #    psi_t_p = cheb_prop_internal_nl_nacme_gpu_fft(r, R, psi_t_p, Enes, K_r, K_R, D_r, D_R, Gs, Vs, vio, Tsup, Tinf, Lambs_r, Lambs_R, n, ef, time1, time2)
        

        #fig2 = plt.figure()
        #ax2 = fig2.add_subplot(111, projection='3d')
        #ax2.plot_surface(R_m, r_m, Nacms_r[0])
        #plt.show()
        #plt.close() 

