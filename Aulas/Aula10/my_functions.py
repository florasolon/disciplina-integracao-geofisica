#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib import gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import diags

mu0 = 4*np.pi*1E-7 # permeabilidade magnetica
z0 = 0  # The depth of the top of the first layer
im = (0.0+1.0j)

def Auxiliars(freq, p0,h):
    
    ''' This function calculates the auxiliars W, gamma, R and
    Impedance Tensor (Z). The entry parameters are frequency (freq) 
    and parameter vector (p0) in resistivity.
    This function returns vectors gamma, w, e and Z which contains 
    the respectivity values for each layer of p0
    
    Inputs:
    freq = frequency (Hz)
    p0 = vector of resistivities'''
    
    M = len(h) # Number of layers
    
    Z = np.zeros(M, dtype = 'complex') 
    gamma = np.zeros(M, dtype = 'complex')
    w = np.zeros(M, dtype = 'complex')
    R = np.zeros(M, dtype = 'complex')
    e = np.zeros(M, dtype = 'complex')
    hc = np.array(h, dtype='complex')

    gamma = np.sqrt((freq*mu0*im)/p0)
    w = gamma*p0
    e = np.exp(-2.0*gamma*h)   # Elemento e[-1] não é utilizado
    
    # Impedance at last layer
    
    gamma[-1] = np.sqrt((freq*mu0*im)/p0[-1])
    w[-1] = gamma[-1]*p0[-1]
    Z[-1] = w[-1]
    e[-1] = 0.0 + 0.0j
        
    # Impedance at each layer
    for i in range(M-1):
        j = M-i-2
        R[j] = (w[j] - Z[j+1])/(w[j] + Z[j+1])
        Z[j] = w[j] * ((1.0-R[j]*e[j])/(1.0+R[j]*e[j]))
 
    return gamma, w, R, e, Z

def Impedance_Tensor(omega, p0,h):
    '''This function calculates the impedance tensor for a given vector 
    of frequencies (omega) and a given vector of resitivities (p0)
    
    Inputs:
    
    omega = vector of frequencies
    p0 = vector of resistivities'''
    
    N = len(omega)
    Zcalc = np.zeros(N, dtype = 'complex')
    
    for k in range(N):
        freq = omega[k]
        gamma, w, R, e, Z = Auxiliars(freq, p0,h)
        Zcalc[k] = Z[0]
    
    return Zcalc

# In[ ]:

def multiplicative_noise(v, mu=0, sigma=0.05):
    '''This function assigns a Gaussian error with zero mean
    a given standard deviation to the input vector v.'''
    
    np.random.seed(10)    
    N = len(v)
    s = np.random.normal(mu, sigma, N)+ np.random.normal(mu, sigma/5., N)*im
    vnoise = (v+(s*v))
       
    return vnoise

def Jacobian_numerica(omega, p0,h):
        
    N = len(omega)
    M = len(p0)
    A = np.zeros((N,M),dtype = 'complex')
    pplus = p0.copy()
    pminus = p0.copy()
    
    for j in range(M):
        pdelta_plus = 10**(np.log10(p0[j]+0.03))
        pdelta_minus = 10**(np.log10(p0[j]-0.03))
        pplus[j] = pdelta_plus
        pminus[j] = pdelta_minus
        for i in range(N):
            freq = omega[i]
            gamma1, w1, R1, e1, Z1 = Auxiliars(freq, pplus,h)
            gamma2, w2, R2, e2, Z2 = Auxiliars(freq, pminus,h)
            Zplus = Z1[0]
            Zminus = Z2[0]
            A[i,j] = (np.log10(Zplus) - np.log10(Zminus))/(2.*0.03)
    
    return A     

def Jacobianline(freq, p0,h, gamma, w, R, e, Z):
    '''This function calculates one line of the Jacobian matrix
    It depends on a given frequency (freq), the parameter vector p0,
    and the auxiliary w, R, e, Z '''
    M = len(h)
    a = np.ones(M, dtype = 'complex')
    
    for i in range(M-1): #expressão 10.3.34
        aux1 = Z[i]/(2.0*p0[i])
        aux2 = 2.0*im*freq*mu0
        aux3 = (1.0+R[i]*e[i])**2.0
        aux4 = (w[i]+Z[i+1])**2.0
        aux5 = (h[i]*(w[i]**2.0 - Z[i+1]**2.0))/p0[i]
        # print aux1, aux2, aux3, aux4, aux5
        
        a[i] = aux1 - ((aux2/(aux3*aux4))*(aux5+Z[i+1]))
    
    a[-1] = 0.5*gamma[-1] #expressão 10.3.38
    
    for i in range(M-1):   # quantidade de quadrados
    
        #calcular quadrado i
        aux1 = 4.0*(w[i]**2)*e[i]
        aux2 = (w[i]+Z[i+1])**2
        aux3 = (1.0+R[i]*e[i])**2
        div = aux1/(aux2*aux3)
    
        #multiplicação do quadrado i pelas colunas
        j = (M-1-i)
        l = i+1
        for k in range(j):
            a[l] = a[l]*div
            l = l+1
    for i in range(M):
        a[i]*=p0[i]
        
    return a
    #return a*np.log(10.)

# In[ ]:

def Jacobian_analitica(omega, p0,h):
    '''Calculation of the Jacobian Matrix (A)'''
    
    M = len(h)
    N = len(omega)
    A = np.ones((N,M), dtype = 'complex')
    
    for k in range(N):
        freq = omega[k]
        gamma, w, R, e, Z = Auxiliars(freq, p0,h)
 
        A[k] = Jacobianline(freq, p0, h, gamma, w, R, e, Z)
    return A

def calc_hessiana_gradiente(metodo,jacobiana,p,residuo,mu,parametro_VT):
    
    m = len(p)
    diagonais=[np.ones(m-1),np.ones(m-1)*-1]
    R = diags(diagonais, [0, 1],shape=(m-1,m)).toarray()  
    
    if metodo == 'suavidade':
        
        RTR = np.dot(R.T,R)
        hessiana = np.dot(jacobiana.T,jacobiana)+(mu*RTR)
        gradiente = np.dot(jacobiana.T,residuo) - mu*np.dot(RTR,(p))
    
    if metodo == 'variacao_total':
        
        # Vetor q da regularização Variação Total
        alfa = parametro_VT
        v = np.dot(R,p)
        aux = np.sqrt(v**2+alfa)
        q = v/aux
                
        #Matriz Q da regularização Variação Total
        Q = np.identity(m-1)*alfa/((np.sqrt(v**2)+alfa)**3)
        
        RTQR = np.dot(R.T,np.dot(Q,R))
        hessiana = np.dot(jacobiana.T,jacobiana)+mu*RTQR
        gradiente = np.dot(jacobiana.T,residuo) - mu*np.dot(R.T,q)
            
    if metodo == 'occam':
        
        # Matriz D da regularizacao
        d1 = np.ones(m)
        d1[0] = 0.
        d2 = np.ones(m-1)*-1
        diagonais = [d1,d2]
        D = diags(diagonais, [0, 1],shape=(m,m)).toarray()
        
        hessiana = mu*np.dot(D.T,D) + np.dot(jacobiana.T,jacobiana) 
        gradiente = np.dot(jacobiana.T,(residuo + np.dot(jacobiana,p)))
        
    return hessiana, gradiente

def phi_val(metodo,residuo,p,mu,parametro_VT):

    m = len(p)
    diagonais=[np.ones(m-1),np.ones(m-1)*-1]
    R = diags(diagonais, [0, 1],shape=(m-1,m)).toarray()
    
    if metodo == 'suavidade':
        rr = np.linalg.norm(residuo)**2 + mu*np.linalg.norm(np.dot(R,(p)))**2

    if metodo == 'variacao_total':
        alfa = parametro_VT
        rr = np.linalg.norm(residuo)**2 + mu*np.linalg.norm(np.sqrt(np.dot(R,p)**2+alfa))**2
    
    if metodo == 'occam':
        # Matriz D da regularizacao
        d1 = np.ones(m)
        d1[0] = 0.
        d2 = np.ones(m-1)*-1
        diagonais = [d1,d2]
        D = diags(diagonais, [0, -1],shape=(m,m)).toarray()
        rr = np.linalg.norm(np.dot(D,p))**2 + (1./mu)*np.linalg.norm(r)**2
        
    return rr

def convergencia(phi0,phi,beta):
    variacao_rel = np.abs((phi0-phi)/phi0)
    #print( variacao_rel)
    return variacao_rel < beta
    
def calc_hessiana_gradiente_vinculo(jacobiana,p,pref,residuo,mu,v = 0.0001):
    
    m = len(p)
    diagonais=[np.ones(m-1),np.ones(m-1)*-1]
    R = diags(diagonais, [0, 1],shape=(m-1,m)).toarray()  
    
    
    RTR = np.dot(R.T,R)
    hessiana = np.dot(jacobiana.T,jacobiana)+(mu*RTR)+v*np.identity(len(p))
    gradiente = np.dot(jacobiana.T,residuo) - mu*np.dot(RTR,(p)) - v*(p-pref)
    
        
    return hessiana, gradiente

def phi_val_vinculo(residuo,p,pref,mu,v = 0.0001):

    m = len(p)
    diagonais=[np.ones(m-1),np.ones(m-1)*-1]
    R = diags(diagonais, [0, 1],shape=(m-1,m)).toarray()
    
    rr = np.linalg.norm(residuo)**2 + mu*np.linalg.norm(np.dot(R,(p)))**2 + v*np.linalg.norm(p-pref)**2

    
    return rr


def marquardt(dado_observado,modelo_inicial,h,omega,mu,metodo,step=10.,Lambda=0.00001,              parametro_VT=0.0001,n_iteracoes=150):
    
    
    #Dados reais
    dobs = np.hstack(((dado_observado.real).copy(),(dado_observado.imag).copy()))

    #chute inicial
    p0 = (modelo_inicial.copy())
    Z0 = Impedance_Tensor(omega, modelo_inicial,h)
    d0 = np.hstack((Z0.real,Z0.imag))
    residuo0 = np.log10(dobs) - np.log10(d0)
    #residuo0 = dobs - d0
    
    parametros =[p0]

    phi0 = phi_val(metodo,residuo0,(p0),mu,parametro_VT)
    phis = [phi0]

    I =np.identity(len(h))
    it_max = n_iteracoes
    beta = 1e-5
    
    for i in range(it_max):
    
        #Jacobiana no ponto i-1
        #jac = Jacobian_numerica(omega,(p0),(h))
        jac = Jacobian_analitica(omega,(p0),(h))
        A = np.vstack((jac.real,jac.imag))

        #Hessiana e gradiente de acordo com o metodo de regularizaçao
               
        hessiana, gradiente = calc_hessiana_gradiente(metodo,A,(p0),residuo0,mu,parametro_VT)
            
        for k in range(10):
            
            #Parametros estimados 
            deltap = np.linalg.solve((hessiana+Lambda*I),gradiente)
            p = p0+deltap
            
            for d in range(len(p)):
                if p[d] < 0:
                    p[d]=p[d]*(-1)

            #Valor da função no ponto i
            Z1 = Impedance_Tensor(omega, p,h)
            d1 = np.hstack((Z1.real,Z1.imag))
               
            #Residuo
            residuo1 = np.log10(dobs) - np.log10(d1)
            #residuo1 = dobs - d1

            phi = phi_val(metodo,residuo1,(p),mu,parametro_VT)

    #       dphi = phi - phi0

            if (phi > phi0):
                Lambda *= step
            else:
                Lambda /= step
                break

            
        if convergencia(phi0,phi,beta):
            break
        else:
            phi0 = phi
            phis.append(phi0)
            p0 = p.copy()
            d0 = d1.copy()
            parametros.append(p0)

    return p,deltap,phis,i,Z1,parametros,A,jac,Lambda
    
    
def marquardt_vinculo(dado_observado,modelo_inicial,modelo_referencia, h,omega,mu,v,step=10.,Lambda=0.00001,n_iteracoes=150):
    
    
    #Dados reais
    dobs = np.hstack(((dado_observado.real).copy(),(dado_observado.imag).copy()))

    #modelo de referÊncia
    pref = modelo_referencia.copy()
    
    #chute inicial
    p0 = (modelo_inicial.copy())
    Z0 = Impedance_Tensor(omega, modelo_inicial,h)
    d0 = np.hstack((Z0.real,Z0.imag))
    residuo0 = np.log10(dobs) - np.log10(d0)
    #residuo0 = dobs - d0
    
    parametros =[p0]

    phi0 = phi_val_vinculo(residuo0,p0,pref,mu,v)
    phis = [phi0]

    I =np.identity(len(h))
    it_max = n_iteracoes
    beta = 1e-6
    
    for i in range(it_max):
    
        #Jacobiana no ponto i-1
        #jac = Jacobian_numerica(omega,(p0),(h))
        jac = Jacobian_analitica(omega,(p0),(h))
        A = np.vstack((jac.real,jac.imag))

        #Hessiana e gradiente de acordo com o metodo de regularizaçao
               
        hessiana, gradiente = calc_hessiana_gradiente_vinculo(A,p0,pref,residuo0,mu,v)
            
        for k in range(10):
            
            #Parametros estimados 
            deltap = np.linalg.solve((hessiana+Lambda*I),gradiente)
            p = p0+deltap
            
            for d in range(len(p)):
                if p[d] < 0:
                    p[d]=p[d]*(-1)

            #Valor da função no ponto i
            Z1 = Impedance_Tensor(omega, p,h)
            d1 = np.hstack((Z1.real,Z1.imag))
               
            #Residuo
            residuo1 = np.log10(dobs) - np.log10(d1)
            #residuo1 = dobs - d1

            phi = phi_val_vinculo(residuo1,p,pref,mu,v)

    #       dphi = phi - phi0

            if (phi > phi0):
                Lambda *= step
            else:
                Lambda /= step
                break

            
        if convergencia(phi0,phi,beta):
            break
        else:
            phi0 = phi
            phis.append(phi0)
            p0 = p.copy()
            d0 = d1.copy()
            parametros.append(p0)

    return p,deltap,phis,i,Z1,parametros,A,jac,Lambda


def rhoap_marquardt_vinculo(dado_observado,modelo_inicial,modelo_referencia, h,omega,mu,v,step=10.,Lambda=0.00001,n_iteracoes=150):
    
    
    #Dados reais
    dobs = np.hstack(((dado_observado[:,0]).copy(),(dado_observado[:,1]).copy()))

    #modelo de referÊncia
    pref = modelo_referencia.copy()
    
    #chute inicial
    p0 = (modelo_inicial.copy())
    Z0 = Impedance_Tensor(omega, modelo_inicial,h)
    
    rhoap0 = (((omega*mu0)**-1.0)*(abs(Z0)**2))
    phase0 = np.arctan(Z0.imag/Z0.real)
    phi0 = (phase0*180)/np.pi
    
    d0 = np.hstack((rhoap0,phi0))
    residuo0 = np.log10(dobs) - np.log10(d0)
    #residuo0 = dobs - d0
    
    parametros =[p0]

    phi0 = phi_val_vinculo(residuo0,p0,pref,mu,v)
    phis = [phi0]

    I =np.identity(len(h))
    it_max = n_iteracoes
    beta = 1e-6
    
    for i in range(it_max):
    
        #Jacobiana no ponto i-1
        #jac = Jacobian_numerica(omega,(p0),(h))
        jac = Jacobian_analitica(omega,(p0),(h))
        A = np.vstack((jac.real,jac.imag))

        #Hessiana e gradiente de acordo com o metodo de regularizaçao
               
        hessiana, gradiente = calc_hessiana_gradiente_vinculo(A,p0,pref,residuo0,mu,v)
            
        for k in range(10):
            
            #Parametros estimados 
            deltap = np.linalg.solve((hessiana+Lambda*I),gradiente)
            p = p0+deltap
            
            for d in range(len(p)):
                if p[d] < 0:
                    p[d]=p[d]*(-1)

            #Valor da função no ponto i
            Z1 = Impedance_Tensor(omega, p,h)
            rhoap1 = (((omega*mu0)**-1.0)*(abs(Z1)**2))
            phase1 = np.arctan(Z1.imag/Z1.real)
            phi1 = (phase1*180)/np.pi
    
            d1 = np.hstack((rhoap1,phi1))
                           
            #Residuo
            residuo1 = np.log10(dobs) - np.log10(d1)
            #residuo1 = dobs - d1

            phi = phi_val_vinculo(residuo1,p,pref,mu,v)

    #       dphi = phi - phi0

            if (phi > phi0):
                Lambda *= step
            else:
                Lambda /= step
                break

            
        if convergencia(phi0,phi,beta):
            break
        else:
            phi0 = phi
            phis.append(phi0)
            p0 = p.copy()
            d0 = d1.copy()
            parametros.append(p0)

    return p,deltap,phis,i,Z1,parametros,A,jac,Lambda