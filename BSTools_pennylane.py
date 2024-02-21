import numpy as np
import pennylane as qml
from scipy import stats
from sympy.combinatorics import GrayCode
import scipy.integrate as integrate
from python_tsp.heuristics import solve_tsp_simulated_annealing



class BS():
    
    def __init__(self, r, K, Smax):
        self.r = r
        self.K = K
        self.Smax = Smax
        self.L = np.log(Smax)
        self.xmax = 2*self.L


    def solve(self, solver, n, sigma, T ,**kwargs):
        self.dev = qml.device("default.qubit", wires=n+2)
        self.n = n
        self.N = 2**n

        self.x = np.linspace(-self.L,self.L,self.N//2)
        self.dx = 2*self.L/(self.N//2-1)

        delta = 2*self.xmax/(self.N-1)   # To check, xmax/(N//2-1) instead ?
        k = np.arange(0,self.N,1)

        self.pk = np.sin(2*np.pi/self.N*k)/delta
        # a = 8
        # b = -1
        # self.pk = (a*np.sin(2*np.pi/self.N*k)+b*np.sin(4*np.pi/self.N*k))/(delta*(a+2*b))
        
        self.sigma = sigma

        if solver=="CN":
            if callable(sigma):
                return self.CrankNicolsonTimeDep(arg[0],sigma,T)
            else:
                return self.CrankNicolson(arg[0],sigma,T)
        elif solver=="QC":
            return self.QC(T, **kwargs)
            # if len(arg)==0:
            #     return self.QC(T)
            # else:
            #     return self.QC(T, arg[0])
        elif solver=="exactp":
            return self.exactp(T, sigma, **kwargs)
        elif solver=="exact":
            return self.exact(T, sigma)[0,:]




    ##### SOLVERS #####
    def exactp(self,T,sigma,NFourier=1000):
        L = self.L
        K = self.K
        r = self.r
        n = self.n
        N = 2**n
        
        coef_ = [K/L*( 2*L/(n*np.pi)*np.sin(n*np.pi/2*(L+np.log(K))/L) - 1/(1+(n*np.pi/(2*L))**2)*(\
        np.cos(n*np.pi/2*(L+np.log(K))/L) + n*np.pi/(2*L)*np.sin(n*np.pi/2*(L+np.log(K))/L) - np.exp(-L)/K) ) for n in range(1,NFourier)]
        
        xx = np.linspace(-2*L,2*L,N)
        S = K/L*(L+np.log(K)-1+np.exp(-L)/K)/2
        
        for nn in range(1,NFourier):
            S += coef_[nn-1]*np.cos(nn*np.pi*(xx+(r-1/2*sigma**2)*T)/(2*L))*np.exp(-sigma**2/2*(np.pi*nn/(2*L))**2*T)
        Fourier = S*np.exp(-r*T)
        return Fourier
    
    def exact(self,t,sigma):
        S = np.exp(np.linspace(-self.L,self.L,self.N//2))
        
        t_ = np.outer(t,np.ones(np.shape(S)))
        S_ = np.outer(np.ones(np.shape(t)), S)
        d1 = (np.log(S_/self.K) + (self.r+(sigma**2)/2)*t_)/(sigma*np.sqrt(t_))
        d2 = (np.log(S_/self.K) + (self.r-(sigma**2)/2)*t_)/(sigma*np.sqrt(t_))
        ST = -S_*stats.norm.cdf(-d1) + self.K*np.exp(-self.r*(t_))*stats.norm.cdf(-d2)
        return ST

    def CrankNicolson(self, Nt,sigma,T):
        Nx = self.N//2
        r = self.r
        K = self.K
        
        x = self.x
        dx = self.dx
        
        t = np.linspace(0,T,Nt)
        dt = t[1]-t[0]

        initx = np.maximum(K-np.exp(x), np.zeros(Nx))
        initx = np.concatenate((np.flip(initx),initx))

        N_ = 2*Nx
        V = initx

        dt = -dt #!
        a = r + sigma**2/dx**2
        b = sigma**2/(2*dx**2)
        c = (r-sigma**2/2)/(2*dx)

        A = (1 - dt/2*a)*np.eye(N_) - dt/2*(-b+c)*np.diag(np.ones(N_-1),-1) - dt/2*(-b-c)*np.diag(np.ones(N_-1),1)
        A[0,-1] = -dt/2*(-b+c)
        A[-1,0] = -dt/2*(-b-c)

        B = (1 + dt/2*a)*np.eye(N_) + dt/2*(-b+c)*np.diag(np.ones(N_-1),-1) + dt/2*(-b-c)*np.diag(np.ones(N_-1),1)
        B[0,-1] = dt/2*(-b+c)
        B[-1,0] = dt/2*(-b-c)

        for i in range(Nt-1):
            # V[:,i+1] = np.linalg.solve(A,np.matmul(B,V[:,i]))
            V = np.linalg.solve(A,np.matmul(B,V))

        return V

    def CrankNicolsonTimeDep(self, Nt,sigma,T):

        Nx = self.N//2
        r = self.r
        K = self.K
        
        x = self.x
        dx = self.dx
        
        t = np.linspace(0,T,Nt)
        dt = t[1]-t[0]

        initx = np.maximum(K-np.exp(x), np.zeros(Nx))
        initx = np.concatenate((np.flip(initx),initx))

        N_ = 2*Nx

        V = initx

        dt = -dt #!

        for i in range(Nt-1):
            a = r + sigma(t[i+1])**2/dx**2
            b = sigma(t[i+1])**2/(2*dx**2)
            c = (r-sigma(t[i+1])**2/2)/(2*dx)

            A = (1 - dt/2*a)*np.eye(N_) - dt/2*(-b+c)*np.diag(np.ones(N_-1),-1) - dt/2*(-b-c)*np.diag(np.ones(N_-1),1)
            A[0,-1] = -dt/2*(-b+c)
            A[-1,0] = -dt/2*(-b-c)

            a = r + sigma(t[i])**2/dx**2
            b = sigma(t[i])**2/(2*dx**2)
            c = (r-sigma(t[i])**2/2)/(2*dx)

            B = (1 + dt/2*a)*np.eye(N_) + dt/2*(-b+c)*np.diag(np.ones(N_-1),-1) + dt/2*(-b-c)*np.diag(np.ones(N_-1),1)
            B[0,-1] = dt/2*(-b+c)
            B[-1,0] = dt/2*(-b-c)

            V = np.linalg.solve(A,np.matmul(B,V))
        return V


    def QC(self, T, M=None, k=None, order=True):
        L = self.L
        n = self.n
        N = 2**n
        
        sim = self.Circuit_node(self.get_coef_H(T),self.get_coef_A(T), k, M, order)
        return sim





    ##### UTILITY #####

    def get_coef_H(self, T):
        n = self.n
        N = 2**n
        coef_arr = []

        for k in range(1,N,2):
            k_bin = np.array(list(np.binary_repr(k,n)), dtype=int)
            if k==1:
                coef = (1/np.tan(np.pi/N))

            elif (np.sum(k_bin)-1)%2==1:
                prod = 1
                for j in range(1,len(k_bin)):
                    if k_bin[-j-1]:
                        prod *= np.tan(np.pi/2**(j+1))
                coef = ((-1)**(int((np.sum(k_bin))/2))*prod)

            else:
                prod = 1
                for j in range(1,len(k_bin)):
                    if k_bin[-j-1]:
                        prod *= np.tan(np.pi/2**(j+1))
                coef = ((-1)**(int((np.sum(k_bin)-1)/2))*prod/np.tan(np.pi/N))

            coef_arr.append(-(N-1)/(2*N)*(self.sigma**2-2*self.r)/self.xmax*coef*T)
        return coef_arr
    
    def get_coef_H_(self,T):
        n = self.n
        N = 2**n
        coef_arr_H = []
        for I in range(1,N,2):
            S = 0
            for k in range(N):
                S += -self.W(k,I,n)*(self.sigma**2/2-self.r)*self.pk[k]
            coef_arr_H.append( S/N*T )
        return coef_arr_H


    def get_coef_A(self, T):
        n = self.n
        N = 2**n
        
        coef_arr_A = []
        for I in range(0,N,2):
            S = 0
            for k in range(N):
                S += self.W(k,I,n)*np.arccos(np.exp(-T*(self.sigma**2/2*self.pk[k]**2 + self.r)))
            coef_arr_A.append( S/N )
        return coef_arr_A
    
    
    

    def Circuit_node(self, coef_H, coef_A, k, M, order):
        N = self.N
        x = np.linspace(-self.L,self.L,N//2)
        initx = np.maximum(self.K-np.exp(x),np.zeros((N//2)))
        init_sym = np.concatenate((np.flip(initx),initx))
        Lambda = np.sqrt(np.sum(np.abs(init_sym)**2))
        
        init = init_sym/Lambda
        init_state = np.zeros((4*N))
        init_state[:N] = init
        
        if k==None:
            k = self.n
        if M==None:
            M = [N//2,N//2]
        
        @qml.qnode(self.dev)
        def Circuit_prob(self, coef_H, coef_A, k, m_H, m_A, order):
            return self.Circuit(coef_H, coef_A, k, m_H, m_A, order)
        
        return Lambda*np.sqrt(Circuit_prob(self, coef_H, coef_A, k, M[0], M[1], order))
    
    
    def Circuit(self, coef_H, coef_A, k, m_H, m_A, order):
        n = self.n
        N = 2**n

        self.wire = list(np.flip(range(n)))
        self.StatePreparation(self.n,k)

        qml.QFT(list(np.flip(np.arange(n))))

        qml.PauliZ(n)
        qml.adjoint(qml.S)(n)
        qml.Hadamard(n)

        if order:
            ind_H = np.arange(1,N,2)
            I = np.flip(np.argsort(np.abs(coef_H)))
            ind_H = ind_H[I[:m_H]]
            coef_H = list(np.array(coef_H)[I[:m_H]])

            ind_A = np.arange(0,N,2)
            I = np.flip(np.argsort(np.abs(coef_A)))
            ind_A = ind_A[I[:m_A]]
            coef_A = list(np.array(coef_A)[I[:m_A]])

            coef = np.concatenate((coef_H, coef_A))
            ind = np.concatenate((ind_H, ind_A))
            permutation = self.order_opt(ind_H, ind_A)

            ind = ind[permutation]
            coef = coef[permutation]

            i = 0
            for k in ind:
                k_bin = np.array(list(np.binary_repr(k,n)), dtype=int)
                if k%2==0:
                    self.Cartan(k_bin,coef[i], antihermitian=True)
                else:
                    self.Cartan(k_bin,-coef[i], antihermitian=False)
                i += 1
            # k_bin_list = [np.array(list(np.binary_repr(k,n)), dtype=int) for k in ind]
            # self.Cartan_list(k_bin_list, coef)

        else:
            ind_H = np.arange(1,N,2)
            I = np.flip(np.argsort(np.abs(coef_H)))

            for i in range(m_H):
                k = ind_H[I[i]]
                k_bin = np.array(list(np.binary_repr(k,n)), dtype=int)
                self.Cartan(k_bin,-coef_H[I[i]])

            ind_A = np.arange(0,N,2)
            I = np.flip(np.argsort(np.abs(coef_A)))

            for i in range(m_A):
                k = ind_A[I[i]]
                k_bin = np.array(list(np.binary_repr(k,n)), dtype=int)
                self.Cartan(k_bin,coef_A[I[i]], antihermitian=True)



        qml.Hadamard(n)
        qml.S(n)

        qml.adjoint(qml.QFT)(list(np.flip(np.arange(n))))
        
        return qml.probs(list(np.flip(np.arange(n+2))))
    

    
    def Cartan(self, bit_str, beta, antihermitian=False):
        n = self.n
        M = len(bit_str)

        for i in range(M):
            if bit_str[i]:
                qml.CNOT([i,n+1])
        if antihermitian:
            qml.CNOT([n,n+1])
        qml.RZ(-2*beta,n+1)
        if antihermitian:
            qml.CNOT([n,n+1])
        for i in range(M):
            if bit_str[M-i-1]:
                qml.CNOT([(M-i-1),n+1])
                
                
    def order_opt(self, ind_H, ind_A):
        ind = np.concatenate((ind_H,ind_A+self.N))
        M = len(ind)
        distance_matrix = np.zeros((M,M))
        
        for i in range(M):
            for j in range(M):
                distance_matrix[i,j] = np.sum(np.abs(np.array(list(np.binary_repr(ind[i],self.n+1)),int) - np.array(list(np.binary_repr(ind[j],self.n+1)),int)))

        distance_matrix[:, 0] = 0

        permutation, distance = solve_tsp_simulated_annealing(distance_matrix)
        return permutation
    

    def W(self, x,k,n):
        x_bin = np.binary_repr(x,n)
        k_bin = np.binary_repr(k,n)

        S = 0
        for i in range(n):
            S += int(x_bin[i])*int(k_bin[n-i-1])
        return (-1)**S
    
    
    
    def StatePreparation(self, n, k0, fun=None):
        self.n = n
        if fun==None:
            fun = lambda x: np.maximum(0, self.K - np.exp(np.abs(x)-self.L))
        self.circuit_FirstAlgorithm(lambda x: fun(x)**2, k0, n)
        
        
        
        
        
        
        
    ##### ----- #####
    
    def func_thetas(self, myfunction2, k, use_inv_matrix=True):
        delta_k = 2*self.xmax/(2**(k-1))
        x_min = -self.xmax
        
        n = self.n
        x = np.linspace(-self.xmax, self.xmax, 2**n)

        thetas = []
        for l in range(2**(k-1)):

            # num = integrate.quad(myfunction2, x_min + l*delta_k, x_min + (l+1/2)*delta_k)[0]
            # denom = integrate.quad(myfunction2, x_min + l*delta_k, x_min + (l+1)*delta_k)[0]
            num =   np.sum( myfunction2(x[l*2**(n-(k-1)):l*2**(n-(k-1))+2**(n-k)]) )
            denom = np.sum( myfunction2(x[l*2**(n-(k-1)):(l+1)*2**(n-(k-1))]) )
            
#             print(num_/denom_, num/denom)
            
            if not (abs(num)<1e-7 and abs(denom)<1e-7):
                ratio = np.divide(num, denom, out=np.zeros_like(num), where=denom!=0)
                if ratio>1:
                    ratio=1
                thetas.append(2*np.arccos(np.sqrt(ratio)))
            else:
                # print('else')
                if use_inv_matrix:
                    thetas.append(0)
                    
            
        if(k==1 or not use_inv_matrix):
            return thetas
        else:
            Minv = self.calculate_inverse_matrix(k-1)
            # print(len(thetas))
            # print(np.shape(Minv))
        return Minv@np.array(thetas)
    
    
    def circuit_Mottonen(self, myfunction2, n_qubits):
        circuit = qk.QuantumCircuit(self.n)
        for k in range(1,n_qubits+1):
            theta = self.func_thetas(myfunction2,k,use_inv_matrix=True)
            self.F_Mottonen(circuit, list(range(k-1)),k-1,theta)
        return circuit
    
    
    def circuit_FirstAlgorithm(self, myfunction2, k0, n_qubits):
        
        for k in range(1,k0+1):
            theta = self.func_thetas(myfunction2,k,use_inv_matrix=True)
            self.F_Mottonen(list(range(k-1)),k-1,theta)

        for k in range(k0,n_qubits):
            theta = self.func_thetas(myfunction2,k+1,use_inv_matrix=False)
            theta_representative = np.mean(theta)

            # delta_k = (x_max - x_min)/(2**(k-1))
            # eta_k = delta_k / 8 * eta
            # if(verbose>=1):
            #     if (np.all(np.abs(theta-theta_representative)< eta_k)):
            #         print("representative ", k," is legit")
            #     else:
            #         print("Warning: representative ", k," is not legit")

            qml.RY(theta_representative,self.wire[k])
        # return circuit

    
    
    def generate_gray_code(self, n):
        a = GrayCode(n)
        l = list(a.generate_gray())+['0'*n]
        
        cnot_positions = []
        for i in range(len(l[:-1])):
            y = int(l[i],2) ^ int(l[i+1],2)
            z =  str('{0:0{1}b}'.format(y,len(l[i])))
            
            cnot_positions.append(z.find('1'))
        return cnot_positions
    
    
    def calculate_inverse_matrix(self, k):
        M = np.zeros((2**k,2**k))
        a = GrayCode(k)
        l = list(a.generate_gray())
        g_list = []
        for el in l:
            b = np.array(list(el),dtype=int)
            g_list.append(b.dot(2**np.arange(b.size)[::-1]))


        for i in range(2**k):
            for j,g_el in enumerate(g_list):
                M[i,j] = (-1)**int(np.sum(np.array(list(np.binary_repr(np.bitwise_and(i,g_el))),dtype=int)))

        M_inv = 2**(-k)*M.transpose()
        return M_inv
    
    
    def F_Mottonen(self, k, n, theta):
        """_summary_

        Args:
            k (list): list of qubits numbers to control
            n (int): qubit number to act rotations on
            theta (list): parameters for the 2**k rotations

        Returns:
            _type_: _description_
        """
        if(k==[] or len(theta)==1):
            qml.RY(theta[0], self.wire[n])

        elif(2**len(k) != len(theta)):
            print("Warning k is not compatible with theta")

        else:
            cnot_positions = [k[el] for el in self.generate_gray_code(len(k))]
            for theta_i,cnot_i in list(zip(theta,cnot_positions)):
                if abs(theta_i)>1e-7:
                    qml.RY(theta_i , self.wire[n])
                qml.CNOT([self.wire[cnot_i], self.wire[n]])
        