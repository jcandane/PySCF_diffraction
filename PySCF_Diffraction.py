import numpy as np 
from scipy.special import hyp1f1

π = np.pi 

def kadsum(Matrix, l):
    ## we then need to contract n_ab -> n_αβ
    
    l   = l.astype(np.int32)
    out = np.zeros((len(l)-1, len(l)-1))
    for i in range(len(l)-1):
        for j in range(len(l)-1):
            out[i][j] = np.sum( Matrix[ l[i]:l[i+1], l[j]:l[j+1] ] )

    return out

def molcoefficients(mol, exact=True):
    
    l    = np.array([])
    ζ    = np.array([])
    R_αx = np.zeros( (mol.nao, 3) )
    d_a  = []
    AOi  = np.zeros(1)

    aos = 0 ## to count AO basis-functions
    for i in range(mol.nbas): ## for-loop over shells
        d_aA    = mol.bas_ctr_coeff(i)
        l_shell = int( mol.bas_angular(i) )
        ζ_shell = mol.bas_exp(i)
        R_shell = 0.529177211 * mol.bas_coord(i) ## to get in Å
        for Z in range(d_aA.shape[1]): ## for-loop over Zeta-Basis
            for α in range(2 * l_shell + 1): ## for-loop over AO basis
                R_αx[aos] = R_shell
                d_a  = np.append( d_a , d_aA[:,Z] )
                ζ    = np.append( ζ , ζ_shell )
                aos += 1
                l    = np.append( l , l_shell * np.ones(len(ζ_shell)) )
                AOi  = np.append( AOi , AOi[-1] + len(ζ_shell) )
    
    ζ_ab  =  np.einsum("A, B -> AB", ζ, np.ones(len(ζ))) + np.einsum("B, A -> AB", ζ, np.ones(len(ζ)))
    l_ab  = (np.einsum("A, B -> AB", l, np.ones(len(l))) + np.einsum("B, A -> AB", l, np.ones(len(l))) + 3)/2
    R_xαβ =  np.einsum("Ax, B -> xAB", R_αx, np.ones(R_αx.shape[0])) - np.einsum("Bx, A -> xAB", R_αx, np.ones(R_αx.shape[0]))

    if exact:
        return R_xαβ, ζ_ab, l_ab, AOi
    else:
        A = kadsum( 2 * l_ab / ( 12 * ζ_ab ) , AOi) 
        B = kadsum( 2 * l_ab * (l_ab + 1) / (60 * ζ_ab**2), AOi)
        return R_xαβ, A, B, AOi

def getf_q(A, B, AOi, q, exact=True):
    q2 = np.dot(q,q)
    if exact:
        return kadsum(  hyp1f1(B, 1.5, - q2/(4*A) ) , AOi)
    else:
        return 1 + A * q2 + B * q2 * q2

def get_diffpattern(mol, detector, D_sαβ):
    """
    Given: mol (pyscf object), detector (custom object), D_sαβ (AO density-matrix)
    Get  : I_q (Diffraction Intensity)
    """
    DS = (D_sαβ[0] + D_sαβ[1]) @ (mol.intor("int1e_ovlp"))

    R, A, B, AOi = molcoefficients(mol, exact=True) ### get molecular information
    q_Dx = detector.q
    I_q  = np.zeros(len(q_Dx))
    for i in range(len(q_Dx)):
        I_q[i] = np.sum( DS @ getf_q(A, B, AOi, q_Dx[i], exact=True) @ np.cos(q_Dx[i,0]*R[0] + q_Dx[i,1]*R[1] + q_Dx[i,2]*R[2]) )

    return I_q

def diffpatternX(mol, D, detector, dQ=0.01, exact=True):

    R, A, B, AOi = molcoefficients(mol, exact=exact)
    S    = mol.intor("int1e_ovlp")

    Q  = np.arange(0., int( np.linalg.norm(detector.q[0]) + 1 ), dQ) ##np.max( np.linalg.norm( detector2.q , axis=1) )
    f_Q = np.zeros((len(Q), len(S), len(S)))
    for i in range(len(Q)):
        f_Q[i] = getf_q(A, B, AOi, np.array([Q[i],0.,0.]), exact=True)

    ## given q (magnitude) find 2d array.
    q  = np.linalg.norm(detector.q, axis=1)
    q *= 1/dQ
    N, dx = q.astype(int), q-(q).astype(int)
    
    return np.einsum("sAB, BC, qCD, qDE -> q", D, S, f_Q[N] + (f_Q[N+1] - f_Q[N] ) * dx[:,None,None], np.cos(np.einsum("qx, xAB -> qAB", detector.q, R)))


class diffraction_detector(object):
    ''' A class of diffraction detector instrament '''
    def __init__(self, λ=1.0, L=1e8, pixelsize=1e5, N_H=100, N_V=1, q_center=np.zeros(3), k_in=np.array([0.,0.,1.])):
        """This class assumes a rectangular detector array with square pixels"""
        self.λ         = λ         ## in-wavelength (in Å)
        self.L         = L         ## detector distance from slit (in Å)
        self.pixelsize = pixelsize ## detector physical size (in Å)
        self.N_H       = N_H ## detector pixel size 
        self.N_V       = N_V ## detector pixel size
        self.k_in      = 2*np.pi/self.λ * k_in
        self.center    = q_center

        detectorH = np.linspace(0, self.pixelsize*self.N_H, self.N_H, endpoint=False) - self.pixelsize*(self.N_H - 1)/2 + (self.center + self.k_in)[0]
        detectorV = np.linspace(0, self.pixelsize*self.N_V, self.N_V, endpoint=False) - self.pixelsize*(self.N_V - 1)/2 + (self.center + self.k_in)[1]
        H, V      = np.meshgrid(detectorH, detectorV)
        detector2 = np.asarray([H, V, self.L*np.ones(H.shape)]).T
        k_Dx      = detector2.reshape((detector2[:,:,0].size,3), order="F")
        k_out     = np.einsum("Dx, D -> Dx", k_Dx, np.linalg.norm(self.k_in)/np.linalg.norm(k_Dx, axis=1) ) ## normalize k_out to match k_in
        self.q    = k_out - self.k_in

    def qq(self):
        return np.reshape(self.q, (self.N_H, self.N_V,3), order="F")

