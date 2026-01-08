import numpy as np
from functools import wraps


np.seterr(all='raise', under="warn")

of = 10
g = 1



def D_mat(nx):
    D = np.zeros((nx+1,nx+1))
    for i in range(nx+1):
        D[i,(i+1)%(nx+1)] = 1
        D[i,(i-1)%(nx+1)] = -1
    return D

def B_mat(nx):
    B = np.zeros((nx+1,nx+1))
    for i in range(nx+1):
        B[i,(i+1)%(nx+1)] = 1
        B[i,i] = -1
    return B




def scheme_wrapper(f):
    def decorator(scheme_func):
        @wraps(scheme_func)
        def wrapped(uOld,hOld,c,nx):
            error = False
            try: uNew, hNew = scheme_func(uOld,hOld,c,nx)
            except FloatingPointError as e:
                print(e)
                error = True
                
            if max(uNew.max(),uNew.min(),key=abs)>of or max(hNew.max(),hNew.min(),key=abs)>of:
                error = True
                
            return uNew, hNew, error
        return wrapped
    return decorator


@scheme_wrapper("ftcs_u")
def ftcs_u(uOld,hOld,c,nx): # u first
    c = c/2.
    D = D_mat(nx)
    
    uNew = uOld - c * (g * D @ hOld + np.diagflat(uOld) @ D @ uOld)
    hNew = hOld - c * (np.diagflat(uNew) @ D @ hOld + np.diagflat(hOld) @ D @ uNew)
    
    return uNew,hNew


@scheme_wrapper("ftcs_h")
def ftcs_h(uOld,hOld,c,nx): # h first
    c = c/2.
    D = D_mat(nx)
    
    hNew = hOld - c * (np.diagflat(uOld) @ D @ hOld + np.diagflat(hOld) @ D @ uOld)
    uNew = uOld - c * (g * D @ hNew + np.diagflat(uOld) @ D @ uOld)
    
    return uNew,hNew


@scheme_wrapper("ftcs_sim")
def ftcs_sim(uOld,hOld,c,nx):
    c = c/2.
    D = D_mat(nx)
    
    hNew = hOld - c * (np.diagflat(uOld) @ D @ hOld + np.diagflat(hOld) @ D @ uOld)
    uNew = uOld - c * (g * D @ hOld + np.diagflat(uOld) @ D @ uOld)
    
    return uNew,hNew


@scheme_wrapper("ftbs_u")
def ftbs_u(uOld,hOld,c,nx): # u first
    B = B_mat(nx)
    
    uNew = uOld - c * (g * B @ hOld + np.diagflat(uOld) @ B @ uOld)
    hNew = hOld - c * (np.diagflat(uNew) @ B @ hOld + np.diagflat(hOld) @ B @ uNew)
    
    return uNew,hNew


@scheme_wrapper("ftbs_h")
def ftbs_h(uOld,hOld,c,nx): # h first
    B = B_mat(nx)
    
    hNew = hOld - c * (np.diagflat(uOld) @ B @ hOld + np.diagflat(hOld) @ B @ uOld)
    uNew = uOld - c * (g * B @ hNew + np.diagflat(uOld) @ B @ uOld)
    
    return uNew,hNew


@scheme_wrapper("ftbs_sim")
def ftbs_sim(uOld,hOld,c,nx):
    B = B_mat(nx)
    
    hNew = hOld - c * (np.diagflat(uOld) @ B @ hOld + np.diagflat(hOld) @ B @ uOld)
    uNew = uOld - c * (g * B @ hOld + np.diagflat(uOld) @ B @ uOld)
    
    return uNew,hNew


@scheme_wrapper("ftcs_semi_impl_u_1")
def ftcs_semi_impl_u_1(uOld,hOld,c,nx): # u implicit, u first
    c = c/2.
    uNew = uOld.copy()
    hNew = hOld.copy()
    
    for j in range(0,nx+1):
        uNew[j] = (uOld[j] - g*c*(hOld[(j+1)%(nx+1)]-hOld[(j-1)%(nx+1)])) / (1 + c*(uOld[(j+1)%(nx+1)]-uOld[j-1]))
    for j in range(0,nx+1):
        hNew[j] = hOld[j] - c*(uNew[j]*(hOld[(j+1)%(nx+1)]-hOld[j-1]) + hOld[j]*(uNew[(j+1)%(nx+1)]-uNew[j-1]))
    
    return uNew,hNew


@scheme_wrapper("ftcs_semi_impl_u_2")
def ftcs_semi_impl_u_2(uOld,hOld,c,nx): # h implicit, u first
    c = c/2.
    uNew = uOld.copy()
    hNew = hOld.copy()
    
    for j in range(0,nx+1):
        uNew[j] = uOld[j] - c*((uOld[j]*(uOld[(j+1)%(nx+1)]-uOld[j-1])) + g*(hOld[(j+1)%(nx+1)]-hOld[j-1]))
    for j in range(0,nx+1):
        hNew[j] = (hOld[j] - c*(uNew[j]*(hOld[(j+1)%(nx+1)]-hOld[j-1]))) / (1 + c*(uNew[(j+1)%(nx+1)]-uNew[j]))
        
    return uNew,hNew


@scheme_wrapper("ftcs_semi_impl_u_3")
def ftcs_semi_impl_u_3(uOld,hOld,c,nx): # u,h implicit, u first
    c = c/2.
    uNew = uOld.copy()
    hNew = hOld.copy()
    
    for j in range(0,nx+1):
        uNew[j] = (uOld[j] - g*c*(hOld[(j+1)%(nx+1)]-hOld[j])) / (1 + c*(uOld[(j+1)%(nx+1)]-uOld[j-1]))
    for j in range(0,nx+1):
        hNew[j] = (hOld[j] - c*(uNew[j]*(hOld[(j+1)%(nx+1)]-hOld[j-1]))) / (1 + c*(uNew[(j+1)%(nx+1)]-uNew[j]))
    
    return uNew,hNew


@scheme_wrapper("ftcs_semi_impl_h_1")
def ftcs_semi_impl_h_1(uOld,hOld,c,nx): # u implicit, h first
    c = c/2.
    uNew = uOld.copy()
    hNew = hOld.copy()
    
    for j in range(0,nx+1):
        hNew[j] = hOld[j] - c*(uOld[j]*(hOld[(j+1)%(nx+1)]-hOld[j-1]) + hOld[j]*(uOld[(j+1)%(nx+1)]-uOld[j-1]))
    for j in range(0,nx+1):
        uNew[j] = (uOld[j] - g*c*(hNew[(j+1)%(nx+1)]-hNew[j])) / (1 + c*(uOld[(j+1)%(nx+1)]-uOld[j-1]))
    
    return uNew,hNew


@scheme_wrapper("ftcs_semi_impl_h_2")
def ftcs_semi_impl_h_2(uOld,hOld,c,nx): # h implicit, h first
    c = c/2.
    uNew = uOld.copy()
    hNew = hOld.copy()
    
    for j in range(0,nx+1):
        hNew[j] = (hOld[j] - c*(uOld[j]*(hOld[(j+1)%(nx+1)]-hOld[j-1]))) / (1 + c*(uOld[(j+1)%(nx+1)]-uOld[j]))
    for j in range(0,nx+1):
        uNew[j] = uOld[j] - c*((uOld[j]*(uOld[(j+1)%(nx+1)]-uOld[j-1])) + g*(hNew[(j+1)%(nx+1)]-hNew[j-1]))
    
    return uNew,hNew


@scheme_wrapper("ftcs_semi_impl_h_3")
def ftcs_semi_impl_h_3(uOld,hOld,c,nx): # u,h implicit, h first
    c = c/2.
    uNew = uOld.copy()
    hNew = hOld.copy()
    
    for j in range(0,nx+1):
        hNew[j] = (hOld[j] - c*(uOld[j]*(hOld[(j+1)%(nx+1)]-hOld[j-1]))) / (1 + c*(uOld[(j+1)%(nx+1)]-uOld[j]))
    for j in range(0,nx+1):
        uNew[j] = (uOld[j] - g*c*(hNew[(j+1)%(nx+1)]-hNew[j])) / (1 + c*(uOld[(j+1)%(nx+1)]-uOld[j-1]))
        
    return uNew,hNew


@scheme_wrapper("ftcs_semi_impl_sim_1")
def ftcs_semi_impl_sim_1(uOld,hOld,c,nx): # u implicit
    c = c/2.
    uNew = uOld.copy()
    hNew = hOld.copy()
    
    for j in range(0,nx+1):
        hNew[j] = hOld[j] - c*(uOld[j]*(hOld[(j+1)%(nx+1)]-hOld[j-1]) + hOld[j]*(uOld[(j+1)%(nx+1)]-uOld[j-1]))
    for j in range(0,nx+1):
        uNew[j] = (uOld[j] - g*c*(hOld[(j+1)%(nx+1)]-hOld[j])) / (1 + c*(uOld[(j+1)%(nx+1)]-uOld[j-1]))
    
    return uNew,hNew


@scheme_wrapper("ftcs_semi_impl_sim_2")
def ftcs_semi_impl_sim_2(uOld,hOld,c,nx): # h implicit
    c = c/2.
    uNew = uOld.copy()
    hNew = hOld.copy()
    
    for j in range(0,nx+1):
        hNew[j] = (hOld[j] - c*(uOld[j]*(hOld[(j+1)%(nx+1)]-hOld[j-1]))) / (1 + c*(uOld[(j+1)%(nx+1)]-uOld[j]))
    for j in range(0,nx+1):
        uNew[j] = uOld[j] - c*((uOld[j]*(uOld[(j+1)%(nx+1)]-uOld[j-1])) + g*(hOld[(j+1)%(nx+1)]-hOld[j-1]))
    
    return uNew,hNew


@scheme_wrapper("ftcs_semi_impl_sim_3")
def ftcs_semi_impl_sim_3(uOld,hOld,c,nx): # u,h implicit
    c = c/2.
    uNew = uOld.copy()
    hNew = hOld.copy()
    
    for j in range(0,nx+1):
        hNew[j] = (hOld[j] - c*(uOld[j]*(hOld[(j+1)%(nx+1)]-hOld[j-1]))) / (1 + c*(uOld[(j+1)%(nx+1)]-uOld[j]))
    for j in range(0,nx+1):
        uNew[j] = (uOld[j] - g*c*(hOld[(j+1)%(nx+1)]-hOld[j])) / (1 + c*(uOld[(j+1)%(nx+1)]-uOld[j-1]))
        
    return uNew,hNew




@scheme_wrapper("ftcs_semi_impl_mat_u_1")
def ftcs_semi_impl_mat_u_1(uOld,hOld,c,nx): # u implicit, u first
    c = c/2.
    D = D_mat(nx)
    
    A = np.eye(nx+1) + c * np.diagflat(uOld) @ D
    uNew = A.I @ (uOld - c * g * D @ hOld)
    hNew = hOld - c * (np.diagflat(uNew) @ D @ hOld + np.diagflat(hOld) @ D @ uNew)
    
    return uNew,hNew


@scheme_wrapper("ftcs_semi_impl_mat_u_2")
def ftcs_semi_impl_mat_u_2(uOld,hOld,c,nx): # h implicit, u first
    c = c/2.
    D = D_mat(nx)
    
    uNew = uOld - c * (g * D @ hOld + np.diagflat(uOld) @ D @ uOld)
    B = np.eye(nx+1) + c * np.diagflat(uNew) @ D
    hNew = B.I @ (hOld - c * np.diagflat(hOld) @ D @ uNew)
            
    return uNew,hNew


@scheme_wrapper("ftcs_semi_impl_mat_u_3")
def ftcs_semi_impl_mat_u_3(uOld,hOld,c,nx): # u,h implicit, u first
    c = c/2.
    D = D_mat(nx)
    
    A = np.eye(nx+1) + c * np.diagflat(uOld) @ D
    uNew = A.I @ (uOld - c * g * D @ hOld)
    B = np.eye(nx+1) + c * np.diagflat(uNew) @ D
    hNew = B.I @ (hOld - c * np.diagflat(hOld) @ D @ uNew)
    
    return uNew,hNew


@scheme_wrapper("ftcs_semi_impl_mat_h_1")
def ftcs_semi_impl_mat_h_1(uOld,hOld,c,nx): # u implicit, h first
    c = c/2.
    D = D_mat(nx)
    
    hNew = hOld - c * (np.diagflat(uOld) @ D @ hOld + np.diagflat(hOld) @ D @ uOld)
    A = np.eye(nx+1) + c * np.diagflat(uOld) @ D
    uNew = A.I @ (uOld - c * g * D @ hNew)
    
    return uNew,hNew


@scheme_wrapper("ftcs_semi_impl_mat_h_2")
def ftcs_semi_impl_mat_h_2(uOld,hOld,c,nx): # h implicit, h first
    c = c/2.
    D = D_mat(nx)
    
    B = np.eye(nx+1) + c * np.diagflat(uOld) @ D
    hNew = B.I @ (hOld - c * np.diagflat(hOld) @ D @ uOld)
    uNew = uOld - c * (g * D @ hNew + np.diagflat(uOld) @ D @ uOld)
    
    return uNew,hNew


@scheme_wrapper("ftcs_semi_impl_mat_h_3")
def ftcs_semi_impl_mat_h_3(uOld,hOld,c,nx): # u,h implicit, h first
    c = c/2.
    D = D_mat(nx)
    
    B = np.eye(nx+1) + c * np.diagflat(uOld) @ D
    hNew = B.I @ (hOld - c * np.diagflat(hOld) @ D @ uOld)
    A = np.eye(nx+1) + c * np.diagflat(uOld) @ D
    uNew = A.I @ (uOld - c * g * D @ hNew)
    
    return uNew,hNew

@scheme_wrapper("ftcs_semi_impl_mat_sim_1")
def ftcs_semi_impl_mat_sim_1(uOld,hOld,c,nx): # u implicit
    c = c/2.
    D = D_mat(nx)
    
    hNew = hOld - c * (np.diagflat(uOld) @ D @ hOld + np.diagflat(hOld) @ D @ uOld)
    A = np.eye(nx+1) + c * np.diagflat(uOld) @ D
    uNew = A.I @ (uOld - c * g * D @ hOld)
    
    return uNew,hNew


@scheme_wrapper("ftcs_semi_impl_mat_sim_2")
def ftcs_semi_impl_mat_sim_2(uOld,hOld,c,nx): # h implicit
    c = c/2.
    D = D_mat(nx)
    
    B = np.eye(nx+1) + c * np.diagflat(uOld) @ D
    hNew = B.I @ (hOld - c * np.diagflat(hOld) @ D @ uOld)
    uNew = uOld - c * (g * D @ hOld + np.diagflat(uOld) @ D @ uOld)
    
    return uNew,hNew


@scheme_wrapper("ftcs_semi_impl_mat_sim_3")
def ftcs_semi_impl_mat_sim_3(uOld,hOld,c,nx): # u,h implicit
    c = c/2.
    D = D_mat(nx)
    
    B = np.eye(nx+1) + c * np.diagflat(uOld) @ D
    hNew = B.I @ (hOld - c * np.diagflat(hOld) @ D @ uOld)
    A = np.eye(nx+1) + c * np.diagflat(uOld) @ D
    uNew = A.I @ (uOld - c * g * D @ hOld)
    
    return uNew,hNew




