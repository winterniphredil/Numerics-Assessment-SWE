import numpy as np
import matplotlib.pyplot as plt
import time

from quantities import mass, momentum

from schemes import ftcs_semi_impl_u_1,ftcs_semi_impl_u_2,ftcs_semi_impl_u_3,ftcs_semi_impl_h_1,ftcs_semi_impl_h_2,ftcs_semi_impl_h_3,ftcs_u,ftcs_h,ftbs_u,ftbs_h
from schemes import ftcs_semi_impl_mat_u_1,ftcs_semi_impl_mat_u_2,ftcs_semi_impl_mat_u_3,ftcs_semi_impl_mat_h_1,ftcs_semi_impl_mat_h_2,ftcs_semi_impl_mat_h_3


fns = [ftcs_u,ftcs_h,ftbs_u,ftbs_h,ftcs_semi_impl_u_1,ftcs_semi_impl_u_2,ftcs_semi_impl_u_3,ftcs_semi_impl_h_1,ftcs_semi_impl_h_2,ftcs_semi_impl_h_3]
mat_fns = [ftcs_semi_impl_mat_u_1,ftcs_semi_impl_mat_u_2,ftcs_semi_impl_mat_u_3,ftcs_semi_impl_mat_h_1,ftcs_semi_impl_mat_h_2,ftcs_semi_impl_mat_h_3]
all_fns = fns + mat_fns

run_times = {}
fail_times = {}
nx = 160 
nt = 80
dx = 1./nx
dt = 1./nt

x = np.linspace(0.0, 1.0, nx+1)

def plot(h,u,n,fn):
    plt.cla()
    plt.plot(x, h, 'k', label='h at time '+str(n*dt))
    plt.plot(x, u, 'g', label='u at time '+str(n*dt))
    plt.title(fn.__name__)
    plt.legend(loc='best')
    plt.ylabel('h / u')
    plt.ylim([-1,1])
    plt.pause(0.005)
    
def plot_m_m(mass_vec,mom_vec,fn):
    t = len(mass_vec)
    plt.cla()
    plt.plot(np.linspace(0.,(t-1)*dt,t),np.array(mass_vec),'b', label = "mass")
    plt.title(fn.__name__)
    plt.legend(loc='best')
    plt.ylabel('quantity')
    plt.pause(0.005)
    
    plt.cla()
    plt.plot(np.linspace(0.,(t-1)*dt,t),np.array(mom_vec),'r', label = "momentum")
    plt.title(fn.__name__)
    plt.legend(loc='best')
    plt.ylabel('quantity')
    plt.pause(0.005)


def run_and_plot(fn,u_0,h_0):
    m = [mass(h_0)]
    mom = [momentum(h_0.T,u_0.T)]
    fail_time = 1.
    for n in range(nt):
        u_0,h_0,failed = fn(u_0, h_0, dt/dx, nx)
        if failed:
            fail_time = n*dt
            break
        #plot(h_0,u_0,n,fn)
        m.append(mass(h_0))
        mom.append(momentum(h_0.T,u_0.T))
    #plot_m_m(m,mom,fn)
    return fail_time

def plot_times(r, f):
    plt.cla()
    
    x = np.arange(len(r))
    width = 0.4
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    keys = list(r.keys())
    
    ax.bar(x-0.2, list(r.values()), width, color = 'b', label = "actual time taken to run (no plots)") # change label if plotting
    ax.bar(x+0.2, list(f.values()), width, color = 'r', label = "time until error occurred, or 1 if no error occurred")
    
    ax.set_ylabel('Time (s)')
    ax.set_title('Time')
    ax.set_xticks(x, list(r.keys()))
    plt.legend(loc='best')
    plt.xticks(rotation=90)
    plt.show()
    return

for fn in all_fns:
    
    h_0 = np.mat(np.where(x%1. < 0.5, np.power(np.sin(2*x*np.pi)*0.5, 2), 0.)).T
    u_0 = np.mat(np.full(nx+1,0.0, dtype=float)).T
    
    start_time = time.time()
    fail_time = run_and_plot(fn,u_0,h_0)
    run_times[fn.__name__] = time.time()-start_time
    fail_times[fn.__name__] = fail_time

plot_times(run_times, fail_times)


