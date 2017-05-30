import os, sys
import numpy as np
import math
import string
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import rc
from optparse import OptionParser

def process_options():
    usage = "usage: %prog [option1] arg1 [option2] arg2 [option3] arg3"
    parser = OptionParser(usage=usage)
    parser.add_option("-z", dest="coord", help="Coordination number of the lattice", default="")
    parser.add_option("-e", dest="energy", help="Energy of interaction as a factor f to kT (i.e. e = f kT).", default="")
    parser.add_option("-c", dest="bias", help="Bias for clustering (no bias: c = 1, strictly biased: c = 0).", default="")

    [options, args] = parser.parse_args()
    
    return int(options.coord), float(options.energy), float(options.bias)

if __name__ == '__main__':
    coord_no, eps_f, neighbor_bias = process_options()
    
    D_em = []
    beta = []
    rT = []
    DC = []
    phi_n = []
    
    kBT = 0.0259
    global eps_0
    eps_0 = eps_f
    global eps
    
    global t_rate
    t_rate = 40.0/0.55
    global t_max
    t_max = 300.0
    
    #neighbor_bias = 1.0
    
    last_phi_2 = 0.0
    
    #coord_no = 6
    limit = 1.0 - 2.0/float(coord_no)
    
    phi_set = np.linspace(0.0,limit,1000)
    
    DC_set = np.linspace(0.0,1.0,1000)
    
    e_value = str(eps_f).split('.')
    e_str = e_value[0]+'_'+e_value[1]
    
    c_value = str(neighbor_bias).split('.')
    c_str = c_value[0]+'_'+c_value[1]
    
    filename = 'z'+str(coord_no)+'e'+e_str+'c'+c_str+'.csv'
    
    ofile = open(filename,'w')
    
    phi_cr = 0.0
    
    global g_bars
    global g_means
    global phis
    g_bars = np.zeros(6)
    g_means = np.zeros(3)
    phis = np.zeros(3)
    
    def compute_eps(temp_factor,cluster_factor):
        eps = eps_0*cluster_factor
        eps *= (300.0/temp_factor)
        
        return eps
    
    def compute_g_means(eps):
        g_means[0] = 1.0 
        g_means[1] = math.exp(-eps)
        g_means[2] = math.exp(-2*eps) 
        
    def compute_coeffs(g_em):
        numer, denom = 0.0, 0.0
        
        for i in xrange(0,3):
            numer += phis[i]*(g_em - g_means[i])/(2*g_em + g_means[i])
            denom += phis[i]/(2*g_em + g_means[i])
            
        return numer, denom
        
    def compute_D_eff(phi_ins,eps):
        phis[0], phis[1], phis[2] = phi_ins[0], phi_ins[1], phi_ins[2]
        
        compute_g_means(eps)
        
        g_em = g_means[0]*phi_ins[0] + g_means[1]*phi_ins[1] + g_means[2]*phi_ins[2]
        
        numer, denom = compute_coeffs(g_em)
        
        delta_g_em = -numer/denom
        
        g_em += delta_g_em
        
        while abs(delta_g_em/g_em)>1e-6:
            numer, denom = compute_coeffs(g_em)
            
            delta_g_em = -numer/denom
        
            g_em += delta_g_em
        
        return g_em
        
    def compute_neighbor_prob(phi_3):
        empty_prob = 1 - phi_3
        
        total_odds = 6.0*phi_3*(empty_prob**5) + 15.0*(phi_3**2)*(empty_prob**4)
        total_odds += 20.0*(phi_3**3)*(empty_prob**3) + 15.0*(phi_3**4)*(empty_prob**2)
        total_odds += 6.0*(phi_3**5)*empty_prob + phi_3**coord_no
        
        return total_odds*empty_prob
    
    def get_sites_from_DC(DC):
        M2 = DC**2
        
        M1 = 2*DC - 2*M2
        
        sites = M1 + M2
        
        return M2
    
    def estimate_phi2(phi_3,neighbor_closed):
        # Probability that a site has no polymer blocking it at side
        empty_prob = (1.0 - phi_3)
            
        neighbor_open = 1.0 - neighbor_closed
        
        # Fraction of nodes blocked from all sides
        all_closed = neighbor_closed**coord_no
        
        # Number of nodes that can act as neighbors
        all_neighbors = phi_3*(1.0 - all_closed)
        
        # An empty site can have 2 types of neighbors
        # Another empty neighbot
        empty_prob_n = empty_prob/(empty_prob+all_neighbors)
        # A blocked neighbor
        occupied_prob_n = all_neighbors/(empty_prob+all_neighbors)
        
        # An empty sight has coord-no of neighbors
        # We need to calculate the probability that atleast one of them but not all of them are blocked sites
        total_odds = 1.0 - empty_prob_n**6 - occupied_prob_n**6
        
        # Total number of neighboring sites
        phi_2 = total_odds*empty_prob
        
        return phi_2
    
    def get_DC_from_phi(phi):
        
        DC = math.sqrt(phi)
        
        sites = 2*DC - DC**2
        
        return DC, sites
    
    def plot_results(DC,rT):
        """This function plots the results.
        """
        plt.plot(DC,rT,marker='o',ms=0.0,lw=5.0)
        plt.minorticks_on()
        plt.grid()
        plt.xlim(0.0,max(DC))
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.xlabel(r'DC',size=20)
        plt.ylabel(r'$D_{\mathrm{em}}$',size=20)
        plt.yscale('log')
        plt.show()
    
    for this_DC in DC_set[1:-2]:
    #for phi in phi_set[1:-2]:
        #this_DC, this_sites = get_DC_from_phi(phi)
        #if phi<=limit:
        crosslinked_sites = this_DC**2
        all_sites = 2*this_DC - crosslinked_sites
        regular_sites = all_sites - crosslinked_sites
            
        if crosslinked_sites<1.0:
            #cl_neighbors = estimate_phi2(crosslinked_sites,neighbor_bias)
            neighbor_closed = all_sites**neighbor_bias
            all_neighbors = estimate_phi2(all_sites,neighbor_closed)
            
            cl_closed = crosslinked_sites**neighbor_bias#*(crosslinked_sites/all_sites)
            
            cl_neighbors = estimate_phi2(crosslinked_sites,cl_closed)
            
            regular_neighbors = all_neighbors - cl_neighbors
            regular_effected = regular_neighbors + regular_sites
            
            total = regular_effected + cl_neighbors + crosslinked_sites
            
            if total>1.0:
                print 'Error'
                
            phi_ins = np.zeros(3)
            
            phi_ins[1] = regular_effected
            phi_ins[2] = cl_neighbors + crosslinked_sites
            phi_ins[0] = 1.0 - phi_ins[1] - phi_ins[2]
            
            if phi_ins[2]<limit:
                temp_factor = 300 + t_rate*this_DC
                t_max = max(temp_factor,t_max)
                cluster_factor = neighbor_closed
                eps = compute_eps(temp_factor,cluster_factor)
            else:
                temp_factor = 300.0 + (t_max - 300.0)*math.exp(-5*(phi_ins[2] - limit))
                cluster_factor = neighbor_closed
                eps = compute_eps(temp_factor,cluster_factor)
            
            
            this_D_em = compute_D_eff(phi_ins,eps)
            
            D_em.append(this_D_em)
            #if D_em[-1]>0.0:
            print >> ofile, this_DC, ',', crosslinked_sites, ',', phi_ins[0], ',', phi_ins[1], ',', phi_ins[2], ',', 1.0/D_em[-1], ',', temp_factor, ',', eps
            rT.append(1.0/D_em[-1])
            DC.append(100*this_DC)
    
    ofile.close()
    
    plot_results(DC,rT)