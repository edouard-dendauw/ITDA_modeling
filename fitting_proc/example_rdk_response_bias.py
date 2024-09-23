# -*- coding: utf-8 -*-
"""
Update: December 21 2023
@author: edendauw
@credit: mservant

# ------- COMMENTS --------------------------------------------------

upper bound : right response
lower bound : left response

# order conditions
--------    ----------------    -- 
unbiased    rightward motion    00        
unbiased    rightward motion    04   
unbiased    rightward motion    10   
unbiased    rightward motion    40

unbiased    leftward motion     00        
unbiased    leftward motion     04   
unbiased    leftward motion     10   
unbiased    leftward motion     40
--------    ----------------    --   
biased L    rightward motion    00        
biased L    rightward motion    04   
biased L    rightward motion    10   
biased L    rightward motion    40

biased L    leftward motion     00        
biased L    leftward motion     04   
biased L    leftward motion     10   
biased L    leftward motion     40    
--------    ----------------    -- 
biased R    rightward motion    00        
biased R    rightward motion    04   
biased R    rightward motion    10   
biased R    rightward motion    40

biased R    leftward motion     00        
biased R    leftward motion     04   
biased R    leftward motion     10   
biased R    leftward motion     40 
--------    ----------------    -- 
"""

CLUSTER1_ = 'path/to/the/fits'

FITS_FOLDER_PATH = CLUSTER1_ 


# -------------------------------------------------------------------
# ------- IMPORTS ---------------------------------------------------

import time
import sys
import os
import tracemalloc
import ctypes
from copy import deepcopy
import datetime

import numpy as np
import scipy.optimize as opt
from scipy.stats import norm

# parameters ----------------------------
os.chdir(FITS_FOLDER_PATH)
np.set_printoptions(formatter={'float': '{:0.4f}'.format})


# -------------------------------------------------------------------
# ------- CONSTANT VALUES -------------------------------------------

# experimental conditions
BIAS = np.array(['NB', 'BTL', 'BTR'])
MDIRS = np.array(['RW', 'LW'])
MCS = np.array([.00, .04, .10, .40])

# model
mydll = ctypes.cdll.LoadLibrary('models/GCDM.dll')  # use .so file if Linux OS or macOS
ALGO = mydll.GCDM

# set boundaries on parameters
x0_B    = (.0,   .9)    # ok
k_B     = (0.,   3.)    # ok
dc_B    = (-.4,  .4)    # >>> NEW <<<
r_B     = (.0,  .1965)  # ok
g_B     = (.0,  .1965)  # ok
Te_B    = (.1,  .8)     # ok
Tr_B    = (.0,  .3)     # ok
xi_B    = (.0,  .2)     # ok
leak_B  = (.25, 1000.)  # ok
u_B     = (0.,  10.)    # >>> NEW <<<

drift_guess_B = (0., .4)  # >>> NEW <<<

# between-trial variability 
sx0_B   = (.0, .9) # as a percentage of 2*g
sv_B    = (.0, .329)
sTe_B   = (.0, .9) # as a percentage of 2*Te
sTr_B   = (.0, .9) # as a percentage of 2*Tr
sleak_B = (.0, .9) # as a percentage of 2*leak


# -------------------------------------------------------------------
# ------- CLASS -----------------------------------------------------

class params:
    def __init__(self, x0, v, dc, r, g, Te, Tr, xi, leak, u, drift_guess, sx0, sv, sTe, sTr, sLeak, n_sim_trials):
        
        self.x0 = x0 * g
        self.v = v + dc 
        self.r = r  
        self.g = g
        self.Te = Te 
        self.Tr = Tr 
        self.xi = xi
        self.leak = leak
        self.u = u
        self.drift_guess = drift_guess

        self.sx0 = sx0 * 2*g
        self.sv = sv
        self.sTe = sTe * 2*Te
        self.sTr = sTr * 2*Tr
        self.sLeak = sLeak * 2*leak

        self.s = .1
        self.dt = .001
        self.n = n_sim_trials

        self.resp = np.zeros(self.n)
        self.RT = np.zeros(self.n)
        self.PMT = np.zeros(self.n)
        self.MT = np.zeros(self.n)
        self.firstHit = np.zeros(self.n)
        self.firstHitLoc = np.zeros(self.n)
        self.coactiv = np.zeros(self.n)

        self.maxiter = 15000
        # LUT parameters (see Evans, 2019)
        # build LUT table for gaussian random number generator
        interval = .0001
        gran = np.arange(interval, 1, interval)
        use_table = norm.ppf(gran)
        self.randomTable = use_table
        self.rangeLow = 0
        self.rangeHigh = len(self.randomTable)


# -------------------------------------------------------------------
# ------- FUNCTIONS -------------------------------------------------

def runIndvFit(s, x0, maxiter):
    args = [qPMT_LL_obs, qPMT_RL_obs, qPMT_pL_obs, qPMT_RR_obs, qPMT_LR_obs, qPMT_pR_obs, 
             qMT_LL_obs,  qMT_RL_obs,  qMT_pL_obs,  qMT_RR_obs,  qMT_LR_obs,  qMT_pR_obs, 
               f_LL_obs,    f_RL_obs,    f_pL_obs,    f_RR_obs,    f_LR_obs,    f_pR_obs,  20000]
    
    def callback_(xk, convergence): 
        print(xk)
        sys.stdout.flush()
    fit = opt.differential_evolution(getFullGsq, bounds, args=args, callback=callback_, disp=True,
                                     updating='deferred', strategy='best1bin', init='latinhypercube', polish=False,
                                     popsize=15, tol=1e-6, recombination=.7, maxiter=maxiter, workers=10, x0=x0)
    
    args[-1] = 500000
    full_G2_500k = getFullGsq(fit.x, *args) # returns G2 based on 500,000 simulated trials
    output = [fit.x, full_G2_500k]
    return output


def getFullGsq(optPar, \
               qPMT_LL_one, qPMT_RL_one, qPMT_pL_one, qPMT_RR_one, qPMT_LR_one, qPMT_pR_one, 
                qMT_LL_one,  qMT_RL_one,  qMT_pL_one,  qMT_RR_one,  qMT_LR_one,  qMT_pR_one, 
                  f_LL_one,    f_RL_one,    f_pL_one,    f_RR_one,    f_LR_one,    f_pR_one,  n_sim_trials):

    fullPar = makeFullPar(optPar)
    fullGsq = 0

    # loop through conditions and add up g-squared
    counter = 0
    for b in range(len(BIAS)):
        for d in range(len(MDIRS)):
            for mc in range(len(MCS)):
                # add 0 and large number to quantiles to make intervals
                qPMT_LL = np.insert(qPMT_LL_one[counter,:], [0,5], [0, 1000])
                qPMT_RL = np.insert(qPMT_RL_one[counter,:], [0,5], [0, 1000])
                qPMT_pL = np.insert(qPMT_pL_one[counter,:], [0,5], [0, 1000])
                qPMT_RR = np.insert(qPMT_RR_one[counter,:], [0,5], [0, 1000])
                qPMT_LR = np.insert(qPMT_LR_one[counter,:], [0,5], [0, 1000])
                qPMT_pR = np.insert(qPMT_pR_one[counter,:], [0,5], [0, 1000])
                
                qMT_LL = np.insert(qMT_LL_one[counter,:], [0,5], [0, 1000])
                qMT_RL = np.insert(qMT_RL_one[counter,:], [0,5], [0, 1000])
                qMT_pL = np.insert(qMT_pL_one[counter,:], [0,5], [0, 1000])
                qMT_RR = np.insert(qMT_RR_one[counter,:], [0,5], [0, 1000])
                qMT_LR = np.insert(qMT_LR_one[counter,:], [0,5], [0, 1000])
                qMT_pR = np.insert(qMT_pR_one[counter,:], [0,5], [0, 1000])
                
                f_LL = f_LL_one[counter,:]
                f_RL = f_RL_one[counter,:]
                f_pL = f_pL_one[counter,:]
                f_RR = f_RR_one[counter,:]
                f_LR = f_LR_one[counter,:]
                f_pR = f_pR_one[counter,:]
                
                # set of parameters in this specific condition
                par = fullPar[b,d,mc,:]  # x0, v, dc, r, g, Te, Tr, xi, leak, u, drift_guess, sx0, sv, sTe, sTr, sLeak
                assert len(par) == 16

                # Gsq function
                debug = gsqfun(par, \
                            qPMT_LL, qPMT_RL, qPMT_pL, qPMT_RR, qPMT_LR, qPMT_pR, 
                             qMT_LL,  qMT_RL,  qMT_pL,  qMT_RR,  qMT_LR,  qMT_pR, 
                               f_LL,    f_RL,    f_pL,    f_RR,    f_LR,    f_pR,  n_sim_trials)
                fullGsq = fullGsq + debug
                
                # next condition
                counter += 1
    return fullGsq


def gsqfun(par, \
           qPMT_LL, qPMT_RL, qPMT_pL, qPMT_RR, qPMT_LR, qPMT_pR, 
            qMT_LL,  qMT_RL,  qMT_pL,  qMT_RR,  qMT_LR,  qMT_pR, 
              f_LL,    f_RL,    f_pL,    f_RR,    f_LR,    f_pR,  n_sim_trials):  
    obj = params(*par, n_sim_trials)

    # -----------------------------------
    get_min_dist = min( np.abs(obj.g-obj.x0), np.abs(-obj.g-obj.x0) ) 
    if obj.sx0 >= 2*get_min_dist:
        gsq = 1000000000000000000000000000000000000000000000000.
        return gsq 
    # ----------------------------------- 

    ALGO(
        ctypes.c_double(obj.x0), ctypes.c_double(obj.v), ctypes.c_double(obj.r), ctypes.c_double(obj.g), ctypes.c_double(obj.Te), ctypes.c_double(obj.Tr),
        ctypes.c_double(obj.xi), ctypes.c_double(obj.leak), ctypes.c_double(obj.u), ctypes.c_double(obj.drift_guess),
        ctypes.c_double(obj.sx0), ctypes.c_double(obj.sv), ctypes.c_double(obj.sTe), ctypes.c_double(obj.sTr), ctypes.c_double(obj.sLeak),

        ctypes.c_double(obj.s), ctypes.c_double(obj.dt),
        
        ctypes.c_void_p(obj.resp.ctypes.data), ctypes.c_void_p(obj.RT.ctypes.data), 
        ctypes.c_void_p(obj.PMT.ctypes.data), ctypes.c_void_p(obj.MT.ctypes.data),  
        ctypes.c_void_p(obj.firstHit.ctypes.data), ctypes.c_void_p(obj.firstHitLoc.ctypes.data), 
        ctypes.c_void_p(obj.coactiv.ctypes.data),

        ctypes.c_int(obj.n), ctypes.c_int(obj.maxiter), 
        ctypes.c_int(obj.rangeLow), ctypes.c_int(obj.rangeHigh), ctypes.c_void_p(obj.randomTable.ctypes.data)
    )   
    
    pred_Resp = deepcopy(obj.resp)
    pred_PMT = deepcopy(obj.PMT)
    pred_MT = deepcopy(obj.MT)
    pred_firstHit = deepcopy(obj.firstHit)
    pred_firstHitLoc = deepcopy(obj.firstHitLoc)

    # -----------------------------------
    # compute partial bursts
    pred_isPartial = []
    for p in range(len(pred_PMT)):
        if (pred_PMT[p] - pred_firstHit[p]) < .00001: pred_isPartial.append(0)
        else: pred_isPartial.append(1) # there is a partial burst
    pred_isPartial = np.array(pred_isPartial)
    
    # -----------------------------------
    # only keep trials that have converged and gives correct values
    keep_condition = (pred_Resp > 0) & (pred_PMT > 0) & (pred_MT > 0) # /!\ --- ACHTUNG --- /!\

    pred_Resp = pred_Resp[keep_condition] 
    pred_PMT = pred_PMT[keep_condition]
    pred_MT = pred_MT[keep_condition]
    pred_firstHit = pred_firstHit[keep_condition]
    pred_firstHitLoc = pred_firstHitLoc[keep_condition] 
    pred_isPartial = pred_isPartial[keep_condition]

    # -----------------------------------
    # we now need to compute the predicted proportion in each bin of the joint distribution of PMT and MT 
    pred_PMT_right = pred_PMT[pred_Resp==1]
    pred_PMT_left  = pred_PMT[pred_Resp==2]
    
    n_simulated = len(pred_PMT_right) + len(pred_PMT_left)   
    if n_simulated == 0:
        gsq = 1000000000000000000000000000000000000000000000000.
        return gsq

    # let's go ! compute_f_ function for all the trial-types !
    # right responses : RR, LR, pureR
    pred_f_RR, final_f_RR = compute_f_( f_RR, qPMT_RR, qMT_RR, pred_PMT, pred_MT, n_simulated, condition=((pred_Resp==1)&(pred_isPartial==1)&(pred_firstHitLoc==1)) ) 
    pred_f_LR, final_f_LR = compute_f_( f_LR, qPMT_LR, qMT_LR, pred_PMT, pred_MT, n_simulated, condition=((pred_Resp==1)&(pred_isPartial==1)&(pred_firstHitLoc==2)) )
    pred_f_pR, final_f_pR = compute_f_( f_pR, qPMT_pR, qMT_pR, pred_PMT, pred_MT, n_simulated, condition=((pred_Resp==1)&(pred_isPartial==0)) ) 

    # left responses : LL, RL, pureL
    pred_f_LL, final_f_LL = compute_f_( f_LL, qPMT_LL, qMT_LL, pred_PMT, pred_MT, n_simulated, condition=((pred_Resp==2)&(pred_isPartial==1)&(pred_firstHitLoc==2)) ) 
    pred_f_RL, final_f_RL = compute_f_( f_RL, qPMT_RL, qMT_RL, pred_PMT, pred_MT, n_simulated, condition=((pred_Resp==2)&(pred_isPartial==1)&(pred_firstHitLoc==1)) ) 
    pred_f_pL, final_f_pL = compute_f_( f_pL, qPMT_pL, qMT_pL, pred_PMT, pred_MT, n_simulated, condition=((pred_Resp==2)&(pred_isPartial==0)) ) 

    # the Gsq part
    obs_all  = np.concatenate((final_f_LL, final_f_RL, final_f_pL, final_f_RR, final_f_LR, final_f_pR))
    pred_all = np.concatenate(( pred_f_LL,  pred_f_RL,  pred_f_pL,  pred_f_RR,  pred_f_LR,  pred_f_pR))

    pred_all[(pred_all < .00001) & (obs_all > 0)] = .00001 # here we add 1 simulated trial so that E > 0 when O > 0
    
    N_obs = np.sum(obs_all)
    N_pred = np.sum(pred_all)

    log_condition = (obs_all>0) & (pred_all>0) # Gsq implementation constrains that O > 0 and E > 0
    obs_all = obs_all[log_condition]
    pred_all = pred_all[log_condition]
    
    gsq = 2 * np.sum( obs_all * np.log(obs_all / (pred_all*N_obs / N_pred)) )  # see Servant et al. (2021) : (pred_n_ijkl * N_i) / simul_N_i
    return gsq


def compute_f_(f_, qPMT, qMT, pred_PMT, pred_MT, n_simulated, condition):
    if np.sum(f_) >= 10:
        pred_f_ = np.zeros(len(qPMT) * len(qMT), dtype=np.float64)
        final_f_ = np.zeros(len(qPMT) * len(qMT), dtype=np.float64)
        counter = 0
        for j in range (1, len(qPMT)):
            for k in range (1, len(qMT)): 
                n_pred = np.sum(condition & (pred_PMT>=qPMT[j-1]) & (pred_PMT<qPMT[j]) & (pred_MT>=qMT[k-1]) & (pred_MT<qMT[k])) / n_simulated
                pred_f_[counter] = n_pred
                final_f_[counter] = f_[counter]
                counter += 1

    elif (np.sum(f_) >= 5) and (np.sum(f_) < 10):
        pred_f_ = np.array([ np.sum( condition & (pred_PMT < qPMT[3])  & (pred_MT < qMT[3])), \
                             np.sum( condition & (pred_PMT < qPMT[3])  & (pred_MT >= qMT[3])),
                             np.sum( condition & (pred_PMT >= qPMT[3]) & (pred_MT < qMT[3])),
                             np.sum( condition & (pred_PMT >= qPMT[3]) & (pred_MT >= qMT[3])) 
                        ]) / n_simulated 
        final_f_ = np.array([ np.sum(f_[:3] + f_[6:9] + f_[12:15]), \
                              np.sum(f_[3:6] + f_[9:12] + f_[15:18]),
                              np.sum(f_[18:21] + f_[24:27] + f_[30:33]),
                              np.sum(f_[21:24] + f_[27:30] + f_[33:36]) ])

    elif np.sum(f_) < 5:
        pred_f_ = np.array([ np.sum(condition) / n_simulated ])
        final_f_ = np.array([ np.sum(f_) ])

    else: raise ValueError
    return pred_f_, final_f_


# --------------------------- APPLY VARIATIONS HERE ------------------------------------- TODO --- \_(O_o)_/
def makeFullPar(optPar): 
        fullPar = np.zeros( (len(BIAS), len(MDIRS), len(MCS), 16) )
        # experimental conditions
        # BIAS = np.array(['NB', 'BTL', 'BTR'])
        # MDIRS = np.array(['RW', 'LW'])
        # MCS = np.array([.00, .04, .10, .40])

        # ---------- x0 ---------------------------------------------
        fullPar[0, :, :, 0] = 0  # NB
        fullPar[1, :, :, 0] = -optPar[0]  # BTL
        fullPar[2, :, :, 0] = optPar[0]  # BTR

        # ---------- v from motion coherence ------------------------
        fullPar[:, 0, 0, 1] = optPar[1] * MCS[0]  # rightward stim
        fullPar[:, 0, 1, 1] = optPar[1] * MCS[1]
        fullPar[:, 0, 2, 1] = optPar[1] * MCS[2]
        fullPar[:, 0, 3, 1] = optPar[1] * MCS[3]
        fullPar[:, 1, 0, 1] = -optPar[1] * MCS[0]  # leftward stim
        fullPar[:, 1, 1, 1] = -optPar[1] * MCS[1]
        fullPar[:, 1, 2, 1] = -optPar[1] * MCS[2]
        fullPar[:, 1, 3, 1] = -optPar[1] * MCS[3]

        fullPar[0, :, :, 2] = 0  # NB
        fullPar[1, :, :, 2] = -optPar[2]  # BTL
        fullPar[2, :, :, 2] = optPar[2]  # BTR
        
        fullPar[:, :, :, 3] = optPar[3] # r 
        
        fullPar[:, :, :, 4] = optPar[4] # g

        fullPar[:, :, :, 5] = optPar[5] # Te

        fullPar[:, :, :, 6] = optPar[6] # Tr

        fullPar[:, :, :, 7] = optPar[7] # xi

        fullPar[:, :, :, 8] = optPar[8] # leak

        fullPar[:, :, :, 9] = optPar[9] # u

        fullPar[:, :, :,10] = 0 # no drift guess

        # no between-trial variability
        fullPar[:, :, :,11] = 0 # no sx0
        fullPar[:, :, :,12] = 0 # no sv
        fullPar[:, :, :,13] = 0 # no sTe
        fullPar[:, :, :,14] = 0 # no sTr
        fullPar[:, :, :,15] = 0 # no sLeak
        
        return fullPar


# -------------------------------------------------------------------
# ------- __main__ RUNNING ZONE -------------------------------------

if __name__=='__main__':
    tracemalloc.start()
    start_time = time.perf_counter()
    subject = int(sys.argv[1])

    fit_name = 'GCDMu_dc'

    # differential evolution parameters
    # if x0 (DE initial guess) != None:
    n_rounds = 1
    maxiter_DE = 200
    # else:
    # n_rounds = 3
    # maxiter_DE = 150

    # model parameters boundaries
    bounds = [
        x0_B, 
        k_B, 
        dc_B,
        r_B,  
        g_B, 
        Te_B,  
        Tr_B,
        xi_B, 
        leak_B,  
        u_B
        # drift_guess_B,

        # sx0_B,
        # sv_B,
        # sTe_B,
        # sTr_B,
        # sleak_B
    ]
    
    parent_name = 'GCDMu_' 
    parent_params = np.load(f'{parent_name}/params/params_s{subject}.npy') # ['x0', 'k', 'r', 'g', 'Te', 'Tr', 'xi', 'leak', 'u']
    x0 = np.concatenate([parent_params[:2], [0], parent_params[2:]]) 

    # ----------------------------------------------------------------------------------- __(X_x)__
    # load observed data (shape: [s, cond, data]) --------
    f_LL_obs = np.load('data_for_fit/f_LL.npy',    allow_pickle=True)[subject-1]
    f_RL_obs = np.load('data_for_fit/f_RL.npy',    allow_pickle=True)[subject-1]
    f_pL_obs = np.load('data_for_fit/f_pureL.npy', allow_pickle=True)[subject-1]
    f_RR_obs = np.load('data_for_fit/f_RR.npy',    allow_pickle=True)[subject-1]
    f_LR_obs = np.load('data_for_fit/f_LR.npy',    allow_pickle=True)[subject-1]
    f_pR_obs = np.load('data_for_fit/f_pureR.npy', allow_pickle=True)[subject-1]

    qPMT_LL_obs = np.load('data_for_fit/q_LL_PMT.npy',    allow_pickle=True)[subject-1]
    qPMT_RL_obs = np.load('data_for_fit/q_RL_PMT.npy',    allow_pickle=True)[subject-1]
    qPMT_pL_obs = np.load('data_for_fit/q_pureL_PMT.npy', allow_pickle=True)[subject-1]
    qPMT_RR_obs = np.load('data_for_fit/q_RR_PMT.npy',    allow_pickle=True)[subject-1]
    qPMT_LR_obs = np.load('data_for_fit/q_LR_PMT.npy',    allow_pickle=True)[subject-1]
    qPMT_pR_obs = np.load('data_for_fit/q_pureR_PMT.npy', allow_pickle=True)[subject-1]

    qMT_LL_obs = np.load('data_for_fit/q_LL_MT.npy',    allow_pickle=True)[subject-1]
    qMT_RL_obs = np.load('data_for_fit/q_RL_MT.npy',    allow_pickle=True)[subject-1]
    qMT_pL_obs = np.load('data_for_fit/q_pureL_MT.npy', allow_pickle=True)[subject-1]
    qMT_RR_obs = np.load('data_for_fit/q_RR_MT.npy',    allow_pickle=True)[subject-1]
    qMT_LR_obs = np.load('data_for_fit/q_LR_MT.npy',    allow_pickle=True)[subject-1]
    qMT_pR_obs = np.load('data_for_fit/q_pureR_MT.npy', allow_pickle=True)[subject-1]

    # run differential evolution --------
    optim_all = np.zeros((n_rounds, len(bounds)+1))
    for round_ in range(n_rounds):
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%d-%m-%Y %H:%M:%S")
        print(f'[{formatted_datetime}] run {round_+1}/{n_rounds} ----------\n')
        sys.stdout.flush() 
        
        optim_round = runIndvFit(subject, x0=x0, maxiter=maxiter_DE) 
        optim_all[round_, :len(bounds)] = optim_round[0] # params
        optim_all[round_, len(bounds)] = optim_round[1] # G2 with 500,000 simulated trials

        finish_time = time.perf_counter()
        print(f'Time elapsed is seconds: {finish_time - start_time:.6f}s\n')
        sys.stdout.flush()
    
    best_Gsq_idx = np.argmin(optim_all[:, len(bounds)])
    best_Gsq = optim_all[best_Gsq_idx, len(bounds)]
    best_params = optim_all[best_Gsq_idx, :len(bounds)]

    best_AIC = best_Gsq + 2 * len(bounds)
    best_BIC = best_Gsq + len(bounds) * np.log( np.round(np.sum(f_LL_obs)+np.sum(f_RL_obs)+np.sum(f_pL_obs)+np.sum(f_RR_obs)+np.sum(f_LR_obs)+np.sum(f_pR_obs)) )

    # save everything -------------------
    np.save(f'{fit_name}/params/params_s{subject}.npy', best_params)
    np.save(f'{fit_name}/goodness/scores_s{subject}.npy', np.array([best_Gsq, best_AIC, best_BIC]))

    # show some stats -----------------------
    current, peak = tracemalloc.get_traced_memory()
    finish_time = time.perf_counter()
    print(f'\nMemory usage: {current / 10**6:.6f} MB')
    print(f'Peak memory usage: {peak / 10**6:.6f} MB') 
    print(f'Time elapsed is seconds: {finish_time - start_time:.6f}s')
    print('--- end reached with success --- (O_o)')
    tracemalloc.stop()
#[X]
