/*
 * File: GCDM.c
 * Description: "All models are wrong but some are useful", George Box (1976), Journal of the American Statistical Association.
 * Authors: E. Dendauw, t. Gajdos, N. Evans, & M. Servant 
 * Version: 2.1
 * Date: 2024-09-23
 */

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>

/* Function: GCDM
 * --------------------
 * Simulate n trials following the formalization of the Gated Cascade Diffusion Model (Dendauw et al., 2024) and latter improvements.
 * See: Dendauw, E., Evans, N. J., Logan, G. D., Haffen, E., Bennabi, D., Gajdos, T., & Servant, M. (2024). The gated cascade diffusion model: An integrated theory of decision making, motor preparation, and motor execution. Psychological Review. https://doi.org/10.1037/rev0000464
 * 
 * Main parameters :
 * x0   : starting point of the accumulation process
 * v    : drift rate
 * r    : response bound
 * g    : gating inhibition
 * Te   : mean duration of sensory encoding and corticomusculuar delay 
 * Tr   : mean duration of residual motor components related to force production 
 * xi   : transmission noise amplitude
 * leak : leakage
 * u    : urgency
 * drift_guess : prior bias
 * 
 * Between-trial variability parameters :
 * sx0   : between-trial variability in starting point (range of uniform with mean x0)
 * sv    : between-trial variability in drift rate     (SD of normal distribution with mean v)
 * sTe   : between-trial variability in Te             (range of uniform with mean Te)
 * sTr   : between-trial variability in Tr             (range of uniform with mean Tr)
 * sleak : between-trial variability in leak           (range of uniform with mean leak)
 * 
 * Other important variables :
 * s  : diffusion coefficient (scaling parameter)
 * dt : step size dt (in seconds)
 * n  : number of simulated trials
 * maxiter : maximum allocated number of time steps to reach a response threshold (else considered as "no response", i.e., resp=-1)
 * 
 * resp : vector containing the predicted accuracy for each trial (1 if upper response threshold hit, 2 if lower, -1 if the sample path did not hit any response bound within the allocated number of time steps
 * RT, PMT, MT : vector containing the predicted RT, PMT, MT for each trial
 * firstHit, firstHitLoc : vector containing the predicted latency of the first gate overcoming (in seconds) and the location (1 upper gating threshold, 2 lower, -1 if the signal never went over any gating threshold, e.g., drift=0)
 * coactiv : vector containing the predicted presence of bilateral EMG activities (1: there is, else 0)
 * 
 * Notes
 * --------------------
 * The variables rangeLow, rangeHigh, and randomTable serve to simulate a random draw from a standard normal distribution (see python code and corresponding method described in Evans (2019). 
 * `rand()/(1.0 + RAND_MAX)` simulates a random number from a uniform distribution bounded at 0 and 1
 * `randomTable[randIndex]` simulates a random number from the standard normal distribution
 */
void GCDM(double x0, double v, double r, double g, double Te, double Tr, 
          double xi, double leak, double u, double drift_guess,
          double sx0, double sv, double sTe, double sTr, double sleak, 
          double s, double dt, 
          double *resp, double *RT, 
          double *PMT, double *MT,
          double *firstHit, double *firstHitLoc, 
          double *coactiv,         
          int n, int maxiter, 
          int rangeLow, int rangeHigh, double *randomTable)
{
    // type declarations 
    double dv, x, y, urg, zU, zL, randNum, sampleV, sampleTe, sampleTr, sampleLeak;
    int i, iter, randIndex, outOfGate, hit;

    // randomize seed of random number generator
    struct timeval t1;
    gettimeofday(&t1, NULL);
    srand(t1.tv_usec * t1.tv_sec);

    // simulate n trials
    for (i=0; i<n; i++) {

        if (sx0 < 0.00001) { dv = x0; } 
        else {
            randNum = rand()/(1.0 + RAND_MAX);  // random number from a uniform distribution bounded at 0 and 1
            dv = x0 + (randNum*sx0) - (sx0/2); 
        }
        x = dv;
        y = dv;
        
        if (sv < 0.00001) { sampleV = v; } 
        else {
            randNum = rand()/(1.0 + RAND_MAX); 
            randIndex = (randNum * (rangeHigh - rangeLow + 1)) + rangeLow;
            randNum = randomTable[randIndex]; //randNum is a random number from the standard normal distribution
            sampleV = v + (sv*randNum);
        }
        
        sampleTe = (sTe < 0.00001) ? Te : Te + ((rand()/(1.0 + RAND_MAX))*sTe ) - (sTe/2);  // (condition) ? value_if_true : value_if_false
        sampleTr = (sTr < 0.00001) ? Tr : Tr + ((rand()/(1.0 + RAND_MAX))*sTr ) - (sTr/2);
        sampleLeak = (sleak < 0.00001) ? leak : leak + ((rand()/(1.0 + RAND_MAX))*sleak) - (sleak/2);
        
        // prepare trial output variables 
        resp[i] = -1.0;
        RT[i] = -1.0;
        PMT[i] = -1.0;
        MT[i] = -1.0;
        firstHit[i] = -1.0;
        firstHitLoc[i] = -1.0;
        coactiv[i] = 0.0;

        iter = 0;  
        outOfGate = 0;  // if both neural drives zU and zL are between g and -g, then outOfGate=0, else 1
        hit = 0;  // first time one gate is crossed (i.e., first time zL>0 or zU>0)
        
        do {
            iter = iter+1;  // direct increment so that iter starts at t_1 because at t_0: dv, x and y are initialized (l77-83)
            
            // ---------- decision variable ----------
            randNum = rand()/(1.0 + RAND_MAX);
            randIndex = (randNum * (rangeHigh - rangeLow + 1)) + rangeLow;
            randNum = randomTable[randIndex];
            dv = dv + (sampleV*dt) + (sqrt(dt)*s*randNum);
            
            // ---------- corrupted decision variable ----------
            randNum = rand()/(1.0 + RAND_MAX);
            randIndex = (randNum * (rangeHigh - rangeLow + 1)) + rangeLow;
            randNum = randomTable[randIndex];
            x = dv + (xi/sqrt(dt))*randNum;  // if (xi != 0) then there is a noisy transmission corruption else not corrupted
            
            // ---------- motor preparation variable ----------
            y = y + sampleLeak*(x - y)*dt + drift_guess*dt;  // constant gain

            // ---------- urgency signal ----------
            urg = u*iter*dt;  // if (u != 0) then urg is constant at 0
            
            // ---------- neural drives ----------
            zU = y + urg - g;  
            zL = -y + urg - g; 

            // if both neural drives are superior to 0, because of the urgency, then we consider it as an EMG bilateral activation
            if ((zU > 0) && (zL > 0) && (coactiv[i] == 0.0)) { coactiv[i] = 1.0; }  
            
            // the motor preparation variable is now located in the region between an EMG bound and the corresponding response bound
            if (((zU > 0) || (zL > 0)) && (outOfGate == 0)) {  
                if (hit == 0) {  // this is the first time the gate is overcome
                    hit = 1; 
                    firstHit[i] = ((iter*dt) - (dt/2.0)) + sampleTe; 
                    firstHitLoc[i] = (zU > 0) ? 1.0 : 2.0; 
                }
                outOfGate = 1; 
                PMT[i] = ((iter*dt) - (dt/2.0)) + sampleTe; // in case several EMG bound hits occur, PMT will store the latency of the last hit before the response 
            }
            
            // the motor preparation variable is now located in the region between both EMG bounds
            if ((outOfGate == 1) && (zU <= 0) && (zL <= 0)) { outOfGate = 0; }
            
            // correct (or right) response threshold hit
            if (zU >= r) {
                resp[i] = 1.0;
                RT[i] = (iter*dt) - (dt/2.0) + sampleTe + sampleTr;
                MT[i] = RT[i] - PMT[i];
                break;
            }

            // inccorect (or left) response threshold hit
            if (zL >= r) {
                resp[i] = 2.0; 
                RT[i] = (iter*dt) - (dt/2.0) + sampleTe + sampleTr;
                MT[i] = RT[i] - PMT[i];
                break;
            }
        } while (iter<maxiter); //simulate trial until max number of dt steps allowed (defined by maxiter)
        // ---------- WARNING ----------
        // if iter reaches maxiter:
        //     resp[i], RT[i] and MT[i] will be -1;
        //     however, firstHit[i], firstHitLoc[i] and PMT[i] may be different from -1;
    }
}