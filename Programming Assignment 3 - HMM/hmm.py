from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here

        # Initial
        alpha[:, 0] = self.pi * self.B[:, self.obs_dict[Osequence[0]]]

        # Rest of Alpha
        for t in range(1, L):
            for j in range(S):
                alpha[j, t] = np.sum(np.multiply(alpha[:, t-1], self.A[:, j]))*self.B[j, self.obs_dict[Osequence[t]]]

        ###################################################
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here

        # Initial
        beta[:, L-1] = 1

        # Rest of Beta going backwards
        for t in reversed(range(0, L-1)):
            mult = np.multiply(beta[:, t+1], self.B[:, self.obs_dict[Osequence[t+1]]])
            beta[:, t] = np.dot(self.A, mult)

        ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here

        # Calculate the alpha
        alpha = self.forward(Osequence)
        prob = np.sum(alpha[:, -1])
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        # Edit here

        # Calculate the numerator
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        
        # Calculate the denominator
        den = self.sequence_prob(Osequence)
        prob = np.divide(np.multiply(alpha, beta), den)

        ###################################################
        return prob
        
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        # Edit here

        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        den = self.sequence_prob(Osequence)

        for i in range(S):
            for j in range(S):
                for t in range(L-1):
                    prob[i,j,t] = (alpha[i, t] * self.A[i, j] * self.B[j, self.obs_dict[Osequence[t+1]]] * beta[j,t+1]) / den


        ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Q3.3 Edit here

        S = len(self.pi)
        L = len(Osequence)

        # Set up a delta matrix and initialize initial values
        delta = np.zeros([S,L])
        os = self.obs_dict[Osequence[0]]
        delta[:,0] = self.pi * self.B[:, os]

        # Set up a states matrix to keep track of states from max
        states = np.zeros([S,L])

        # Iterate through the times
        for t in range(1,L):
            os = self.obs_dict[Osequence[t]]

            for j in range(S):
                # Calculate new delta using max of all Emissions, transitions, and previous deltas
                delta[j,t] = self.B[j, os] * np.max(self.A[:, j] * delta[:, t-1])
                # Save max state
                states[j,t] = np.argmax(self.A[:, j] * delta[:, t-1])

        x = int(np.argmax(delta[:,L-1]))
        for key, value in self.state_dict.items():
            if value == x:
                path.append(key)
        
        for i in range(L-1,0,-1):
            x = int(states[x,i])
            for key, value in self.state_dict.items():
                if value == x:
                    path.append(key)


        path.reverse()
        ###################################################
        return path
