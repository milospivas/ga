# -*- coding: utf-8 -*-
"""Genetski_algoritmi_mk4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JBE5wctP_Lf1M0jNINAhIlLqxO3o4U8J

# Primer
## Tražimo maksimum funkcije:
$$
f(x,y)=sin(\omega x)^2 cos(\omega y)^2 e^{\frac{x+y}{\sigma}}
$$

## Vizualizacija preko konturnih linija:
"""

# %matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML

# parameters for the 'hill space'

# class Parameters:

# parameters of the SearchSpace
class SearchSpace:
    dim = 2

    # fitness function parameters
    w = 1
    num_peaks_sqrt =  4     #@param {type : 'integer'}
    sigma = 10 + 2.5*(num_peaks_sqrt - 4)

    num_peaks = num_peaks_sqrt**2
    n_per_peak = 25
    with_moat = False

    # search space parameters
    start = 0
    stop = num_peaks_sqrt*np.pi
    N = num_peaks_sqrt * n_per_peak
    
    f_normal = lambda x,y : np.power(np.sin(SearchSpace.w*x) * np.sin(SearchSpace.w*y),2) * np.exp((x+y)/SearchSpace.sigma)

    @classmethod
    def f_moat(cls, xv,yv):    
        if xv.shape != yv.shape:
            print("Warning! Different shapes of x and y.")

        s = cls.stop
        nps = cls.num_peaks_sqrt
        z = np.empty(xv.shape, dtype = float)
        if (len(xv.shape) < 2):
            nx = xv.shape[0]
            for i in range(nx):
                x,y = xv[i], yv[i]
                z[i] = cls.f_normal(x,y) if ((x < s/nps) or (y < s/nps) or ((x > (1-1/nps)*s) and (y > (1-1/nps)*s))) else 0
        else:
            ny,nx = xv.shape
            for i in range(nx):
                for j in range(ny):
                    x,y = xv[i,j], yv[i,j]
                    z[i,j] = cls.f_normal(x,y) if ((x < s/nps) or (y < s/nps) or ((x > (1-1/nps)*s) and (y > (1-1/nps)*s))) else 0
        
        return z
    
    @classmethod
    def plot2d(cls, ax):
        # making the hill
        x = np.linspace(SearchSpace.start, SearchSpace.stop, num=SearchSpace.N)
        xv, yv = np.meshgrid(x, x)
        f = cls.f_moat if cls.with_moat else cls.f_normal
        z = f(xv,yv) # (np.sin(w*x)**2) * (np.sin(w*y)**2) * np.exp((x+y)/sigma)
        _ = ax.contour(x.reshape(-1), x.reshape(-1), z, levels=12, linewidths=0.5, colors='k', extend='both')
        _ = ax.contourf(x.reshape(-1), x.reshape(-1), z, levels=12, cmap='PuBu', extend='both')

        pass
    


SearchSpace.with_moat = False #@param {type:"boolean"}

fig, ax = plt.subplots()
ax.set_aspect('equal')
SearchSpace.plot2d(ax)
fig.show()

"""## Hromozom:"""

class Chromosome:
    num_genes = 2
    genes_lower_bound = SearchSpace.start * np.ones(num_genes)
    genes_upper_bound = SearchSpace.stop * np.ones(num_genes)
    p_mutation = 0.1
    p_crossover = 0.75
    mutation_step = 3
    fitness = SearchSpace.f_normal # fitness function
    hash_precision = 4
    
    @classmethod
    def plot(cls, c, ax):
        ax.plot(c[0], c[1], 'ko', ms=3)# 1+2*c[-1])
    
    @classmethod
    def prehash(cls, c):
        c = np.around(10**Chromosome.hash_precision*c, decimals=0)
        s = ""
        for g in c[:Chromosome.num_genes]:
            s += str(g)[:-2]
        return s

"""## Definisanje populacije"""

class Population:
    default_cap = 100
    start = SearchSpace.start
    stop = SearchSpace.stop/SearchSpace.num_peaks_sqrt
    p_selection_in_tournament = 0
    
    # new random population
    def __init__(self, cap=None):
        if cap is None:
            cap = Population.default_cap
            
        self.cap = cap
        self.max_size = 2*cap**2
        self.default_tournament_size = cap
        
        self.hash = set()
        self.gen = np.empty((self.max_size, Chromosome.num_genes+1), dtype=float)
        self.last = 0
        while self.last < self.cap:
            self.add(Population.start+(Population.stop - Population.start)*np.random.rand(Chromosome.num_genes))
        
    def size(self):
        return self.last
    
    def __len__(self):
        return self.last
    
    def add(self, c):
        if self.last == self.max_size:
            print("ERROR. Adding to a full generation matrix")
            return
        if (c[0] < Population.start) or (c[1] < Population.start):
            print("SOMETHING IS VERY WRONG")

        s = Chromosome.prehash(c)
        if not s in self.hash:
            self.gen[self.last, :Chromosome.num_genes] = c
            self.last += 1
            self.hash.add(s)
#             print('added a gene')

        
    def __str__(self):
        s = ""
        for c in self.gen:
            s += str(c) + '\n'
        return s
            
    def plot(self, ax):
        for c in self.gen[:self.last, :2]:
            Chromosome.plot(c, ax)
        pass

    # monogenic mutation
    def mutate(self):
        old_size = self.size()
        
        for i in range(old_size):
            if np.random.rand() < Chromosome.p_mutation:
                c = np.copy(self.gen[i,:Chromosome.num_genes]) #Chromosome.try_mutate(self.generation[i])
                gene_idx = np.random.randint(0, Chromosome.num_genes)
                step = (2*np.random.rand()-1)*Chromosome.mutation_step
                c[gene_idx] += step
                if c[gene_idx] < SearchSpace.start:
                    c[gene_idx] -= 2*step
                if c[gene_idx] > SearchSpace.stop:
                    c[gene_idx] -= 2*step
                self.add(c)

    def crossover(self):
        old_size = self.last
        for i in range(old_size):
            for j in range(i+1, old_size):
                if (np.random.rand() < Chromosome.p_crossover):
                    c_idx = np.random.randint(1,Chromosome.num_genes)
                    c1 = np.copy(self.gen[i,:Chromosome.num_genes])
                    c1[c_idx:] = self.gen[j,c_idx:Chromosome.num_genes]
                    c2 = np.copy(self.gen[j,:Chromosome.num_genes])
                    c2[c_idx:] = self.gen[i,c_idx:Chromosome.num_genes]

                    self.add(c1)
                    self.add(c2)
    
    def calc_fitness(self):
        # fitness calc
        self.gen[:self.last, 2] = Chromosome.fitness(self.gen[:self.last, 0], self.gen[:self.last,1])

    def sort(self):
        # fitness based sort
        idx = self.gen[:self.last, 2].argsort()[::-1]
        self.gen[:self.last] = self.gen[idx]

    def preselect(self):
        self.calc_fitness()
        self.sort()

    # TODO correct
    def truncate(self, start=None):
        ''' Simple truncation '''
        if start is None:
            start = self.cap
        if start != self.cap:
            print("Warning! Number of selected chromosomes is different from popcap.")
            return
        # remove elements from hashset
        for c in self.gen[start:self.last, :Chromosome.num_genes]:
            s = Chromosome.prehash(c)
            self.hash.remove(s)
        # move buffer tail to popcap
        self.last = self.cap
    
    # TODO expand docstring
    def rearange(self, mask):
        ''' Rearange chromosomes in generation according to mask. '''
        N = np.count_nonzero(mask)
        
        if N != self.cap:
            print("Warning! Number of selected chromosomes is different from popcap.")
        
        aux_mask = np.zeros(self.max_size, dtype=bool)
        aux_mask[:self.last] = True

        # TODO delete
        # print(self.last-np.count_nonzero(mask)-np.count_nonzero(mask!=aux_mask))

        self.gen[:N], self.gen[N:self.last] = self.gen[mask], self.gen[mask!=aux_mask]
        return N
    
    # TODO write docstring
    def keep(self, mask):
        N = self.rearange(mask)
        self.truncate(N)
    
    def truncation_select(self):
        ''' Keeps the best individuals. '''
        if self.size() > self.cap:
            self.preselect()
            self.truncate()
        
    def fps_select(self):
        ''' Fitness Proportionate Selection, aka "Roulette wheel" selection '''
        if self.size() > self.cap:
            self.preselect()

            f = np.copy(self.gen[:self.last, Chromosome.num_genes]) # first column after chromosome values is fitness
            f = f / np.sum(f)
            for i in range(1,self.last):
                f[i] += f[i-1]

            mask = np.zeros(self.max_size, dtype=bool)
            n_selected = 0
            while n_selected != self.cap:
                rnd = np.random.rand()
                idx = np.searchsorted(f, rnd) # binary search
                if not mask[idx]:
                    mask[idx] = True
                    n_selected += 1
            self.keep(mask)
    
    # TODO implement
    def sus_select(self):
        ''' Stoachastic Universal Sampling method. '''
        if self.size() > self.cap:
            self.preselect()

            
            f = np.copy(self.gen[:self.last, Chromosome.num_genes]) # first column after chromosome values is fitness
            F = np.sum(f) # should be 1
            N = self.cap
            P = F/N
            mask = np.zeros(self.max_size, dtype=bool)
            
            Start = P*np.random.rand()
            Pointers = Start + P*np.arange(N)

            # cumulative sum
            # f = f / F
            for i in range(1,self.last):
                f[i] += f[i-1]

            i = 0
            for P in Pointers:
                while (f[i] < P) and (self.last != i):
                    i += 1
                if not mask[i]:
                    mask[i] = True
                else:
                    # edge case where the collision happens
                    # pick a side randomly
                    if np.random.rand() < 0.5:
                        # go forward:
                        for j in range(i+1, i+N):
                            k = j % N
                            if not mask[k]:
                                mask[k] = True   
                    else:
                        # or, go back
                        for j in range(i-1, i-N, -1):
                            k = j % N
                            if not mask[k]:
                                mask[k] = True          
            self.keep(mask)          

    def tournament_select(self, tsize=None):
        ''' Tournament selection. '''
        if self.size() > self.cap:
            self.preselect()
            mask = np.zeros(self.max_size, dtype=bool)

            if tsize is None:
                tsize = self.default_tournament_size

            n_selected = 0
            while n_selected < self.cap:
                tournament = np.sort(np.random.choice(self.last, tsize))
                for idx in tournament:
                    if (np.random.rand() < Population.p_selection_in_tournament) and (not mask[idx]):
                        mask[idx] = True
                        n_selected += 1
                        break                        
            self.keep(mask)

    def select(self, type='tournament'):
        function = {'trunc':self.truncation_select,
                    'fps':self.fps_select,
                    'sus':self.sus_select,
                    'tournament':self.tournament_select,
                    }
        
        function[type]()
            
    def max(self):
        idx = np.argmax(self.gen[:self.last, Chromosome.num_genes])
        f = self.gen[idx, Chromosome.num_genes]
        c = self.gen[idx, :Chromosome.num_genes]
        return f, c

"""## Pokretanje simulacije"""

def run_ga(p, num, selection):
    max_fitness = np.zeros(num, dtype=float)
    max_chromosome = np.zeros((num, Chromosome.num_genes), dtype=float)
    for i in range(num):
        p.crossover()
        # print('        size after crossover: ', p.size())
        p.mutate()
        # print('            size after mutation: ', p.size())
        p.select(selection)
        # print('                size after selection: ', p.size())
        max_fitness[i], max_chromosome[i] = p.max()
        # print("Generation", i," max fitness =", max_fitness[i])
    return max_fitness, max_chromosome

def plot_ga(max_fitness, max_chromosome, num, selection):
    fig = plt.figure()
    ax = plt.subplot(1, 2, 1)
    ax.set_aspect('equal')
    SearchSpace.plot2d(ax)
    p.plot(ax)
    plt.title(selection+", "+str(num)+" generacija")
    ax = plt.subplot(1, 2, 2)
    ax.plot(np.arange(num), max_fitness)
    plt.xlabel("#generacija")
    plt.title("max fitness")

def run_and_plot(p, num, selection):
    max_fitness, max_chromosome = run_ga(p, num, selection)
    plot_ga(max_fitness, max_chromosome, num, selection)

    return max_fitness, max_chromosome

with_moat = False #@param {type:"boolean"}
SearchSpace.with_moat = with_moat
Chromosome.fitness = SearchSpace.f_moat if with_moat else SearchSpace.f_normal
Chromosome.p_mutation = 0.1   #@param {type:'number'}
Chromosome.p_crossover = 0.75     #@param {type:'number'}
Chromosome.mutation_step = 3.15 #@param {type:'number'}
Population.p_selection_in_tournament = 0.8 #@param {type: 'number'}
num_generations =  42#@param {type: 'integer'}
popcap = 200 #@param {type: 'integer'}
# selection = "trunc" #@param {type:"string"}


p = Population(popcap)
for selection in ['fps', 'sus', 'tournament', 'trunc']:
    max_fitness, max_chromosome = run_and_plot(p, num_generations, selection)