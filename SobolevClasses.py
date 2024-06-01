# Imports 
import numpy as np
import sympy as sp
from findiff import FinDiff
from itertools import product

class SobolevSpace:
    def __init__(self, domain, r, p):
        """
        Initialize the Sobolev Space with a given domain, derivative order r, and norm p.

        Parameters:
        - domain: a list of tuples, each specifying the domain of integration for each variable.
        - r: the order of derivatives to consider.
        - p: the norm order (p=2 for L^2 norm, etc.).
        """
        self.domain = domain
        self.r = r
        self.p = p

    def multi_indices_upto_order(self, order):
        # Generate all tuples of dimension dim with entries summing to less than or equal to order
        dim = len(self.domain)
        def multi_index_order(dim, total):
            return [tup for tup in product(range(total + 1), repeat=dim) if sum(tup) == total]
        
        return [tup for i in range(order + 1) for tup in multi_index_order(dim, i)]
     
    def multi_index_derivatives(self, f, variables, order):
        if order == 0:
            return f
        else:
            derivative = f
            for var, power in zip(variables, order):
                derivative = sp.diff(derivative, var, power)
            return derivative
        
    def all_derivatives(self, f, variables, order):
        # Generate all derivatives of f up to total order r
        derivatives = []
        for order in self.multi_indices_upto_order(self.r):
            derivative = self.multi_index_derivatives(f, variables, order)
            derivatives.append(derivative)
        return derivatives


    def norm(self, f, variables):
        """
        Calculate the Sobolev norm of a function f in this space.

        Parameters:
        - f: the function, expressed as a SymPy expression in terms of variables.
        - variables: a list of SymPy symbols representing the variables of f.

        Returns:
        - The Sobolev norm of the function.
        """
        # Generate all derivatives of f up to total order r
        derivatives = []
        for order in self.multi_indices_upto_order(self.r):
            derivative = self.multi_index_derivatives(f, variables, order)
            # print(f"Derivative of order {order}: {derivative}")
            derivatives.append(derivative)

        # Print length of derivatives
        # print(f"Length of derivatives: {len(derivatives)}")
        
        # Calculate the L^p norm of each derivative
        Lp_norms = []
        for derivative in derivatives:
            integrand = sp.Abs(derivative)**self.p
            integrated = integrand
            for var, dom in zip(variables, self.domain):
                integrated = sp.integrate(integrated, (var, dom[0], dom[1]))
                # print(f"Integrating {integrated} over {var} in {dom}")
            Lp_norms.append(sp.root(integrated, self.p))
        
        # print(f"Lp norms: {Lp_norms}")

        # Compute the generalized Sobolev norm
        sobolev_norm = sp.Pow(sum(norm**self.p for norm in Lp_norms), 1/self.p)
        
        return sobolev_norm
    
class DiscreteSobolevSpace:
    def __init__(self, domain, r, p, grid_res=None):
        """
        Initialize the Discrete Sobolev Space with a given domain, derivative order r, norm p, and number of grid points N.

        Parameters:
        - domain: a list of tuples, each specifying the domain of integration for each variable.
        - r: the order of derivatives to consider.
        - p: the norm order (p=2 for L^2 norm, etc.).
        - N: the number of grid points in each dimension.
        """
        self.domain = domain
        self.r = r
        self.p = p
        self.grid_res = grid_res if grid_res else [100]*len(domain)

    def multi_indices_upto_order(self, order):
        dim = len(self.domain)
        def multi_index_order(dim, total):
            return [tup for tup in product(range(total + 1), repeat=dim) if sum(tup) == total]
        
        multi_indices = [tup for i in range(order + 1) for tup in multi_index_order(dim, i)]
        
        # Convert to FinDiff format
        def convert_to_findiff(deriv_tuple, step_sizes=self.grid_res):
            result = []
            
            for i, order in enumerate(deriv_tuple):
                if order > 0:
                    # Append [axis index, step size, derivative order]
                    result.append((i, (self.domain[i][1] - self.domain[i][0])/self.grid_res[i], order))
            
            return result
        
        # print("Multi-indices: ", multi_indices)
        # print("Converted: ", [convert_to_findiff(tup) for tup in multi_indices])
        return [convert_to_findiff(tup) for tup in multi_indices]
        
    
    def multi_index_derivatives(self, f, multi_index):
        # Multi-index of format: (dim, grid_spacing, order)

        if multi_index == []:
            grid = [np.linspace(dom[0], dom[1], res) for dom, res in zip(self.domain, self.grid_res)]
            vals = f(*np.meshgrid(*grid, indexing='ij'))
            return vals
        
        # Create meshgrid for evaluation
        grid = [np.linspace(dom[0], dom[1], res) for dom, res in zip(self.domain, self.grid_res)]
        vals = f(*np.meshgrid(*grid, indexing='ij'))
        # Create derivative operator
        deriv = FinDiff(*multi_index)

        return deriv(vals)
    
    def all_derivatives_upto(self, f):
        # Generate all derivatives of f up to total order r
        orders = self.multi_indices_upto_order(self.r)
        derivatives = []
        for order in orders:
            # print(f"Order: {order}")
            derivative = self.multi_index_derivatives(f, order)
            derivatives.append(derivative)
        # print(f"Length of derivatives: {len(derivatives)}")
        return derivatives
    
    def norm(self, f):

        # Generate all derivatives of f up to total order self.r
        derivatives = self.all_derivatives_upto(f)

        # Calculate the L^p norm of each derivative
        Lp_norms = []
        for derivative in derivatives:
            integrand = np.abs(derivative)**self.p
            integrated = integrand
            for dom, res in zip(self.domain, self.grid_res):
                integrated = np.trapz(integrated, dx=(dom[1] - dom[0])/res, axis=0)
            Lp_norms.append(np.power(integrated, 1/self.p))

        # print(f"Lp norms: {Lp_norms}")

        # Compute the generalized Sobolev norm
        sobolev_norm = np.power(np.sum(np.power(Lp_norms, self.p)), 1/self.p)

        return sobolev_norm