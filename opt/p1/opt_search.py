# -*- coding: utf-8 -*-
"""
Tarea 1. 2. Actividades:
 - Codificar el método de búsqueda Dicotómica.
 - Codificar el método de búsqueda de la Sección Áurea.

@author: <maria.barrosoh@estudiante.uam.es>
"""
import numpy as np


def dichotomic_search(func, lower_bound, upper_bound, uncertainty_length, epsilon, max_iter = 100):

	for it_k in range(max_iter):
		
		if upper_bound - lower_bound < uncertainty_length:
			break

		mid_point = 0.5 * (upper_bound + lower_bound)
		lambda_k = mid_point - epsilon
		mu_k = mid_point + epsilon

		if func(lambda_k) < func(mu_k):
			upper_bound = mu_k
		else:
			lower_bound = lambda_k

	if it_k == max_iter:
		print(f'Dichotomic search has divergenced.')

	return lower_bound, upper_bound, it_k

def golden_search(func, lower_bound, upper_bound, uncertainty_length, max_iter = 100):
	
	golden_alpha = 0.618
	
	def update_lambda(lower_bound, upper_bound, golden_alpha): return lower_bound + (1 - golden_alpha)*(upper_bound - lower_bound)
	def update_mu(lower_bound, upper_bound, golden_alpha): return lower_bound + golden_alpha*(upper_bound - lower_bound)
	
	lambda_k = update_lambda(lower_bound, upper_bound, golden_alpha)
	mu_k = update_mu(lower_bound, upper_bound, golden_alpha)

	for it_k in range(max_iter):

		if upper_bound - lower_bound < uncertainty_length:
			break

		if func(lambda_k) > func(mu_k):
			lower_bound = lambda_k
			lambda_k = mu_k
			mu_k = update_mu(lower_bound, upper_bound, golden_alpha)
		else:
			upper_bound = mu_k
			mu_k = lambda_k
			lambda_k = update_lambda(lower_bound, upper_bound, golden_alpha)
		

	if it_k == max_iter:
		print(f'Dichotomic search has divergenced.')

	return lower_bound, upper_bound, it_k


# https://programmerclick.com/article/68471751254/
def HJ_search(func, xk, lambd = 1.0, alpha = 1.0, beta = 0.5, epsilon = 1e-3, max_iter = 100):
	"""
		xk : punto inicial 
		lambd : paso de búsqueda de detección inicial
		alfa (alfa> = 1) : factor de aceleración
		beta (0 <beta <1) : tasa de reducción beta
		epsilon  (epsilon> 0) : error permisible
	"""
	yk = xk.copy()
	n = len(xk)
	k = 1
	while lambd > epsilon:

		if k == max_iter:
			print(f'Hooke and Jeeves search has divergenced.')
			return xk, k

		for i in range(n):
			d = np.zeros(n)
			d[i] = 1

			if func(yk + lambd * d) < func(yk):
				yk = yk + lambd * d
			elif func(yk - lambd * d) < func(yk):
				yk = yk - lambd * d

		if func(yk) < func(xk):
			xk = yk
			yk = yk + alpha * (yk - xk)
		else:
			lambd, yk = lambd * beta, xk

		k += 1

	return xk, k
