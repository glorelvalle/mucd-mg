# -*- coding: utf-8 -*-
"""
Tarea 1. 2. Actividades:
 - Codificar el método de búsqueda Dicotómica.
 - Codificar el método de búsqueda de la Sección Áurea.

@author: <maria.barrosoh@estudiante.uam.es>
"""


def dichotomic_search(func, lower_bound, upper_bound, uncertainty_length, epsilon, max_iter = 100):

	for k in range(max_iter):
		
		if upper_bound - lower_bound < uncertainty_length:
			break

		mid_point = 0.5 * (upper_bound + lower_bound)
		lambda_k = mid_point - epsilon
		mu_k = mid_point + epsilon

		if func(lambda_k) < func(mu_k):
			upper_bound = mu_k
		else:
			lower_bound = lambda_k

	if k == max_iter:
		print(f'Dichotomic search has divergenced.')

	return lower_bound, upper_bound

def golden_search(func, lower_bound, upper_bound, uncertainty_length, max_iter = 100):
	
	golden_alpha = 0.618
	
	def update_lambda(lower_bound, upper_bound, golden_alpha): return lower_bound + (1 - golden_alpha)*(upper_bound - lower_bound)
	def update_mu(lower_bound, upper_bound, golden_alpha): return lower_bound + golden_alpha*(upper_bound - lower_bound)
	
	lambda_k = update_lambda(lower_bound, upper_bound, golden_alpha)
	mu_k = update_mu(lower_bound, upper_bound, golden_alpha)

	for k in range(max_iter):

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
		

	if k == max_iter:
		print(f'Dichotomic search has divergenced.')

	return lower_bound, upper_bound