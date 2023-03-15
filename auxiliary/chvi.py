# import pygmo
from sklearn.preprocessing import normalize

def calculate_contrib_hvi(objectives):
    raise NotImplemented('CUrrenly not implemented')
#     hv = pygmo.hypervolume(objectives) 
#     reference_points = [4,10,10] 
#     hv.compute(reference_points)
#     return hv.contributions(reference_points)

def calculate_normalised_contrib_hvi(objectives):
    raise NotImplemented('CUrrenly not implemented')
#     objectives = normalize(objectives, axis=0, norm='max')
#     hv = pygmo.hypervolume(objectives) 
#     reference_points = [4,16,8] 
#     hv.compute(reference_points)
#     return hv.contributions(reference_points)