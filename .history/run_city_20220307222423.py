# from __future__ import print_function

# d_n=$(date +'%Y_%m_%d_%H:%M')
# python run_city.py --max_conv_layers 25 --batch_size 4 --exp_name ${d_n}20p_20g_25l_pss10 --epochs 10 --dataset city --pop_size 20 --generations 20 --number_of_runs 1 --seed 82 --type_problem ss --pss 10
import sys
sys.path.append("..")

import operator
import random
import pickle
import math
import traceback
import os
import argparse
import numpy as np
from ne import PymooGenomeReduced


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)

#Used for calculating contributing HVI and normalised CHVI
from auxiliary.chvi import calculate_contrib_hvi, calculate_normalised_contrib_hvi

from datasets import get_dataset


from tensorflow.keras import backend as K

parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument("--pop_size")
parser.add_argument('--batch_size')
parser.add_argument("--generations")
parser.add_argument("--number_of_runs")
parser.add_argument("--fitness_fn")
parser.add_argument("--seed")
parser.add_argument("--pss")
parser.add_argument('--nasbench')
parser.add_argument('--type_problem')
parser.add_argument('--exp_name')
parser.add_argument('--naswot')
parser.add_argument('--epochs')
parser.add_argument('--max_conv_layers')
parser.add_argument('--synflow')
args = parser.parse_args()

METRIC_OPS = [operator.__lt__, operator.__gt__]
METRIC_OBJECTIVES = [min, max]


BATCH_NORMALIZATION = False

# Setting default values and fetching arguments
DATASET = args.dataset
TYPE_PROBLEM = 'classification'
BATCH_SIZE = 0
EXPERIMENT_NAME = 'exp404'
NASWOT = False
SYNFLOW = False
EPOCHS = 3
MAX_CONV_LAYERS = 20

# Weather to use NASWOT or not
if(args.naswot):
  NASWOT = True
else:
  NASWOT = False

# Weather to use SYNFLOW or not
if(args.synflow):
      SYNFLOW = True
else:
  SYNFLOW = False

# Specify the problem type possible ones for now: 'classification', 'ss' - semantic segmentation, 'ae' - autoencoder 
if (args.type_problem):
  TYPE_PROBLEM = args.type_problem

# TODO make as user input
if(TYPE_PROBLEM =='classification'):
      BATCH_NORMALIZATION = True


if(args.pop_size):
  POP_SIZE = int(args.pop_size)
else:
  POP_SIZE = 20

if(args.pop_size):
  BATCH_SIZE = int(args.batch_size)
else:
  BATCH_SIZE = 32

if(args.generations):
  GENERATIONS = int(args.generations)
else:
  GENERATIONS = 20
  
if(args.number_of_runs):
  NUMBER_OF_RUNS = int(args.number_of_runs)
else:
  NUMBER_OF_RUNS = 1
if(args.fitness_fn):
  FITNES_FN = args.fitness_fn
else:
  FITNES_FN = 'CHVI'
if(args.seed):
  SEED = int(args.seed)
else:
  SEED = random.randint(1,100)
if(args.epochs):
  EPOCHS = int(args.epochs)
if(args.max_conv_layers):
  MAX_CONV_LAYERS = int(args.max_conv_layers)
if(args.pss):
  PSS = int(args.pss)
else:
  PSS = False
if(args.nasbench):
  NASBENCH = True
else:
  NASBENCH = False

if(args.exp_name):
  EXPERIMENT_NAME = args.exp_name
else:
  raise Exception('Please specify experiment name to avoid mess!')
print(f'==== RUNNING EXPERIMENT {EXPERIMENT_NAME} =====')


# Verbosity is now 0

physical_devices = tf.config.experimental.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  print("Invalid device or cannot modify virtual devices once initialized.")
  pass


# **Prepare dataset**



fitnes_fns = dict()
fitnes_fns['CHVI'] = calculate_contrib_hvi
fitnes_fns['CHVI_norm'] = calculate_normalised_contrib_hvi


dataset, input_shape,n_classes, TRAIN_WITH_GEN, TRAIN_WITH_LOGITS, batch_size, normalize, multilabel = get_dataset(DATASET,batch_size=BATCH_SIZE, PSS=PSS)
seed = SEED
seeds_used = []
result = None

for run_num in range(NUMBER_OF_RUNS):
  seeds_used.append(seed)
  np.random.seed(0)
  np.random.seed(seed)
  random.seed(seed)
  tf.random.set_seed(seed)
    # 6->10
    # 256 -> 512
  problem = PymooGenomeReduced(max_conv_layers=MAX_CONV_LAYERS, 
                                  max_dense_layers=0,
                                  max_nodes=256,
                                  max_filters=512,
                                  input_shape=input_shape,
                                  n_classes=n_classes,
                                  dropout=False,
                                  type_problem=TYPE_PROBLEM,
                                  batch_size = BATCH_SIZE,
                                  TRAIN_WITH_LOGITS = TRAIN_WITH_LOGITS,
                                  batch_normalization=BATCH_NORMALIZATION,
                                  NASWOT=NASWOT,
                                  SYNFLOW=SYNFLOW
                                  )
  problem.feed_data(
        train_with_gen=TRAIN_WITH_GEN,
        dataset = dataset,
        num_generations=GENERATIONS,
        pop_size = POP_SIZE,
        pss = PSS,
        multilabel=True,
        metric = 'loss',
        batch_size = BATCH_SIZE,
        gen_to_tf_data = True,
        epochs=EPOCHS,
        normalize = normalize)
  # from pymoo.algorithms.moead import MOEAD
  from pymoo.algorithms.moo.nsga2 import NSGA2
  from pymoo.factory import get_sampling, get_crossover, get_mutation,get_reference_directions, get_selection

  ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=10)
  # from pymoo.algorithms.nsga3 import NSGA3
  algorithm = NSGA2(
        pop_size=POP_SIZE,
        n_offsprings=None,
        sampling=get_sampling("int_random"),
        crossover=get_crossover("int_k_point", n_points=3,prob=0.9),
        mutation=get_mutation("int_pm",eta=0.01, prob=0.05),
        eliminate_duplicates=True
    )

  from pymoo.factory import get_termination

  termination = get_termination("n_gen", GENERATIONS)

  from pymoo.optimize import minimize

  res = None
  try:
    res = minimize(problem,
                    algorithm,
                    termination,
                    seed=SEED,
                    save_history=True,
                    verbose=True)
    print('==========================')
    print(res.F)
    print('==========================')
    print(res.X)
    print('==========================')
  except Exception as error:
    traceback.print_exc()
    print(error)
    pass

  
  import pickle
  
  with open(r"l{}-{}-F.pkl".format(DATASET,EXPERIMENT_NAME),'wb') as f:
      pickle.dump(res.F,f)

  with open(r"l{}-{}-X.pkl".format(DATASET,EXPERIMENT_NAME),'wb') as f:
      pickle.dump(res.X,f)

# Code taken and adapted from https://pymoo.org/

  from pymoo.visualization.scatter import Scatter

  # get the pareto-set and pareto-front for plotting
  ps = problem.pareto_set(use_cache=False, flatten=False)
  pf = problem.pareto_front(use_cache=False, flatten=False)

  # Objective Space
  plot = Scatter(title = "Objective Space")
  plot.add(res.F)
  if pf is not None:
      plot.add(pf, plot_type="line", color="black", alpha=0.7)
  plot.show()



  n_evals = []    # corresponding number of function evaluations\
  F = []          # the objective space values in each generation
  cv = []         # constraint violation in each generation


  # iterate over the deepcopies of algorithms
  for algorithm in res.history:

      # store the number of function evaluations
      n_evals.append(algorithm.evaluator.n_eval)

      # retrieve the optimum from the algorithm
      opt = algorithm.opt

      # store the least contraint violation in this generation
      cv.append(opt.get("CV").min())

      # filter out only the feasible and append
      feas = np.where(opt.get("feasible"))[0]
      _F = opt.get("F")[feas]
      F.append(_F)

  import matplotlib.pyplot as plt

  k = min([i for i in range(len(cv)) if cv[i] <= 0])
  first_feas_evals = n_evals[k]
  print(f"First feasible solution found after {first_feas_evals} evaluations")

  plt.plot(n_evals, cv, '--', label="CV")
  plt.scatter(first_feas_evals, cv[k], color="red", label="First Feasible")
  plt.xlabel("Function Evaluations")
  plt.ylabel("Constraint Violation (CV)")
  plt.legend()
  plt.show()

  import matplotlib.pyplot as plt
  from pymoo.performance_indicator.hv import Hypervolume

  # MODIFY - this is problem dependend
  ref_point = np.array([1.0, 1.0])

  # create the performance indicator object with reference point
  metric = Hypervolume(ref_point=ref_point, normalize=False)

  # calculate for each generation the HV metric
  hv = [metric.calc(f) for f in F]

  # visualze the convergence curve
  plt.plot(n_evals, hv, '-o', markersize=4, linewidth=2)
  plt.title("Convergence")
  plt.xlabel("Function Evaluations")
  plt.ylabel("Hypervolume")
  plt.show()

  from pymoo.util.running_metric import RunningMetric

  running = RunningMetric(delta_gen=GENERATIONS/2,
                          n_plots=2,
                          only_if_n_plots=True,
                          key_press=False,
                          do_show=True)

  for algorithm in res.history[:GENERATIONS]:
      running.notify(algorithm)
