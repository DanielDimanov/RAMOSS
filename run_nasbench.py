# python run_nashbench.py --pop_size 20 --generations 50 --exp_name NASBENCH_20_50_run
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
from ne import PymooNASBenchGenome


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
parser.add_argument("--pop_size")
parser.add_argument('--batch_size')
parser.add_argument("--generations")
parser.add_argument("--number_of_runs")
parser.add_argument("--seed")
parser.add_argument("--exp_name")
args = parser.parse_args()

METRIC_OPS = [operator.__lt__, operator.__gt__]
METRIC_OBJECTIVES = [min, max]


BATCH_NORMALIZATION = False


DATASET = 'cifar10'
TYPE_PROBLEM = 'classification'
BATCH_SIZE = 0
EXPERIMENT_NAME = 'exp404'
NASWOT = False
SYNFLOW = False




# TODO make as user input
if(TYPE_PROBLEM =='classification'):
      BATCH_NORMALIZATION = True


if(args.pop_size):
  POP_SIZE = int(args.pop_size)
else:
  POP_SIZE = 20


if(args.generations):
  GENERATIONS = int(args.generations)
else:
  GENERATIONS = 20
  
if(args.number_of_runs):
  NUMBER_OF_RUNS = int(args.number_of_runs)
else:
  NUMBER_OF_RUNS = 1
if(args.seed):
  SEED = int(args.seed)
else:
  SEED = random.randint(1,100)
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
seed = SEED
seeds_used = []
result = None
from ne import PymooNASBenchGenome
import numpy as np

from nasbench import api
NASBENCH_TFRECORD  = "./nasbench_full.tfrecord"

nasbench_api = api.NASBench(NASBENCH_TFRECORD)
# nasbench_api = None

for run_num in range(NUMBER_OF_RUNS):
  seeds_used.append(seed)
  np.random.seed(0)
  np.random.seed(seed)
  random.seed(seed)
  tf.random.set_seed(seed)
    # 6->10
    # 256 -> 512

  from tensorflow.keras.datasets import cifar10
  input_shape = (32,32,3)
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  x_train = x_train.reshape(x_train.shape[0], 32, 32, 3).astype('float32') / 255
  x_test = x_test.reshape(x_test.shape[0], 32, 32, 3).astype('float32') / 255
  n_classes = 10
  batch_size = 32
  TRAIN_WITH_LOGITS = False
  POP_SIZE = 20
  GENERATIONS = 50

  problem = PymooNASBenchGenome(max_conv_layers=6, 
                            max_filters=128,
                            input_shape=input_shape,
                            n_classes=n_classes,
                            dropout=False,
                            type_problem='nasbench',
                            batch_size = batch_size,
                            TRAIN_WITH_LOGITS = TRAIN_WITH_LOGITS,
                            nasbench_api = nasbench_api
                            )
  # from pymoo.algorithms.nsga2 import NSGA2
  from pymoo.algorithms.moo.nsga2 import NSGA2
  from pymoo.factory import get_sampling, get_crossover, get_mutation,get_reference_directions, get_selection

  ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=10)
  # from pymoo.algorithms.nsga3 import NSGA3
  algorithm = NSGA2(
        pop_size=POP_SIZE,
        n_offsprings=None,
        sampling=get_sampling("int_random"),
        crossover=get_crossover("int_k_point", n_points=3,prob=0.9),
        mutation=get_mutation("int_pm",eta=50, prob=0.02),
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
  
  with open(r"NASBENCHl{}-{}-F.pkl".format(DATASET,EXPERIMENT_NAME),'wb') as f:
      pickle.dump(res.F,f)

  with open(r"NASBENCHl{}-{}-X.pkl".format(DATASET,EXPERIMENT_NAME),'wb') as f:
      pickle.dump(res.X,f)



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
