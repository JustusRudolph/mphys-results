import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import os, sys
import argparse

cwd = os.getcwd()  # for any future imports

def parse_args():
  """
  Go through all possible arguments to the programme and returns the args object
  """
  descr = "Analysing file to output from traccc_seq_example_cuda"
  parser = argparse.ArgumentParser(description=descr)
  parser.add_argument("-s", "--single_module", default=0)
  parser.add_argument("-p", "--path", default="out.txt")
  args = parser.parse_args()

  return args


def show_module_cells(cells, activation_vals, moduleNO, cell_cluster_numbers=None):
  """
  This function plots all the cells in a module as a heatmap based
  on the value of the activation value of the cells. Since cells are
  stored as a sparse matrix in an array, this function will autofill all
  other parts of the matrix with zeroes for the heatmap.

  cells is np.array otherwise this doesnt work

  cell_cluster_numbers are essentially labels for the cells, if not given
  then they are simply ignored. The cluster number is to the corresponding
  cell at the same index.
  """
  x_vals = cells[:, 0]
  y_vals = cells[:, 1]
  x_range = (np.min(x_vals), np.max(x_vals))
  y_range = (np.min(y_vals), np.max(y_vals))

  # remember that (x, y) is (col, row), so must instantiate
  # the grid size to be (y, x) for row and column dimensions
  # add +1 to account for array[max] not being out of range
  cell_grid = np.zeros((y_range[1]+1-y_range[0], x_range[1]+1-x_range[0]))
  
  for i, cell in enumerate(cells):
    x = cell[0]
    y = cell[1]
    act_val = activation_vals[i]

    # again, points in 2d array are (y,x). Also, need to shift
    # back down with the minimum to not be out of range
    cell_grid[y - y_range[0], x - x_range[0]] = act_val

  if (cell_cluster_numbers is not None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,15))
    # get the grid painted
    ax1.imshow(cell_grid, aspect=1/3, origin="lower")  # image origin normally top
    fig.colorbar(cm.ScalarMappable(), ax=ax1, label="Activation value")
    ax1.set_title(f"Module {moduleNO}, Number of cells activated: {len(cells)}")
  
    max_cluster_number = np.max(cell_cluster_numbers)
    min_cluster_number = np.min(cell_cluster_numbers)

    # Ensure distinguishable colours for the N different clusters
    cmap = cm.nipy_spectral  # pick colourmap
    n_plots = max_cluster_number - min_cluster_number + 1  # include boundaries
    colours = [cmap(i) for i in np.linspace(0, 1, n_plots)]
    #ax2.set_prop_cycle(colours)  # write the colours to the axis

    for cluster_number in range(min_cluster_number, max_cluster_number+1):
      cell_indices = np.where(cell_cluster_numbers == cluster_number)[0]
      relevant_cells = cells[cell_indices]

      x_vals_c = relevant_cells[:, 0]  # c in the end to denote for this cluster
      y_vals_c = relevant_cells[:, 1]

      # now shift them so they align with the placement in the module grid
      x_vals_c_shifted = x_vals_c - x_range[0]
      y_vals_c_shifted = y_vals_c - y_range[0]

      colour = colours[cluster_number-min_cluster_number]  # 0 indexed
      ax2.plot(x_vals_c_shifted, y_vals_c_shifted, "o", label=cluster_number,
               c=colour)
      
    ax2.legend()  # only want to set legend once

  else:  # only paint the grid (aka the module heatmap)
    plt.rcParams["figure.figsize"] = [20, 15]
    plt.imshow(cell_grid, aspect=1/5, origin="lower")  # image origin normally top
    plt.colorbar(label="Activation value")
    plt.title(f"Module {moduleNO}, Number of cells activated: {len(cells)}")
  
  plt.show()


def read_all_lines(lines):
  """
  Go through all lines from the output file, passed as a list of lines to
  this function. This function is easily appendable with more lists that become
  relevant to look at.

  Returns all relevant values that the user wants to look at, currently this
  includes the activation values, cell points (x,y) in the module, the number of
  all modules for where this particular cell point occurs, and in which thread
  this module number was run in.
  """
  activation_values = []
  cell_points_dict = {}  # keys are cell indices, values are size 2 tuples (x,y)
  module_numbers = []
  thread_points = []  # will be array of size 2 tuples (blockIdx, threadIdx)

  n_clusters = 0  # will be overwritten when reading "After" lines
  cell_cluster_numbers_dict = {}  # key is the cell number, the value is which
                                  # number cluster that cell is part of

  for i, line in enumerate(lines):
    if "activation" in line:
      end_of_global_idx = line.find(",")
      global_idx = int(line[5: end_of_global_idx])

      rest_of_line = line[end_of_global_idx+1:]  # skip comma

      start_block = rest_of_line.find("(") + 1
      end_block = rest_of_line.find(",")
      start_thread = end_block + 2
      end_thread = rest_of_line.find(")")

      blockIdx = int(rest_of_line[start_block: end_block])
      threadIdx = int(rest_of_line[start_thread: end_thread])
      thread_point = (blockIdx, threadIdx)

      rest_of_line = rest_of_line[end_thread+1:]  # move to next
      start_cell_idx = rest_of_line.find("x") + 2
      end_cell_idx = rest_of_line.find("a") - 1
      start_cellx = rest_of_line.find("(") + 1
      end_cellx = rest_of_line.find(",")
      start_celly = end_cellx + 2
      end_celly = rest_of_line.find(")")
      cell_idx = int(rest_of_line[start_cell_idx: end_cell_idx])
      cell_x = int(rest_of_line[start_cellx: end_cellx])
      cell_y = int(rest_of_line[start_celly: end_celly])
      cell_point = (cell_x, cell_y)

      rest_of_line = rest_of_line[end_celly + 1:]  # move to next
      start_activation = rest_of_line.find("n") + 3
      activation_value = float(rest_of_line[start_activation: -1])  # miss \n

      activation_values.append(activation_value)
      cell_points_dict[cell_idx] = cell_point
      module_numbers.append(global_idx)
      thread_points.append(thread_point)


    elif ("After" in line):
      relevant_line = line[line.find("Clusters"):]
      relevant_line = relevant_line[(relevant_line.find(":")+2):]
      # now relevant line starts from the number
      n_clusters_end = relevant_line.find(",")
      n_clusters = int(relevant_line[:n_clusters_end])

      cell_idx_start = n_clusters_end + 7
      cell_idx_end = relevant_line.find("b") - 1
      cluster_no_start = relevant_line.find("r") + 2

      cell_idx = int(relevant_line[cell_idx_start: cell_idx_end])
      cluster_no = int(relevant_line[cluster_no_start: -1])  # skip \n

      cell_cluster_numbers_dict[cell_idx] = cluster_no

  
  activation_values = np.array(activation_values)
  module_numbers = np.array(module_numbers)
  thread_points = np.array(thread_points)

  return (activation_values, cell_points_dict, module_numbers, thread_points,
          n_clusters, cell_cluster_numbers_dict)


def sort_dict_values_by_keys(dictionary):
  """
  Helper function which sorts the values in the dictionary by the keys
  and returns the sorted values

  This is used for the cell points & cluster number which can
  be linked by their index
  """
  keys = np.array(list(dictionary.keys()))
  values = np.array(list(dictionary.values()))
  values_sorted_indices = np.argsort(keys)
  values_sorted = values[values_sorted_indices]

  return values_sorted


def analyse_several(activation_values, cell_points_dict, module_numbers,
                    n_activated_cutoff=100):
  """
  If there are several modules being considered, store everything as arrays
  where sorting is relevant.

  Reads, analyses and plots the cells and their activation value for each
  module that has more cells activated than the cutoff specified (defaults
  to 100).
  """
  # ATTENTION: The following line is a temporary solution because currently
  # the several module analysis contains only looking at the cell points,
  # not the pre and post clusterisation values.
  cell_points = np.array(list(cell_points_dict.values()))

  # sorted_unique_global_indices  are the module numbers
  sorted_unique_global_indices = np.sort(np.unique(module_numbers))
  n_modules = sorted_unique_global_indices.size

  cells_per_module = []  # 2d of points: (x,y) for each cell for each module
  n_cells_per_module = []  # tracks the above, just gives the length
  activation_val_per_module = []

  for i in range(n_modules):
    global_idx = sorted_unique_global_indices[i]
    # now get indices where this is the global index
    global_idx_indices = np.where(module_numbers == global_idx)[0]

    cell_pts = cell_points[global_idx_indices]
    act_vals = activation_values[global_idx_indices]

    cells_per_module.append(cell_pts)
    n_cells_per_module.append(len(cell_pts))
    activation_val_per_module.append(act_vals)

  plt.hist(n_cells_per_module)
  plt.title("Number of cells per module with activation above 0.01")
  plt.yscale("log")
  plt.show()

  plt.plot(np.arange(len(n_cells_per_module)), n_cells_per_module)
  plt.title("Number of activated cells by module")
  plt.yscale("linear")
  plt.show()

  # print(sorted_unique_global_indices)
  plt.rcParams["figure.figsize"] = [20, 15]
  plt.rcParams["figure.autolayout"] = True
  for i, cells in enumerate(cells_per_module):
    if (len(cells) >= n_activated_cutoff):
      module_number = sorted_unique_global_indices[i]
      show_module_cells(cells, activation_val_per_module[i], module_number)


def analyse_single(activation_values, cell_points_dict, module_number, 
                   n_clusters, cell_cluster_numbers_dict):
  """
  This function is used for when only one module is considered. More detailed
  things will be analysed here, but no general pictures for all modules are shown.
  """
  cell_points_sorted = sort_dict_values_by_keys(cell_points_dict)
  cell_cluster_numbers_sorted = sort_dict_values_by_keys(cell_cluster_numbers_dict)
  # Now index i in both cell_cluster_numbers_sorted and cell_points_sorted
  # correspond to the same cell, so they can be used together

  show_module_cells(cell_points_sorted, activation_values, module_number,
                    cell_cluster_numbers=cell_cluster_numbers_sorted)
  



if __name__ == "__main__":
  # go into the main function for the functionality decisions
  # start by parsing command line arguments/flags
  args = parse_args()
  single_module_only = args.single_module
  path = args.path

  with open(path) as file:
    lines = file.readlines()

  (activation_values, cell_points_dict, module_numbers, thread_points,
   n_clusters, cell_cluster_numbers_dict) = read_all_lines(lines)

  if (single_module_only):
    # look at just one module
    module_number = module_numbers[0]  # they will all be identical
    assert(np.all(module_numbers == module_number))  # ensure identical

    analyse_single(activation_values, cell_points_dict, module_number,
                   n_clusters, cell_cluster_numbers_dict)

  else:
    analyse_several(activation_values, cell_points_dict, module_numbers)
