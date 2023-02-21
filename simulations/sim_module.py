import numpy as np
import argparse

def parse_args():
  """
  Go through all possible arguments to the programme and returns the args object
  """
  descr = "Simulation of a 2D Ising Model. First Checkpoint of MVP at UoE."
  parser = argparse.ArgumentParser(description=descr)
  parser.add_argument("-s", "--size", default=10)
  parser.add_argument("-p", "--cell_probability", default=0.3)
  args = parser.parse_args()

  return args


class CellGrid:
  
  def __init__(self, size, cell_probability):
    self.size=size
    cell_array = np.zeros(size**2, dtype=np.int32)
    self.n_activated_cells = int(np.floor(cell_probability * size**2))
    
    # now pick n sites at random to set to 1
    cell_array[0:self.n_activated_cells] = 1
    labels_array = cell_array.copy()
    # give each one a unique label
    labels_array[0:self.n_activated_cells] = np.arange(1,self.n_activated_cells+1)
    permutation = np.random.permutation(size**2)

    shuffled_cell_array = cell_array[permutation]
    shuffled_labels_array = labels_array[permutation]

    # the grid stores the position of the cells
    self.grid = shuffled_cell_array.reshape((size, size))
    self.labels = shuffled_labels_array.reshape((size, size))

    # there is an easier way to do this, but get all positions
    # in the intuitive readable way.
    activated_cell_positions = []
    for i in range(size):
      for j in range(size):
        if (self.grid[i][j] == 1):
          activated_cell_positions.append([i,j])

    self.activated_cell_positions = np.array(activated_cell_positions)
    np.random.shuffle(self.activated_cell_positions)  # arbitrary order
    self.current_iteration = 0  # for propagation

    self.print_grid()
    self.print_labels()


  def print_grid(self):
    full_str = "Cell Grid:\n"
    full_str += "_" * self.size*3 + "\n"
    for row in range(self.size):
      full_str += "|"
      for col in range(self.size):
        if (self.grid[row][col] != 0):
          full_str += " X "
        else:
          full_str += " O "

      full_str += "\n"  # start next row

    print(full_str)
    
    
  def print_labels(self):
    full_str = "Cell Labels:\n"
    full_str += "_" * self.size*3 + "\n"
    for row in range(self.size):
      full_str += "|"
      for col in range(self.size):
        label = self.labels[row][col]
        if (label >= 100):
          full_str += str(label)
        elif (label >= 10):
          full_str += str(label) + " "
        elif (label > 0):
          full_str += " " + str(label) + " "
        else:
          full_str += " O "

      full_str += "\n"  # start next row

    print(full_str)
    
    
  def propagate_state(self, cell_point):
    row = cell_point[0]
    col = cell_point[1]
    curr_label = self.labels[row][col]


    if (row == 0):
      if (col == 0):
        return  # when in top corner, do nothing
      else:
        left_label = self.labels[row][col-1]
        if (left_label != 0):
          # can have max one
          # always overwrite the bigger one
          if (left_label > curr_label):
            self.labels[row][col-1] = curr_label
          else:
            self.labels[row][col] = left_label
    
    elif (col == 0):
      # top left is already covered
      above_label = self.labels[row-1][col]
      if (above_label > 0):
        # don't do anything with diagonal
        if (above_label > curr_label):
          self.labels[row-1][col] = curr_label
        else:
          self.labels[row][col] = above_label

      else:  # above is not a cell, do diagonal
        dia_right_label = self.labels[row-1][col+1]
        if (dia_right_label > 0):
          if (dia_right_label > curr_label):
            self.labels[row-1][col+1] = curr_label
          else:
            self.labels[row][col] = dia_right_label

        else:
          return  # nothing above or diagonally above

    elif (col == (size-1)):  # are on the right edge
      # top is already covered
      # do the same as for col==0 but with left diag instead
      # of the right diag
      above_label = self.labels[row-1][col]
      if (above_label > 0):
        # don't do anything with diagonal
        if (above_label > curr_label):
          self.labels[row-1][col] = curr_label
        else:
          self.labels[row][col] = above_label

      else:  # above is not a cell, do diagonal
        dia_left_label = self.labels[row-1][col-1]
        if (dia_left_label > 0):
          if (dia_left_label > curr_label):
            self.labels[row-1][col-1] = curr_label
          else:
            self.labels[row][col] = dia_left_label

        else:
          return  # nothing above or diagonally above

    else:  # we are in the body of the grid, can take all 4 neighbours
      above_label = self.labels[row-1][col]
      left_label = self.labels[row][col-1]
      if (above_label * left_label > 0):  # above and left
        min_label = min(above_label, left_label, curr_label)
        self.labels[row-1][col] = min_label
        self.labels[row][col-1] = min_label
        self.labels[row][col] = min_label

      elif (above_label > 0):  # only one above
        if (above_label > curr_label):
          # set above to curr
          self.labels[row-1][col] = curr_label
        else:
          # set curr to above
          self.labels[row][col] = above_label

      elif (left_label > 0):  # only one left
          if (left_label > curr_label):
            # set left to curr
            self.labels[row][col-1] = curr_label
          else:
            # set curr to left
            self.labels[row][col] = left_label

      else:
        # check if anything on the diagonals
        dia_right_label = self.labels[row-1][col+1]
        dia_left_label = self.labels[row-1][col-1]
        if (dia_right_label * dia_left_label > 0):  # dia_right and dia_left
          min_label = min(dia_right_label, dia_left_label, curr_label)
          # set both diagonals and current to the smallest one
          self.labels[row-1][col+1] = min_label
          self.labels[row-1][col-1] = min_label
          self.labels[row][col] = min_label

        elif (dia_right_label > 0):  # only one: dia_right
          if (dia_right_label > curr_label):
            self.labels[row-1][col+1] = curr_label
          else:
            self.labels[row][col] = dia_right_label

        elif (dia_left_label > 0):  # only one: dia_left
            if (dia_left_label > curr_label):
              self.labels[row-1][col-1] = curr_label
            else:
              self.labels[row][col] = dia_left_label

        else:
          return  # if nothing on the diagonals either, do nothing


  def run(self):
    for i in range(self.n_activated_cells):
      cell_point = self.activated_cell_positions[self.current_iteration]
      print(f"Checking for point {cell_point}")
      self.propagate_state(cell_point)
      self.print_labels()
      self.current_iteration += 1  # move to next cell afterwards
      input()

if __name__ == "__main__":
  args = parse_args()
  size = int(args.size)
  prob_cells = float(args.cell_probability)

  cells = CellGrid(size, prob_cells)
  cells.run()

  