

import sys
import numpy
import matplotlib . pyplot as pyplot

import tensorflow


# ------------ cmd line args ---------------------------------------------------

nb_args = len (sys . argv)

if (nb_args == 1):
  variables = [  "PovPct", ]
  target = "Brth15to17"
elif (nb_args == 2):
  if (sys . argv [1] == "--help"):
    print ("ARGS: target variable_1 ... variable_N")
    sys . exit (0)
  variables = [  "PovPct", ]
  target = sys . argv [1]
else:
  target = sys . argv [1]
  variables = sys . argv [ 2 : ]


# ------------ params ----------------------------------------------------------

#variables = [  "PovPct", "ViolCrime" ]


N = len (variables)

# ------------ reading the data -------------------------------------------------

data_file = open ("./poverty.txt", "r")

rows = []

for line in data_file:
  if (line [-1] == "\n"):
    line = line [:-1]
  rows . append (line . split ("\t"))


row_names = rows [0]
rows = rows [1:]

m = len (rows)

variables_indexes = [ row_names . index (var)  for var in variables ]
target_index = row_names . index (target)

#print (variables_indexes)
#print (row_names)
#print (len (rows))
#print (target_index)
#print (rows [0])

# No will automatically fill the last column with ones
A_data = numpy . ones (( m, N + 1), dtype = "float32")
b_data = numpy . zeros (( m, 1), dtype = "float32")

#print ("b_data="+str(b_data))

for j in range (m):
  #print (j)
  #print (rows [j])
  b_data [ j, 0 ] = float (rows [ j ] [ target_index ])
  for v in range (N):
    A_data [ j, v ] = float (rows [ j ] [ variables_indexes [ v ] ])

#print (b_feed)
#print (A_feed)

#raise Exception ("stop")

# --------------- creating the graph ---------------------------------------------
A = tensorflow . placeholder (shape = (None, None), dtype = "float32")
b = tensorflow . placeholder (shape = (None, 1), dtype = "float32")


A_transposed = tensorflow . transpose (A)
A_transposed_times_A = tensorflow . matmul (A_transposed, A)
rhs = tensorflow . matmul (A_transposed, b)

# likely to work - might be slower given the small number of variables
chol_dec = tensorflow . cholesky (A_transposed_times_A)
x = tensorflow . cholesky_solve (chol_dec, rhs)

input_feed_dict = {
  A : A_data,
  b : b_data,
}



# --------- solving ---------------------------------------------------------------

tf_session = tensorflow . Session ()
tf_session . run (tensorflow . global_variables_initializer ())

try:
  solutions = tf_session . run (x, feed_dict = input_feed_dict)
except (InvalidArgumentError):
  # trying again with the general solver in case this is an issue with using cholesky
  print ("🏳  Second attempt...")
  x = tensorflow . solve (A_transposed_times_A, rhs)
  # variables have already been initialized
  solutions = tf_session . run (x, feed_dict = input_feed_dict)

coefficients = [ s [0] for s in solutions ]

#print (solutions)
#print (coefficients)


equation_text = ""
coefficients_text = ""
for variable_index in range (len (variables)):
  coeff_name = "a_" + str (variable_index)
  equation_text += coeff_name + " * " + variables [variable_index] + " + "
  coefficients_text += "  " + coeff_name + " = " + str (coefficients [variable_index]) + ",\n"

equation_text += "b,\n"
coefficients_text += "  b = " + str (coefficients [-1]) + ".\n"

print ("Equation:\n  " + target + " = " + equation_text + "with:\n" + coefficients_text)



if (len (variables) == 1):
  x_max_range = 1.15 * numpy . amax (A_data)
  x_min_range = 0.85 * numpy . amin (A_data)
  x_plot = numpy . arange (x_min_range, x_max_range, 0.0088 * x_max_range)
  y_plot = coefficients [0] * x_plot + coefficients [1]
  pyplot . plot (A_data [ :, 0 ], b_data [ :, 0 ], "o",  x_plot, y_plot, "-")
  pyplot . show ()


