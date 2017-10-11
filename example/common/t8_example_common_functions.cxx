/*
  This file is part of t8code.
  t8code is a C library to manage a collection (a forest) of multiple
  connected adaptive space-trees of general element types in parallel.

  Copyright (C) 2015 the developers

  t8code is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  t8code is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with t8code; if not, write to the Free Software Foundation, Inc.,
  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
*/

/** \file t8_example_common_functions.cxx Provide real valued functions
  * that are used in more than one example. */

#include <example/common/t8_example_common.h>

T8_EXTERN_C_BEGIN ();

double
t8_constant_one (const double x[3], double t)
{
  return 1;
}

double
t8_constant_zero (const double x[3], double t)
{
  return 0;
}

double
t8_project_x (const double x[3], double t)
{
  return x[0];
}

double
t8_exp_distribution (const double x[3], double t)
{
  double              dummy, X;

  /* Get fractional part of t. t is thus periodically
   * mapped to the unit interval */
  t = modf (t, &dummy);
  X = x[0] - .5;
  return exp (-4 * X * X);
}

/* This function is =1 if 0.25 <= x <= 0.75 and 0 else */
double
t8_step_function (const double x[3], double t)
{
  return 0.25 <= x[0] && x[0] <= 0.75;
}

/* This function is =1 if 0.25 <= x <= 0.75,
 * it is 0 outside of 0.25-eps and 0.75+eps,
 * it interpolates linearly in between. */
double
t8_almost_step_function (const double x[3], double t)
{
  double              eps = 0.1;

  /* interpolate in [0.25-eps,0.25+eps] */
  if (0.25 - eps < x[0] && x[0] < 0.25) {
    return (x[0] - 0.25 + eps) / (eps);
  }
  /* interpolate in [0.75-eps,0.75+eps] */
  else if (0.75 < x[0] && x[0] < 0.75 + eps) {
    return 1 - (x[0] - 0.75) / (eps);
  }
  /* 1 inside [0.25,0.75], 0 outside */
  return 0.25 <= x[0] && x[0] <= 0.75;
}

double
t8_sinx (const double x[3], double t)
{
  return sin (2 * M_PI * x[0]) + 1;
}

double
t8_sinx_cosy (const double x[3], double t)
{
  return sin (2 * M_PI * x[0]) * cos (2 * M_PI * x[1]);
}

double
t8_sint (const double x[3], double t)
{
  return sin (2 * M_PI * t);
}

void
t8_constant_one_vec (const double x[3], double t, double x_out[3])
{
  x_out[0] = x_out[1] = x_out[2] = 1;
}

void
t8_constant_one_x_vec (const double x[3], double t, double x_out[3])
{
  x_out[0] = 1;
  x_out[1] = x_out[2] = 0;
}

void
t8_constant_one_xy_vec (const double x[3], double t, double x_out[3])
{
  x_out[0] = x_out[1] = 1;
  x_out[2] = 0;
}

T8_EXTERN_C_END ();
