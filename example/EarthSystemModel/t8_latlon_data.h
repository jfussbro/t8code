/*
  This file is part of t8code.
  t8code is a C library to manage a collection (a forest) of multiple
  connected adaptive space-trees of general element classes in parallel.

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

/** file t8_latlon_data.h
 */

#ifndef T8_LATLON_DATA_H
#define T8_LATLON_DATA_H

#include <t8.h>
#include <t8_forest.h>

/* If we associate data on an X x Y subgrid, it can be 
 * sorted in memory to match the gridcells in different ways.
 */
typedef enum
{
  T8_LATLON_DATA_XSTRIPE,       /* Row-wise storage data[y * X + x] gives data of (x,y). */
  T8_LATLON_DATA_YSTRIPE,       /* Column-wise storage data[x * Y + y] gives data of (x,y). */
  T8_LATLON_DATA_MORTON         /* Morton SFC storage. The data is sorted according to the 
                                 * Morton SFC index in the surrounding quad forest (not the subgrid). */
} T8_LATLON_DATA_NUMBERING;

/* Describes an X by Y subchunk of the grid with
 * data on it. */
typedef struct
{
  t8_locidx_t         x_start;  /* Starting x coordinate. */
  t8_locidx_t         y_start;  /* Starting y coordinate. */
  t8_locidx_t         x_length; /* Number of subgrid cells in x dimension. */
  t8_locidx_t         y_length; /* Number of subgrid cells in y dimension. */
  int                 dimension;        /* Dimensionality of the data (1, 2, 3). */
  int                 level;    /* The smallest uniform refinement level of a forest that can have the
                                 * grid (not the subgrid) as submesh. */
  T8_LATLON_DATA_NUMBERING numbering;   /* Numbering scheme of data. */
  double             *data;     /* x_length x y_length x dimension many data items.
                                   for each data item dimension many entries. */
  t8_linearidx_t     *data_ids; /* If numbering is T8_LATLON_DATA_MORTON then for
                                   each data item (x_lenght x y_length) its element's 
                                   Morton index. */
  const char         *description;      /* The name of this dataset. */
} t8_latlon_data_chunk_t;

T8_EXTERN_C_BEGIN ();

/* function declarations */

/* Given x and y coordinates in an X by Y grid compute
 * the Morton linear id according to a given level of the 
 * element associated with (x,y).
 */
t8_linearidx_t
     t8_latlon_to_linear_id (t8_gloidx_t x, t8_gloidx_t y, int level);

void                t8_latlon_data_test (t8_locidx_t x_start,
                                         t8_locidx_t y_start,
                                         t8_locidx_t x_length,
                                         t8_locidx_t y_length, int dimension,
                                         int level,
                                         T8_LATLON_DATA_NUMBERING numbering,
                                         t8_gloidx_t x_length_global,
                                         t8_gloidx_t y_length_global);

T8_EXTERN_C_END ();

#endif /* !T8_LATLON_DATA_H */
