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

/** \file t8_cmesh_vtk_reader.hxx
* Header for the vtk-reader. 
*/

#ifndef T8_CMESH_VTK_READER
#define T8_CMESH_VTK_READER

#include <t8_cmesh.h>

#if T8_WITH_VTK
#include <vtkSmartPointer.h>
#include <vtkCellData.h>
#include <vtkDataSet.h>
#endif

T8_EXTERN_C_BEGIN ();

/**
 * Construct a cmesh given a filename.
 * The \a filename should point to file containing an unstructured Grid.
 * \warning: This is in production and this header will probably change! Update
 * as the function progresses
 * 
 * \param[in] filename      The name of the file 
 * \param[in] comm          The communicator used 
 * \return t8_cmesh_t       The cmesh described by the files or NULL if the cmesh could not be read.
 */
t8_cmesh_t          t8_cmesh_read_from_vtk_unstructured (const char *filename,
                                                         sc_MPI_Comm comm);

/**
 * Construct a cmesh given a filename.
 * The \a filename should point to file containing vtkPolyData. As polyData can consist of
 * polyhedra, the vtkTriangleFilter is used to break the cells down into triangles.
 * \warning: This is in production and this header will probably change! Update
 * as the function progresses
 * 
 * \param[in] filename      The name of the file 
 * \param[in] comm          The communicator used 
 * \return t8_cmesh_t       The cmesh described by the files or NULL, if t8code is not linked against vtk
 */
t8_cmesh_t          t8_cmesh_read_from_vtk_poly (const char *filename,
                                                 sc_MPI_Comm comm);

T8_EXTERN_C_END ();

#endif /* T8_CMESH_VTK_READER */
