#include <t8.h>
#include <t8_cmesh.h>
#include <t8_cmesh/t8_cmesh_examples.h>
#include <t8_forest.h>
#include <t8_schemes/t8_default/t8_default_cxx.hxx>
#include <t8_vec.h>

/* Create a hypercube cmesh out of 3 pryamid elements. */

static t8_cmesh_t
t8_build_pyracube_coarse_mesh (sc_MPI_Comm comm)
{
  t8_cmesh_t cmesh;

  cmesh = t8_cmesh_new_hypercube (T8_ECLASS_PYRAMID, comm, 0, 0, 0);

  return cmesh;
}

static double *
t8_create_element_data (t8_forest_t forest)
{
  t8_locidx_t num_local_elements;
  double *element_data;

  T8_ASSERT (t8_forest_is_committed(forest));

  num_local_elements = t8_forest_get_local_num_elements (forest);

  /* Allocate memory to store the data for all elements.
   * Fill the data array by iterating over every element in the tree and 
   * computing the volume for each one.
   */

  element_data = T8_ALLOC (double, num_local_elements);
  {
    t8_locidx_t itree, num_local_trees;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;
    const t8_element_t *element;

    num_local_trees = t8_forest_get_num_local_trees (forest);
    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree)
    {
      num_elements_in_tree = t8_forest_get_tree_num_elements (forest, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index)
      {
        element = t8_forest_get_element_in_tree (forest, itree, ielement);
        element_data[current_index] = t8_forest_element_volume(forest, itree, element);
      }
    }
  }
  return element_data;
}

/* Write the forest as vtu and also write the element's volume in the file. */

static void t8_output_data_to_vtu (t8_forest_t forest,
                                   double *data,
                                   const char *prefix)
{
  int num_data = 1;
  t8_vtk_data_field_t vtk_data;
  vtk_data.type = T8_VTK_SCALAR;
  strcpy (vtk_data.description, "Element volume");
  vtk_data.data = data;

  {
    int write_treeid = 1;
    int write_mpirank = 1;
    int write_level = 1;
    int write_element_id = 1;
    int write_ghosts = 0;
    t8_forest_write_vtk_ext (forest, prefix, write_treeid, write_mpirank,
                             write_level, write_element_id, write_ghosts,
                             0, 0, num_data, &vtk_data);
  }
}

int main (int argc, char **argv)
{
  int mpiret;
  sc_MPI_Comm comm;
  t8_cmesh_t cmesh;
  t8_forest_t forest;
  const char *prefix_forest_with_volumes = "t8_practice_file_forest_with_volumes";
  const int level = 0;
  double *data;

  mpiret = sc_MPI_Init (&argc, &argv);
  SC_CHECK_MPI (mpiret);

  sc_init (sc_MPI_COMM_WORLD, 1, 1, NULL, SC_LP_ESSENTIAL);
  t8_init (SC_LP_PRODUCTION);
  comm = sc_MPI_COMM_WORLD;

  cmesh = t8_build_pyracube_coarse_mesh (comm);
  t8_global_productionf ("Created coarse mesh (pyracube).\n");

  forest = t8_forest_new_uniform (cmesh, t8_scheme_new_default_cxx(), level, 0, comm);
  t8_global_productionf ("Created uniform forest out of cmesh.\n");
  t8_global_productionf ("Refinement level:\t\t\t%i\n", level);

  data = t8_create_element_data(forest);

  t8_global_productionf ("Computed level and volume data for local elements.\n");

  t8_output_data_to_vtu (forest, data, prefix_forest_with_volumes);
  t8_global_productionf ("Wrote forest and volume data to %s*.\n", prefix_forest_with_volumes);

  T8_FREE (data);

  t8_forest_unref (&forest);
  t8_global_productionf ("Destroyed forest.\n");

  sc_finalize ();

  mpiret = sc_MPI_Finalize ();
  SC_CHECK_MPI (mpiret);

  return 0;
}
