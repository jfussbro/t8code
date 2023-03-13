#include <cmath>
#include <t8.h>
#include <t8_vec.h>
#include <t8_cmesh.h>
#include <t8_cmesh/t8_cmesh_examples.h>
#include <t8_forest.h>
#include <t8_schemes/t8_default/t8_default_cxx.hxx>

static t8_cmesh_t
t8_build_pyracube_coarse_mesh (sc_MPI_Comm comm)
{
  t8_cmesh_t          cmesh;

  cmesh = t8_cmesh_new_hypercube (T8_ECLASS_PRISM, comm, 0, 0, 0);

  return cmesh;
}

struct t8_adapt_data
{
  double              midpoint[3];
  double              refine_if_inside_radius;
  double              coarsen_if_outside_radius;
};

int
t8_adapt_callback (t8_forest_t forest,
                   t8_forest_t forest_from,
                   t8_locidx_t which_tree,
                   t8_locidx_t lelement_id,
                   t8_eclass_scheme_c *ts,
                   const int is_family,
                   const int num_elements, t8_element_t *elements[])
{
  double              centroid[3];
  const struct t8_adapt_data *adapt_data =
    (const struct t8_adapt_data *) t8_forest_get_user_data (forest);
  double              dist;
  T8_ASSERT (adapt_data != NULL);
  t8_forest_element_centroid (forest_from, which_tree, elements[0], centroid);
  dist = t8_vec_dist (centroid, adapt_data->midpoint);
  if (dist < adapt_data->refine_if_inside_radius) {
    return 1;
  }
  else if (is_family && dist > adapt_data->coarsen_if_outside_radius) {
    return -1;
  }
  return 0;
}

static t8_forest_t
t8_adapt_partition_balance_ghost (t8_forest_t forest)
{
  t8_forest_t         new_forest;
  T8_ASSERT (t8_forest_is_committed (forest));
  struct t8_adapt_data adapt_data = {
    {1, 1, 1},
    0.3,
    0.6
  };
  t8_forest_init (&new_forest);
  t8_forest_set_user_data (new_forest, &adapt_data);
  t8_forest_set_adapt (new_forest, forest, t8_adapt_callback, 0);
  t8_forest_set_partition (new_forest, NULL, 0);
  t8_forest_set_balance (new_forest, NULL, 0);
  t8_forest_set_ghost (new_forest, 1, T8_GHOST_FACES);
  t8_forest_commit (new_forest);

  return new_forest;
}

void
t8_print_forest_information (t8_forest_t forest)
{
  t8_locidx_t         local_num_elements;
  t8_gloidx_t         global_num_elements;

  T8_ASSERT (t8_forest_is_committed (forest));
  local_num_elements = t8_forest_get_local_num_elements (forest);
  global_num_elements = t8_forest_get_global_num_elements (forest);
  t8_global_productionf ("Local number of elements:\t\t%i\n",
                         local_num_elements);
  t8_global_productionf ("Global number of elements:\t%li\n",
                         global_num_elements);
}

struct t8_data_per_element
{
  int                 level;
  double              volume;
};

static struct t8_data_per_element *
t8_create_element_data (t8_forest_t forest)
{
  T8_ASSERT (t8_forest_is_committed (forest));

  t8_locidx_t         num_local_elements =
    t8_forest_get_local_num_elements (forest);
  t8_locidx_t         num_ghost_elements = t8_forest_get_num_ghosts (forest);
  struct t8_data_per_element *element_data;

  element_data =
    T8_ALLOC (struct t8_data_per_element,
              num_local_elements + num_ghost_elements);
  {
    t8_locidx_t         itree, num_local_trees;
    t8_locidx_t         current_index;
    t8_locidx_t         ielement, num_elements_in_tree;
    t8_eclass_t         tree_class;
    t8_eclass_scheme_c *eclass_scheme;
    const t8_element_t *element;

    num_local_trees = t8_forest_get_num_local_trees (forest);
    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      tree_class = t8_forest_get_tree_class (forest, itree);
      eclass_scheme = t8_forest_get_eclass_scheme (forest, tree_class);
      num_elements_in_tree = t8_forest_get_tree_num_elements (forest, itree);
      for (ielement = 0; ielement < num_elements_in_tree;
           ++ielement, ++current_index) {
        element = t8_forest_get_element_in_tree (forest, itree, ielement);
        element_data[current_index].level =
          eclass_scheme->t8_element_level (element);
        element_data[current_index].volume =
          t8_forest_element_volume (forest, itree, element);
      }
    }
  }
  return element_data;
}

static void
t8_exchange_ghost_data (t8_forest_t forest, struct t8_data_per_element *data)
{
  sc_array           *sc_array_wrapper;
  t8_locidx_t         num_elements =
    t8_forest_get_local_num_elements (forest);
  t8_locidx_t         num_ghosts = t8_forest_get_num_ghosts (forest);

  sc_array_wrapper =
    sc_array_new_data (data, sizeof (struct t8_data_per_element),
                       num_elements + num_ghosts);

  t8_forest_ghost_exchange_data (forest, sc_array_wrapper);

  sc_array_destroy (sc_array_wrapper);
}

static void
t8_output_data_to_vtu (t8_forest_t forest, struct t8_data_per_element *data,
                       const char *prefix)
{
  t8_locidx_t         num_elements =
    t8_forest_get_local_num_elements (forest);
  t8_locidx_t         ielem;
  double             *element_volumes = T8_ALLOC (double, num_elements);
  int                 num_data = 1;
  t8_vtk_data_field_t vtk_data;
  vtk_data.type = T8_VTK_SCALAR;
  strcpy (vtk_data.description, "Element volume");
  vtk_data.data = element_volumes;

  for (ielem = 0; ielem < num_elements; ++ielem) {
    element_volumes[ielem] = data[ielem].volume;
  }
  int                 write_treeid = 1;
  int                 write_mpirank = 1;
  int                 write_level = 1;
  int                 write_element_id = 1;
  int                 write_ghosts = 0;
  t8_forest_write_vtk_ext (forest, prefix, write_treeid, write_mpirank,
                           write_level, write_element_id, write_ghosts,
                           0, 0, num_data, &vtk_data);
  T8_FREE (element_volumes);
}

int
main (int argc, char **argv)
{
  int                 mpiret;
  sc_MPI_Comm         comm;
  t8_cmesh_t          cmesh;
  t8_forest_t         forest;
  const char         *prefix_forest = "t8_practice_file_forest";
  const char         *prefix_forest_with_data =
    "t8_practice_file_forest_with_data";
  const int           level = 3;
  t8_data_per_element *data;

  mpiret = sc_MPI_Init (&argc, &argv);
  SC_CHECK_MPI (mpiret);

  sc_init (sc_MPI_COMM_WORLD, 1, 1, NULL, SC_LP_ESSENTIAL);
  t8_init (SC_LP_PRODUCTION);
  comm = sc_MPI_COMM_WORLD;

  cmesh = t8_build_pyracube_coarse_mesh (comm);
  t8_global_productionf ("Created coarse mesh (pyracube).\n");

  forest =
    t8_forest_new_uniform (cmesh, t8_scheme_new_default_cxx (), level, 0,
                           comm);
  t8_global_productionf ("Created uniform forest out of cmesh.\n");
  t8_global_productionf ("Refinement level:\t\t\t%i\n", level);
  t8_print_forest_information (forest);

  t8_global_productionf
    ("Adapting, repartitioning, balancing this forest and creating a ghost layer.\n");
  forest = t8_adapt_partition_balance_ghost (forest);
  t8_global_productionf
    ("Adapted, repartitioned, balanced forest and build ghost layer.\n");
  t8_print_forest_information (forest);
  t8_forest_write_vtk (forest, prefix_forest);
  t8_global_productionf
    ("Wrote adapted, pardtitioned and balanced forest with ghost elements to vtu files: %s*\n",
     prefix_forest);

  data = t8_create_element_data (forest);

  t8_global_productionf
    ("Computed level and volume data for local elements.\n");
  if (t8_forest_get_local_num_elements (forest) > 0) {
    t8_global_productionf ("Element 0 has level %i and volume %e. \n",
                           data[0].level, data[0].volume);
  }

  t8_exchange_ghost_data (forest, data);
  t8_global_productionf ("Exchanged ghost data.\n");

  if (t8_forest_get_num_ghosts (forest) > 0) {
    t8_locidx_t         first_ghost_index =
      t8_forest_get_local_num_elements (forest);
    t8_global_productionf ("Ghost 0 has level %i and volume %e.\n,",
                           data[first_ghost_index].level,
                           data[first_ghost_index].volume);
  }

  t8_output_data_to_vtu (forest, data, prefix_forest_with_data);
  t8_global_productionf ("Wrote forest and volume data to %s*.\n",
                         prefix_forest_with_data);

  T8_FREE (data);

  t8_forest_unref (&forest);
  t8_global_productionf ("Destroyed forest.\n");

  sc_finalize ();

  mpiret = sc_MPI_Finalize ();
  SC_CHECK_MPI (mpiret);

  return 0;
}
