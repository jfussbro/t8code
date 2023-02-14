#include <t8.h>
#include <t8_cmesh.h>
#include <t8_cmesh/t8_cmesh_examples.h>
#include <t8_forest.h>
#include <t8_schemes/t8_default/t8_default_cxx.hxx>
#include <t8_vec.h>

static t8_cmesh_t
t8_build_pyracube_coarse_mesh (sc_MPI_Comm comm)
{
    t8_cmesh_t cmesh;

    cmesh = t8_cmesh_new_hypercube (T8_ECLASS_PYRAMID, comm, 0, 0, 0);
        
    return cmesh;
}

static t8_forest_t
t8_build_uniform_forest (sc_MPI_Comm comm, t8_cmesh_t cmesh, int level)
{
    t8_forest_t         forest;
    t8_scheme_cxx_t     *scheme;

    scheme = t8_scheme_new_default_cxx ();
    forest = t8_forest_new_uniform (cmesh, scheme, level, 0, comm);

    return forest;
}

struct t8_adapt_data
{
    double          midpoint[3];
    double          refine_if_inside_radius;
    double          coarsen_if_outside_radius;
};

int
t8_adapt_callback   (t8_forest_t forest,
                     t8_forest_t forest_from,
                     t8_locidx_t which_tree,
                     t8_locidx_t lelement_id,
                     t8_eclass_scheme_c *ts,
                     const int is_family,
                     const int num_elements,
                     t8_element_t *elements[])
{
    double          centroid[3];
    const struct t8_adapt_data *adapt_data =
        (const struct t8_adapt_data *) t8_forest_get_user_data (forest);
    double          dist;
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

t8_forest_t
t8_adapt_forest (t8_forest_t forest)
{
    t8_forest_t             forest_adapt;
    struct t8_adapt_data    adapt_data = {
        {1, 1, 1},
        0.3,
        0.6
    };
    T8_ASSERT (t8_forest_is_committed (forest));
    forest_adapt =
        	t8_forest_new_adapt (forest, t8_adapt_callback, 0, 0, &adapt_data);
    return forest_adapt;
}

static t8_forest_t
t8_partition_balance_ghost (t8_forest_t forest)
{
    t8_forest_t         new_forest;
    T8_ASSERT (t8_forest_is_committed (forest));
    t8_forest_init (&new_forest);
    t8_forest_set_partition (new_forest, forest, 0);
    t8_forest_set_balance (new_forest, forest, 0);
    t8_forest_set_ghost (new_forest, 1, T8_GHOST_FACES);
    t8_forest_commit (new_forest);

    return new_forest;
}

static void
t8_write_forest_vtk (t8_forest_t forest, const char *prefix_forest)
{
    t8_forest_write_vtk (forest, prefix_forest);
}

static void
t8_destroy_forest (t8_forest_t forest)
{
    t8_forest_unref (&forest);
}

void
t8_print_forest_information (t8_forest_t forest)
{
    t8_locidx_t         local_num_elements;
    t8_gloidx_t         global_num_elements;

    T8_ASSERT (t8_forest_is_committed (forest));
    local_num_elements = t8_forest_get_local_num_elements(forest);
    global_num_elements = t8_forest_get_global_num_elements (forest);
    t8_global_productionf ("Local number of elements:\t\t%i\n",
                           local_num_elements);
    t8_global_productionf ("Global number of elements:\t%li\n",
                           global_num_elements);
}

int
main (int argc, char **argv)
{
    int             mpiret;
    sc_MPI_Comm     comm;
    t8_cmesh_t      cmesh;
    t8_forest_t     forest;
    const char      *prefix_uniform = "t8_practice_file_uniform_forest";
    const char      *prefix_adapt = "t8_practice_file_adapted_forest";
    const char      *prefix_partition_balance_ghost = "t8_partitioned_balance_ghost_forest";
    const int       level = 5;

    mpiret = sc_MPI_Init (&argc, &argv);
    SC_CHECK_MPI (mpiret);

    sc_init (sc_MPI_COMM_WORLD, 1, 1, NULL, SC_LP_ESSENTIAL);
    t8_init (SC_LP_PRODUCTION);
    comm = sc_MPI_COMM_WORLD;

    cmesh = t8_build_pyracube_coarse_mesh (comm);
    t8_global_productionf ("Created coarse mesh (pyracube).\n");
    
    forest = t8_forest_new_uniform (cmesh, t8_scheme_new_default_cxx (), level, 0, comm);
    t8_global_productionf ("Created uniform forest out of cmesh.\n");
    t8_global_productionf ("Refinement level:\t\t\t%i\n", level);
    t8_print_forest_information (forest);
    t8_forest_write_vtk (forest, prefix_uniform);
    t8_global_productionf ("Wrote uniform forest to vtu files:\t%s*\n",
                           prefix_uniform);

    forest = t8_adapt_forest (forest);
    t8_global_productionf ("Adapted forest.\n");
    t8_print_forest_information (forest);
    t8_write_forest_vtk (forest, prefix_adapt);
    t8_global_productionf ("Wrote adapted forest to vtu files:\t%s*\n",
                           prefix_adapt);

    t8_global_productionf ("Repartitioning and balancing this forest and creating a ghost layer.\n");
    forest = t8_partition_balance_ghost (forest);
    t8_global_productionf ("Repartitioned and balanced forest and build ghost layer.\n");
    t8_print_forest_information(forest);
    t8_forest_write_vtk (forest, prefix_partition_balance_ghost);

    t8_forest_unref (&forest);
    t8_global_productionf ("Destroyed forest.\n");

    sc_finalize ();

    mpiret = sc_MPI_Finalize ();
    SC_CHECK_MPI (mpiret);

    return 0;
}
