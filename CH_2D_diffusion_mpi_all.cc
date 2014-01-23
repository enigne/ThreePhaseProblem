#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/work_stream.h>
#include "timer.cc"
#include <deal.II/base/parameter_handler.h>
 
#include <lac/full_matrix.h>
#include <lac/solver_bicgstab.h>
#include <lac/solver_cg.h>
#include <lac/solver_gmres.h>
#include <lac/constraint_matrix.h>
#include <lac/block_sparsity_pattern.h>
#include <lac/trilinos_block_vector.h>
#include <lac/trilinos_sparse_matrix.h>
#include <lac/trilinos_block_sparse_matrix.h>
#include <lac/trilinos_precondition.h>
#include <lac/trilinos_solver.h>

#include <lac/precondition.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>
#include <locale>
#include <string>
#include <algorithm> 

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/base/index_set.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>


using namespace dealii;
#define NUMBEROFPHASES 3
#define TOTALSPREAD_OFF

namespace EquationData
{
  const double Reno = 1.0;
  const double Cahn = 0.01;
  const double Pe = 100;  
  const double velocity[] = {0,0,0};
  const double alpha = 1; 
  const double Eps = 0.01;
  
#ifdef TOTALSPREAD_ON
  const double s_t[] = {3,1,1};// Total spread
  const double Lambda = 7;
#else
  const double s_t[] = {1,0.8,1.4}; // Partial spread
  const double Lambda = 0;
#endif
  const double sigma[] = { (s_t[1] + s_t[2]- s_t[0]),
  (s_t[0] + s_t[2]- s_t[1]),
  (s_t[0] + s_t[1]- s_t[2])};
  const double sigmaT =  3/(1/sigma[1]+1/sigma[2]+1/sigma[0]);
}

////////////////////////linear solver////////////////////////////////////
namespace LinearSolvers
{
  unsigned int cg_it;

class IdentityPreconditioner : public Subscriptor
  {
    public:
      IdentityPreconditioner ()
    {}

      void vmult (TrilinosWrappers::MPI::Vector &dst,
                  const TrilinosWrappers::MPI::Vector &src) const
    { dst=src;}
  };

//  template <class Matrix>
  template <class PreconditionerA>
  class InverseMatrix : public Subscriptor
  {
    public:
      InverseMatrix (const TrilinosWrappers::SparseMatrix &m,
                     const PreconditionerA &Apreconditioner); //Matrix &m);


      void vmult (TrilinosWrappers::MPI::Vector &dst,
                  const TrilinosWrappers::MPI::Vector &src) const;

    private:
      const SmartPointer<const TrilinosWrappers::SparseMatrix> matrix;
      const PreconditionerA  &a_preconditioner;
  };


//  template <class Matrix>
  template <class PreconditionerA>
  InverseMatrix<PreconditionerA>::
  InverseMatrix (const TrilinosWrappers::SparseMatrix &m,
                 const PreconditionerA &Apreconditioner)
                  :
                  matrix (&m),
                  a_preconditioner  (Apreconditioner)
  {}

//  template <class Matrix>
  template <class PreconditionerA>
  void
  InverseMatrix<PreconditionerA>::
  vmult (TrilinosWrappers::MPI::Vector &dst,
         const TrilinosWrappers::MPI::Vector &src) const
  {
     SolverControl solver_control1 (src.size(),
                                1e-3*src.l2_norm(),false,false);
     SolverCG<TrilinosWrappers::MPI::Vector> cg (solver_control1);

     dst.reinit(src);
     
     cg.solve (*matrix, dst, src, a_preconditioner);

     cg_it = cg_it + solver_control1.last_step();
  }

  template <class PreconditionerA>
  class BlockPreconditioner : public Subscriptor
  {
    public:
      BlockPreconditioner (const TrilinosWrappers::BlockSparseMatrix  &S,
                           const InverseMatrix<PreconditionerA> &AMGinv,
                           const double zep);

      void vmult (TrilinosWrappers::MPI::BlockVector &dst,
                  const TrilinosWrappers::MPI::BlockVector &src) const;

    private:
      const SmartPointer<const TrilinosWrappers::BlockSparseMatrix> system_matrix;
      const SmartPointer<const InverseMatrix<PreconditionerA>> AMG_inverse;
      const double zeps;
      mutable TrilinosWrappers::MPI::Vector tmp1, tmp2;
      mutable TrilinosWrappers::MPI::Vector g1, g2;
  };

  template <class PreconditionerA>
  BlockPreconditioner<PreconditionerA>::
  BlockPreconditioner(const TrilinosWrappers::BlockSparseMatrix  &S,
                      const InverseMatrix<PreconditionerA> &AMGinv,
                      const double zep)
                  :
                  system_matrix           (&S),
                  AMG_inverse             (&AMGinv),
                  zeps                    (zep),
                  g1                      (system_matrix->block(0,0).range_partitioner()),
                  g2                      (system_matrix->block(0,0).range_partitioner()),
                  tmp1                    (system_matrix->block(0,0).range_partitioner()),
                  tmp2                    (system_matrix->block(0,0).range_partitioner())

  {}

  template <class PreconditionerA>
  void BlockPreconditioner<PreconditionerA>::vmult (
    TrilinosWrappers::MPI::BlockVector &dst,
    const TrilinosWrappers::MPI::BlockVector &src) const
  {
    tmp1 = src.block(0);
    tmp1 *= zeps;
    tmp2 = tmp1;
    tmp2 += src.block(1);
    AMG_inverse->vmult (g1, tmp2);
    tmp2 = 0;
    system_matrix->block(0,0).vmult (tmp2, g1);
    tmp2 *= -1;
    tmp1 +=tmp2;
    AMG_inverse->vmult(g2, tmp1);
    dst.block(1) = g2;
    dst.block(1) *= -1;
    g2 *=(1/zeps);
    g1 *=(1/zeps);
    dst.block(0) = g2;
    dst.block(0) += g1; 
  }
}

//=============== Initial Conditions ==================

 template <int dim>
  class InitialValuesC1 : public Function<dim>
  {
  public:
    InitialValuesC1 () : Function<dim>() {}
    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };
  template <int dim>
  class InitialValuesC2 : public Function<dim>
  {
  public:
    InitialValuesC2 () : Function<dim>() {}
    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };
  
  template <int dim>
  double InitialValuesC1<dim>::value (const Point<dim>  &p,
                                     const unsigned int component) const
  {
    Assert (component == 0, ExcInternalError());
    double c1;
    c1 = 0.5*(1+std::tanh(2.0/EquationData::Eps*std::min(p.norm()-0.1,p[1])));

    return c1;
  }  

  template <int dim>
  double InitialValuesC2<dim>::value (const Point<dim>  &p,
                                     const unsigned int component) const
  {
    Assert (component == 0, ExcInternalError());
    double c2;
    c2 = 0.5*(1-std::tanh(2.0/EquationData::Eps*std::max(-p.norm()+0.1,p[1])));
   return c2;
  }


/////////////The MultiPhaseFlowProblem class///////////
template <int dim>
class MultiPhaseFlowProblem
{
  public:
    MultiPhaseFlowProblem ();
    void run (int n_refs);

  private:
    void make_grid_and_dofs ();
    void assemble_constant_matrix ();
    void assemble_system_replace ();
    void solve_replace();
    void output_results(const int) const;

    ConditionalOStream pcout;

    parallel::distributed::Triangulation<dim>   triangulation;
    FE_Q<dim>         fe;
    FESystem<dim>     system_fe;
    DoFHandler<dim>   dof_handler; //do we need both fe and system_fe and the two DoFHandlers??
    DoFHandler<dim>   system_dof_handler;
    ConstraintMatrix  matrix_constraints; //we might need that when calling some 
					//functions that take *MPI* arguments too
    
    unsigned int n_refinement_steps;

    double time_step;
    double beps, aeps, ceps, zeps;
    unsigned int timestep_number;
    unsigned int nonlin_it;
    unsigned int lin_it;

    TrilinosWrappers::BlockSparseMatrix system_matrix;

    TrilinosWrappers::MPI::BlockVector solution[NUMBEROFPHASES-1];
    TrilinosWrappers::MPI::BlockVector old_solution[NUMBEROFPHASES-1];
    TrilinosWrappers::MPI::BlockVector lin_solution[NUMBEROFPHASES-1];
    TrilinosWrappers::MPI::BlockVector system_rhs[NUMBEROFPHASES-1];
    
    TrilinosWrappers::SparseMatrix AMG_matrix;

    std_cxx1x::shared_ptr<TrilinosWrappers::PreconditionAMG>    Amg_preconditioner;

    TimerOutput computing_timer; //new member, have to put it to use as in step-32
};

/////////////////////make_grid_and_dofs////////////////////////////////
template <int dim>
 MultiPhaseFlowProblem<dim>::MultiPhaseFlowProblem ()
		:
		pcout (std::cout,
			(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) //new in the distributed version
			 == 0)),

		triangulation (MPI_COMM_WORLD,
				typename Triangulation<dim>::MeshSmoothing
				(Triangulation<dim>::smoothing_on_refinement |
				Triangulation<dim>::smoothing_on_coarsening)), //new in the distributed version
		fe (1),
                system_fe(FE_Q<dim>(1),1,
                           FE_Q<dim>(1),1),
                system_dof_handler(triangulation),
                dof_handler (triangulation),
		computing_timer (pcout,
				 TimerOutput::summary,
				 TimerOutput::wall_times)  //new in the distributed version	

 {}

template <int dim>
 void MultiPhaseFlowProblem<dim>::make_grid_and_dofs ()
 {
   GridGenerator::hyper_rectangle(triangulation, Point<dim>(-0.3,0.15), Point<dim>(0.3,-0.15));
   triangulation.refine_global (n_refinement_steps);
 
   dof_handler.distribute_dofs (fe);
   system_dof_handler.distribute_dofs (system_fe);
   DoFRenumbering::component_wise (system_dof_handler);
    
   const unsigned int n_u = dof_handler.n_dofs();
   const unsigned int n_s = system_dof_handler.n_dofs();

   pcout << "Number of active cells: "
             << triangulation.n_global_active_cells()
             << std::endl
             << "Number of degrees of freedom: "
             << n_u				
             << std::endl
             << "Number of degrees of freedom(system): "
             << n_s			
             << std::endl;

   std::vector<IndexSet> system_partitioning, system_relevant_partitioning;
   IndexSet system_index_set, system_relevant_set, unknowns_partitioning, unknowns_relevant_partitioning;
	//IndexSet unknowns_partitioning (n_u), unknowns_relevant_partitioning (n_u);

   system_index_set = system_dof_handler.locally_owned_dofs();
   system_partitioning.push_back(system_index_set.get_view(0,n_u));
   system_partitioning.push_back(system_index_set.get_view(n_u,2*n_u));
   
   DoFTools::extract_locally_relevant_dofs (system_dof_handler,
					    system_relevant_set);
   system_relevant_partitioning.push_back(system_relevant_set.get_view(0,n_u));
   system_relevant_partitioning.push_back(system_relevant_set.get_view(n_u,2*n_u));

   unknowns_partitioning = dof_handler.locally_owned_dofs();
   DoFTools::extract_locally_relevant_dofs (dof_handler,
					    unknowns_relevant_partitioning);

   matrix_constraints.clear ();
   matrix_constraints.reinit (system_relevant_set);
/* we might actually not need that at all in our case  
   DoFTools::make_hanging_node_constraints (dof_handler,  //or system_dof_handler??? in step-32 classes of
				// constraint objects are the same, as well as function calls to initialize them...
                                             matrix_constraints);
*/
   matrix_constraints.close ();

   TrilinosWrappers::BlockSparsityPattern sparsity_pattern (system_partitioning,
					   			MPI_COMM_WORLD);
   const unsigned int n_couplings = system_dof_handler.max_couplings_between_dofs();
 
   DoFTools::make_sparsity_pattern (system_dof_handler,
				     sparsity_pattern,
				     matrix_constraints, false,
				     Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
   sparsity_pattern.compress();

   TrilinosWrappers::SparsityPattern sp (unknowns_partitioning,
					 MPI_COMM_WORLD);
   DoFTools::make_sparsity_pattern (dof_handler, sp,
				     matrix_constraints, false,
				     Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
   sp.compress();

   system_matrix.reinit (sparsity_pattern);
   AMG_matrix.reinit (sp);  

   // Initialize vectors for solutions in nonlinear iterations
   for (int i=0; i < NUMBEROFPHASES-1; i++) {
   	system_rhs[i].reinit (system_partitioning, MPI_COMM_WORLD);
   	solution[i].reinit (system_relevant_partitioning, MPI_COMM_WORLD);
   	old_solution[i].reinit (system_rhs[i]);
   	lin_solution[i].reinit (system_rhs[i]);
   }
 }
 
/////////////////////assemble_constant_matrix(K,M,V)///////////////
template <int dim>
 void MultiPhaseFlowProblem<dim>::assemble_constant_matrix ()
 {
   system_matrix.block(0,0) = 0;
   system_matrix.block(1,0) = 0;
   system_matrix.block(1,1) = 0;
   AMG_matrix = 0;
 
   QGauss<dim>   quadrature_formula (2);
   FEValues<dim> fe_values (fe, quadrature_formula,
                                        update_values    | update_gradients |
                                        update_JxW_values);
 
   const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
   const unsigned int   n_q_points      = quadrature_formula.size();
 
   FullMatrix<double>   local_mass_matrix (dofs_per_cell, dofs_per_cell);
   FullMatrix<double>   local_stiffness_matrix (dofs_per_cell, dofs_per_cell);
   FullMatrix<double>   local_convection_matrix (dofs_per_cell, dofs_per_cell);
   FullMatrix<double>   local_AMG_matrix (dofs_per_cell, dofs_per_cell);
   FullMatrix<double>   local_D_matrix (dofs_per_cell, dofs_per_cell);
   FullMatrix<double>   local_11_matrix(dofs_per_cell, dofs_per_cell),
                        local_10_matrix(dofs_per_cell, dofs_per_cell),
                        local_01_matrix(dofs_per_cell, dofs_per_cell),
                        local_00_matrix(dofs_per_cell, dofs_per_cell);

   std::vector<unsigned int> local_dof_indices (dofs_per_cell);
   std::vector<unsigned int> local_system_dof_indices (system_fe.dofs_per_cell);
 
   std::vector<double>         phi       (dofs_per_cell);
   std::vector<Tensor<1,dim> > grad_phi  (dofs_per_cell);
   typename DoFHandler<dim>::active_cell_iterator
   cell = dof_handler.begin_active(),
   endc = dof_handler.end();
   
   for (; cell!=endc; ++cell)
     {
      if (cell->is_locally_owned())
      {	
       local_mass_matrix = 0;
       local_stiffness_matrix = 0;
       local_convection_matrix = 0; 
       local_11_matrix = 0;
       local_10_matrix = 0;
       local_00_matrix = 0;
       local_01_matrix = 0;
       local_AMG_matrix =0;

       fe_values.reinit (cell);
  
       for (unsigned int q=0; q<n_q_points; ++q)
         {
           for (unsigned int k=0; k<dofs_per_cell; ++k)
             {
               grad_phi[k] = fe_values.shape_grad (k,q);
               phi[k]      = fe_values.shape_value (k, q);
               
             }
 
           for (unsigned int i=0; i<dofs_per_cell; ++i)
             for (unsigned int j=0; j<dofs_per_cell; ++j)
               {
                 local_mass_matrix(i,j)
                   += (phi[i] * phi[j]
                       *
                       fe_values.JxW(q));
                 local_stiffness_matrix(i,j)
                   += (grad_phi[i] * grad_phi[j]
                       *
                       fe_values.JxW(q));
                 for (unsigned int d=0; d<dim; ++d)
                 local_convection_matrix(i,j)
                   += ((EquationData::velocity[d] * grad_phi[j][d]) 
                       * 
                       phi[i]
                       *
                       fe_values.JxW(q));
               }
         }
 
       cell->get_dof_indices (local_dof_indices);
  
       for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=0; j<dofs_per_cell; ++j)
        {
	  local_11_matrix(i, j) =  (1/time_step)*local_mass_matrix(i,j);
	  local_00_matrix(i, j) =  (1/time_step)*local_mass_matrix(i,j);
	  local_10_matrix(i, j) =  (1/EquationData::Pe)*local_stiffness_matrix(i,j);
          local_01_matrix(i, j) =  -aeps*local_stiffness_matrix(i,j);
          local_AMG_matrix(i,j) = (1/time_step)*local_mass_matrix(i,j)
                                  + beps*local_stiffness_matrix(i,j);
        }                            
 
       matrix_constraints.distribute_local_to_global (local_11_matrix,
                                                           local_dof_indices,
                                                           system_matrix.block(1,1));
       matrix_constraints.distribute_local_to_global (local_10_matrix,
                                                           local_dof_indices,
                                                           system_matrix.block(1,0));
       matrix_constraints.distribute_local_to_global (local_00_matrix,
                                                           local_dof_indices,
                                                           system_matrix.block(0,0));
       matrix_constraints.distribute_local_to_global (local_01_matrix,
                                                           local_dof_indices,
                                                           system_matrix.block(0,1));
       matrix_constraints.distribute_local_to_global (local_AMG_matrix,
                                                           local_dof_indices,
                                                         AMG_matrix);
       }
     }
     system_matrix.compress(VectorOperation::add);
     AMG_matrix.compress(VectorOperation::add);
 }

template <int dim>
 void MultiPhaseFlowProblem<dim>::assemble_system_replace ()
 {
   for (int i = 0; i < NUMBEROFPHASES - 1; i++) {
       system_rhs[i].block(0) = 0;
       system_rhs[i].block(1) = 0;
   }

   QGauss<dim>   quadrature_formula (2);
   FEValues<dim> fe_values (fe, quadrature_formula,
                                        update_values    | update_gradients |
                                        update_JxW_values);

   const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
   const unsigned int   n_q_points      = quadrature_formula.size();

   Vector<double>       local_rhs1 (dofs_per_cell);
   Vector<double>       local_rhs2 (dofs_per_cell);

   std::vector<unsigned int> local_dof_indices (dofs_per_cell);
   std::vector<double>       solution_values(dofs_per_cell);
  
   // temprary variables used in 3-phase
   // for nonlinear term
   std::vector<double>       solution_df1(dofs_per_cell);
   std::vector<double>       solution_df2(dofs_per_cell);
   std::vector<double>       solution_df3(dofs_per_cell);

   TrilinosWrappers::MPI::Vector tmp1 (system_rhs[0].block(0));
   TrilinosWrappers::MPI::Vector tmp2 (system_rhs[0].block(0));
   TrilinosWrappers::MPI::Vector tmp3 (system_rhs[0].block(0));
   double F1,F2;

   std::vector<double>         phi       (dofs_per_cell);
   std::vector<double>         phi_f       (dofs_per_cell);
   std::vector<Tensor<1,dim> > grad_phi  (dofs_per_cell);

   typename DoFHandler<dim>::active_cell_iterator
     cell = dof_handler.begin_active(),
     endc = dof_handler.end();

   for (; cell!=endc; ++cell)
     {
      if (cell->is_locally_owned())
      {
       local_rhs1 = 0;
       local_rhs2 = 0;

       fe_values.reinit (cell);

      cell->get_dof_indices (local_dof_indices);

       for (unsigned int k=0; k<dofs_per_cell; ++k)
       {
           // Compute components of nonlinear terms
           double c1,c2,c3;
           c1 = solution[0].block(1)(local_dof_indices[k]);
           c2 = solution[1].block(1)(local_dof_indices[k]);
           c3 = 1 - c1 - c2;
           solution_df1[k] = EquationData::sigma[0]*c1*(c2+c3)*(c2+c3) +
		 	     EquationData::sigma[1]*c2*c2*(c1+c3) + 
			     EquationData::sigma[2]*c3*c3*(c1+c2) +
                             6*EquationData::Lambda*c1*c2*c2*c3*c3;

           solution_df2[k] = EquationData::sigma[0]*c1*c1*(c2+c3) + 
			     EquationData::sigma[1]*c2*(c1+c3)*(c1+c3) + 
                             EquationData::sigma[2]*c3*c3*(c1+c2) +
                             6*EquationData::Lambda*c1*c1*c2*c3*c3;

           solution_df3[k] = EquationData::sigma[0]*c1*c1*(c2+c3) + 
                             EquationData::sigma[1]*c2*c2*(c1+c3) + 
			     EquationData::sigma[2]*c3*(c1+c2)*(c1+c2) +
                             6*EquationData::Lambda*c1*c1*c2*c2*c3;
          
       }
       for (unsigned int q=0; q<n_q_points; ++q)
         {
           F1 = 0.0;
           F2 = 0.0;
           for (unsigned int k=0; k<dofs_per_cell; ++k)
             {
               phi[k]      = fe_values.shape_value (k, q);
               grad_phi[k] = fe_values.shape_grad (k,q);

               F1 += (4*(EquationData::sigmaT/EquationData::sigma[0])*(1/EquationData::sigma[1]*(solution_df1[k]-solution_df2[k]) + 
				(1/EquationData::sigma[2])*(solution_df1[k]-solution_df3[k])))*phi[k];
               F2 += (4*(EquationData::sigmaT/EquationData::sigma[1])*(1/EquationData::sigma[0]*(solution_df2[k]-solution_df1[k]) + 
				(1/EquationData::sigma[2])*(solution_df2[k]-solution_df3[k])))*phi[k];

             }

           for (unsigned int i=0; i<dofs_per_cell; ++i)
           {

            local_rhs1(i) += (F1 * phi[i] * fe_values.JxW(q));
            local_rhs2(i) += (F2 * phi[i] * fe_values.JxW(q));
           }
          }

       matrix_constraints.distribute_local_to_global (local_rhs1,
                                                      local_dof_indices,
						      system_rhs[0].block(0));
       
       matrix_constraints.distribute_local_to_global (local_rhs2,
                                                      local_dof_indices,
						      system_rhs[1].block(0));
      }
   }

   // Update right hand sides
   for (int i = 0; i < NUMBEROFPHASES - 1; i ++) {
   system_rhs[i].block(0).compress(VectorOperation::add);

   TrilinosWrappers::MPI::BlockVector
	distributed_system_solution (system_rhs[i]);
      distributed_system_solution = solution[i];

   system_matrix.block(0,0).vmult(tmp1, distributed_system_solution.block(0));
   system_matrix.block(1,0).vmult(tmp2, distributed_system_solution.block(1));
   tmp2 *= -aeps*EquationData::Pe;
   system_rhs[i].block(0) *= -ceps;
   system_rhs[i].block(0) += tmp1;
   system_rhs[i].block(0) += tmp2;

   system_matrix.block(1,1).vmult(tmp1,distributed_system_solution.block(1));
   system_matrix.block(1,0).vmult(tmp2,distributed_system_solution.block(0));
   system_matrix.block(0,0).vmult(tmp3,old_solution[i].block(1));
   system_rhs[i].block(1) = tmp3;
   system_rhs[i].block(1) *= -1;
   system_rhs[i].block(1) += tmp2;
   system_rhs[i].block(1) += tmp1;
   }
 }


template <int dim>
void MultiPhaseFlowProblem<dim>::solve_replace ()
{
   double norm_crit  = 1e+3;
   double nonlin_eps = 1e-6;
   unsigned int nonlin_it = 0;
   //unsigned int lin_it = 0;
   LinearSolvers::cg_it=0;

   const LinearSolvers::InverseMatrix<TrilinosWrappers::PreconditionAMG>
   AMG_inverse (AMG_matrix,*Amg_preconditioner);

   const LinearSolvers::BlockPreconditioner<TrilinosWrappers::PreconditionAMG>
      preconditioner (system_matrix, AMG_inverse, zeps);

   while (norm_crit>nonlin_eps)
   {
       nonlin_it = nonlin_it +1;
       assemble_system_replace ();
       double temp_norm = 0.0;

       for (int i = 0; i < NUMBEROFPHASES-1; i++) {
       TrilinosWrappers::MPI::BlockVector  temp_sol(solution[i]);

       computing_timer.enter_section("Multiply preconditioner");
       preconditioner.vmult(lin_solution[i],  system_rhs[i]);
       computing_timer.exit_section("Multiply preconditioner");
       // Compute total errors for two phases
       temp_norm += lin_solution[i].l2_norm()*lin_solution[i].l2_norm();

        temp_sol = lin_solution[i];
        solution[i] -= temp_sol;
      }
      norm_crit = std::sqrt(temp_norm);
   }
   pcout << "   "
             << nonlin_it
             << " nonlinear iterations for system."
             << std::endl
             << "   "
             << "Average "
             << LinearSolvers::cg_it/nonlin_it/4
             << " CG iterations for system."
             << std::endl;
      
}

template <int dim>
void MultiPhaseFlowProblem<dim>::output_results (const int pI) const
{
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
{
   std::vector<std::string> solution_names;

   solution_names.push_back ("c"+std::to_string(pI));

   DataOut<dim> data_out;

   data_out.attach_dof_handler (system_dof_handler);
   data_out.add_data_vector (solution[pI-1].block(1), solution_names);

   data_out.build_patches ();

   std::ostringstream filename;
   filename << "solution-c" << pI << "-" << timestep_number << ".gpl";

   std::ofstream output (filename.str().c_str());
   data_out.write_gnuplot (output);
   output.close();
}
}

template <int dim>
void MultiPhaseFlowProblem<dim>::run (int n_refs)
{
  n_refinement_steps = n_refs;
  pcout << "Number of refinements: " <<n_refinement_steps<< std::endl;
  
  time_step = std::pow(0.5, double(n_refinement_steps))*0.15*0.5;
  pcout << "Mesh size: " << time_step<< std::endl;
  time_step = time_step/10.0;//*time_step;
  pcout << "Time step: " <<time_step<< std::endl;

  aeps = 0.75*(EquationData::Cahn*EquationData::Cahn)/time_step;
  beps = std::sqrt(aeps/EquationData::Pe);
  ceps = EquationData::alpha/time_step;
  zeps = beps/aeps;

  //computing_timer.enter_section("Setup dof systems");
  make_grid_and_dofs ();
  //computing_timer.exit_section();

  //computing_timer.enter_section("Assemble constant matrices");
  assemble_constant_matrix ();
  //computing_timer.exit_section();

  TrilinosWrappers::MPI::BlockVector  init_sol[NUMBEROFPHASES-1];

  for (int Index=0; Index < NUMBEROFPHASES-1; Index++) {
    // Set up the initial values 
    old_solution[Index].block(0)= 0.0;
    init_sol[Index].reinit(solution[Index]);
    typename DoFHandler<dim>::active_cell_iterator
       cell = dof_handler.begin_active(),
       endc = dof_handler.end();

     unsigned int dofs_per_cell = fe.dofs_per_cell;
     double c1,c2; 
     Vector<double> local_init (dofs_per_cell);
     std::vector<unsigned int> local_dof_indices (dofs_per_cell);
     Point<dim> p;

     for (; cell!=endc; ++cell)
       {
        if (cell->is_locally_owned())
        {	
  	     for(unsigned int i=0; i<dofs_per_cell;i++)
  	     { 
  	       p = cell->vertex(i);
               c1 = 0.5*(1+std::tanh(2.0/EquationData::Eps*std::min(p.norm()-0.1,p[1])));
               c2 = 0.5*(1-std::tanh(2.0/EquationData::Eps*std::max(-p.norm()+0.1,p[1])));
               if (p[0] == -0.3 || p[0] == 0.3 || p[1] == -0.15 || p[1] == 0.15) {
                   if ((p[0] == -0.3 || p[0] == 0.3) && (p[1] == -0.15 || p[1] == 0.15))
                       local_init(i) = 1;
                   else
                       local_init(i) = 0.5;
               }
               else 
  	           local_init(i)= 0.25;//((Index == 0) ? c1:c2);
               local_init(i) *= ((Index == 0)?c1:c2);
  	     }
  	     cell->get_dof_indices (local_dof_indices);
  	     matrix_constraints.distribute_local_to_global (local_init,
                    local_dof_indices, old_solution[Index].block(1));
        }
      }


    
    old_solution[Index].compress(VectorOperation::add);
    init_sol[Index] = old_solution[Index];
    solution[Index] = old_solution[Index];

  }

  double time;

  // Prepare AMG preconditioner
  Amg_preconditioner.reset (new TrilinosWrappers::PreconditionAMG());

  TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
  Amg_data.elliptic = true;
  Amg_data.higher_order_elements = true;
  Amg_data.smoother_sweeps = 2;
  Amg_data.aggregation_threshold = 0.02;

  Amg_preconditioner->initialize(AMG_matrix, Amg_data);

  int repeat = 1;
  // Start time loop
  pcout<<"BEGIN SECTION inexact Newton with direct replace"<<std::endl;
  repeat = 1;
  do{
    for (int Index=0; Index<NUMBEROFPHASES-1;Index++) {
      solution[Index] = init_sol[Index];
      old_solution[Index] = init_sol[Index];
    }
    pcout << "Repeat number " << repeat
          << std::endl;
    time = 0;
    timestep_number = 1;
    // Nonlinear iterations
    do
      {
	pcout << "Timestep " << timestep_number
	      << std::endl;
	
	computing_timer.enter_section("Solve nonlinear system");
	solve_replace ();
	computing_timer.exit_section("Solve nonlinear system");
        
	// Update solution from nonlinear method 
        for (int Index=0; Index<NUMBEROFPHASES-1;Index++) 
	   old_solution[Index] = solution[Index];

	time += time_step;
	++timestep_number;
	pcout << "   Now at t=" << time
	      << ", dt=" << time_step << '.'
	      << std::endl
	      << std::endl;

      }
    while (timestep_number <= 10);
    computing_timer.print_summary ();
    ++repeat;
  }
  while(repeat<=3);
//  output_results(1);
//  output_results(2);
}

int main (int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
  int n_refs;
  if (argc>=2)
    n_refs = atoi(argv[1]);
  else
    n_refs = 6;
  //std::cout<<n_refs<<std::endl;
  deallog.depth_console(0);
  MultiPhaseFlowProblem<2> Multi_Phase_Flow_Problem;
  Multi_Phase_Flow_Problem.run (n_refs);

  return 0;
}

