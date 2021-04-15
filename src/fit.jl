


function fit!(model::MatFacModel, A::AbstractMatrix;
              max_iter::Int=100,
              lr::Float64=0.001, abs_tol::Float64=1e-6, rel_tol::Float64=1e-5)

    # Setup
    # initialize loss

    # While iter < max_iter and abs(loss - new_loss) > abs_tol and abs(loss - new_loss)/loss > rel_tol

        # Update X columns
        # Loop over instance colors
            # Loop over columns
                # Accumulate loss gradients
                # Add regularizer gradient
                # Update with gradient

        # Update Y columns 
        # Loop over feature colors
            # Loop over columns
                # Accumulate loss gradients
                # Add regularizer gradient
                # Update with gradient


end


