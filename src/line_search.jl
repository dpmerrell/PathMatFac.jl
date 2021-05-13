

function grad_descent_line_search!(x_view, grad_x_view, f, grad_f; 
                                   alpha=1e-3, c1=1e-5, c2=0.5, 
                                   grow=1.5, shrink=0.5, max_iter=10)

    cur_x = copy(x_view)

    cur_val = f(x_view)
    grad_x_view .= grad_f(x_view)
    direction = -copy(grad_x_view)
    println("DIRECTION: ", size(direction))
    cur_prod = sum(grad_x_view .* direction)

    new_val = Inf

    for i=1:max_iter
        x_view .= cur_x + alpha.*direction
        new_val = f(x_view)
        println("\talpha: ", alpha, "\tLoss: ", new_val)

        # First Wolfe condition
        if new_val <= cur_val + c1*alpha*cur_prod
            println("\t\tFirst Wolfe condition satisfied!")
            grad_x_view .= grad_f(x_view)
            new_prod = sum(grad_x_view .* direction)

            println("\t\tnew prod: ", new_prod, "\tcur prod: ", cur_prod)
            # Second Wolfe condition
            if new_prod >= c2*cur_prod
                println("\t\tSecond Wolfe condition satisfied!")
                # Success! Use this update.
                break
            else
                alpha *= grow
            end
        else
            alpha *= shrink
        end
    end

    # Note: it modifies x_view and grad_x_view inplace
    return alpha, new_val
end


