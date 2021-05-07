

function grad_descent_line_search!(x_view, grad_x_view, f, grad_f; alpha=1e-3, c1=1e-5, c2=0.9, 
                                                                   grow=1.5, shrink=0.5, max_iter=10)

    cur_val = f(x_view)
    cur_grad = grad_f(x_view)
    direction = - cur_grad ./ norm(cur_grad)
    cur_prod = sum(cur_grad .* direction)

    alpha_diff = alpha
    alpha_new = alpha

    for i=1:max_iter
        x_view .+= alpha_diff .* direction
        new_val = f(x_view)

        # First Wolfe condition
        if new_val <= cur_val + c1*alpha*cur_prod

            grad_x_view .= grad_f(x_view)
            new_prod = sum(grad_x_view .* direction)

            # Second Wolfe condition
            if -new_prod <= -c2*cur_prod
                # Success! Use this update.
                break
            else
                alpha_new = alpha*grow
                alpha_diff = alpha_new - alpha
                alpha = alpha_new
            end
        else
            alpha_new = alpha*shrink
            alpha_diff = alpha_new - alpha
            alpha = alpha_new
        end
    end
    return alpha
end


