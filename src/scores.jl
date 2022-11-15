
"""
    average_precision(y_pred, y_true)

Compute the average precision binary classification score
for a vector of predictions (y_pred), given the 
ground truth (y_true). The entries of y_pred may take arbitrary
numeric values; higher values indicate higher "confidence"
in a positive classification.
"""
function average_precision(y_pred::AbstractVector, y_true::AbstractVector)

    srt_idx = sortperm(y_pred; rev=true)
    pred_srt = y_pred[srt_idx]
    true_srt = y_true[srt_idx]

    cur_threshold = pred_srt[1] 
    TP_total = sum(y_true)
    TP = 0
    FP = 0
    R = 0.0
    ave_prec = 0.0

    # Iterate through thresholds (decreasing order)
    for (i,yp) in enumerate(pred_srt)

        # Update sums if we reach a new threshold
        if yp != cur_threshold
            Rnew = TP / TP_total
            P = TP / (TP + FP)
            ave_prec += P*(Rnew - R)

            R = Rnew
            cur_threshold = yp 
        end

        # Increment the counts
        if true_srt[i]
            TP += 1
        else
            FP += 1
        end
    end
  
    Rnew = 1.0
    P = TP / (TP + FP)
    ave_prec += P*(Rnew - R) 
    
    return ave_prec
end


function model_Y_average_precs(model::MultiomicModel)

    Y_pred = cpu(abs.(model.matfac.Y))
    Y_true = cpu(map( v->(!).(v), model.matfac.Y_reg.l1_reg.l1_idx))
    K = size(Y_pred, 1)
    pathway_av_precs = [average_precision(Y_pred[i,:], Y_true[i,:]) for i=1:K]

    return pathway_av_precs
end

