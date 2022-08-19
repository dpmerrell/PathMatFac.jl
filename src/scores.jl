
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
    FN = TP_total
    R = 0.0
    ave_prec = 0.0

    for (i,yp) in enumerate(pred_srt)

        if yp != cur_threshold
            Rnew = TP / TP_total
            P = TP / (TP + FP)
            ave_prec += P*(Rnew - R)

            R = Rnew
            cur_threshold = yp 
        end

        # Increment/decrement the counts
        if true_srt[i]
            TP += 1
            FN -= 1
        else
            FP += 1
        end
    end
  
    Rnew = 1.0
    P = TP / (TP + FP)
    ave_prec += P*(Rnew - R) 
    
    return ave_prec
end




