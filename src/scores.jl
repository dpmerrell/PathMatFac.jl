
"""
    average_precision(y_pred, y_true)

Compute the average precision binary classification score
for a vector of predictions (y_pred), given the 
ground truth (y_true). The entries of y_pred may take arbitrary
numeric values; higher values indicate higher "confidence"
in a positive classification.
"""
function average_precision(y_pred::AbstractVector, y_true::AbstractVector)

    srt_idx = argsort(y_pred)
    pred_srt = y_pred[srt_idx]
    true_srt = y_true[srt_idx]

    TP = sum(y_true)
    FP = length(y_true) - TP
    FN = 0
    Rminus = 1.0
    ave_prec = 0.0
    for i=1:length(y_true)
        if true_srt[i]
            FN += 1
            TP -= 1
        else
            FP -= 1
        end
        R = TP/(TP + FN)
        prec = TP/(TP + FP)
        ave_prec += (Rminus - R)*prec
        Rminus = R
    end
    return ave_prec
end





