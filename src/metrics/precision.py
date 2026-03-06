def calculate_precision(pred, target):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    TP = (pred_flat * target_flat).sum()
    FP = (pred_flat * (1 - target_flat)).sum()
    
    precision = TP / (TP + FP + 1e-6)
    
    return precision.item()