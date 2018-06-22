from fitness_function import*
from input_output import*
from teste import*

pset = init_pset(75)
ind = txt_to_individual('EXPR_ROCGP_EEG_K7__1.txt', pset)

t, tt, x1, y1, ttt = import_data('data/wav_all_seg_ex1.csv', 0, 0)
t, tt, x2, y2, ttt = import_data('data/wav_all_seg_ex2.csv', 0, 0)
pred1 = knn_feature_selection(ind, 7, x1, y1, x1, pset)
pred2 = knn_feature_selection(ind, 7, x1, y1, x1, pset)