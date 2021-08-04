import numpy
import pandas as pd
from pycaret.classification import *

# Data Parse
Fishdata = np.load("Fishdata.npy", allow_pickle=True).item()
pandaData = pd.concat([pd.DataFrame(Fishdata['datas']), pd.DataFrame(data=Fishdata['labels'], columns=['labels'])], axis=1)
test1 = setup(pandaData, target='labels', train_size = 0.7, log_experiment=True)
# best_models = compare_models(n_select=3)
gbm = create_model('lightgbm')
tuned_gbm = tuned_model(gbm)
final_gbm = finalize_model(tuned_gbm)
fish = predict_model(final_gbm)

from pycaret.utils import check_metric
check_metric(metric='Accuracy')
