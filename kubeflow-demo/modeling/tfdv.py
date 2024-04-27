
import tensorflow_data_validation as tfdv

stats = tfdv.generate_statistics_from_dataframe(data, tfdv.StatsOptions(label_feature='price'))
tfdv.visualize_statistics(stats)
