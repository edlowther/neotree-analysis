source('./.Rprofile')
install.packages('renv')
renv::init()
install.packages('pacman')
pacman::p_load(dplyr, tidyr, readr, pROC)

df <- read_csv('./output/delong_df.csv')

lr_df <- df %>%
  filter(name == 'lr')

knn_df <- df %>%
  filter(name == 'knn')

lgb_df <- df %>%
  filter(name == 'lgb')

lr_roc <- roc(lr_df$y_test, lr_df$y_pred)
knn_roc <- roc(knn_df$y_test, knn_df$y_pred)
lgb_roc <- roc(lgb_df$y_test, lgb_df$y_pred)

roc.test(lr_roc, knn_roc)
roc.test(lr_roc, lgb_roc)
roc.test(knn_roc, lgb_roc)

ci.auc(lr_roc)
ci.auc(knn_roc)
ci.auc(lgb_roc)


full_df <- read_csv('./output/full_dataset_lgb_delong_df.csv') %>%
  filter(reduce_cardinality)

full_roc <- roc(full_df$y_test, full_df$y_pred)
ci.auc(full_roc)
