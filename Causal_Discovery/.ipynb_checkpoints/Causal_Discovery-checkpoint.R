

library(pcalg)
library(Rgraphviz)

# Select Dataset: 
# Bank dataset
data_path <-'./subjects/datasets/Bank' 
# Bank dataset -> data_path <-'./subjects/datasets/Bank' 
# Adult dataset -> data_path <-'./subjects/datasets/adult_org-Copy1.csv' 
# Compas dataset -> data_path <-'./subjects/datasets/compas-Copy1' 
# Bank dataset -> data_path <-'./subjects/datasets/Bank' 
# Bank dataset -> data_path <-'./subjects/datasets/Bank' 

data <- read.csv(data_path, stringsAsFactors = TRUE)
suffStat <- list(dm = data, adaptDF = FALSE)
pc_fit <- pc(suffStat, indepTest = disCItest, alpha = 0.05, labels = colnames(data))
ges_fit <- ges(data_numeric, score = "bde", labels = colnames(data))
simy_fit <- simy(data, alpha = 0.05, labels = colnames(data))
plot(pc_fit, main = "Causal Graph Using PC Algorithm")
plot(ges_fit, main = "Causal Graph Using GES Algorithm")
plot(simy_fit, main = "Causal Graph Using SIMY Algorithm")
pc_matrix <- as(pc_fit@graph, "matrix")
pc_df <- as.data.frame(pc_matrix)
write.csv(pc_df, "./Bank_Analysis\pc\DAGs\Bank_pc.csv", row.names = TRUE)

ges_matrix <- as(ges_fit$essgraph@graph, "matrix")
ges_df <- as.data.frame(ges_matrix)
write.csv(pc_df, "./Bank_Analysis\ges\DAGs\Bank_ges.csv", row.names = TRUE)

simy_matrix <- as(simy_fit@graph, "matrix")
simy_df <- as.data.frame(simy_matrix)
write.csv(simy_df, "./Bank_Analysis\simy\DAGs\Bank_simy.csv", row.names = TRUE)
