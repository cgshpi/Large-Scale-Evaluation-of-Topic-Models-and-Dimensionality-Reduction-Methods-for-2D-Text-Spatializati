# setwd('C:/Users/Daniel Atzberger/Documents/GitRepos/Benchmark-IEEE-Vis/Analysis_Visualization')

args <- commandArgs(trailingOnly = TRUE)
base_path <- args[1]

df <- read.csv(base_path)
save_path <- gsub(".csv", ".pdf", base_path)

library(ggplot2)

cols <- c("#a6cee3", "#1f78b4", "#b2df8a", "#33a02c",
          "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00")

ggplot(df, aes(x = x, y = y, color = factor(category))) +
  geom_point(size = 0.1) +
  # scale_color_manual(values = cols) +
  theme(panel.grid.major = element_blank(),
        panel.background = element_blank(),
        legend.position = "none",
        axis.ticks.length = unit(0, "pt"),
        axis.text.x = element_text(size = 0),
        axis.text.y = element_text(size = 0)) + 
  labs(x="", y="")

ggsave(filename = save_path, device = "pdf")

